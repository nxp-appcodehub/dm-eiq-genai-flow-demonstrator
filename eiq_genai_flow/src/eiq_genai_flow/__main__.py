# Copyright 2023-2026 NXP
# NXP Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.

# =============================================================================
# IMPORTS AND ENVIRONMENT SETUP
# =============================================================================
import os
import threading
import random
import subprocess
import sys
import platform
import signal
import time
import logging
import typer
import warnings
import importlib.util
from colorama import Fore, Style
from eiq_genai_flow.config import Config
from eiq_genai_flow.gui.config import end_token, stop_token, vit_token
from eiq_genai_flow.adapters.event_manager import EventManager, EventType, Event
if importlib.util.find_spec("shared_utils"):
    from eiq_genai_flow.utils.argument_manager import ArgumentManager
    from shared_utils.utils import get_number_of_cores, setup_logging
    from eiq_genai_flow.utils.cpu_governor_manager import setup_cpu_governor, restore_cpu_governor
    from eiq_genai_flow.utils.utils import overwrite_config, get_soc_id, get_machine, get_revision, \
        suppress_stderr, skip
else:
    raise ModuleNotFoundError("Shared utils not available, please install nxp_eiq_shared_utils")


os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Suppress PyTorch ONNX registration warnings
warnings.filterwarnings("ignore", message=".*Symbolic function.*already registered.*")
warnings.filterwarnings("ignore", message=".*CUDA initialization.*")
warnings.filterwarnings("ignore", message=".*NVIDIA driver.*")

logging.getLogger("torio._extension.utils").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# =============================================================================
# MAIN PIPELINE CLASS
# =============================================================================


class eIQGenAIFlow:
    """
    Main pipeline class for eIQ GenAI Flow conversational AI system.

    Orchestrates the integration of VIT (Voice Intelligence Technology for wakeword support),
    STT (Speech To Text, aka ASR), RAG (Retrieval-Augmented Generation), LLM (Large Language Model),
    and TTS (Text-To-Speech) components.
    """

    def __init__(
        self,
        config,
        input_mode,
        capture_device,
        wake_word_model,
        gui_config_class,
        llm_model,
        use_rag,
        system_prompt,
        output_mode,
        playback_device,
        continuous,
        stt_model,
        use_neutron,
        use_voice_id,
        benchmark,
        verbose,
    ):
        """Initialize the eIQ GenAI Flow pipeline with all components."""
        # Store all parameters as instance attributes
        for key, value in locals().items():
            if key != "self":
                setattr(self, key, value)

        # Resolve the effective system prompt once: use the user-provided
        # value (-p/--system-prompt) and fall back to the config default.
        self.system_prompt = self.system_prompt or self.config.default_system_prompt

        # Register signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        # Initialize basic attributes
        self.stop_threads = False
        self.mq_to_c = None
        self.mq_from_c = None
        self.tts_mq_reader_thread = None
        self.stt_init_time = 0
        self.rag_init_time = 0
        self.llm_init_time = 0
        self.tts_init_time = 0

        # Only set benchmark_logger to None if not already set (by child class)
        if not hasattr(self, "benchmark_logger"):
            self.benchmark_logger = None

        self.llm_gui_dnpu_client = False

        # Initialize event manager before adapters
        self.event_manager = EventManager()
        self.event_manager.start()
        # self.event_manager.subscribe(EventType.LISTENING, self._on_listening)
        self.event_manager.subscribe(EventType.TIMEOUT, self._on_timeout)

        # Setup system configuration
        self.setup_system_config()

        # Setup audio manager
        self.setup_audio_manager()

        # Initialize all components
        self.initialize_components()

        # Start the system
        self.start_system()

    # =========================================================================
    # SYSTEM SETUP AND INITIALIZATION
    # =========================================================================

    def setup_system_config(self):
        """Setup system-specific configuration."""
        self.device = "PC" if platform.machine() == "x86_64" else get_soc_id()
        self.machine = "PC" if platform.machine() == "x86_64" else get_machine()
        self.revision = "PC" if platform.machine() == "x86_64" else get_revision()
        self.full_machine = "PC" if platform.machine() == "x86_64" else f"{self.machine} rev{self.revision}"

        self.device_name = "iMX"
        if "8" in self.device:
            self.device_name += "8"
        elif "9" in self.device:
            self.device_name += "9"

        self.thread_num = get_number_of_cores()

        if self.config.set_cpu_governor and self.device != "PC":
            setup_cpu_governor(self.config.cpu_governor)

        logger.info(f"Target Device: {self.device}")
        self.kasr_wake = self.input_mode == "kasr"

    def setup_audio_manager(self):
        """Setup centralized audio manager using factory pattern."""
        import os
        from audio_manager.audio_factory import create_audio_manager
        from audio_manager.audio_manager_base import CaptureConfig, PlaybackConfig

        # Determine backend from environment variable
        backend = os.getenv("AUDIO_BACKEND", "auto")  # "auto", "alsa", or "gstreamer"

        # =============================================================================
        # AUDIO CONFIGURATION
        # =============================================================================
        # For codecs with shared MCLK (e.g., WM896x):
        #   - sample_rate MUST match between capture/playback (shared clock)
        #   - playback channels MUST be 2 (hardware is stereo, channels=1 causes L/R issues)
        #   - format SHOULD match (reduces driver overhead)
        # =============================================================================

        # Create separate capture and playback configurations
        capture_config = CaptureConfig(
            capture_device=self.capture_device,
            sample_rate=16000,
            channels=2,
            format="S32LE",
            frame_duration_ms=30,
            buffer_duration_sec=10,
            save_audio=self.config.save_audio_capture,
            audio_save_path=self.config.audio_save_path,
            keep_device_open=self.config.keep_capture_device_open,
        )

        playback_config = PlaybackConfig(
            playback_device=self.playback_device,
            sample_rate=16000,
            channels=2,
            format="S32LE",
            frame_duration_ms=30,
            save_audio=self.config.save_audio_playback,
            audio_save_path=self.config.audio_save_path,
            keep_device_open=self.config.keep_playback_device_open,
        )

        # Create audio manager with auto-selected or specified backend
        self.audio_manager = create_audio_manager(
            backend=backend,
            capture_config=capture_config,
            playback_config=playback_config,
        )

        logger.debug(f"Audio Manager created with backend: {backend}")

    @suppress_stderr
    def initialize_components(self):
        """Initialize all system components based on configuration."""
        # GUI component
        if self.gui_config_class and not self.benchmark:
            self.gui_init()
        else:
            self.gui = None

        # Initialize earcon manager if playback is available and not in silent benchmark mode
        if self.playback_device and not (self.benchmark and self.config.silent_benchmark):
            self.earcon_init()
        else:
            self.earcon = skip()

        # STT (ASR) component
        if self.gui or "asr" in self.input_mode:
            self.vad_init()
            start_time = time.time()
            self.stt_init()
            self.stt_init_time = time.time() - start_time
        else:
            self.vad = skip()
            self.stt = skip()
            self.stt_init_time = 0

        # VIT component - Skip in KASR benchmark mode, but keep for VASR
        if self.gui or self.input_mode == "vasr":
            self.vit_init()
            self.keyboard = skip()

            # Voice ID component
            if self.use_voice_id:
                start_time = time.time()
                self.voice_id_init()
                self.voice_id_init_time = time.time() - start_time
            else:
                self.voice_id = skip()
                self.voice_id_init_time = 0

        else:
            self.vit = skip()
            self.voice_id = skip()
            self.keyboard_init()
            if self.use_voice_id:
                logger.warning(
                    f"Voice ID disabled: not compatible with '{self.input_mode}' input mode."
                )

        if self.input_mode == "keyb" or self.voice_id:
            if self.continuous:
                logger.warning("Continuous mode disabled: not compatible with voice-id or keyboard input mode.")
            self.continuous = False

        # RAG component
        if self.use_rag:
            start_time = time.time()
            self.rag_init()
            self.rag_init_time = time.time() - start_time
        else:
            self.retriever = None
            self.query_classifier = None
            self.rag_init_time = 0

        # LLM component
        if self.llm_model is not None:
            start_time = time.time()
            self.llm_init()
            self.llm_init_time = time.time() - start_time

            if self.use_neutron:
                if self.llm is None:
                    self.use_neutron = False
                    logger.info("Neutron acceleration is only available for LLMs, will use CPU only")
                elif "Neutron" not in str(self.llm.actual_providers):
                    self.use_neutron = False
                    logger.warning("Neutron not available on this platform, falling back to CPU")
        else:
            self.llm = None
            self.llm_init_time = 0

        # TTS component
        if "tts" in self.output_mode:
            start_time = time.time()
            # In benchmark mode with silent_benchmark enabled, use quiet mode
            quiet_mode = self.benchmark and self.config.silent_benchmark
            self.tts_init(quiet=quiet_mode, lava_test=True if self.benchmark else False)
            self.tts_init_time = time.time() - start_time
        else:
            self.tts = skip()
            self.tts_init_time = 0

        # Triggered when vad triggers
        self.wake_event = threading.Event()
        self.wake_type = None
        self.init_event = threading.Event()
        self.speaker_verification_event = threading.Event()
        self.speaker_verification_event_type = None

        self.end_of_input_event = threading.Event()
        self.input_text = ""

    def earcon_init(self):
        """Initialize earcon manager and load sounds based on config."""
        from eiq_genai_flow.adapters.earcon import EarconAdapter, EarconConfig
        import os

        # Get project root directory (where eiq_genai_flow.py is located)
        project_root = os.path.dirname(os.path.abspath(__file__))
        assets_path = os.path.join(project_root, "assets")

        logger.debug(f"Initializing Earcon with assets path: {assets_path}")

        earcon_config = EarconConfig()
        earcon_config.wakeword_earcon = self.config.play_wake_word_detect_sound
        earcon_config.tts_start_earcon = self.config.play_tts_start_sound
        earcon_config.intent_earcon = self.config.play_intent_detect_sound
        earcon_config.tts_filler_earcon = self.config.play_audio_filler

        self.earcon = EarconAdapter(audio_manager=self.audio_manager,
                                    event_manager=self.event_manager,
                                    assets_path=assets_path,
                                    config=earcon_config)

        logger.info("Earcon initialization complete")

    def start_system(self):
        """Start the system and play initial message."""

        if self.gui_config_class or "asr" in self.input_mode:
            # Start capture regardless of keep_device_open mode
            self.audio_manager.start_capture()
            logger.info("Audio Manager capture started")

        if self.gui:
            self.gui.start()
            self.gui.send_connect()

        with self.tts:
            self.tts_process(f"{self.config.tts_start_text}", eos=True)

            if self.gui:
                words = self.config.tts_start_text.split()
                self.gui.send_rsp(words[0])
                time.sleep(0.1)
                for word in words[1:]:
                    self.gui.send_rsp(" " + word)
                    time.sleep(0.1)
                self.gui.send_rsp(end_token)

            self.wait_tts()

    # =========================================================================
    # COMPONENT INITIALIZATION METHODS
    # =========================================================================

    def keyboard_init(self):
        logger.info("Keyboard initialization...")
        from eiq_genai_flow.adapters.keyboard import KeyboardAdapter
        self.keyboard = KeyboardAdapter(event_manager=self.event_manager, wake_only=self.kasr_wake)

    def gui_init(self):
        """Initialize GUI component."""
        logger.info(f"Loading {self.gui_config_class.__name__} GUI I/F module")
        from eiq_genai_flow.gui.generic_gui_interface import GenericGuiInterface
        self.gui = GenericGuiInterface(callback=self.gui_callback, user_config=self.gui_config_class)

    def vit_init(self):
        """Initialize VIT component."""
        logger.info("VIT initialization...")
        assert self.capture_device, "no audio audio capture device!"

        from eiq_genai_flow.adapters.vit import VITAdapter, VITConfig

        vit_config = VITConfig(
            wake_word_model=self.wake_word_model,
            operating_mode="wakeword",
            noise_floor=-80.0,
            noise_threshold=10.0,
            save_audio_vit=self.config.save_audio_vit,
            channel_indices=self.config.vit_channel_indices,
        )

        self.vit = VITAdapter(
            config=vit_config,
            audio_manager=self.audio_manager,
            event_manager=self.event_manager,
        )

    def rag_init(self):
        """Initialize RAG component."""
        logger.info("RAG initialization...")

        from retriever import QueryClassifier, Retriever, DEFAULT_DB_PATH, Config as RAGConfig

        rag_config = RAGConfig()

        rag_db_path = self.config.rag_db_path
        if not os.path.isfile(rag_db_path):
            logger.debug(f"RAG database not found: {rag_db_path}")
            rag_db_path = DEFAULT_DB_PATH
            logger.info(f"Using default database: {rag_db_path}")
        if self.benchmark:
            rag_db_path = DEFAULT_DB_PATH
            logger.info(f"Loading benchmark database: {rag_db_path}")

        self.retriever = Retriever(config=rag_config, embedding_model="all-MiniLM-L6-v2", rag_db=rag_db_path)
        self.query_classifier = QueryClassifier(config=rag_config,
                                                retriever=self.retriever,
                                                similarity_threshold=self.config.similarity_threshold)
        self.egf_print(f"Embedding model used: {self.retriever.embedding_model.name}", color=Fore.LIGHTGREEN_EX)

    def llm_init(self):
        """Initialize LLM component."""
        logger.info("LLM initialization...")

        if self.llm_model == "gui-dnpu-client":
            from eiq_genai_flow.llm_backends.gui_dnpu_client.gui_dnpu_client import GuiDNPUClient as llm

            try:
                self.llm = llm(self.device_name, prompt=self.system_prompt)
                self.llm_gui_dnpu_client = True
                # FIXME: Handle the connect/disconnect on both server and client sides.
                # If the server is already initialized, we wait for ever.
                # The lines below can be commented out to avoid waiting for the already initialized server.
                if self.llm.wait_for_server_init():
                    self.egf_print("Got Server INITED message", color=Fore.YELLOW)
                    self.llm.send_inited()
                logger.info("GUI DNPU client initialized")
            except ConnectionError as e:
                logger.error(f"GUI DNPU connection failed: {e}")
                exit(1)
        elif self.llm_model.endswith("-ara"):
            from eiq_genai_flow.llm_backends.ara_dnpu_client.ara_dnpu_client import AraDNPUClient as llm

            try:
                logger.info(f"Initializing ARA DNPU with model: {self.llm_model}")
                logger.info("Note: This may take up to 10 minutes if connector is starting")
                self.egf_print(
                    "Initializing AAF Connector (may take up to 10 minutes if eiq-aaf-connector is starting)",
                    color=Fore.LIGHTGREEN_EX,
                )

                self.llm = llm(model_name=self.llm_model, device_name=self.device_name, prompt=self.system_prompt)

                logger.info("✓ ARA DNPU initialized successfully")

            except ConnectionError as e:
                logger.error("ARA DNPU Connection Failed")
                logger.error(f"Error: {e}")
                logger.error("The connector did not become ready in time.")
                logger.error("You can try again or check the troubleshooting steps above.")
                exit(1)
        else:
            from llm.modeling_llm import make_LLM
            from llm.config.user_config import Config as LLMUserConfig

            # Create user config
            config_params = {
                "n_threads": self.thread_num,
                "use_neutron": self.use_neutron,
                "verbose": self.verbose,
                "prompt": self.system_prompt,
            }
            user_config = LLMUserConfig(**config_params)

            # Create LLM instance
            self.llm = make_LLM(
                self.llm_model,  # model
                user_params=user_config,  # user-defined configuration
            )

            # Overwrite default config by applying eiq config.py and log configuration
            default_config = self.llm.model_config
            self.llm.model_config = overwrite_config(default_config, self.config.LLMConfig, self.llm.name)

        self.egf_print(f"LLM used: {self.llm.name}", color=Fore.LIGHTGREEN_EX)

    def tts_init(self, quiet=False, lava_test=False):
        """Initialize TTS component."""
        logger.info("TTS initialization...")
        assert self.playback_device, "no audio audio playback device!"
        from eiq_genai_flow.adapters.tts import TTSAdapter, TTSAdapterConfig
        from tts.config import MultiSpeakerTTS16kHzQuantConfig as TTSConfig

        config_params = {"speed": self.config.tts_speed,
                         "speaker_id": self.config.tts_speaker_id}
        tts_config = TTSConfig(**config_params)

        adapter_config = TTSAdapterConfig(
            tts_config=tts_config,
            playback_device=self.playback_device,
            quiet=quiet,
            mode=self.config.tts_mode,
            lava_test=lava_test,
        )

        self.tts = TTSAdapter(config=adapter_config,
                              audio_manager=self.audio_manager,
                              event_manager=self.event_manager)

        if self.config.play_audio_filler and self.tts and self.earcon:
            # create audio fillers
            self.tts.generate_audio_fillers(
                dir_path=self.config.audio_filler_path, sentences=self.config.audio_filler_sentences
            )

            with self.earcon:
                for i in range(len(self.config.audio_filler_sentences)):
                    audio_path = os.path.join(self.config.audio_filler_path, f"{i}.wav")
                    self.event_manager.publish(Event(EventType.AUDIO_FILLER_REGISTER,
                                                     source="eIQGenAIFlow",
                                                     data=audio_path))

        self.tts_complete_event = threading.Event()

        logger.info(f"Playback Device used: {self.playback_device}")
        self.egf_print(f"TTS model used: {self.tts.model_name}", color=Fore.LIGHTGREEN_EX)

    def vad_init(self):
        """Initialize VAD component."""
        from eiq_genai_flow.adapters.vad import VADAdapter, VADAdapterConfig
        logger.info("VAD initialization...")
        assert self.capture_device, "no audio audio capture device!"
        vad_config = VADAdapterConfig(
            channel_indices=self.config.vad_channel_indices if hasattr(self.config, "vad_channel_indices") else [0],
        )

        self.vad = VADAdapter(
            config=vad_config,
            audio_manager=self.audio_manager,
            event_manager=self.event_manager,
        )

    def stt_init(self):
        """Initialize STT component."""
        logger.info("STT initialization...")
        assert self.capture_device, "no audio audio capture device!"
        from eiq_genai_flow.adapters.stt import STTAdapter, STTAdapterConfig

        stt_config = STTAdapterConfig(
            model_name=self.stt_model,
            channel_indices=self.config.stt_channel_indices,
            stream_print=False,
            audio_chunk_duration=3.0,
            inactivity_timeout=self.config.stt_inactivity_timeout,
            timer_print=False,
        )

        self.stt = STTAdapter(
            config=stt_config,
            audio_manager=self.audio_manager,
            event_manager=self.event_manager,
            vad_window_size_sec=self.vad.window_size_sec,
        )

    def voice_id_init(self):
        """Initialize ASR component."""
        logger.info("Voice ID initialization...")
        assert self.capture_device, "no audio audio capture device!"
        from eiq_genai_flow.adapters.voice_id import VoiceIDAdapter, VoiceIDConfig

        voice_id_config = VoiceIDConfig()
        self.voice_id = VoiceIDAdapter(
            config=voice_id_config,
            audio_manager=self.audio_manager,
            vad_window_size_sec=self.vad.window_size_sec,
            event_manager=self.event_manager,
        )

    # =========================================================================
    # MAIN EXECUTION LOOP
    # =========================================================================

    def _on_input_text(self, event: Event):
        self.input_text = event.data
        if self.gui and self.input_text:
            self.gui.send_qst(self.input_text)

    def _on_end_of_input(self, event: Event):
        self.end_of_input_event.set()

    def _on_keyboard_keypress(self, event: Event):
        if self.gui:
            self.gui.send_qst(event.data["buffer"])

    def _display_ready_message(self):
        """Display the appropriate ready message based on current mode."""
        need_wake = not self.wake_event.is_set()
        if (self.voice_id or self.vit) and need_wake:
            message = self.config.start_voice_id_info if self.voice_id else self.config.start_vasr_info
            print(message)
            if self.gui:
                self.gui.send_vis()
        elif self.keyboard and self.stt and need_wake:
            print(self.config.start_kasr_info)
        elif self.keyboard and need_wake:
            print(self.config.prompt)

    def _on_timeout(self, event: Event):
        logger.info("Timeout occurred")
        if event.source == "VoiceIDAdapter":
            # Unblock wait_wake_event function in case of VoiceID timeout
            self.wake_event.set()
        self.end_of_input_event.set()
        self.speaker_verification_event.set()
        if self.continuous:
            self.wake_event.clear()
        if self.gui:
            self.gui.send_tmo()

    def _on_stop_command(self, event: Event):
        logger.info("Speaker unknown during command, stopping command !")
        self.end_of_input_event.set()
        self.wake_event.set()
        self.speaker_verification_event.set()

        # Update interface
        if self.gui:
            self.gui.send_tmo()

        # Update message
        message = self.config.start_voice_id_info if self.voice_id else self.config.start_vasr_info
        print(message)

    def wait_user_input(self, timeout=None):
        if self.stt and not self.voice_id:
            print(self.config.listening_info)

        self.end_of_input_event.wait(timeout=timeout)  # blocking call

        self.event_manager.unsubscribe(EventType.KEYBOARD_KEYPRESS, self._on_keyboard_keypress)
        self.event_manager.unsubscribe(EventType.INPUT_TEXT, self._on_input_text)
        self.event_manager.unsubscribe(EventType.END_OF_INPUT, self._on_end_of_input)
        self.event_manager.unsubscribe(EventType.VAD_SPEECH_START, self._on_vad_speech_start_gui)

        if self.gui:
            self.gui.send_qst(end_token)

        text = self.input_text
        self.input_text = ""
        self.end_of_input_event.clear()

        return text

    # voice_id for STT-speaker verification
    def _on_stt_speaker_verification(self, event: Event):
        self.speaker_verification_event_type = event.event_type
        self.speaker_verification_event.set()

    def wait_speaker_verification(self, timeout=None) -> bool:
        logger.debug("wait_speaker_verification")

        self.speaker_verification_event.wait(timeout=timeout)  # blocking call

        self.event_manager.unsubscribe([EventType.VERIFIED_SPEAKER, EventType.UNVERIFIED_SPEAKER],
                                       self._on_stt_speaker_verification)
        self.speaker_verification_event.clear()

        if self.speaker_verification_event_type == EventType.VERIFIED_SPEAKER:
            return True

        else:
            # EventType.UNVERIFIED_SPEAKER
            logger.info("Speaker not enrolled, please use wake-word to continue")
            return False

    # vit for wake-up
    def _on_wake(self, event: Event):
        if event.event_type == EventType.VIT_WAKE:
            if self.gui:
                self.gui.send_wwd("")
                self.gui.send_speech_activity("<SPEECH_AFTER_WAKE>")
            if self.voice_id:
                print(self.config.listening_info)
        self.wake_event.set()
        self.wake_type = event.event_type

    # gui vad activities
    def _on_vad_speech_start_gui(self, event: Event):
        if self.gui:
            self.gui.send_speech_activity("<SPEECH>")

    def _on_vad_speech_end_gui(self, event: Event):
        if self.gui:
            self.gui.send_speech_activity("<NO_SPEECH>")

    # gui voice_id activities
    def _on_speaker_verification_gui(self, event: Event):
        if self.gui:
            if event.event_type == EventType.VOICE_ID_WAKE:
                self.gui.send_speaker_activity('<VERIFIED>')
            elif event.event_type == EventType.VOICE_ID_NO_WAKE:
                self.gui.send_speaker_activity('<UNVERIFIED>')
            else:
                raise NotImplementedError

    def wait_wake_event(self, timeout=None):
        self._display_ready_message()

        self.event_manager.subscribe([EventType.VIT_WAKE, EventType.VOICE_ID_WAKE,
                                      EventType.KEYBOARD_WAKE], self._on_wake)
        self.event_manager.subscribe(EventType.KEYBOARD_KEYPRESS, self._on_keyboard_keypress)
        self.event_manager.subscribe(EventType.INPUT_TEXT, self._on_input_text)
        self.event_manager.subscribe(EventType.END_OF_INPUT, self._on_end_of_input)
        self.event_manager.subscribe(EventType.VAD_SPEECH_START, self._on_vad_speech_start_gui)
        self.event_manager.subscribe(EventType.VAD_SPEECH_END, self._on_vad_speech_end_gui)
        self.event_manager.subscribe([EventType.VOICE_ID_WAKE, EventType.VOICE_ID_NO_WAKE],
                                     self._on_speaker_verification_gui)
        self.event_manager.subscribe([EventType.VERIFIED_SPEAKER, EventType.UNVERIFIED_SPEAKER],
                                     self._on_stt_speaker_verification)

        self.wake_event.wait(timeout=timeout)
        if not self.continuous:
            self.wake_event.clear()
        else:
            self.event_manager.publish(Event(EventType.CONTINUOUS_WAKE, source="eIQGenAIFlow"))

        self.event_manager.subscribe(EventType.VOICE_ID_STOP_COMMAND, self._on_stop_command)

        self.event_manager.unsubscribe([EventType.VIT_WAKE, EventType.VOICE_ID_WAKE,
                                        EventType.KEYBOARD_WAKE], self._on_wake)
        self.event_manager.unsubscribe(EventType.VAD_SPEECH_END, self._on_vad_speech_end_gui)
        self.event_manager.unsubscribe([EventType.VOICE_ID_WAKE, EventType.VOICE_ID_NO_WAKE],
                                       self._on_speaker_verification_gui)

    def run(self):
        """Main execution loop."""
        with self.earcon, self.tts:
            while not self.stop_threads:
                question = None
                with self.keyboard, self.vit, self.vad, self.voice_id, self.stt:

                    self.wait_wake_event()

                    question = self.wait_user_input()
                    verified = self.wait_speaker_verification() if self.voice_id else True

                # Handle the question
                if question and verified:
                    self.handle_question(question)

    # =========================================================================
    # QUESTION HANDLING AND PROCESSING
    # =========================================================================

    def handle_question(self, question):
        """Handle user question through the pipeline."""

        print(f"User: {question}")
        prompt = self.system_prompt
        rag_time = 0

        if question != "":
            if self.query_classifier:
                start_rag_time = time.perf_counter()
                query_category, chunk_list, _, metadata_list = self.query_classifier(query=question)
                prompt = (prompt, chunk_list)
                logger.debug(f"RAG prompt: {prompt}")
                rag_time = time.perf_counter() - start_rag_time
                match query_category:
                    case "CENSORED":
                        self.send_domain_response(random.choice(self.config.out_of_domain_response_list))
                    case "INTENT":
                        self.execute_intent(metadata_list)
                    case "REJECTED":
                        self.send_domain_response(random.choice(self.config.out_of_domain_response_list))
                    case "AMBIGUOUS":
                        self.send_domain_response(random.choice(self.config.ambiguous_response_list))
                    case "ACCEPTED":
                        if self.gui and "image" in metadata_list[0]:
                            self.gui.send_rsp(metadata_list[0]["image"])
                            self.gui.send_rsp(end_token)
                        if self.llm:
                            self.get_llm_output(question, prompt)
                        else:
                            self.process_complete_response(chunk_list[0])
            else:
                if self.llm:
                    logger.debug(f"LLM prompt: {prompt}")
                    self.get_llm_output(question, prompt)
                else:
                    self.process_complete_response(question)

        # Wait for ALL TTS playback to complete
        self.wait_tts()

        self.add_benchmark_stats(rag_time=rag_time)

        if self.gui:
            self.gui.send_thf()

        # Stop capture in on-demand mode (before VIT re-enables)
        if not self.config.keep_capture_device_open:
            if self.audio_manager.is_capture_running():
                logger.info("Stopping capture device (on-demand mode)")
                self.audio_manager.stop_capture()

    def get_llm_output(self, question, prompt):
        """Process question through LLM."""
        start_time = time.time()
        time_to_first_token, token_count = 0, 0
        full_response = ""

        for token_count, decoded_token in enumerate(self.llm(query=question, prompt=prompt), start=1):
            if token_count == 1:
                time_to_first_token = time.time() - start_time
                self.event_manager.publish(Event(EventType.AUDIO_FILLER_PLAY, source="eIQGenAIFlow"))

            if self.gui:
                self.gui.send_rsp(decoded_token)
            self.egf_print(decoded_token, color=Fore.GREEN, append=True)

            full_response += decoded_token

            # when answer is too long
            if decoded_token == self.llm.long_token:
                break
            elif decoded_token:
                self.tts_process(decoded_token)

        logger.info(f"LLM output: {full_response}")

        self.egf_print("\n")
        if self.gui:
            self.gui.send_rsp(end_token)
        self.tts_process(eos=True)  # end of sequence

        if token_count and time_to_first_token and (self.verbose or self.benchmark):
            total_time = time.time() - start_time
            gen_time = total_time - time_to_first_token
            token_per_second = token_count / gen_time if gen_time > 0 else 0
            self.add_benchmark_stats(llm_ttft=time_to_first_token, llm_tps=token_per_second, llm_time=total_time)
            if self.benchmark_logger:
                self.benchmark_logger.log(
                    f"{self.llm.name}: {self.llm.llm_input_size} input tokens, "
                    f"gen {token_count} tokens in {total_time:0.2f}s "
                    f"=> {token_per_second:0.2f}tok/s, ttft = {time_to_first_token:0.2f}s\n"
                )

    def process_complete_response(self, response: str):
        """Process complete response (no LLM OR no LLM & no RAG)."""
        self.tts_process(response, eos=True)
        self.egf_print(response, color=Fore.GREEN)
        if self.gui:
            self.gui.send_rsp(response)
            self.gui.send_rsp(end_token)

    # =========================================================================
    # DOMAIN AND INTENT HANDLING
    # =========================================================================

    def send_domain_response(self, response_text):
        """Send domain-related response through TTS and GUI."""
        # Output to console and TTS
        self.egf_print(response_text + "\n", color=Fore.GREEN)
        self.tts_process(response_text, eos=True)

        # Send to GUI word by word if available
        if self.gui:
            words = response_text.split()
            if words:
                self.gui.send_rsp(words[0])
                time.sleep(0.1)
                for word in words[1:]:
                    self.gui.send_rsp(" " + word)
                    time.sleep(0.1)
                self.gui.send_rsp(end_token)

    def execute_intent(self, metadata_list: list):
        """Execute intent-based commands if detected in metadata."""
        try:
            intent = metadata_list[0]["intent"]
            self.egf_print(f"RAG: intent detected >>>>> {intent}\n", color=Fore.GREEN)

            self.event_manager.publish(Event(EventType.INTENT_DETECTED, source="eIQGenAIFlow", data=intent))

            if self.gui:
                self.gui.send_intent(intent)

        except (ValueError, KeyError) as e:
            logger.error(f"Failed to execute intent: {e}")

    # =========================================================================
    # GUI CALLBACK HANDLING
    # =========================================================================

    def gui_callback(self, msg):
        print("GUI gui_callback : " + msg)
        """Handle GUI messages and route them appropriately."""
        if msg == stop_token:
            if self.llm and self.llm._running:
                self.llm.stop()
                logger.error("LLM generation interrupted")
            if self.stt and self.stt.is_running:
                self.stt.disable()

        elif msg == vit_token:
            if not self.vit.is_running:
                print("busy, rejecting request")
                return
            self.vit.bypass()

        elif msg == self.gui.user_config.connect_sig:
            self.gui.send_connect()
            print("GUI send_connect")

        else:
            if not self.vit.is_running:
                print("busy, rejecting request")
                return
            # When we get a keyboard question from gui, we simulate a vit wake word +
            # stt recognition to pass through the pipeline
            self.vit.bypass(bypass_asr=True)
            self.handle_question(msg)

    # =========================================================================
    # AUDIO AND TTS METHODS
    # =========================================================================

    def tts_process(self, text=None, eos=False):
        """Process text through TTS adapter."""
        if self.tts:
            self.event_manager.publish(Event(EventType.TTS_PROCESS,
                                             source="eIQGenAIFlow",
                                             data={"text": text,
                                                   "eos": eos}))
            self.tts_complete_event.clear()
            self.event_manager.subscribe(EventType.TTS_COMPLETE, self._on_tts_complete)

    def _on_tts_complete(self, event: Event):
        if event and event.event_type == EventType.TTS_COMPLETE:
            if event.data.get("tts_process_source") == "eIQGenAIFlow":
                self.event_manager.unsubscribe(EventType.TTS_COMPLETE, self._on_tts_complete)
                self.tts_complete_event.set()

    def wait_tts(self):
        if self.tts:
            self.tts_complete_event.wait()

    # =========================================================================
    # UTILITY AND OUTPUT METHODS
    # =========================================================================

    def egf_print(self, message, append=False, color=Fore.RESET, style=Style.NORMAL):
        """Print a message with optional color formatting and benchmark logging support."""
        # Benchmark logging
        if self.benchmark_logger:
            if append:
                self.benchmark_logger.append_print(message)
            else:
                self.benchmark_logger.log(message)

            # Only print to console in benchmark mode if verbose is enabled
            if not self.verbose:
                return

        # Print with formatting
        print(f"{color}{style}{message}{Style.RESET_ALL}", end="" if append else "\n")

        # Flush output for append mode
        if append:
            sys.stdout.flush()
        else:
            # Log to standard logger (avoid flooding in append mode)
            logger.info(message)

    # =========================================================================
    # CLEANUP AND SIGNAL HANDLING
    # =========================================================================
    def clean_up(self):
        """Clean up all resources."""
        cleanup_actions = [
            (self.stt, lambda: self.stt.shutdown()),
            (self.gui, lambda: self.gui.send_disconnect()),
            (self.vit, lambda: self.vit.shutdown()),
            (self.tts, lambda: self.tts.disable()),
            (self.earcon, lambda: self.earcon.disable()),
            (self.audio_manager, lambda: self.audio_manager.shutdown()),
        ]

        for resource, action in cleanup_actions:
            if resource:
                try:
                    action()
                except Exception as e:
                    logger.error(f"Error during cleanup: {e}")

        self.stop_threads = True

        if self.config.restore_cpu_governor_on_exit and self.device != "PC":
            restore_cpu_governor()

    def signal_handler(self, sig, frame):
        """Handle system signals for graceful shutdown."""
        logger.warning("Ctrl-C detected. Attempting graceful shutdown...")

        # Prevent re-entry
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)

        # Reset terminal
        try:
            subprocess.run(["stty", "echo"], check=False, timeout=1)
            subprocess.run(["stty", "sane"], check=False, timeout=1)
        except Exception:
            pass

        try:
            # Set stop flags immediately
            self.stop_threads = True
            adapters = [
                (self.keyboard, "KEYBOARD"),
                (self.vit, "VIT"),
                (self.vad, "VAD"),
                (self.voice_id, "VOICE_ID"),
                (self.stt, "STT"),
                (self.tts, "TTS"),
                (self.earcon, "EARCON")
            ]

            for adapter, _ in adapters:
                if adapter and hasattr(adapter, "_stop_event"):
                    adapter._stop_event.set()

            # Wait for threads to actually stop
            timeout = 0.5
            for adapter, name in adapters:
                if adapter and hasattr(adapter, "_worker_thread"):
                    thread = adapter._worker_thread
                    if thread and thread.is_alive():
                        thread.join(timeout=timeout)
                        if not thread.is_alive():
                            logger.debug(f"{name} stopped")

            if self.gui:
                # Send disconnect message to gui
                self.gui.send_disconnect()

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        finally:
            logger.info("Exiting now...")
            os._exit(0)

    # =========================================================================
    # BENCHMARK METHODS (PLACEHOLDER)
    # =========================================================================

    def add_benchmark_stats(self, *args, **kwargs):
        """Add benchmark statistics (implemented in benchmark subclass)."""
        pass  # This method is overridden in the Benchmark class


# =============================================================================
# MODULE CONFIGURATION AND CLI SETUP
# =============================================================================


def process_arguments(
    input_mode, capture_device, llm_model, output_mode, playback_device, stt_model, logging_level, module_configs
):
    """Process and normalize command line arguments."""
    return {
        "input_mode": input_mode.value,
        "capture_device": capture_device.value if capture_device else None,
        "llm_model": llm_model.value if llm_model and llm_model.value != "no_llm" else None,
        "output_mode": output_mode.value,
        "playback_device": playback_device.value if playback_device else None,
        "stt_model": stt_model.value if stt_model != "none" else None,
        "logging_level": logging._nameToLevel[logging_level.value],
        "gui_config_class": module_configs["gui"]["config_classes"].get(input_mode.value)
        if input_mode.value in module_configs["gui"]["modules"]
        else None,
    }


def create_pipeline(
    config, processed_args, wake_word_model, system_prompt, use_rag, use_neutron, use_voice_id,
    continuous, benchmark, verbose
):
    """Create the appropriate pipeline instance."""
    if benchmark:
        from eiq_genai_flow.benchmark.benchmark import Benchmark

        PipelineClass = Benchmark
    else:
        PipelineClass = eIQGenAIFlow

    return PipelineClass(
        config,
        processed_args["input_mode"],
        processed_args["capture_device"],
        wake_word_model,
        processed_args["gui_config_class"],
        processed_args["llm_model"],
        use_rag,
        system_prompt,
        processed_args["output_mode"],
        processed_args["playback_device"],
        continuous,
        processed_args["stt_model"],
        use_neutron,
        use_voice_id,
        benchmark,
        verbose,
    )


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def main():
    """Main entry point for eIQ GenAI Flow Demo application."""

    # ===== EARLY LOGGING SETUP (only if DEBUG requested) =====
    import sys

    debug_requested = False
    for i, arg in enumerate(sys.argv):
        if arg in ["-l", "--logging-level"]:
            # Check next argument
            if i + 1 < len(sys.argv) and sys.argv[i + 1] == "DEBUG":
                debug_requested = True
                break

    # Only set up early logging if DEBUG was requested
    if debug_requested:
        early_handler = logging.StreamHandler(sys.stdout)
        early_handler.setLevel(logging.DEBUG)
        early_formatter = logging.Formatter("%(levelname)-8s - %(name)s - %(message)s")
        early_handler.setFormatter(early_formatter)

        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(early_handler)
    # ==========================================================

    app = typer.Typer(
        name="eIQ GenAI Flow Demo",
        no_args_is_help=True,
        add_completion=False,
        context_settings={"help_option_names": ["-h", "--help"]},
    )

    config = Config()
    directory_path = os.path.dirname(__file__)

    args_manager = ArgumentManager(config=config)
    module_configs = args_manager.get_module_configs()
    arg_enums, cli_options = args_manager.get_arguments_and_options()

    # Extract enums from dictionary for type annotations
    InputModesArgs = arg_enums["InputModesArgs"]
    CaptureDeviceArgs = arg_enums["CaptureDeviceArgs"]
    LlmArgs = arg_enums["LlmArgs"]
    OutputModesArgs = arg_enums["OutputModesArgs"]
    PlaybackDeviceArgs = arg_enums["PlaybackDeviceArgs"]
    STTArgs = arg_enums["STTArgs"]
    LoggingLevel = arg_enums["LoggingLevel"]

    @app.command()
    def parse_args(
        input_mode: InputModesArgs = cli_options["input_mode"],
        capture_device: CaptureDeviceArgs = cli_options["capture_device"],
        llm_model: LlmArgs = cli_options["llm_model"],
        system_prompt: str = cli_options["system_prompt"],
        output_mode: OutputModesArgs = cli_options["output_mode"],
        playback_device: PlaybackDeviceArgs = cli_options["playback_device"],
        stt_model: STTArgs = cli_options["stt_model"],
        use_rag: bool = cli_options["use_rag"],
        use_neutron: bool = typer.Option(False, "--use-neutron", "-n", help="Use Neutron ONNX Execution Provider."),
        use_voice_id: bool = cli_options["voice_id"],
        wake_word_model: str = cli_options["wake_word_model"],
        continuous: bool = typer.Option(
            False, "--continuous", "-c", help="Continuous mode where ASR is always listening."
        ),
        benchmark: bool = typer.Option(
            False, "--benchmark", "-b", help=("Benchmark mode - take a list of questions and store the results.")
        ),
        verbose: bool = typer.Option(False, "--verbose", "-v", help="Display information (e.g. inference times)."),
        logging_level: LoggingLevel = typer.Option(
            "WARNING", "--logging-level", "-l", help="Level of displayed information."
        ),
        use_traceback: bool = typer.Option(
            False, "--use-traceback", help="Activate Typer traceback of exceptions and errors."
        ),
    ):
        """Parse command line arguments and run the pipeline."""
        app.pretty_exceptions_enable = use_traceback

        # Process arguments
        processed_args = process_arguments(
            input_mode,
            capture_device,
            llm_model,
            output_mode,
            playback_device,
            stt_model,
            logging_level,
            module_configs,
        )

        # Remove early debug handler if it exists
        if debug_requested:
            root_logger = logging.getLogger()
            # Remove all existing handlers (the early one)
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)

        # Setup logging
        setup_logging(level=processed_args["logging_level"], root_path=directory_path)

        # Create and run pipeline
        pipeline = create_pipeline(
            config, processed_args, wake_word_model, system_prompt, use_rag, use_neutron, use_voice_id,
            continuous, benchmark, verbose
        )
        pipeline.run()

    app()


if __name__ == "__main__":
    main()
