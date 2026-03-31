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

import random
import subprocess
import sys
import os
import platform
import select
import signal
import time
import logging
import typer
from colorama import Fore, Style
from gui.config import end_token, stop_token, vit_token

from config import Config
from shared_utils.utils import get_number_of_cores, setup_logging
from utils.utils import overwrite_config, get_soc_id, get_machine, get_revision
from utils.cpu_governor_manager import setup_cpu_governor, restore_cpu_governor
from utils.argument_manager import ArgumentManager

import warnings

# Suppress PyTorch ONNX registration warnings
warnings.filterwarnings("ignore", message=".*Symbolic function.*already registered.*")
logging.getLogger("torio._extension.utils").setLevel(logging.WARNING)

# Environment configuration
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["ORT_LOGGING_LEVEL"] = "3"


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
        benchmark,
        verbose,
    ):
        """Initialize the eIQ GenAI Flow pipeline with all components."""
        # Store all parameters as instance attributes
        for key, value in locals().items():
            if key != "self":
                setattr(self, key, value)

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

        self.earcon_manager = None

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

        logger.info(f"Audio Manager created with backend: {backend}")
        self.frame_duration_s = self.audio_manager.capture_config.frame_duration_ms / 1000.0

        if self.capture_device and (self.gui_config_class or "asr" in self.input_mode):
            if self.config.keep_capture_device_open:
                logger.info("Audio Manager capture initialized (will start after components ready)")
            else:
                logger.info("Audio Manager capture initialized (on-demand mode)")

    def initialize_components(self):
        """Initialize all system components based on configuration."""
        # GUI component
        if self.gui_config_class and not self.benchmark:
            self.gui_init()
        else:
            self.gui = None

        # Initialize earcon manager if playback is available and not in silent benchmark mode
        if self.playback_device and not (self.benchmark and self.config.silent_benchmark):
            self._init_earcon_manager()
        else:
            self.earcon_manager = None

        # VIT component - Skip in KASR benchmark mode, but keep for VASR
        if self.capture_device and (self.gui or self.input_mode == "vasr"):
            self.vit_init()
        else:
            self.vit = None

        # STT (ASR) component
        if self.capture_device and (self.gui or "asr" in self.input_mode):
            start_time = time.time()
            self.stt_init()
            self.stt_init_time = time.time() - start_time
        else:
            self.stt = None
            self.stt_init_time = 0

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
        if self.playback_device and (self.gui_config_class or "tts" in self.output_mode):
            start_time = time.time()
            # In benchmark mode with silent_benchmark enabled, use quiet mode
            quiet_mode = self.benchmark and self.config.silent_benchmark
            self.tts_init(quiet=quiet_mode, lava_test=True if self.benchmark else False)
            self.tts_init_time = time.time() - start_time
        else:
            self.tts = None
            self.tts_init_time = 0

    def _init_earcon_manager(self):
        """Initialize earcon manager and load sounds based on config."""
        from utils.earcon_manager import EarconManager
        import os

        # Get project root directory (where eiq_genai_flow.py is located)
        project_root = os.path.dirname(os.path.abspath(__file__))
        assets_path = os.path.join(project_root, "assets")

        logger.info(f"Initializing EarconManager with assets path: {assets_path}")

        self.earcon_manager = EarconManager(audio_manager=self.audio_manager)

        # Register wake word earcon if configured
        if self.config.play_wake_word_detect_sound:
            ww_path = os.path.join(assets_path, "ww_earcon.wav")
            logger.info(f"Registering wake word earcon: {ww_path}")
            if self.earcon_manager.register_earcon("wake_word", ww_path, enabled=True):
                logger.info("Wake word earcon registered and enabled")
            else:
                logger.warning("Failed to register wake word earcon")

        # Register TTS start earcon if configured
        if self.config.play_tts_start_sound:
            tts_path = os.path.join(assets_path, "tts_earcon.wav")
            logger.info(f"Registering TTS earcon: {tts_path}")
            if self.earcon_manager.register_earcon("tts_start", tts_path, enabled=True):
                logger.info("TTS earcon registered and enabled")
            else:
                logger.warning("Failed to register TTS earcon")

        # Register intent earcon if configured
        if self.config.play_intent_detect_sound:
            intent_path = os.path.join(assets_path, "intent_earcon.wav")
            logger.info(f"Registering intent earcon: {intent_path}")
            if self.earcon_manager.register_earcon("intent_detected", intent_path, enabled=True):
                logger.info("Intent earcon registered and enabled")
            else:
                logger.warning("Failed to register intent earcon")

        logger.info("EarconManager initialization complete")

    def play_ww_detect_earcon(self):
        """Play Wake Word detection earcon sound."""
        if self.earcon_manager and not (self.benchmark and self.config.silent_benchmark):
            self.earcon_manager.play_earcon("wake_word")

    def play_tts_start_earcon(self):
        """Play TTS start earcon sound."""
        if self.earcon_manager and not (self.benchmark and self.config.silent_benchmark):
            self.earcon_manager.play_earcon("tts_start")

    def play_intent_detect_earcon(self):
        """Play Intent detection earcon sound."""
        if self.earcon_manager and not (self.benchmark and self.config.silent_benchmark):
            self.earcon_manager.play_earcon("intent_detected")

    def play_audio_filler_earcon(self):
        if self.earcon_manager and not (self.benchmark and self.config.silent_benchmark):
            idx = random.randint(0, len(self.config.audio_filler_sentences) - 1)
            self.earcon_manager.play_earcon(f"audio_filler_{idx}")
            self.egf_print(self.config.audio_filler_sentences[idx], color=Fore.GREEN)

    def start_system(self):
        """Start the system and play initial message."""

        if self.capture_device and (self.gui_config_class or "asr" in self.input_mode):
            # Start capture regardless of keep_device_open mode
            logger.info("Starting audio capture - pipeline fully ready")
            self.audio_manager.start_capture()
            logger.info("Audio Manager capture started")

        if self.gui:
            self.gui.start()
            self.gui.send_connect()

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

        if self.vit and self.stt:
            logger.info("Enabling both VIT and STT readers for synchronized audio")

            # VIT starts listening
            self.vit.enable(clear_buffer=False)

            # ASR reader enabled but won't process until triggered
            stt_reader = self.audio_manager.get_reader("STT")
            if stt_reader:
                # Enable with sync to current to align with VIT's buffer
                stt_reader.enable(sync_to_current=True)
                logger.info("VIT and STT readers synchronized")
        elif self.vit:
            logger.info("Enabling VIT reader")
            self.vit.enable(clear_buffer=False)

        if not self.benchmark:
            if self.kasr_wake:
                print(self.config.start_kasr_info)
            elif self.vit:
                print(self.config.start_vasr_info)

    # =========================================================================
    # COMPONENT INITIALIZATION METHODS
    # =========================================================================

    def gui_init(self):
        """Initialize GUI component."""
        logger.debug(f"Loading {self.gui_config_class.__name__} GUI I/F module")

        from gui.generic_gui_interface import GenericGuiInterface

        self.gui = GenericGuiInterface(callback=self.gui_callback, user_config=self.gui_config_class)

    def vit_init(self):
        """Initialize VIT component."""
        from adapters.vit.vit_adapter import VITAdapter, VITConfig

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
            verbose=self.verbose,
        )

        # Get status
        vit_status = self.vit.get_status()
        logger.info(f"VIT Status: {vit_status}")

    def stt_init(self):
        """Initialize ASR component."""
        from adapters.stt.stt_adapter import STTAdapter, STTAdapterConfig

        logger.info("stt_init")

        stt_config = STTAdapterConfig(
            model_name=self.stt_model,
            channel_indices=self.config.stt_channel_indices,
            stream_print=False,
            audio_chunk_duration=3.0,
            inactivity_timeout=20.0,
            vad_threshold=0.3,
            vad_min_silence_duration_ms=200,
        )

        self.stt = STTAdapter(
            config=stt_config,
            audio_manager=self.audio_manager,
            verbose=self.verbose,
        )

        logger.info(f"Capture Device used: {self.capture_device}")
        self.egf_print(f"STT model used: {self.stt_model}", color=Fore.LIGHTGREEN_EX)

    def get_stt_output(self) -> str:
        text = ""
        for text in self.stt.mic_to_text():
            if self.gui and text:
                self.gui.send_qst(text)

        if self.gui:
            self.gui.send_qst(end_token)
        return text

    def rag_init(self):
        """Initialize RAG component."""
        from rag.retrieval import QueryClassifier, Retriever
        from rag.config import Config as RAGConfig

        rag_config = RAGConfig()
        self.retriever = Retriever(config=rag_config,
                                   embedding_model="all-MiniLM-L6-v2",
                                   rag_db=self.config.rag_database_path)
        self.query_classifier = QueryClassifier(
            config=rag_config, retriever=self.retriever, similarity_threshold=self.config.similarity_threshold
        )
        self.egf_print(f"Embedding model used: {self.retriever.embedding_model.name}", color=Fore.LIGHTGREEN_EX)

    def llm_init(self):
        """Initialize LLM component."""
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
        from adapters.tts.tts_adapter import TTSAdapter, TTSAdapterConfig
        from tts.config import MultiSpeakerTTS16kHzQuantConfig as TTSConfig

        config_params = {"speed": 0.55, "speaker_id": 24}
        tts_config = TTSConfig(**config_params)

        adapter_config = TTSAdapterConfig(
            tts_config=tts_config,
            playback_device=self.playback_device,
            quiet=quiet,
            mode=self.config.tts_mode,
            lava_test=lava_test,
        )

        self.tts = TTSAdapter(config=adapter_config, audio_manager=self.audio_manager, verbose=self.verbose)

        if self.config.play_audio_filler and self.tts:
            # create audio fillers
            self.tts.generate_audio_fillers(
                dir_path=self.config.audio_filler_path, sentences=self.config.audio_filler_sentences
            )
            for i in range(len(self.config.audio_filler_sentences)):
                audio_path = os.path.join(self.config.audio_filler_path, f"{i}.wav")
                if self.earcon_manager.register_earcon(f"audio_filler_{i}", audio_path, enabled=True):
                    logger.info(f"Audio filler {i} earcon registered and enabled")

        logger.info(f"Playback Device used: {self.playback_device}")
        self.egf_print(f"TTS model used: {self.tts.model_name}", color=Fore.LIGHTGREEN_EX)

    # =========================================================================
    # MAIN EXECUTION LOOP
    # =========================================================================

    def run(self):
        """Main execution loop."""
        self.first_keyb_wake = True
        self.first_vit_wake = True

        while not self.stop_threads:
            question = self.get_user_input()

            # Update wake flags
            if self.kasr_wake and question != "":
                self.first_keyb_wake = False
            elif self.vit and self.continuous and question != "":
                self.first_vit_wake = False

            if question == "TIMEOUT":
                logger.debug("Timeout occurred, resetting wake flags")
                self.restart_wake(reset=True)
                continue

            # Handle the question
            self.handle_question(question)

    # =========================================================================
    # INPUT HANDLING METHODS
    # =========================================================================

    def get_user_input(self):
        """Get input from ASR or keyboard."""
        if not self.stt:
            return self.get_keyboard_input()

        if self.kasr_wake:
            return self.get_kstt_input()
        elif self.vit:
            return self.get_vit_input()
        else:
            return self.process_stt_output()

    def get_keyboard_input(self):
        """Get keyboard input with error handling."""
        self.wait_tts()
        print(f"\n{self.config.prompt}")

        while True:
            try:
                return input()
            except ValueError:
                logger.error("ValueError")
                continue
            except KeyboardInterrupt:
                if self.llm:
                    self.llm.close()
                if self.tts:
                    self.tts.exit()
                logger.error("\nExiting...")
                exit()

    def get_kstt_input(self):
        """Get keyboard-triggered ASR input with blocking read."""
        if self.first_keyb_wake or not self.continuous:
            while not self.stop_threads:
                # Block on keyboard with timeout for responsiveness
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)

                if ready:
                    key = sys.stdin.read(1)
                    if key == "\n":
                        print(self.config.listening_info)
                        return self.process_stt_output()

            return ""  # Stop requested
        else:
            print(self.config.listening_info)
            return self.process_stt_output()

    def get_vit_input(self):
        """Get VIT wake word triggered ASR input."""
        if self.first_vit_wake:
            vit_info_string = self.wait_for_wake_word()
            if vit_info_string == "":
                return ""

            self.play_ww_detect_earcon()
            print(self.config.listening_info)

            if vit_info_string != self.vit.bypass_vit_asr_wwd:
                if self.gui:
                    self.gui.send_wwd(vit_info_string)
                return self.process_stt_output()

            return ""
        else:
            print(self.config.listening_info)
            return self.process_stt_output()

    def wait_for_wake_word(self):
        """Wait for wake word detection."""
        vit_info_string = ""
        while "WWD" not in vit_info_string:
            vit_info_string = self.vit.wait_for_wake_word(stop_requested=self.stop_threads)

            if "VIS" in vit_info_string and self.gui:
                self.gui.send_vis()

        logger.debug("Wakeword received: " + vit_info_string)

        # Get detailed info (non-blocking)
        detection_info = self.vit.get_detailed_info()

        if detection_info:
            ww_end_abs_idx = detection_info["ww_end_abs_index"]

            # Get STT reader
            stt_reader = self.audio_manager.get_reader("STT")
            if stt_reader:
                # Set read index to RIGHT AFTER wake word
                stt_reader.read_index = ww_end_abs_idx

                if not stt_reader.enabled:
                    stt_reader.enable(sync_to_current=False)
                    logger.info(f"STT reader enabled and positioned at index {stt_reader.read_index}")
                else:
                    logger.debug(f"STT reader repositioned to index {stt_reader.read_index}")

        # Disable VIT for next detection
        if self.vit.is_running:
            self.vit.disable()

        return vit_info_string

    def process_stt_output(self):
        """Process ASR output and provide user feedback."""
        question = self.get_stt_output()

        if question == "":
            question = "TIMEOUT"
            self.egf_print("ASR: No speech detected", color=Fore.YELLOW)
        else:
            self.egf_print(f"ASR: {question}")

        return question

    # =========================================================================
    # QUESTION HANDLING AND PROCESSING
    # =========================================================================

    def handle_question(self, question):
        """Handle user question through the pipeline."""
        prompt = self.config.default_system_prompt
        rag_time = 0

        if question != "":
            if self.query_classifier:
                start_rag_time = time.perf_counter()
                query_category, chunk_list, _, metadata_list = self.query_classifier(query=question)
                prompt = (self.config.default_system_prompt, chunk_list)
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

        # Restart wake detection (VIT will restart capture if needed)
        self.restart_wake()

    def restart_wake(self, reset=False):
        """Restart wake mode after ASR processing."""
        if self.stt:
            if not self.continuous or reset:
                if reset:
                    self.first_keyb_wake = True
                    self.first_vit_wake = True
                if self.kasr_wake and not self.benchmark:
                    print(self.config.start_kasr_info)
                elif self.vit:
                    # In on-demand mode, restart capture before re-enabling VIT
                    if not self.config.keep_capture_device_open:
                        if not self.audio_manager.is_capture_running():
                            logger.info("Restarting capture device for VIT (on-demand mode)")
                            self.audio_manager.start_capture()

                    # VIT.enable() will start capture if in on-demand mode
                    logger.info("Re-enabling VIT wake word detection after TTS completion")
                    self.vit.enable(clear_buffer=True)
                    print(self.config.start_vasr_info)

    def get_llm_output(self, question, prompt):
        """Process question through LLM."""
        self.play_tts_start_earcon()

        start_time = time.time()
        time_to_first_token, token_count = 0, 0
        full_response = ""

        for token_count, decoded_token in enumerate(self.llm(query=question, prompt=prompt), start=1):
            if token_count == 1:
                time_to_first_token = time.time() - start_time
                if self.tts and self.config.play_audio_filler:
                    self.play_audio_filler_earcon()

            if self.gui:
                self.gui.send_rsp(decoded_token)
            self.egf_print(decoded_token, color=Fore.GREEN, append=True)

            full_response += decoded_token

            # when answer is too long
            if decoded_token is self.llm.long_token:
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
        self.play_tts_start_earcon()
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
        self.play_tts_start_earcon()
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

            self.play_intent_detect_earcon()

            if self.gui:
                self.gui.send_cmd(input_string=intent)
                self.tts_process(intent, eos=True)

        except (ValueError, KeyError) as e:
            logger.error(f"Failed to execute intent: {e}")

    # =========================================================================
    # GUI CALLBACK HANDLING
    # =========================================================================

    def gui_callback(self, msg):
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
            self.tts.process(text, eos=eos)

    def wait_tts(self):
        """Wait until TTS is completely finished."""
        if self.tts:
            self.tts.wait_for_completion()

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
            (self.tts, lambda: self.tts.shutdown()),  # Use adapter shutdown
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
            if self.vit:
                self.vit._stop_event.set()
            if self.stt:
                self.stt._stop_event.set()
            if self.tts:
                self.tts._stop_event.set()

            # Wait for threads to actually stop
            timeout = 0.5
            for adapter, name in [(self.vit, "VIT"), (self.stt, "STT"), (self.tts, "TTS")]:
                if adapter and hasattr(adapter, "_worker_thread"):
                    thread = adapter._worker_thread
                    if thread and thread.is_alive():
                        thread.join(timeout=timeout)
                        if not thread.is_alive():
                            logger.debug(f"{name} stopped")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
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
        "stt_model": stt_model.value if stt_model else None,
        "logging_level": logging._nameToLevel[logging_level.value],
        "gui_config_class": module_configs["gui"]["config_classes"].get(input_mode.value)
        if input_mode.value in module_configs["gui"]["modules"]
        else None,
    }


def create_pipeline(
    config, processed_args, wake_word_model, system_prompt, use_rag, use_neutron, continuous, benchmark, verbose
):
    """Create the appropriate pipeline instance."""
    if benchmark:
        from tests.benchmark.benchmark import Benchmark

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
        benchmark,
        verbose,
    )


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def main():
    """Main entry point for eIQ GenAI Flow Demo application."""
    app = typer.Typer(
        name="eIQ GenAI Flow Demo",
        no_args_is_help=True,
        add_completion=False,
        context_settings={"help_option_names": ["-h", "--help"]},
    )

    config = Config()
    directory_path = os.path.dirname(__file__)

    args_manager = ArgumentManager(config=config, directory_path=directory_path)
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

        # Setup logging
        setup_logging(level=processed_args["logging_level"], root_path=directory_path)

        # Create and run pipeline
        pipeline = create_pipeline(
            config, processed_args, wake_word_model, system_prompt, use_rag, use_neutron, continuous, benchmark, verbose
        )
        pipeline.run()

    app()


if __name__ == "__main__":
    main()
