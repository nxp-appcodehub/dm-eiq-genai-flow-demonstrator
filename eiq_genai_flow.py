# Copyright 2023-2025 NXP
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
import re
import platform
import select
import signal
import time
import logging
import soundfile as sf
import typer
from colorama import Fore, Style
from gui.config import end_token, stop_token, vit_token

from config import Config
from shared_utils.utils import get_number_of_cores, setup_logging, set_audio_config, AlsaPlayer
from utils.utils import get_soc_id, get_machine,get_revision
from utils.cpu_governor_manager import setup_cpu_governor, restore_cpu_governor
from utils.argument_manager import ArgumentManager

# Environment configuration
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)


# =============================================================================
# MAIN PIPELINE CLASS
# =============================================================================


class eIQGenAIFlow:
    """
    Main pipeline class for eIQ GenAI Flow conversational AI system.

    Orchestrates the integration of VIT (Voice Intelligence Technology for wakeword support), ASR (Automatic Speech Recognition),
    RAG (Retrieval-Augmented Generation), LLM (Large Language Model), and TTS (Text-To-Speech) components.
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
        asr_model,
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
        self.asr_init_time = 0
        self.rag_init_time = 0
        self.llm_init_time = 0
        self.tts_init_time = 0
        self.benchmark_logger = None

        # Setup system configuration
        self.setup_system_config()

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

    def initialize_components(self):
        """Initialize all system components based on configuration."""
        # GUI component
        if self.gui_config_class and not self.benchmark:
            self.gui_init()
        else:
            self.gui = None

        # Audio component
        if self.capture_device and (self.gui or "asr" in self.input_mode):
            set_audio_config("capture", self.capture_device)
            logger.info("Capture device set to " + self.capture_device)

        # VIT component
        if not self.benchmark and self.capture_device and (self.gui or self.input_mode == "vasr"):
            self.vit_init()
        else:
            self.vit = None

        # ASR component
        if self.capture_device and (self.gui or "asr" in self.input_mode):
            start_time = time.time()
            self.asr_init()
            self.asr_init_time = time.time() - start_time
        else:
            self.asr = None
            self.asr_init_time = 0

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
            self.tts_init(quiet=True if self.benchmark else False, lava_test=True if self.benchmark else False)
            self.tts_init_time = time.time() - start_time
        else:
            self.tts = None
            self.tts_init_time = 0

    def start_system(self):
        """Start the system and play initial message."""
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
        from vit.wake_word import VIT

        self.vit = VIT(
            capture_device=self.capture_device,
            wake_word_model=self.wake_word_model,
            py_to_c_queue=self.config.py_to_c_queue,
            c_to_py_queue=self.config.c_to_py_queue,
            verbose=self.verbose,
        )
        if self.config.play_wake_word_sound:
            file_path = os.path.dirname(os.path.abspath(__file__))
            self.ww_sound, _ = sf.read(os.path.join(file_path, "assets", "ww_earcon.wav"))

    def asr_init(self):
        """Initialize ASR component."""
        from asr.streaming.speech_to_text import SpeechToText

        self.asr = SpeechToText(
            self.asr_model,
            language="English",
            task="transcribe",
            source='file' if self.benchmark else 'mic',
            audio_card_name=self.capture_device,
            stream_print=False
        )
        logger.info(f"Capture Device used: {self.capture_device}")

        self.egf_print(f"ASR model used: {self.asr_model}", color=Fore.LIGHTGREEN_EX)

    def rag_init(self):
        """Initialize RAG component."""
        from rag.retrieval import QueryClassifier, Retriever
        from rag.config import Config as RAGConfig

        rag_config = RAGConfig()
        src_dir_path = os.path.dirname(os.path.abspath(__file__))
        rag_db_path = os.path.join(src_dir_path, "rag", "src", "data", "rag_database.pkl")
        self.retriever = Retriever(config=rag_config, embedding_model="all-MiniLM-L6-v2", rag_db=rag_db_path)
        self.query_classifier = QueryClassifier(config=rag_config,
                                                retriever=self.retriever,
                                                similarity_threshold=self.config.similarity_threshold)
        self.egf_print(f"Embedding model used: {self.retriever.embedding_model.name}", color=Fore.LIGHTGREEN_EX)

    def llm_init(self):
        """Initialize LLM component."""
        from llm.modeling_llm import make_LLM
        from llm.config.user_config import Config as LLMUserConfig

        config_params = {"n_threads": self.thread_num, "use_neutron": self.use_neutron, "verbose": self.verbose, "prompt": self.system_prompt}

        user_config = LLMUserConfig(**config_params)
        self.llm = make_LLM(self.llm_model, user_params=user_config)

        self.egf_print(f"LLM used: {self.llm.name}", color=Fore.LIGHTGREEN_EX)

    def tts_init(self, quiet=False, lava_test=False):
        """Initialize TTS component."""
        from tts.inference import TTSPlayer
        from tts.config import MultiSpeakerTTS16kHzQuantConfig as TTSConfig

        config_params = {"speed": 0.55, "speaker_id": 24}
        tts_config = TTSConfig(**config_params)

        self.tts = TTSPlayer(config=tts_config, playback_device=self.playback_device, quiet=quiet, lava_test=lava_test)

        if self.config.play_tts_sound:
            file_path = os.path.dirname(os.path.abspath(__file__))
            self.tts_sound, _ = sf.read(os.path.join(file_path, "assets", "tts_earcon.wav"))

        logger.info(f"Playback Device used: {self.playback_device}")
        self.egf_print(f"TTS model used: {self.tts.model_name}", color=Fore.LIGHTGREEN_EX)

    # =========================================================================
    # MAIN EXECUTION LOOP
    # =========================================================================

    def run(self):
        """Main execution loop."""
        first_keyb_wake = True
        first_vit_wake = True

        if self.vit:
            self.vit.enable()

        while not self.stop_threads:
            question = self.get_user_input(first_keyb_wake, first_vit_wake)

            # Update wake flags
            if self.kasr_wake and question != "":
                first_keyb_wake = False
            if self.vit and self.continuous and question != "":
                first_vit_wake = False

            # Handle the question
            self.handle_question(question)

    # =========================================================================
    # INPUT HANDLING METHODS
    # =========================================================================

    def get_user_input(self, first_keyb_wake, first_vit_wake):
        """Get input from ASR or keyboard."""
        if not self.asr:
            return self.get_keyboard_input()

        # Small sleep to avoid 100% CPU usage
        time.sleep(0.01)

        if self.kasr_wake:
            return self.get_kasr_input(first_keyb_wake)
        elif self.vit:
            return self.get_vit_input(first_vit_wake)
        else:
            return self.process_asr_output()

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

    def get_kasr_input(self, first_keyb_wake):
        """Get keyboard-triggered ASR input."""
        if first_keyb_wake or not self.continuous:
            if select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1)
                if key == "\n":
                    print(self.config.listening_info)
                    return self.process_asr_output()
            return ""
        else:
            print(self.config.listening_info)
            return self.process_asr_output()

    def get_vit_input(self, first_vit_wake):
        """Get VIT wake word triggered ASR input."""
        if first_vit_wake:
            vit_info_string = self.wait_for_wake_word()
            if vit_info_string == "":
                return ""

            self.play_ww_sound()

            if vit_info_string != self.vit.bypass_vit_asr_wwd:
                if self.gui:
                    self.gui.send_wwd(vit_info_string)
                return self.process_asr_output()

            self.vit.disable()
            return ""
        else:
            print(self.config.listening_info)
            return self.process_asr_output()

    def wait_for_wake_word(self):
        """Wait for wake word detection."""
        vit_info_string = ""
        while "WWD" not in vit_info_string:
            vit_info_string = self.vit.get_info(self.stop_threads)
            if "VIS" in vit_info_string and self.gui:
                self.gui.send_vis()

        logger.debug("Wakeword received: " + vit_info_string)
        self.vit.disable()
        return vit_info_string

    def process_asr_output(self):
        """Process ASR output and provide user feedback."""
        question = self.get_asr_output()

        if question == "":
            question = "stop"
            self.egf_print("ASR: No speech detected", color=Fore.YELLOW)
        else:
            self.egf_print(f"ASR: {question}")

        return question

    def get_asr_output(self) -> str:
        text = ""
        for text in self.asr.mic_to_text():
            if self.gui and text:
                self.gui.send_qst(text)

        if self.gui:
            self.gui.send_qst(end_token)
        return text

    # =========================================================================
    # QUESTION HANDLING AND PROCESSING
    # =========================================================================

    def handle_question(self, question):
        """Handle user question through the pipeline."""
        prompt = self.config.default_system_prompt

        if question != "":
            if self.query_classifier:
                start_rag_time = time.time()
                query_category, chunk_list, _, metadata_list = self.query_classifier(query=question)
                prompt = (self.config.default_system_prompt, chunk_list)
                rag_time = time.time() - start_rag_time
                self.add_benchmark_stats(rag_time=rag_time)
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
                        if self.llm:
                            self.get_llm_output(question, prompt)
                        else:
                            self.process_complete_response(chunk_list[0])  # RAG only response
            else:
                if self.llm:
                    self.get_llm_output(question, prompt)
                else:
                    self.process_complete_response(question)  # Parrot back question if no LLM & no RAG

            self.wait_tts()
            if self.gui:
                self.gui.send_thf()
            if self.asr and not self.continuous:
                if self.kasr_wake and not self.benchmark:
                    print(self.config.start_kasr_info)
                elif self.vit:
                    self.vit.enable()
                    print(self.config.start_vasr_info)

    def get_llm_output(self, question, prompt):
        """Process question through LLM."""
        self.tts_play_sound()

        start_time = time.time()
        time_to_first_token, token_count = 0, 0
        full_response = ""

        for token_count, decoded_token in enumerate(self.llm(query=question, prompt=prompt), start=1):
            if token_count == 1:
                time_to_first_token = time.time() - start_time

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
                self.benchmark_logger.log(f"{self.llm.name}: {self.llm.llm_input_size} input tokens, "
                                          f"gen {token_count} tokens in {total_time:0.2f}s "
                                          f"=> {token_per_second:0.2f}tok/s, ttft = {time_to_first_token:0.2f}s\n")

    def process_complete_response(self, response: str):
        """Process complete response (no LLM OR no LLM & no RAG)."""
        self.tts_play_sound()
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
        self.tts_play_sound()
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
            if self.gui:
                self.gui.send_cmd(intent)
                self.tts_process(intent, eos=True)
                self.tts_play_sound()

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
            if self.asr and self.asr.running.is_set():
                self.asr.stop_threads()

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
            # When we get a keyboard question from gui, we simulate a vit wake word + asr recognition to pass through the pipeline
            self.vit.bypass(bypass_asr=True)
            self.handle_question(msg)

    # =========================================================================
    # AUDIO AND TTS METHODS
    # =========================================================================

    def play_ww_sound(self):
        """Play Wake Word detection notification sound with AlsaPlayer."""
        if self.config.play_wake_word_sound:
            logger.debug("AlsaPlayer initialized")
            player = AlsaPlayer(device_name=self.playback_device, samplerate=16000)
            player.play_stereo(self.ww_sound)
            logger.debug("Wake Word earcon played")
            player.close()
            logger.debug("AlsaPlayer closed")

    def tts_play_sound(self):
        """Play TTS notification sound if configured."""
        if self.tts and self.config.play_tts_sound:
            self.tts(waveform=self.tts_sound)

    def tts_process(self, text=None, eos=False):
        """Process text through TTS (start threads if necessary and process text)."""
        if self.tts:
            if eos and text is not None:
                segments = re.split(r"(?<=[.,;:!?])\s*(?=[A-Za-z])", text)
                for seg in segments:
                    self.tts(" " + seg)
                self.tts(eos=eos)
            else:
                self.tts(text, eos=eos)

    def wait_tts(self):
        """Wait until the speech is completely played."""
        if self.tts:
            self.tts.join()
            if self.gui:
                self.gui.send_thf()

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
            (self.asr, lambda: self.asr.stop_threads()),
            (self.gui, lambda: self.gui.send_disconnect()),
            (self.vit, lambda: self.vit.shutdown()),
            (self.mq_to_c, lambda: self.mq_to_c.close()),
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
        logger.warning("Ctrl-C or termination signal detected. Exiting...")

        # Force terminal reset
        try:
            subprocess.run(["stty", "echo"], check=False, timeout=1)
            subprocess.run(["stty", "sane"], check=False, timeout=1)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        if self.gui:
            self.gui.send_disconnect()
        self.clean_up()
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

def process_arguments(input_mode, capture_device, llm_model, output_mode, playback_device, asr_model, logging_level, module_configs):
    """Process and normalize command line arguments."""
    return {
        "input_mode": input_mode.value,
        "capture_device": capture_device.value if capture_device else None,
        "llm_model": llm_model.value if llm_model and llm_model.value != "no_llm" else None,
        "output_mode": output_mode.value,
        "playback_device": playback_device.value if playback_device else None,
        "asr_model": asr_model.value if asr_model else None,
        "logging_level": logging._nameToLevel[logging_level.value],
        "gui_config_class": module_configs["gui"]["config_classes"].get(input_mode.value) if input_mode.value in module_configs["gui"]["modules"] else None,
    }


def create_pipeline(config, processed_args, wake_word_model, system_prompt, use_rag, use_neutron, continuous, benchmark, verbose):
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
        processed_args["asr_model"],
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

    args_manager = ArgumentManager(
        config=config,
        directory_path=directory_path
    )
    module_configs = args_manager.get_module_configs()
    arg_enums, cli_options = args_manager.get_arguments_and_options()

    @app.command()
    def parse_args(
        input_mode: arg_enums["InputModesArgs"] = cli_options["input_mode"],
        capture_device: arg_enums["CaptureDeviceArgs"] = cli_options["capture_device"],
        llm_model: arg_enums["LlmArgs"] = cli_options["llm_model"],
        system_prompt: str = cli_options["system_prompt"],
        output_mode: arg_enums["OutputModesArgs"] = cli_options["output_mode"],
        playback_device: arg_enums["PlaybackDeviceArgs"] = cli_options["playback_device"],
        asr_model: arg_enums["AsrArgs"] = cli_options["asr_model"],
        use_rag: bool = cli_options["use_rag"],
        use_neutron: bool = typer.Option(False, "--use-neutron", "-n", help="Use Neutron ONNX Execution Provider."),
        wake_word_model: str = cli_options["wake_word_model"],
        continuous: bool = typer.Option(False, "--continuous", "-c", help="Continuous mode where ASR is always listening."),
        benchmark: bool = typer.Option(False, "--benchmark", "-b", help="Benchmark mode - take a list of questions and store the results."),
        verbose: bool = typer.Option(False, "--verbose", "-v", help="Display information (e.g. inference times)."),
        logging_level: arg_enums["LoggingLevel"] = typer.Option("WARNING", "--logging-level", "-l", help="Level of displayed information."),
        use_traceback: bool = typer.Option(False, "--use-traceback", help="Activate Typer traceback of exceptions and errors."),
    ):
        """Parse command line arguments and run the pipeline."""
        app.pretty_exceptions_enable = use_traceback

        # Process arguments
        processed_args = process_arguments(input_mode, capture_device, llm_model, output_mode, playback_device, asr_model, logging_level, module_configs)


        # Setup logging
        setup_logging(level=processed_args["logging_level"], root_path=directory_path)


        # Create and run pipeline
        pipeline = create_pipeline(config, processed_args, wake_word_model, system_prompt, use_rag, use_neutron, continuous, benchmark, verbose)
        pipeline.run()

    app()


if __name__ == "__main__":
    main()
