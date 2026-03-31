# Copyright 2025-2026 NXP
# NXP Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.

from datetime import datetime
import json
import os
import subprocess
import sys
import threading
import time
import wave
import contextlib
import logging
import onnxruntime as ort
import psutil
from eiq_genai_flow import eIQGenAIFlow
from tests.benchmark.questions_to_wav import generate_wav_files
from utils.utils import get_git_commit_sha, get_installed_versions, get_linux_version, get_sha256, get_neutron_info

logger = logging.getLogger(__name__)


class Benchmark(eIQGenAIFlow):
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
        # =====================================================================
        # STEP 1: Initialize benchmark-specific attributes FIRST
        # =====================================================================
        logger.info("Initializing Benchmark mode...")

        # Store input mode parameters
        self.benchmark_input_mode = input_mode
        self.is_vasr_mode = input_mode == "vasr"
        self.is_kasr_mode = input_mode == "kasr"

        # Initialize benchmark logger
        self.benchmark_logger = BenchmarkLogger()

        # Initialize benchmark metrics
        self._init_benchmark_attributes()

        logger.info(
            f"Benchmark mode: {input_mode} "
            f"({'VASR - VIT wake word' if self.is_vasr_mode else 'KASR - keyboard trigger'})"
        )
        # =====================================================================
        # STEP 2: Setup virtual audio for VASR/KASR modes (before parent init)
        # =====================================================================
        if self.is_vasr_mode or self.is_kasr_mode:
            self._setup_virtual_audio_loopback()
            # Override capture device to use loopback
            original_capture_device = capture_device
            capture_device = "plughw:CARD=Loopback"
            logger.debug(
                f"{'VASR' if self.is_vasr_mode else 'KASR'} mode: "
                f"overriding capture device from '{original_capture_device}' to '{capture_device}'"
            )
        # =====================================================================
        # STEP 3: Call parent constructor
        # =====================================================================
        logger.debug("Calling parent eIQGenAIFlow constructor...")
        super().__init__(
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
        )

        #  Define wake word for benchmark mode (TODO: get it from the model)
        self.wake_word = "Hey NXP"

        # Benchmark-specific state for run loop
        self.benchmark_questions = []
        self.benchmark_wav_files = []
        self.current_question_idx = 0
        self.benchmark_stats = None
        self.benchmark_start_time = None

        logger.debug("Benchmark initialization complete")

    def _init_benchmark_attributes(self):
        """Initialize benchmark-specific tracking variables."""
        logger.debug("Initializing benchmark tracking attributes...")

        # RAG metrics
        self.rag_sum_time = 0

        # LLM metrics
        self.llm_sum_ttft = 0
        self.llm_min_ttft = 1000
        self.llm_max_ttft = 0
        self.llm_sum_tps = 0
        self.llm_min_tps = 1000
        self.llm_max_tps = 0
        self.llm_inf_count = 0
        self.llm_sum_time = 0

        # VIT metrics (for VASR mode)
        self.vit_sum_time = 0
        self.wakeword_missed = 0

        # STT metrics
        self.error_rate_computer = None

        # Additional tracking
        self.stt_sum_time = 0
        self.ttfa_sum_time = 0
        self.ttfa_question_count = 0
        self.tts_sum_time = 0
        self.wav_sum_duration = 0

        # Initialize TTFA tracking variable
        self.ttfa_start_time = None

        logger.debug("Benchmark attributes initialized")

    def _setup_virtual_audio_loopback(self):
        """Setup virtual audio loopback for STT benchmark mode."""
        logger.debug("Setting up virtual audio loopback for STT benchmark...")

        # Find setup script
        setup_script = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "setup_audio_loopback.sh"
        )

        if not os.path.exists(setup_script):
            error_msg = f"Setup script not found: {setup_script}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            # Make script executable
            subprocess.run(["chmod", "+x", setup_script], check=True)

            # Run setup script
            logger.debug(f"Running setup script: {setup_script}")
            result = subprocess.run([setup_script], capture_output=True, text=True, check=True)

            logger.info("✓ Virtual audio loopback setup complete")
            logger.debug(f"Setup output:\n{result.stdout}")

        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to setup virtual audio: {e.stderr}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def run(self):
        """Run benchmark - wraps parent run() with benchmark setup/teardown."""
        # =====================================================================
        # PRE-BENCHMARK SETUP
        # =====================================================================
        logger.info("Starting benchmark run...")
        # Read questions and filter out empty lines
        with open(self.config.benchmark_questions_file) as file:
            self.benchmark_questions = [line.rstrip() for line in file if line.strip()]
        bench_len = len(self.benchmark_questions)

        self.benchmark_logger.log(f"Total questions: {bench_len}")

        # =====================================================================
        # GENERATE WAV FILES
        # =====================================================================
        if self.stt:
            from speech_to_text.utils.utils import ErrorRateComputer

            self.error_rate_computer = ErrorRateComputer()

            if self.is_vasr_mode:
                # VASR: Generate wake-word-prefixed WAV files with noise and prefix (VIT requirement)
                logger.info(f"Generating VASR WAV files with wake word: '{self.wake_word}'")
                self.benchmark_wav_files = generate_wav_files(
                    wav_dir=self.config.tests_data_path,
                    text_file_path=self.config.benchmark_questions_file,
                    text_file_len=bench_len,
                    wake_word="Hey NXP",
                    add_noise=True,
                    snr_db=40.0,
                    noise_type="pink",
                    noise_prefix_duration=0.8,
                )
                self.benchmark_logger.log(
                    f"{len(self.benchmark_wav_files)} wake-word prefixed audio files (wake word: '{self.wake_word}')"
                )
            else:
                # KASR: Standard WAV generation (no wake word)
                logger.info("Generating KASR WAV files (no wake word)")

                # Generate WAV files without noise prefix
                self.benchmark_wav_files = generate_wav_files(
                    wav_dir=self.config.tests_data_path,
                    text_file_path=self.config.benchmark_questions_file,
                    text_file_len=bench_len,
                    wake_word=None,
                    add_noise=True,
                    snr_db=40.0,
                    noise_type="pink",
                    noise_prefix_duration=0,
                )
                self.benchmark_logger.log(f"{len(self.benchmark_wav_files)} audio files found")

        # Setup monitoring
        self.benchmark_stats = {
            "cpu_usage": [],
            "memory_used": [],
            "memory_percent": [],
            "running": True,
        }
        monitor_thread = threading.Thread(target=monitor_system, args=(self.benchmark_stats,))
        monitor_thread.start()

        # Setup logging
        filename = self._generate_benchmark_filename()
        self.benchmark_logger.set_log_file(filename + ".log")
        self.benchmark_logger.clear_log_file()
        self._log_benchmark_header()

        # Start timer
        self.benchmark_start_time = time.perf_counter()

        # =====================================================================
        # RUN PARENT'S MAIN LOOP
        # =====================================================================
        logger.info("Starting parent's run() loop...")
        super().run()

        # =====================================================================
        # POST-BENCHMARK TEARDOWN
        # =====================================================================
        total_time = time.perf_counter() - self.benchmark_start_time
        self.benchmark_stats["running"] = False
        monitor_thread.join()

        # Print summary and save results
        self._print_benchmark_summary(self.benchmark_stats, total_time, bench_len, filename)

        # Cleanup and exit
        self.clean_up()
        os._exit(0)

    def get_user_input(self):
        """
        Override parent method to provide benchmark input.

        This method is called by the parent's run() loop.
        It replaces keyboard/mic input with file-based input for benchmarking.
        """
        # Check if we've processed all questions
        if self.current_question_idx >= len(self.benchmark_questions):
            logger.info("All benchmark questions processed")
            self.stop_threads = True  # This will exit parent's while loop
            return ""

        question_text = self.benchmark_questions[self.current_question_idx]

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Question {self.current_question_idx + 1}/{len(self.benchmark_questions)}: {question_text}")
        logger.info(f"{'=' * 60}")

        self._reset_tts_state()

        self.benchmark_logger.log(f"\n{'=' * 80}")
        self.benchmark_logger.log(f"Question {self.current_question_idx + 1}: {question_text}")

        if self.is_vasr_mode or self.is_kasr_mode:
            # STT modes - retrieve WAV file
            wav_file = self.benchmark_wav_files[self.current_question_idx]

            if self.is_vasr_mode:
                return self._get_vasr_input(wav_file, question_text)
            else:  # self.is_kasr_mode
                return self._get_kasr_input(wav_file, question_text)
        else:
            # Keyboard text input mode - just return the question
            self.current_question_idx += 1
            return question_text

    def _get_vasr_input(self, wav_file, question_text):
        """
        VASR mode: inject audio, wait for VIT, then let parent handle STT.

        Args:
            wav_file: Path to the WAV file to process
            question_text: Expected transcription text
        """
        # Handle VIT wake word detection
        if not self._wait_and_handle_wake_word(wav_file):
            return ""  # Wake word missed

        # Process STT (audio already injected during VIT detection)
        self.current_question_idx += 1
        question = self._process_stt_after_wake_word(wav_file, question_text)

        # Set TTFA start time to when user STOPPED speaking (VAD end timestamp)
        result_info = self.stt.get_detailed_info()
        if result_info and "end_timestamp" in result_info:
            self.ttfa_start_time = result_info["end_timestamp"]
            logger.debug(
                f"[TTFA] Question {self.current_question_idx}: "
                f"TTFA start time set to speech end: {self.ttfa_start_time:.4f}s"
            )
        else:
            logger.warning(f"[TTFA] Question {self.current_question_idx}: No end_timestamp available from STT")

        return question

    def _get_kasr_input(self, wav_file, question_text):
        """
        KASR mode: sync STT to current, then immediately inject audio.

        Args:
            wav_file: Path to the WAV file to process
            question_text: Expected transcription text

        Returns:
            str: Transcribed question or "TIMEOUT"
        """
        # Ensure capture is running before injection
        if not self.audio_manager.is_capture_running():
            logger.info("Capture not running, restarting...")
            self.audio_manager.start_capture()

            # Wait for capture to be ready
            max_wait = 2.0
            wait_interval = 0.1
            elapsed = 0
            while not self.audio_manager.is_capture_running() and elapsed < max_wait:
                time.sleep(wait_interval)
                elapsed += wait_interval

            if not self.audio_manager.is_capture_running():
                logger.error("Capture failed to start within timeout!")
                self.current_question_idx += 1
                return "TIMEOUT"

            # Give pipeline time to stabilize
            time.sleep(0.3)

        # Ensure STT is enabled and synced
        if self.stt:
            if not self.stt.is_running:
                logger.info("Enabling STT adapter and syncing to current buffer...")
                self.stt.enable(sync_to_current=True)
            else:
                # STT is already running, but re-sync the reader to current buffer
                logger.info("STT already running, re-syncing reader to current buffer...")
                stt_reader = self.audio_manager.get_reader("STT")
                if stt_reader and stt_reader.enabled:
                    # Sync to current to skip any stale buffered data
                    current_idx = self.audio_manager.write_index
                    stt_reader.read_index = current_idx
                    logger.debug(f"Reader synced to index {current_idx}")

        # Now inject audio
        logger.info(f"Injecting audio: {os.path.basename(wav_file)}")
        inject_proc = subprocess.Popen(
            ["aplay", "-D", "fake_input", wav_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        # Small delay to ensure audio starts flowing
        time.sleep(0.1)

        # Process STT
        self.current_question_idx += 1
        stt_start = time.perf_counter()
        question = self.process_stt_output()
        stt_time = time.perf_counter() - stt_start
        self.stt_sum_time += stt_time

        logger.info(f"STT completed in {stt_time:.2f}s")

        # Calculate WER
        if question and question != "TIMEOUT":
            self._calculate_wer(question, question_text)
            logger.info(f"Expected: {question_text}")
            logger.info(f"Got: {question}")

        # Cleanup injection
        self._cleanup_audio_injection(inject_proc)

        # Set TTFA timestamp
        result_info = self.stt.get_detailed_info()
        if result_info and "end_timestamp" in result_info:
            self.ttfa_start_time = result_info["end_timestamp"]
            logger.debug(f"[TTFA] KASR: TTFA start time = {self.ttfa_start_time:.4f}s")

        return question if question else "TIMEOUT"

    # =========================================================================
    # HELPER METHODS FOR INPUT PROCESSING
    # =========================================================================

    def _reset_tts_state(self):
        """Reset TTS state at the beginning of each question."""

        # Simply reset to None (always exists now)
        self.ttfa_start_time = None
        logger.debug("Reset ttfa_start_time from previous question")

        # Reset TTS state if TTS is enabled
        if self.tts:
            try:
                del self.tts.timestamp_ttfa
            except AttributeError:
                pass  # Not set yet

            self.tts.start_play = True
            logger.debug("[TTFA] Question %d: TTS state reset", self.current_question_idx + 1)

    def add_benchmark_stats(self, rag_time=0, llm_ttft=0, llm_tps=0, llm_time=0):
        """Add benchmark statistics (called by parent's handle_question)."""
        # LLM metrics
        if self.llm and llm_ttft and llm_tps and llm_time:
            self.llm_inf_count += 1
            self.llm_sum_ttft += llm_ttft
            self.llm_min_ttft = min(self.llm_min_ttft, llm_ttft)
            self.llm_max_ttft = max(self.llm_max_ttft, llm_ttft)
            self.llm_sum_tps += llm_tps
            self.llm_min_tps = min(self.llm_min_tps, llm_tps)
            self.llm_max_tps = max(self.llm_max_tps, llm_tps)
            self.llm_sum_time += llm_time

        # RAG metrics
        if self.retriever and rag_time:
            self.rag_sum_time += rag_time

        # TTFA calculation
        if not self.tts:
            return  # No TTS, skip TTFA calculation

        # Get TTS timestamp (may not exist if TTS hasn't run yet)
        try:
            tts_timestamp = self.tts.timestamp_ttfa
        except AttributeError:
            logger.debug(
                "[TTFA] Question %d: TTS hasn't run yet - skipping TTFA calculation", self.current_question_idx
            )
            return

        # Calculate TTFA if we have both timestamps
        if self.ttfa_start_time is not None:
            ttfa = tts_timestamp - self.ttfa_start_time
            self.ttfa_sum_time += ttfa
            self.ttfa_question_count += 1

            logger.info(
                "[TTFA] Question %d: ✓ CALCULATED TTFA = %.4fs | start=%.4f, end=%.4f | Total: %.4fs, Count: %d",
                self.current_question_idx,
                ttfa,
                self.ttfa_start_time,
                tts_timestamp,
                self.ttfa_sum_time,
                self.ttfa_question_count,
            )

            # Reset for next question
            self.ttfa_start_time = None
        else:
            logger.debug(
                "[TTFA] Question %d: ttfa_start_time not set - skipping TTFA calculation", self.current_question_idx
            )

        # TTS metrics
        try:
            self.tts_sum_time += self.tts.inference_time
        except AttributeError:
            pass  # inference_time not set yet

    def _wait_and_handle_wake_word(self, wav_file):
        """
        Handle VIT wake word detection for VASR mode.

        Returns:
            bool: True if wake word detected, False if missed
        """
        # Enable VIT if not already running
        if self.vit and not self.vit.is_running:
            self.vit.enable(clear_buffer=True)

        # Inject audio
        logger.info(f"Injecting: {os.path.basename(wav_file)}")
        vit_start = time.perf_counter()
        inject_proc = subprocess.Popen(
            ["aplay", "-D", "fake_input", wav_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        # Wait for VIT with timeout
        vit_result = self._wait_for_vit_with_timeout(timeout=3.0)

        if not vit_result or "WWD" not in vit_result:
            # Wake word missed - cleanup and prepare for next question
            self._handle_wake_word_miss(inject_proc)
            return False

        # Wake word detected - process result
        self._handle_wake_word_success(vit_result, vit_start)
        return True

    def _handle_wake_word_miss(self, inject_proc):
        """Handle wake word miss scenario - cleanup and reset VIT."""
        self.wakeword_missed += 1
        ww_detected_so_far = (self.current_question_idx + 1) - self.wakeword_missed
        ww_detection_rate = (ww_detected_so_far / (self.current_question_idx + 1)) * 100

        logger.warning(
            f"✗ Wake word missed ({self.wakeword_missed}/{self.current_question_idx + 1}) - "
            f"Detection rate: {ww_detection_rate:.1f}%"
        )

        # Log wake word miss
        self.benchmark_logger.log(
            f"VIT: Wake word MISSED (total missed: {self.wakeword_missed}, "
            f"detection rate: {ww_detection_rate:.1f}%)"
        )

        # Kill audio injection process
        inject_proc.terminate()
        inject_proc.wait()

        # Disable VIT to stop its worker thread
        if self.vit and self.vit.is_running:
            logger.info("Disabling VIT after timeout...")
            self.vit.disable()

        # Reset VIT state for next question
        if self.vit:
            logger.info("Resetting VIT for next question...")
            self.vit.vit.reset()

        # Re-enable VIT for next question
        if self.vit:
            logger.info("Re-enabling VIT for next question...")
            self.vit.enable(clear_buffer=True)

        self.current_question_idx += 1

    def _handle_wake_word_success(self, vit_result, vit_start):
        """Process successful wake word detection."""
        vit_time = time.perf_counter() - vit_start
        self.vit_sum_time += vit_time
        ww_detected_so_far = (self.current_question_idx + 1) - self.wakeword_missed
        ww_detection_rate = (ww_detected_so_far / (self.current_question_idx + 1)) * 100

        logger.info(f"✓ Wake word detected: {vit_result} ({vit_time:.2f}s) - Detection rate: {ww_detection_rate:.1f}%")
        self.benchmark_logger.log(f"VIT: Wake word DETECTED ({vit_time:.2f}s, "
                                  f"detection rate: {ww_detection_rate:.1f}%)")

        # Position STT reader after wake word
        detection_info = self.vit.get_detailed_info()
        if detection_info:
            ww_end_abs_idx = detection_info["ww_end_abs_index"]
            stt_reader = self.audio_manager.get_reader("STT")
            if stt_reader:
                stt_reader.read_index = ww_end_abs_idx
                if not stt_reader.enabled:
                    stt_reader.enable(sync_to_current=False)

        # Disable VIT (will be re-enabled by parent's restart_wake)
        if self.vit.is_running:
            self.vit.disable()

    def _process_stt_after_wake_word(self, wav_file, question_text):
        """
        Process STT after wake word detection (VASR mode only).
        Audio has already been injected during VIT wake word detection.

        Args:
            wav_file: Path to WAV file (for reference/logging)
            question_text: Expected transcription text

        Returns:
            str: Transcribed question or "TIMEOUT"
        """
        # Track STT timing
        stt_start = time.perf_counter()
        question = self.process_stt_output()
        stt_time = time.perf_counter() - stt_start
        self.stt_sum_time += stt_time

        logger.info(f"STT completed in {stt_time:.2f}s")

        # Calculate WER
        if question and question != "TIMEOUT":
            self._calculate_wer(question, question_text)
            logger.info(f"Expected: {question_text}")
            logger.info(f"Got: {question}")

        # Return result
        if question == "":
            logger.warning("STT returned empty result (timeout)")
            return "TIMEOUT"

        return question

    def _calculate_wer(self, transcribed, expected):
        """Calculate Word Error Rate for STT transcription."""
        normalized_transcribed = self.stt.stt.text_normalizer(transcribed).split(" ")
        normalized_expected = self.stt.stt.text_normalizer(expected).split(" ")

        self.error_rate_computer.append(
            ids=[self.current_question_idx - 1],  # -1 because we already incremented
            predict=[normalized_transcribed],
            target=[normalized_expected],
        )

    def _cleanup_audio_injection(self, inject_proc):
        """Cleanup audio injection subprocess."""
        if inject_proc.poll() is None:
            # Process still running
            logger.info("Terminating audio injection process...")
            try:
                inject_proc.terminate()
                inject_proc.wait(timeout=1.0)
                logger.debug("Audio injection stopped cleanly")
            except subprocess.TimeoutExpired:
                logger.warning("Audio injection didn't terminate gracefully, force killing...")
                inject_proc.kill()
                inject_proc.wait()
        else:
            logger.debug("Audio injection completed naturally")

    def _wait_for_vit_with_timeout(self, timeout=3.0):
        """Wait for VIT result with proper timeout."""
        vit_done = threading.Event()
        vit_data = {"result": None}

        def wait_vit():
            vit_data["result"] = self.vit.wait_for_result(timeout=0.1)
            vit_done.set()

        vit_thread = threading.Thread(target=wait_vit, daemon=True)
        vit_thread.start()

        if vit_done.wait(timeout=timeout):
            return vit_data["result"]
        else:
            logger.warning(f"VIT timeout after {timeout}s")
            return None

    def _generate_benchmark_filename(self):
        """Generate benchmark output filename."""
        return (
            f"Benchmark_{self.device}"
            + f"{('_neutron' if self.use_neutron else '_CPU')}"
            + f"{('_' + self.stt_model + '-stt' if self.stt else '')}"
            + f"{('_rag' if self.retriever else '')}"
            + f"{('_' + self.llm.name + '-llm' if self.llm else '_no_llm')}"
            + f"{('_' + self.output_mode if self.tts else '')}"
            + f"{('_' + self.benchmark_input_mode)}"
            + datetime.now().strftime("_%Y%m%d_%H%M%S_%f")
        )

    def _log_benchmark_header(self):
        """Log benchmark header information."""
        self.linux_version = get_linux_version()
        self.benchmark_logger.log(
            f"Benchmarking: {(self.stt_model + ', ' if self.stt else '')}"
            f"{(self.llm.name if self.llm else '_no_llm')}"
            f"{(' with RAG' if self.retriever else '')}"
            f"{(', with TTS' if self.tts else '')} "
            f"in {self.benchmark_input_mode} mode on {self.full_machine} "
            f"{'using neutron' if self.use_neutron else 'using CPU'}"
        )
        print_benchmark_system_info(self.benchmark_logger, self.config)
        if self.llm:
            self.benchmark_logger.log(f"Actual ORT Execution Providers: {self.llm.actual_providers}\n")

    def _print_benchmark_summary(self, stats, total_time, bench_len, filename):
        """Print benchmark summary and save results."""
        # =====================================================================
        # CALCULATE METRICS
        # =====================================================================
        avg_cpu = sum(stats["cpu_usage"]) / len(stats["cpu_usage"]) if stats["cpu_usage"] else 0
        min_cpu = min(stats["cpu_usage"]) if stats["cpu_usage"] else 0
        max_cpu = max(stats["cpu_usage"]) if stats["cpu_usage"] else 0

        # Calculate questions actually processed (excluding missed wake words)
        questions_processed = bench_len - self.wakeword_missed if self.is_vasr_mode else bench_len

        # Calculate TTFA average using only questions that had valid TTFA
        ttfa_avg = self.ttfa_sum_time / self.ttfa_question_count if self.ttfa_question_count > 0 else 0

        logger.info(
            f"[TTFA] Final Summary: ttfa_sum_time={self.ttfa_sum_time:.4f}s, "
            f"ttfa_question_count={self.ttfa_question_count}, ttfa_avg={ttfa_avg:.4f}s"
        )

        # Calculate WER
        stt_wer = 0
        if self.stt and self.error_rate_computer and self.error_rate_computer.scores:
            stt_wer = self.error_rate_computer.summarize("WER")
            file_name = f"WER_model-{self.stt_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            save_path = os.path.join(os.path.dirname(__file__), "results", f"{file_name}.log")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w") as w:
                self.error_rate_computer.write_stats(w)
            print(f"STT WER saved in {save_path}")
        else:
            stt_wer = None

        llm_avg_ttft = 0
        llm_avg_tps = 0
        if self.llm_inf_count:
            llm_avg_ttft = self.llm_sum_ttft / self.llm_inf_count
            llm_avg_tps = self.llm_sum_tps / self.llm_inf_count

        tts_avg_time = 0
        tts_avg_rtf = 0
        if self.tts:
            tts_metrics = self.tts.metrics
            rtf_list = tts_metrics["rtf"]
            if len(rtf_list) > 0:
                tts_avg_rtf = sum(rtf_list) / len(rtf_list)
                tts_avg_time = self.tts_sum_time / questions_processed if questions_processed > 0 else 0

        avg_memory_used = sum(stats["memory_used"]) / len(stats["memory_used"]) if stats["memory_used"] else 0
        avg_memory_percent = (
            sum(stats["memory_percent"]) / len(stats["memory_percent"]) if stats["memory_percent"] else 0
        )

        # =====================================================================
        # PRINT SUMMARY
        # =====================================================================
        self.benchmark_logger.log("\n=== Benchmark Summary ===")
        self.benchmark_logger.log(f"Platform: {self.full_machine}")
        self.benchmark_logger.log(f"Linux Kernel: {self.linux_version}")
        self.benchmark_logger.log(f"Input Mode: {self.benchmark_input_mode}")
        self.benchmark_logger.log(f"Total Benchmark Questions: {bench_len}")
        if self.is_vasr_mode:
            self.benchmark_logger.log(f"Questions Processed: {questions_processed} (missed {self.wakeword_missed})")
            # Report TTFA question count
            self.benchmark_logger.log(f"Questions with Valid TTFA: {self.ttfa_question_count}/{questions_processed}")
        self.benchmark_logger.log(f"NPU: {'ON' if self.use_neutron else 'OFF'}")

        # Component statuses
        self.benchmark_logger.log("\n--- Component Status ---")
        if self.is_vasr_mode:
            ww_detected = bench_len - self.wakeword_missed
            ww_detection_rate = (ww_detected / bench_len) * 100 if bench_len > 0 else 0
            self.benchmark_logger.log(
                f"VIT: Enabled (wake word: '{self.wake_word}') | "
                f"Detection Rate: {ww_detected}/{bench_len} ({ww_detection_rate:.1f}%)"
            )
            if self.wakeword_missed > 0:
                self.benchmark_logger.log(f"     Wake Words Missed: {self.wakeword_missed}")

        self.benchmark_logger.log(f"STT: {self.stt_model if self.stt else 'OFF'}")

        # Calculate RAG-only answers (questions processed but not answered by LLM)
        if self.retriever and questions_processed > 0:
            rag_only_answered = questions_processed - self.llm_inf_count
            self.benchmark_logger.log(
                f"RAG: {self.retriever.embedding_model.name} | "
                f"Answered Directly: {rag_only_answered}/{questions_processed}"
            )
        else:
            self.benchmark_logger.log("RAG: OFF")

        # LLM statistics
        if self.llm:
            if questions_processed > 0:
                llm_usage_rate = (self.llm_inf_count / questions_processed) * 100
                self.benchmark_logger.log(
                    f"LLM: {self.llm.name} | "
                    f"Answered: {self.llm_inf_count}/{questions_processed} ({llm_usage_rate:.1f}%)"
                )
            else:
                self.benchmark_logger.log(f"LLM: {self.llm.name} | Answered: 0/0 (no questions processed)")
        else:
            self.benchmark_logger.log("LLM: OFF")

        self.benchmark_logger.log(
            f"TTS: {'ON' if self.tts else 'OFF'}"
            + (f" | Model: {self.tts.model_name}, Mode: {self.output_mode}" if self.tts else "")
        )

        # Timing and performance
        self.benchmark_logger.log("\n--- Performance Metrics ---")
        self.benchmark_logger.log(
            f"Benchmark Time: Avg = {total_time / questions_processed if questions_processed > 0 else 0:.2f}s | "
            f"Total = {total_time:.2f}s"
        )
        # Report TTFA with context
        if self.ttfa_question_count > 0:
            self.benchmark_logger.log(
                f"TTFA: Avg = {ttfa_avg:.2f}s (measured for {self.ttfa_question_count}/{questions_processed} questions)"
            )
        else:
            self.benchmark_logger.log("TTFA: N/A (no valid measurements)")

        self.benchmark_logger.log(f"CPU Usage: Avg = {avg_cpu:.2f}% | Min = {min_cpu:.2f}% | Max = {max_cpu:.2f}%")
        self.benchmark_logger.log(f"Memory Usage: Avg = {avg_memory_used:.2f} MB ({avg_memory_percent:.2f}%)")
        # VIT metrics (VASR mode only)
        if self.is_vasr_mode and self.vit_sum_time > 0:
            ww_detected = bench_len - self.wakeword_missed
            vit_avg_time = self.vit_sum_time / ww_detected if ww_detected > 0 else 0
            ww_detection_rate = (ww_detected / bench_len) * 100 if bench_len > 0 else 0

            self.benchmark_logger.log("\n--- VIT Metrics ---")
            self.benchmark_logger.log(f"Wake Word Detection Rate: {ww_detected}/{bench_len} ({ww_detection_rate:.1f}%)")
            self.benchmark_logger.log(f"Wake Words Missed: {self.wakeword_missed}")
            self.benchmark_logger.log(
                f"Detection Time (successful only): Avg = {vit_avg_time:.2f}s | Total = {self.vit_sum_time:.2f}s"
            )

        # STT metrics
        if self.stt:
            self.benchmark_logger.log("\n--- STT Metrics ---")
            self.benchmark_logger.log(f"Init Time: {self.stt_init_time:.2f}s")
            self.benchmark_logger.log(
                "Processing Time: Avg = "
                f"{self.stt_sum_time / questions_processed if questions_processed > 0 else 0:.2f}s | "
                f"Total = {self.stt_sum_time:.2f}s"
            )
            self.benchmark_logger.log(
                f"Word Error Rate: {stt_wer:.2f}%" if stt_wer is not None else "Word Error Rate: N/A"
            )
            self.benchmark_logger.log(
                f"Wave file duration: Avg = {self.wav_sum_duration / bench_len if bench_len > 0 else 0:.2f}s"
            )

        # RAG metrics
        if self.retriever:
            rag_avg_time = self.rag_sum_time / questions_processed if questions_processed > 0 else 0
            self.benchmark_logger.log("\n--- RAG Metrics ---")
            self.benchmark_logger.log(f"Init Time: {self.rag_init_time:.2f}s")
            self.benchmark_logger.log(f"Processing Time: Avg = {rag_avg_time:.2f}s | Total = {self.rag_sum_time:.2f}s")

        # LLM metrics
        if self.llm_inf_count:
            llm_avg_time = self.llm_sum_time / questions_processed if questions_processed > 0 else 0
            self.benchmark_logger.log("\n--- LLM Metrics ---")
            self.benchmark_logger.log(f"Init Time: {self.llm_init_time:.2f}s")
            self.benchmark_logger.log(f"Processing Time: Avg = {llm_avg_time:.2f}s | Total = {self.llm_sum_time:.2f}s")
            self.benchmark_logger.log(
                f"TTFT: Avg = {llm_avg_ttft:.2f}s | Min = {self.llm_min_ttft:.2f}s | Max = {self.llm_max_ttft:.2f}s"
            )
            self.benchmark_logger.log(
                f"Tokens/sec: Avg = {llm_avg_tps:.2f} | Min = {self.llm_min_tps:.2f} | Max = {self.llm_max_tps:.2f}"
            )

        # TTS metrics
        if self.tts:
            self.benchmark_logger.log("\n--- TTS Metrics ---")
            self.benchmark_logger.log(f"Init Time: {self.tts_init_time:.2f}s")
            self.benchmark_logger.log(f"Processing Time: Avg = {tts_avg_time:.2f}s | Total = {self.tts_sum_time:.2f}s")
            self.benchmark_logger.log(f"Real Time Factor: Avg = {tts_avg_rtf:.2f}")

        # =====================================================================
        # SAVE RESULTS
        # =====================================================================
        generator = MetricGenerator(
            machine=self.full_machine,
            llm=self.llm.name if self.llm else "",
            rag=self.retriever.embedding_model.name if self.retriever else "",
            tts=self.tts.model_name if self.tts else "",
            stt=self.stt_model if self.stt else "",
            linux_version=self.linux_version,
            use_npu=self.use_neutron,
        )
        metrics_data = generator.get_full_config_entry(
            ttfa_avg=ttfa_avg,
            avg_time=total_time / questions_processed if questions_processed > 0 else 0,
            avg_cpu=avg_cpu,
            avg_mem=avg_memory_used,
            stt_init_time=self.stt_init_time,
            stt_avg_time=self.stt_sum_time / questions_processed if questions_processed > 0 else 0,
            stt_wer=stt_wer,
            rag_avg_time=self.rag_sum_time / questions_processed if questions_processed > 0 and self.retriever else 0,
            rag_init_time=self.rag_init_time,
            llm_init_time=self.llm_init_time,
            llm_avg_time=self.llm_sum_time / questions_processed
            if questions_processed > 0 and self.llm_inf_count
            else 0,
            llm_tts_avg_time=0,
            llm_avg_ttft=llm_avg_ttft if self.llm_inf_count else 0,
            llm_avg_tps=llm_avg_tps if self.llm_inf_count else 0,
            tts_init_time=self.tts_init_time,
            tts_avg_rtf=tts_avg_rtf if self.tts else 0,
            tts_avg_time=tts_avg_time if self.tts else 0,
        )
        # Create single entry for JSON file
        grouped_metrics_data = {metrics_data["Platform"]: [{k: v for k, v in metrics_data.items() if k != "Platform"}]}

        save_to_json_file(grouped_metrics_data, f"{filename}.json")

        if self.config.update_global_benchmark_json:
            update_json_file(
                f"{filename}.json",
                "metrics.json",
                tolerance=0.05,
                action_on_existing="update_if_improved",
            )

        print(f"\n{'=' * 60}")
        print("Benchmark complete!")
        if self.is_vasr_mode and self.wakeword_missed > 0:
            print(f"Wake words detected: {questions_processed}/{bench_len} ({ww_detection_rate:.1f}%)")
        print(f"Results saved to: {filename}.[log/json]")
        print(f"{'=' * 60}\n")


class BenchmarkLogger:
    """Logger for benchmark output."""

    def __init__(self, log_file_path="benchmark_log.txt"):
        self.log_file_path = log_file_path

    def set_log_file(self, log_file_path):
        """Set the log file path."""
        self.log_file_path = log_file_path
        print(f"Log file path set to '{self.log_file_path}'.")

    def clear_log_file(self):
        """Clear/delete the log file."""
        if os.path.exists(self.log_file_path):
            os.remove(self.log_file_path)
            print(f"Log file '{self.log_file_path}' has been deleted.")

    def log(self, message):
        """Log message to file."""
        with open(self.log_file_path, "a") as log_file:
            print(message, file=log_file)

    def append_print(self, msg):
        """Append message without newline."""
        with open(self.log_file_path, "a") as log_file:
            log_file.write(msg)


def print_benchmark_system_info(logger, config):
    """Print system information for benchmark."""
    if logger:
        logger.log("System Info:")
        logger.log(f"Linux Kernel: {get_linux_version()}")
        logger.log(f"Neutron FW sha256: {get_sha256(config.neutron_fw_path)}")
        logger.log(f"Neutron Info: {get_neutron_info()}")
        logger.log(f"ORT build info: {ort.get_build_info()}")
        logger.log(f"ORT so sha256: {get_sha256(config.ort_lib_path)}")
        logger.log(f"Python packages: {get_installed_versions(config.python_packages_versions_to_display)}")
        logger.log(f"EGF commit sha: {get_git_commit_sha()}")
        logger.log("\n")


def monitor_system(stats, interval=0.5):
    """Continuously monitor CPU and memory usage."""
    while stats["running"]:
        cpu_percent = psutil.cpu_percent(interval=None)
        memory_info = psutil.virtual_memory()

        stats["cpu_usage"].append(cpu_percent)
        stats["memory_used"].append(memory_info.used / (1024**2))  # MB
        stats["memory_percent"].append(memory_info.percent)

        time.sleep(interval)


def get_wav_duration(file_path):
    """Get duration of WAV file in seconds."""
    with contextlib.closing(wave.open(file_path, "r")) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        return duration


def _get_metric_definitions():
    """Returns metric definitions for comparison."""
    return {
        "metrics_to_compare": [
            "time_avg",
            "llm_avg_ttft",
            "llm_avg_tps",
            "cpu_avg",
            "mem_avg",
            "stt_wer",
            "stt_avg_time",
        ],
        "higher_is_better_metrics": {"llm_avg_tps"},
        "lower_is_better_metrics": {
            "ttfa_avg",
            "time_avg",
            "llm_avg_time",
            "llm_avg_ttft",
            "cpu_avg",
            "mem_avg",
            "stt_wer",
            "stt_avg_time",
            "tts_avg_rtf",
            "tts_avg_time",
        },
        "metric_units": {
            "ttfa_avg": "seconds",
            "time_avg": "seconds",
            "cpu_avg": "%",
            "mem_avg": "MB",
            "stt_init_time": "seconds",
            "stt_avg_time": "seconds",
            "stt_wer": "%",
            "rag_init_time": "seconds",
            "rag_avg_time": "seconds",
            "llm_init_time": "seconds",
            "llm_avg_time": "seconds",
            "llm_tts_avg_time": "seconds",
            "llm_avg_ttft": "seconds",
            "llm_avg_tps": "tokens/second",
            "tts_init_time": "seconds",
            "tts_avg_time": "seconds",
            "tts_avg_rtf": "seconds",
        },
    }


def _get_base_model_name(full_name):
    """Extract base name from model path."""
    if not full_name or full_name == "OFF":
        return "OFF"
    return os.path.basename(str(full_name))


class MetricGenerator:
    """Generate metrics entries for JSON output."""

    def __init__(
        self,
        machine,
        llm=None,
        rag=None,
        tts=None,
        stt=None,
        use_npu=False,
        linux_version=None,
        git_sha=None,
    ):
        self.machine = machine
        self.llm = _get_base_model_name(llm)
        self.retriever = _get_base_model_name(rag)
        self.tts = _get_base_model_name(tts)
        self.stt = _get_base_model_name(stt)
        self.npu_status = "ON" if use_npu else "OFF"
        self.linux_version = linux_version
        self.git_sha = git_sha if git_sha is not None else get_git_commit_sha()

    def get_full_config_entry(
        self,
        ttfa_avg,
        avg_time,
        avg_cpu,
        avg_mem,
        stt_init_time,
        stt_avg_time,
        stt_wer,
        rag_init_time,
        rag_avg_time,
        llm_init_time,
        llm_avg_time,
        llm_tts_avg_time,
        llm_avg_ttft,
        llm_avg_tps,
        tts_init_time,
        tts_avg_rtf,
        tts_avg_time,
    ):
        """Generate full configuration entry with metrics."""
        return {
            "Platform": self.machine,
            "Configuration": {
                "LLM": self.llm,
                "RAG": self.retriever,
                "TTS": self.tts,
                "STT": self.stt,
                "NPU": self.npu_status,
                "Linux_Version": self.linux_version,
                "Git_SHA": self.git_sha,
            },
            "Metrics": {
                "ttfa_avg": f"{ttfa_avg:.2f}",
                "time_avg": f"{avg_time:.2f}",
                "cpu_avg": f"{avg_cpu:.2f}",
                "mem_avg": f"{avg_mem:.2f}",
                "stt_init_time": f"{stt_init_time:.2f}",
                "stt_avg_time": f"{stt_avg_time:.2f}",
                "stt_wer": f"{stt_wer:.2f}" if stt_wer is not None else "N/A",
                "rag_init_time": f"{rag_init_time:.2f}",
                "rag_avg_time": f"{rag_avg_time:.2f}",
                "llm_init_time": f"{llm_init_time:.2f}",
                "llm_avg_time": f"{llm_avg_time:.2f}",
                "llm_tts_avg_time": f"{llm_tts_avg_time:.2f}",
                "llm_avg_ttft": f"{llm_avg_ttft:.2f}",
                "llm_avg_tps": f"{llm_avg_tps:.2f}",
                "tts_init_time": f"{tts_init_time:.2f}",
                "tts_avg_time": f"{tts_avg_time:.2f}",
                "tts_avg_rtf": f"{tts_avg_rtf:.2f}",
            },
        }

    def get_unique_config_identifier(self, new_config_entry=None):
        """
        Generates a tuple representing the unique identity of a configuration.
        This is used to check if a configuration already exists in the list.
        If new_config_entry is provided, it extracts details from there;
        otherwise, it uses the instance's attributes (which are already base names).
        """
        if new_config_entry:
            config_details = new_config_entry.get("Configuration", {})
            return (
                new_config_entry.get("Platform"),
                _get_base_model_name(config_details.get("LLM")),
                _get_base_model_name(config_details.get("RAG")),
                _get_base_model_name(config_details.get("TTS")),
                _get_base_model_name(config_details.get("STT")),
                config_details.get("NPU"),
                config_details.get("Linux_Version"),
                # Git_SHA is NOT part of the unique identifier for comparison,
                # as you might want to update metrics for the same config
                # but under a new git SHA. If you want it part of the identifier, uncomment:
                # config_details.get("Git_SHA"),
            )
        else:
            # Attributes are already base names due to __init__ normalization
            return (
                self.machine,
                self.llm,
                self.retriever,
                self.tts,
                self.stt,
                self.npu_status,
                self.linux_version,
            )


def save_to_json_file(data, filename="metrics.json"):
    """Save metrics data to JSON file."""
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Metrics saved to: {filename}")


def update_json_file(
    source_json_filepath,
    target_filename="metrics.json",
    action_on_existing="compare",
    tolerance=0.05,
    lava_test_case=False,
):
    """
    Loads data from an existing JSON file (source_json_filepath), extracts its first entry
    ('Platform'), and then uses that entry to update another JSON file (target_filename).

    If a configuration already exists in target_filename:
        - If action_on_existing is "compare", it prints a comparison.
        - If action_on_existing is "update", it updates the existing entry.
        - If action_on_existing is "update_if_improved", it updates if all priority metrics are better.
    If the configuration is new, it adds it and saves the file.

    Args:
        source_json_filepath (str): The path to the JSON file containing the new
                                      configuration entry (e.g., a benchmark.json).
                                      It's expected to be in the format:
                                      {"Platform1": [entry1, entry2], "Platform2": [entry3]}
                                      We will take the *first* entry found in this file
                                      for the update.
        target_filename (str): The path to the JSON file to be updated.
        action_on_existing (str): Specifies behavior if config exists.
                                  Can be "compare" (default), "update", or "update_if_improved".
        tolerance (float): The percentage tolerance to pass to compare_metrics.
        lava_test_case (bool): If True, reports results using LAVA test case API format for comparisons.
    """

    # Helper to safely convert values to float for comparison
    def to_float(val):
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0.0

    metric_defs = _get_metric_definitions()
    metric_units = metric_defs["metric_units"]

    new_config_entry = {}
    if os.path.exists(source_json_filepath):
        try:
            with open(source_json_filepath, "r") as f:
                content = f.read().strip()
                if content:
                    loaded_source_data = json.loads(content)
                    if isinstance(loaded_source_data, dict):
                        found_entry = False
                        for platform_key, entries_list in loaded_source_data.items():
                            if entries_list:
                                # Ensure we get a deep copy to avoid modifying source_json_filepath's data
                                new_config_entry = entries_list[0].copy()
                                new_config_entry["Platform"] = platform_key
                                found_entry = True
                                print(
                                    f"Using the first entry from '{platform_key}' in "
                                    f"'{source_json_filepath}' for update."
                                )
                                break
                        if not found_entry:
                            print(
                                f"Warning: No valid configuration entries found in '{source_json_filepath}'. "
                                "No update performed."
                            )
                            return
                    else:
                        print(
                            f"Error: '{source_json_filepath}' content is not in the expected grouped dictionary "
                            "format. No update performed."
                        )
                        return
                else:
                    print(f"Warning: '{source_json_filepath}' is empty. No update performed.")
                    return
        except json.JSONDecodeError as e:
            print(f"Error: Could not decode JSON from {source_json_filepath}. No update performed. Error: {e}")
            return
    else:
        print(f"Error: Source file '{source_json_filepath}' not found. No update performed.")
        return

    data = {}

    if os.path.exists(target_filename):
        try:
            with open(target_filename, "r") as f:
                content = f.read().strip()
                if content:
                    loaded_data = json.loads(content)
                    if isinstance(loaded_data, dict):
                        data = loaded_data
                    else:
                        print(
                            f"Warning: '{target_filename}' content is not a dictionary."
                            "Resetting to an empty dictionary."
                        )
                        data = {}
                else:
                    data = {}
        except json.JSONDecodeError as e:
            print(f"Warning: Could not decode JSON from {target_filename}. Starting with empty data. Error: {e}")
            data = {}

    new_platform = new_config_entry.get("Platform")
    if not new_platform:
        print("Error: New configuration entry from source file is missing 'Platform' key. Cannot process.")
        return

    # Extract Git_SHA from new_config_entry directly for use in MetricGenerator
    new_config_git_sha = new_config_entry.get("Configuration", {}).get("Git_SHA")

    # Create MetricGenerator for the new entry, so its internal attributes are normalized
    # and it captures the Git_SHA from the new config entry
    temp_generator = MetricGenerator(
        machine=new_config_entry.get("Platform"),
        llm=new_config_entry.get("Configuration", {}).get("LLM"),
        rag=new_config_entry.get("Configuration", {}).get("RAG"),
        tts=new_config_entry.get("Configuration", {}).get("TTS"),
        stt=new_config_entry.get("Configuration", {}).get("STT"),
        use_npu=new_config_entry.get("Configuration", {}).get("NPU") == "ON",
        linux_version=new_config_entry.get("Configuration", {}).get("Linux_Version"),
        git_sha=new_config_git_sha,  # Pass the Git_SHA from the new config file
    )
    # Get the unique identifier from the temp_generator directly (it has normalized names now)
    new_entry_comparison_tuple = temp_generator.get_unique_config_identifier()

    full_identifier_for_print = new_entry_comparison_tuple

    if new_platform not in data:
        data[new_platform] = []

    platform_entries = data[new_platform]
    entry_found = False

    for i, entry in enumerate(platform_entries):
        # Extract Git_SHA from existing entry for MetricGenerator
        existing_config_git_sha = entry.get("Configuration", {}).get("Git_SHA")

        # Create MetricGenerator for the existing entry. Its internal attributes will also be normalized.
        existing_temp_generator = MetricGenerator(
            machine=entry.get("Platform"),
            llm=entry.get("Configuration", {}).get("LLM"),
            rag=entry.get("Configuration", {}).get("RAG"),
            tts=entry.get("Configuration", {}).get("TTS"),
            stt=entry.get("Configuration", {}).get("STT"),
            use_npu=entry.get("Configuration", {}).get("NPU") == "ON",
            linux_version=entry.get("Configuration", {}).get("Linux_Version"),
            git_sha=existing_config_git_sha,  # Pass the Git_SHA from the existing config
        )
        # Get the unique identifier from existing_temp_generator directly
        existing_entry_comparison_tuple = existing_temp_generator.get_unique_config_identifier()

        if existing_entry_comparison_tuple == new_entry_comparison_tuple:
            entry_found = True

            if action_on_existing == "update":
                # Ensure the new_config_entry's Configuration reflects the normalized model names
                # and the correct Git_SHA before updating.
                new_config_entry["Configuration"]["LLM"] = temp_generator.llm
                new_config_entry["Configuration"]["RAG"] = temp_generator.retriever
                new_config_entry["Configuration"]["TTS"] = temp_generator.tts
                new_config_entry["Configuration"]["STT"] = temp_generator.stt
                new_config_entry["Configuration"]["NPU"] = temp_generator.npu_status
                new_config_entry["Configuration"]["Linux_Version"] = temp_generator.linux_version
                new_config_entry["Configuration"]["Git_SHA"] = temp_generator.git_sha  # Update Git_SHA here

                if "Configuration" in new_config_entry and "Configuration" in platform_entries[i]:
                    platform_entries[i]["Configuration"].update(new_config_entry["Configuration"])
                elif "Configuration" in new_config_entry:
                    platform_entries[i]["Configuration"] = new_config_entry["Configuration"]

                if "Metrics" in new_config_entry and "Metrics" in platform_entries[i]:
                    platform_entries[i]["Metrics"].update(new_config_entry["Metrics"])
                elif "Metrics" in new_config_entry:
                    platform_entries[i]["Metrics"] = new_config_entry["Metrics"]

                print(f"Configuration already exists. Updated metrics for: {full_identifier_for_print}")
                with open(target_filename, "w") as f:
                    json.dump(data, f, indent=4)
                print(f"File {target_filename} has been updated.")

                if lava_test_case:
                    for metric_name, value in new_config_entry.get("Metrics", {}).items():
                        lava_test_name = metric_name
                        units = metric_units.get(metric_name, "unitless")
                        report_lava_test_case(
                            name=lava_test_name,
                            result="pass",
                            measurement=to_float(value),
                            units=units,
                        )

            elif action_on_existing == "compare":
                compare_metrics(
                    platform_entries[i],
                    new_config_entry,
                    full_identifier_for_print,
                    tolerance=tolerance,
                    lava_test_case=lava_test_case,
                )
                print(f"File {target_filename} was NOT modified as a comparison was performed.")
            elif action_on_existing == "update_if_improved":
                metric_defs = _get_metric_definitions()
                priority_metrics = [
                    "stt_wer",
                    "stt_avg_time",
                    "llm_avg_ttft",
                    "llm_avg_tps",
                    "tts_avg_time",
                    "tts_avg_rtf",
                    "ttfa_avg",
                ]
                higher_is_better_metrics_priority = metric_defs["higher_is_better_metrics"]
                lower_is_better_metrics_priority = metric_defs["lower_is_better_metrics"]

                all_priority_metrics_improved_or_equal = True
                improvement_messages = []
                comparison_details = []

                for metric_name in priority_metrics:
                    old_val = to_float(entry.get("Metrics", {}).get(metric_name, 0))
                    new_val = to_float(new_config_entry.get("Metrics", {}).get(metric_name, 0))

                    current_metric_improved = False
                    message = ""
                    status = ""

                    if old_val == 0:
                        if new_val != 0:
                            current_metric_improved = True
                            message = f"New metric or old data missing, new value: {new_val:.2f}"
                            status = "NEW/IMPROVED"
                        else:
                            message = "No data or both zero."
                            status = "NO_DATA"
                    else:
                        diff = new_val - old_val
                        percentage_diff = (diff / old_val) * 100

                        if metric_name in higher_is_better_metrics_priority:
                            if new_val >= old_val * (1 - tolerance):
                                current_metric_improved = True
                                status = "IMPROVED" if percentage_diff > (tolerance * 100) else "STABLE"
                                message = f"Improved/Stable (+{percentage_diff:+.2f}%)"
                            else:
                                status = "REGRESSED"
                                message = f"Regressed ({percentage_diff:+.2f}%)"

                        elif metric_name in lower_is_better_metrics_priority:
                            if new_val <= old_val * (1 + tolerance):
                                current_metric_improved = True
                                status = "IMPROVED" if percentage_diff < (-tolerance * 100) else "STABLE"
                                message = f"Improved/Stable ({percentage_diff:+.2f}%)"
                            else:
                                status = "REGRESSED"
                                message = f"Regressed (+{percentage_diff:+.2f}%)"
                        else:
                            if abs(percentage_diff) <= (tolerance * 100):
                                current_metric_improved = True
                                status = "STABLE"
                                message = f"No significant change ({percentage_diff:+.2f}%)"
                            else:
                                status = "CHANGED_OUTSIDE_TOLERANCE"
                                message = f"Significant change outside tolerance ({percentage_diff:+.2f}%)"

                    if not current_metric_improved:
                        all_priority_metrics_improved_or_equal = False

                    improvement_messages.append(
                        f"{metric_name}: Old={old_val:.2f}, New={new_val:.2f}, Status: {status} ({message})"
                    )
                    comparison_details.append(
                        {
                            "metric": metric_name,
                            "old_val": old_val,
                            "new_val": new_val,
                            "status": status,
                            "message": message,
                            "improved": current_metric_improved,
                        }
                    )

                if all_priority_metrics_improved_or_equal:
                    # Before updating, ensure the stored config values are normalized base names
                    # and that Git_SHA is updated.
                    new_config_entry["Configuration"]["LLM"] = temp_generator.llm
                    new_config_entry["Configuration"]["RAG"] = temp_generator.retriever
                    new_config_entry["Configuration"]["TTS"] = temp_generator.tts
                    new_config_entry["Configuration"]["STT"] = temp_generator.stt
                    new_config_entry["Configuration"]["NPU"] = temp_generator.npu_status
                    new_config_entry["Configuration"]["Linux_Version"] = temp_generator.linux_version
                    new_config_entry["Configuration"]["Git_SHA"] = temp_generator.git_sha  # Update Git_SHA here

                    if "Configuration" in new_config_entry and "Configuration" in platform_entries[i]:
                        platform_entries[i]["Configuration"].update(new_config_entry["Configuration"])
                    elif "Configuration" in new_config_entry:
                        platform_entries[i]["Configuration"] = new_config_entry["Configuration"]

                    if "Metrics" in new_config_entry and "Metrics" in platform_entries[i]:
                        platform_entries[i]["Metrics"].update(new_config_entry["Metrics"])
                    elif "Metrics" in new_config_entry:
                        platform_entries[i]["Metrics"] = new_config_entry["Metrics"]

                    print(
                        f"Configuration already exists. Updated metrics for: {full_identifier_for_print} "
                        "(All priority metrics improved or stable)."
                    )
                    with open(target_filename, "w") as f:
                        json.dump(data, f, indent=4)
                    print(f"File {target_filename} has been updated.")

                    if lava_test_case:
                        for metric_name, value in new_config_entry.get("Metrics", {}).items():
                            lava_test_name = metric_name
                            units = metric_units.get(metric_name, "unitless")
                            report_lava_test_case(
                                name=lava_test_name,
                                result="pass",
                                measurement=to_float(value),
                                units=units,
                            )

                else:
                    print(
                        f"Configuration already exists. Metrics for {full_identifier_for_print} not updated "
                        "as not all priority metrics improved or were stable."
                    )
                    for msg in improvement_messages:
                        print(f"  - {msg}")
                    print(f"File {target_filename} was NOT modified.")

                    if lava_test_case:
                        for detail in comparison_details:
                            metric_name = detail["metric"]
                            lava_test_name = metric_name
                            units = metric_units.get(metric_name, "unitless")
                            report_lava_test_case(
                                name=lava_test_name,
                                result="pass" if detail["improved"] else "fail",
                                measurement=detail["new_val"],
                                units=units,
                            )

                        for metric_name, value in new_config_entry.get("Metrics", {}).items():
                            if metric_name not in priority_metrics:
                                lava_test_name = metric_name
                                units = metric_units.get(metric_name, "unitless")
                                report_lava_test_case(
                                    name=lava_test_name,
                                    result="pass",
                                    measurement=to_float(value),
                                    units=units,
                                )

            else:
                print(
                    f"Invalid action_on_existing: '{action_on_existing}'. Must be 'compare', 'update', or "
                    "'update_if_improved'. No action taken on existing entry."
                )
            break

    if not entry_found:
        # Before adding, ensure the stored config values are normalized base names
        # and include the Git_SHA from temp_generator.
        entry_to_add = {
            "Platform": new_config_entry["Platform"],
            "Configuration": {
                "LLM": temp_generator.llm,
                "RAG": temp_generator.retriever,
                "TTS": temp_generator.tts,
                "STT": temp_generator.stt,
                "NPU": temp_generator.npu_status,
                "Linux_Version": temp_generator.linux_version,
                "Git_SHA": temp_generator.git_sha,  # Include Git_SHA from temp_generator
            },
            "Metrics": new_config_entry.get("Metrics", {}),
        }
        data[new_platform].append(entry_to_add)
        print(f"Added new configuration for platform '{new_platform}': {full_identifier_for_print}")
        with open(target_filename, "w") as f:
            json.dump(data, f, indent=4)
        print(f"File {target_filename} has been updated with a new entry.")
        if lava_test_case:
            for metric_name, value in new_config_entry.get("Metrics", {}).items():
                lava_test_name = metric_name
                units = metric_units.get(metric_name, "unitless")
                report_lava_test_case(
                    name=lava_test_name,
                    result="pass",
                    measurement=to_float(value),
                    units=units,
                )
    sys.exit()

    def get_unique_config_identifier(self, new_config_entry=None):
        """
        Generates a tuple representing the unique identity of a configuration.
        This is used to check if a configuration already exists in the list.
        If new_config_entry is provided, it extracts details from there;
        otherwise, it uses the instance's attributes (which are already base names).
        """
        if new_config_entry:
            config_details = new_config_entry.get("Configuration", {})
            return (
                new_config_entry.get("Platform"),
                _get_base_model_name(config_details.get("LLM")),
                _get_base_model_name(config_details.get("RAG")),
                _get_base_model_name(config_details.get("TTS")),
                _get_base_model_name(config_details.get("STT")),
                config_details.get("NPU"),
                config_details.get("Linux_Version"),
                # Git_SHA is NOT part of the unique identifier for comparison,
                # as you might want to update metrics for the same config
                # but under a new git SHA. If you want it part of the identifier, uncomment:
                # config_details.get("Git_SHA"),
            )
        else:
            # Attributes are already base names due to __init__ normalization
            return (self.machine, self.llm, self.retriever, self.tts, self.stt, self.npu_status, self.linux_version)


def report_lava_test_case(name, result, measurement, units="unitless", extra_args=None):
    """
    Calls the lava-test-case command to report a single metric result in the specified format:
    lava-test-case TEST_CASE_ID --result RESULT --units UNITS --measurement MEASUREMENT

    Args:
        name (str): The name of the test case (TEST_CASE_ID), which is now just the metric name.
        result (str): The result of the test case ("pass" or "fail").
        measurement (float/str): The actual measured value (MEASUREMENT).
        units (str): The units of the measurement (UNITS).
        extra_args (list): This parameter is ignored as per the specified lava-test-case format.
    """
    command = [
        "lava-test-case",
        str(name),
        "--result",
        str(result),
        "--units",
        str(units),
        "--measurement",
        str(measurement),
    ]

    # Print the command being executed for debugging/traceability
    print(f"Executing LAVA command: {' '.join(command)}")

    try:
        # Keep capture_output=True to catch potential errors in stdout/stderr
        # check=True will raise CalledProcessError if the command returns a non-zero exit code
        result_obj = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Successfully reported LAVA test case: {name}")
        # Optionally, print stdout/stderr even on success if needed for verbose logging
        if result_obj.stdout:
            print(f"  LAVA command STDOUT: {result_obj.stdout.strip()}")
        if result_obj.stderr:
            print(f"  LAVA command STDERR: {result_obj.stderr.strip()}")

    except subprocess.CalledProcessError as e:
        print(f"Error reporting LAVA test case {name}: {e}")
        print(f"  STDOUT: {e.stdout.strip()}")
        print(f"  STDERR: {e.stderr.strip()}")
    except FileNotFoundError:
        print("Error: 'lava-test-case' command not found. Please ensure LAVA tools are installed and in your PATH.")


def compare_metrics(old_entry, new_entry, identifier, tolerance=0.05, lava_test_case=False):
    """
    Compares two sets of metrics from nested entries and prints a detailed report,
    indicating whether the change is good (green) or bad (red) based on metric type.
    It expects 'old_entry' and 'new_entry' to be dictionaries containing a 'Metrics'
    sub-dictionary.

    Args:
        old_entry (dict): The dictionary for the old configuration, containing 'Metrics'.
        new_entry (dict): The dictionary for the new configuration, containing 'Metrics'.
        identifier (tuple): A tuple representing the unique identifier of the configuration.
        tolerance (float): The percentage tolerance (e.g., 0.01 for 1%). If the absolute
                           percentage difference is within this tolerance, it's considered
                           "NO SIGNIFICANT CHANGE".
        lava_test_case (bool): If True, reports results using LAVA test case API format.
    """
    print(f"\n--- Comparison for Configuration: {identifier} ---")

    metric_defs = _get_metric_definitions()
    metrics_to_compare = metric_defs["metrics_to_compare"]
    higher_is_better_metrics = metric_defs["higher_is_better_metrics"]
    lower_is_better_metrics = metric_defs["lower_is_better_metrics"]
    metric_units = metric_defs["metric_units"]

    def to_float(val):
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0.0

    old_metrics = old_entry.get("Metrics", {})
    new_metrics = new_entry.get("Metrics", {})

    for metric_name in metrics_to_compare:
        old_val = to_float(old_metrics.get(metric_name, 0))
        new_val = to_float(new_metrics.get(metric_name, 0))

        lava_test_name = metric_name  # As per the new requirement

        units = metric_units.get(metric_name, "unitless")

        if old_val == 0 and new_val == 0:
            print(f"- {metric_name}: No data or both zero.")
            if lava_test_case:
                report_lava_test_case(
                    name=lava_test_name,
                    result="pass",
                    measurement=0.0,
                    units=units,
                )
            continue

        diff = new_val - old_val
        status = "NO CHANGE"
        color_start = "\033[0m"
        percentage_diff = 0.0

        if old_val != 0:
            percentage_diff = (diff / old_val) * 100
        elif diff != 0:
            percentage_diff = float("inf")

        test_result_lava = "fail"

        if abs(percentage_diff) <= (tolerance * 100):
            status = "NO SIGNIFICANT CHANGE"
            color_start = "\033[0m"
            test_result_lava = "pass"

        elif diff > 0:
            if metric_name in higher_is_better_metrics:
                status = "IMPROVED"
                color_start = "\033[92m"
                test_result_lava = "pass"
            elif metric_name in lower_is_better_metrics:
                status = "REGRESSED"
                color_start = "\033[91m"
                test_result_lava = "fail"
            else:
                status = "CHANGED"
                color_start = "\033[0m"
                test_result_lava = "pass"

        elif diff < 0:
            if metric_name in higher_is_better_metrics:
                status = "REGRESSED"
                color_start = "\033[91m"
                test_result_lava = "fail"
            elif metric_name in lower_is_better_metrics:
                status = "IMPROVED"
                color_start = "\033[92m"
                test_result_lava = "pass"
            else:
                status = "CHANGED"
                color_start = "\033[0m"
                test_result_lava = "pass"
        else:
            test_result_lava = "pass"

        if old_val != 0:
            print(
                f"- {metric_name}: Old={old_val:.2f}, New={new_val:.2f}, {status}: "
                f"{color_start}{diff:+.2f} ({percentage_diff:+.2f}%)\033[0m"
            )
            if lava_test_case:
                report_lava_test_case(
                    name=lava_test_name,
                    result=test_result_lava,
                    measurement=new_val,
                    units=units,
                )
        else:
            # Handle cases where old_val is 0 but new_val is not
            print(f"- {metric_name}: Old={old_val:.2f}, New={new_val:.2f}, Change: {diff:+.2f} (N/A % - old was zero)")
            if lava_test_case:
                report_lava_test_case(
                    name=lava_test_name,
                    result="pass",
                    measurement=new_val,
                    units=units,
                )

    print("------------------------------------------\n")
