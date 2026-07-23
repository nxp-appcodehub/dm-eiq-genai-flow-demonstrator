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
import threading
import time
import wave
import contextlib
import logging
import psutil
from eiq_genai_flow.__main__ import eIQGenAIFlow
from eiq_genai_flow.benchmark.questions_to_wav import generate_wav_files
from eiq_genai_flow.utils.audio_loopback_setup import AudioLoopbackSetup
from eiq_genai_flow.adapters.event_manager import EventType, Event
from eiq_genai_flow.utils.utils import (
    suppress_stderr_context,
    get_git_commit_sha,
    get_installed_versions,
    get_linux_version,
    get_sha256,
    get_neutron_info
)
with suppress_stderr_context():
    import onnxruntime as ort
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
        use_voice_id,
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
            use_voice_id,
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

        self._vad_end_event = threading.Event()

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
        self.llm_tts_sum_time = 0
        self._current_rag_time = 0

        # VIT metrics (for VASR mode)
        self.vit_sum_time = 0
        self.wake_word_detected = 0

        # STT metrics
        self.error_rate_computer = None

        # Voice ID metrics
        self.vID_sum_time = 0
        self.speaker_recognized = 0

        self.questions_processed = 0

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

        audio_loopback = AudioLoopbackSetup()
        logger.debug("Running audio loopback setup")
        audio_loopback.setup()
        logger.info("✓ Virtual audio loopback setup complete")

    # =========================================================================
    # MAIN BENCHMARK LOOP
    # =========================================================================

    def run(self):
        """
        Override parent's run() to implement benchmark-specific flow.

        Instead of waiting for keyboard/VIT/STT input, we:
        1. Directly trigger events for each question
        2. Process through the pipeline
        3. Collect metrics
        """
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
        # GENERATE WAV FILES (if STT mode)
        # =====================================================================
        if self.stt:
            from speech_to_text.utils.utils import ErrorRateComputer

            self.error_rate_computer = ErrorRateComputer()

            if self.is_vasr_mode:
                wake_word_once_by_speaker = True if self.voice_id else False
                logger.info(f"Generating VASR WAV files with wake word: '{self.wake_word}'")
                self.benchmark_wav_files = generate_wav_files(
                    wav_dir=self.config.tests_data_path,
                    text_file_path=self.config.benchmark_questions_file,
                    text_file_len=bench_len,
                    wake_word=self.wake_word,
                    wake_word_once_by_speaker=wake_word_once_by_speaker,
                    add_noise=True,
                    snr_db=40.0,
                    noise_type="pink",
                    noise_prefix_duration=0.8,
                )
                self.benchmark_logger.log(
                    f"{len(self.benchmark_wav_files)} wake-word prefixed audio files (wake word: '{self.wake_word}')"
                )
            else:  # KASR mode
                logger.info("Generating KASR WAV files (no wake word)")
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
        # MAIN BENCHMARK LOOP
        # =====================================================================
        logger.info("Starting benchmark question processing loop...")

        with self.earcon, self.tts:
            while not self.stop_threads and self.current_question_idx < len(self.benchmark_questions):
                # Subscribe to events
                self.event_manager.subscribe(EventType.INPUT_TEXT, self._on_input_text)
                self.event_manager.subscribe(EventType.END_OF_INPUT, self._on_end_of_input)
                self.event_manager.subscribe([EventType.VERIFIED_SPEAKER, EventType.UNVERIFIED_SPEAKER],
                                             self._on_stt_speaker_verification)

                # Get question by triggering appropriate events
                question = self._get_next_question()
                verified = self.wait_speaker_verification() if self.voice_id else True

                # Unsubscribe from events
                self.event_manager.unsubscribe(EventType.INPUT_TEXT, self._on_input_text)
                self.event_manager.unsubscribe(EventType.END_OF_INPUT, self._on_end_of_input)

                # Skip if timeout or empty
                if question == "TIMEOUT" or question == "" or not verified:
                    continue

                # Handle the question through the pipeline
                self.questions_processed += 1
                self.handle_question(question)

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

    # =========================================================================
    # QUESTION DISPATCH
    # =========================================================================

    def _get_next_question(self):
        """
        Get the next question by triggering appropriate events.

        Returns:
            str: Question text or "TIMEOUT"
        """
        if self.current_question_idx >= len(self.benchmark_questions):
            logger.info("All benchmark questions processed")
            self.stop_threads = True
            return ""

        question_text = self.benchmark_questions[self.current_question_idx]

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Question {self.current_question_idx + 1}/{len(self.benchmark_questions)}: {question_text}")
        logger.info(f"{'=' * 60}")

        self._reset_tts_state()

        self.benchmark_logger.log(f"\n{'=' * 80}")
        self.benchmark_logger.log(f"Question {self.current_question_idx + 1}: {question_text}")

        # Route to appropriate event trigger based on mode
        if self.is_vasr_mode and not self.voice_id :
            return self._trigger_vasr_events(self.benchmark_wav_files[self.current_question_idx], question_text)
        elif self.is_vasr_mode and self.voice_id :
            return self._trigger_voice_id_events(self.benchmark_wav_files[self.current_question_idx], question_text)
        elif self.is_kasr_mode:
            return self._trigger_kasr_events(self.benchmark_wav_files[self.current_question_idx], question_text)
        else:
            return self._trigger_keyboard_events(question_text)

    def handle_question(self, question):
        """Override to measure actual LLM+TTS wall-clock time."""
        self._current_rag_time = 0
        prev_llm_count = self.llm_inf_count
        handle_start = time.perf_counter()

        super().handle_question(question)

        if self.llm_inf_count > prev_llm_count:
            # LLM was used — wall-clock time minus RAG = actual LLM+TTS time
            llm_tts_time = time.perf_counter() - handle_start - self._current_rag_time
            self.llm_tts_sum_time += llm_tts_time

    # =========================================================================
    # MODE-SPECIFIC EVENT TRIGGERS
    # =========================================================================

    def _on_vad_speech_end(self, event: Event):
        self._vad_end_event.set()

    def _trigger_keyboard_events(self, question_text):
        """
        Trigger keyboard events to simulate user typing.

        Args:
            question_text: The question to process

        Returns:
            str: The question text
        """
        logger.debug("Triggering keyboard events for benchmark")

        self.event_manager.publish(Event(event_type=EventType.INPUT_TEXT, source="Benchmark", data=question_text))
        time.sleep(0.05)
        self.event_manager.publish(Event(event_type=EventType.END_OF_INPUT, source="Benchmark"))

        timeout = 1.0
        if self.end_of_input_event.wait(timeout=timeout):
            result = self.input_text
            self.input_text = ""
            self.end_of_input_event.clear()
        else:
            logger.warning("Timeout waiting for keyboard events to process")
            result = "TIMEOUT"

        self.current_question_idx += 1
        return result

    def _trigger_vasr_events(self, wav_file, question_text):
        """
        Trigger VASR events: inject audio with wake word, wait for VIT_WAKE, then STT.

        Args:
            wav_file: Path to the WAV file with wake word prefix
            question_text: Expected transcription (for WER calculation)

        Returns:
            str: Transcribed question or "TIMEOUT"
        """
        logger.debug("Triggering VASR events for benchmark")

        self.event_manager.subscribe(EventType.VIT_WAKE, self._on_wake)
        self.wake_event.clear()

        # Enable VIT if not already running
        if self.vit and not self.vit.is_running:
            logger.info("Enabling VIT for wake word detection...")
            self.vit.enable(sync_to_current=True)
            time.sleep(0.1)

        # Inject audio with wake word
        logger.info(f"Injecting audio with wake word: {os.path.basename(wav_file)}")
        vit_start = time.perf_counter()

        inject_proc = subprocess.Popen(
            ["aplay", "-D", "fake_input", wav_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        self.wav_sum_duration += get_wav_duration(wav_file)

        # Wait for VIT_WAKE event
        logger.debug("Waiting for VIT to publish VIT_WAKE event...")
        vit_timeout = 5.0
        question = ""

        if self.wake_event.wait(timeout=vit_timeout):
            # Wake word detected
            vit_time = time.perf_counter() - vit_start
            self.vit_sum_time += vit_time
            self.wake_word_detected += 1

            ww_detection_rate = (self.wake_word_detected / (self.current_question_idx + 1)) * 100

            logger.info(f"✓ Wake word detected ({vit_time:.2f}s) - Detection rate: {ww_detection_rate:.1f}%")
            self.benchmark_logger.log(
                f"VIT: Wake word DETECTED ({vit_time:.2f}s, detection rate: {ww_detection_rate:.1f}%)"
            )

            self.wake_event.clear()
            self.event_manager.unsubscribe(EventType.VIT_WAKE, self._on_wake)

            # Enable VAD and STT for speech transcription
            if self.vad:
                self.vad.enable(sync_to_current=True)
                logger.info("VAD adapter enabled after wake word")

            if self.stt:
                self.stt.enable(sync_to_current=True)
                logger.info("STT adapter enabled after wake word")

            # Publish KEYBOARD_WAKE so STT subscribes to VAD events
            self.event_manager.publish(Event(event_type=EventType.KEYBOARD_WAKE, source="Benchmark"))
            time.sleep(0.15)

            # Start STT timing
            stt_start = time.perf_counter()

            # Wait for VAD to end and take this as the starting time for ttfa computation
            self._vad_end_event.clear()
            self.event_manager.subscribe(EventType.VAD_SPEECH_END, self._on_vad_speech_end)
            vad_timeout = 30.0
            if self._vad_end_event.wait(timeout=vad_timeout):
                self.ttfa_start_time = time.perf_counter()
                logger.debug(f"[TTFA] VASR: TTFA start time = {self.ttfa_start_time:.4f}s")
            else:
                logger.warning("[TTFA] VASR: VAD END timeout after wake word detection")
                self.ttfa_start_time = 0

            self.event_manager.unsubscribe(EventType.VAD_SPEECH_END, self._on_vad_speech_end)

            # Now wait for STT to process (INPUT_TEXT and END_OF_INPUT events)
            logger.debug("Waiting for STT to publish INPUT_TEXT and END_OF_INPUT events...")
            stt_timeout = 30.0

            if self.end_of_input_event.wait(timeout=stt_timeout):
                stt_end = time.perf_counter()
                stt_time = stt_end - stt_start
                self.stt_sum_time += stt_time
                logger.info(f"STT completed in {stt_time:.2f}s")

                question = self.input_text
                self.input_text = ""
                self.end_of_input_event.clear()

                if question and question != "TIMEOUT":
                    self._calculate_wer(question, question_text)
                    logger.info(f"Expected: {question_text}")
                    logger.info(f"Got: {question}")
            else:
                logger.warning("STT timeout after wake word detection")
                question = "TIMEOUT"

            # Disable adapters for next question
            if self.stt:
                self.stt.disable()
            if self.vad:
                self.vad.disable()
        else:
            # Wake word missed
            ww_missed = (self.current_question_idx + 1) - self.wake_word_detected
            ww_detection_rate = (self.wake_word_detected / (self.current_question_idx + 1)) * 100

            logger.warning(
                f"✗ Wake word missed ({ww_missed}/{self.current_question_idx + 1}) - "
                f"Detection rate: {ww_detection_rate:.1f}%"
            )
            self.benchmark_logger.log(
                f"VIT: Wake word MISSED (total missed: {ww_missed}, "
                f"detection rate: {ww_detection_rate:.1f}%)"
            )

            self.event_manager.unsubscribe(EventType.VIT_WAKE, self._on_wake)

            # Reset VIT for next question
            if self.vit:
                self.vit.disable()
                time.sleep(0.1)
                self.vit.vit.reset()
                self.vit.enable(sync_to_current=True)

        # Cleanup injection
        self._cleanup_audio_injection(inject_proc)

        self.current_question_idx += 1
        return question if question else "TIMEOUT"

    def _trigger_voice_id_events(self, wav_file, question_text):
        # Ensure capture is running before injection
        if not self.audio_manager.is_capture_running():
            logger.info("Capture not running, restarting...")
            self.audio_manager.start_capture()

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

            time.sleep(0.3)

        self.wav_sum_duration += get_wav_duration(wav_file)

        self.event_manager.subscribe([EventType.VIT_WAKE, EventType.VOICE_ID_WAKE], self._on_wake)
        self.wake_event.clear()

        # Enable VIT if not already running
        if self.vit :
            logger.info("Enabling VIT for wake word detection...")
            self.vit.enable(sync_to_current=True)
            time.sleep(0.1)

        # Enable VAD (required for speech detection)
        if self.vad:
            self.vad.disable()  # clean state from previous question
            self.vad.enable(sync_to_current=True)
            logger.info("VAD adapter enabled")

        # Enable Voice ID
        if self.voice_id:
            self.voice_id.disable()
            self.voice_id.enable(sync_to_current=True)
            logger.info("voice_id adapter enabled after wake word")

        # Enable STT (worker loop exits after each question)
        if self.stt:
            self.stt.disable()  # clean state from previous question
            self.stt.enable(sync_to_current=True)
            logger.info("STT adapter enabled")

        # Inject audio and start STT timing
        stt_start = time.perf_counter()
        logger.info(f"Injecting audio: {os.path.basename(wav_file)}")
        inject_proc = subprocess.Popen(
            ["aplay", "-D", "fake_input", wav_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(0.1)

        # Wait for VIT_WAKE event or Voice ID event
        logger.debug("Waiting for VIT or Voice ID to publish wake event...")
        timeout = 5.0
        start = time.perf_counter()

        if self.wake_event.wait(timeout=timeout):
            if self.wake_type == EventType.VOICE_ID_WAKE:
                logger.info("✓ Wake event detected")
                self.benchmark_logger.log("Speaker recognized")
                self.speaker_recognized += 1
                end = time.perf_counter()
                vID_time = end - start
                self.vID_sum_time += vID_time
            else :
                logger.info("✓ Wake event detected")
                self.benchmark_logger.log("Wake word DETECTED")
                self.wake_word_detected += 1
                end = time.perf_counter()
                vit_time = end - start
                self.vit_sum_time += vit_time

            self.wake_event.clear()
            self.event_manager.unsubscribe([EventType.VIT_WAKE, EventType.VOICE_ID_WAKE], self._on_wake)

        # Wait for STT to publish END_OF_INPUT
        stt_timeout = 30.0
        if self.end_of_input_event.wait(timeout=stt_timeout):
            stt_end = time.perf_counter()
            stt_time = stt_end - stt_start
            self.stt_sum_time += stt_time
            self.ttfa_start_time = stt_end
            logger.info(f"STT completed in {stt_time:.2f}s")
            logger.debug(f"[TTFA] KASR: TTFA start time = {self.ttfa_start_time:.4f}s")

            question = self.input_text
            self.input_text = ""
            self.end_of_input_event.clear()
        else:
            logger.warning(f"STT timeout after {stt_timeout}s")
            question = "TIMEOUT"

        # Calculate WER if we got a result
        if question and question != "TIMEOUT":
            self._calculate_wer(question, question_text)
            logger.info(f"Expected: {question_text}")
            logger.info(f"Got: {question}")

        # Cleanup injection
        self._cleanup_audio_injection(inject_proc)

        # Disable adapters (will be re-enabled for the next question)
        if self.voice_id:
            self.voice_id.disable()
        if self.stt:
            self.stt.disable()
        if self.vad:
            self.vad.disable()
        # Reset VIT for next question
        if self.vit:
            self.vit.disable()

        self.current_question_idx += 1
        return question if question else "TIMEOUT"

    def _trigger_kasr_events(self, wav_file, question_text):
        """
        KASR mode: enable VAD+STT, trigger wake, inject audio, wait for transcription.

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

            time.sleep(0.3)

        self.wav_sum_duration += get_wav_duration(wav_file)

        # 1. Enable VAD (required for speech detection)
        if self.vad:
            self.vad.disable()  # clean state from previous question
            self.vad.enable(sync_to_current=True)
            logger.info("VAD adapter enabled")

        # 2. Enable STT (worker loop exits after each question)
        if self.stt:
            self.stt.disable()  # clean state from previous question
            self.stt.enable(sync_to_current=True)
            logger.info("STT adapter enabled")

        # 3. Publish KEYBOARD_WAKE so STT subscribes to VAD events
        self.event_manager.publish(Event(event_type=EventType.KEYBOARD_WAKE, source="Benchmark"))
        time.sleep(0.15)  # let event propagate through STT._on_wake

        # 4. Inject audio and start STT timing
        stt_start = time.perf_counter()
        logger.info(f"Injecting audio: {os.path.basename(wav_file)}")
        inject_proc = subprocess.Popen(
            ["aplay", "-D", "fake_input", wav_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(0.1)
        # Wait for VAD to end and take this as the starting time for ttfa computation
        self._vad_end_event.clear()
        self.event_manager.subscribe(EventType.VAD_SPEECH_END, self._on_vad_speech_end)
        vad_timeout = 30.0
        if self._vad_end_event.wait(timeout=vad_timeout):
            self.ttfa_start_time = time.perf_counter()
            logger.debug(f"[TTFA] KASR: TTFA start time = {self.ttfa_start_time:.4f}s")
        else:
            logger.warning("[TTFA] VASR: VAD END timeout after wake word detection")
            self.ttfa_start_time = 0

        # 5. Wait for STT to publish END_OF_INPUT
        stt_timeout = 30.0
        if self.end_of_input_event.wait(timeout=stt_timeout):
            stt_end = time.perf_counter()
            stt_time = stt_end - stt_start
            self.stt_sum_time += stt_time
            logger.info(f"STT completed in {stt_time:.2f}s")

            question = self.input_text
            self.input_text = ""
            self.end_of_input_event.clear()
        else:
            logger.warning(f"STT timeout after {stt_timeout}s")
            question = "TIMEOUT"

        # Calculate WER if we got a result
        if question and question != "TIMEOUT":
            self._calculate_wer(question, question_text)
            logger.info(f"Expected: {question_text}")
            logger.info(f"Got: {question}")

        # Cleanup injection
        self._cleanup_audio_injection(inject_proc)

        # Disable adapters (will be re-enabled for the next question)
        if self.stt:
            self.stt.disable()
        if self.vad:
            self.vad.disable()

        self.current_question_idx += 1
        return question if question else "TIMEOUT"

    # =========================================================================
    # BENCHMARK STATS & HELPERS
    # =========================================================================

    def _reset_tts_state(self):
        """Reset TTS state at the beginning of each question."""
        self.ttfa_start_time = None
        logger.debug("Reset ttfa_start_time from previous question")

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
            return

        # RAG metrics
        if self.retriever and rag_time:
            self.rag_sum_time += rag_time
            self._current_rag_time = rag_time

        # TTFA calculation
        if not self.tts:
            return

        try:
            tts_timestamp = self.tts.timestamp_ttfa
        except AttributeError:
            logger.debug(
                "[TTFA] Question %d: TTS hasn't run yet - skipping TTFA calculation", self.current_question_idx
            )
            return

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

            self.ttfa_start_time = None
        else:
            logger.debug(
                "[TTFA] Question %d: ttfa_start_time not set - skipping TTFA calculation", self.current_question_idx
            )

        # TTS metrics
        try:
            self.tts_sum_time += self.tts.inference_time
        except AttributeError:
            pass

    def _calculate_wer(self, transcribed, expected):
        """Calculate Word Error Rate for STT transcription."""
        normalized_transcribed = self.stt.stt.text_normalizer(transcribed).split(" ")
        normalized_expected = self.stt.stt.text_normalizer(expected).split(" ")

        self.error_rate_computer.append(
            ids=[self.current_question_idx],  # not yet incremented at call site
            predict=[normalized_transcribed],
            target=[normalized_expected],
        )

    def _cleanup_audio_injection(self, inject_proc):
        """Cleanup audio injection subprocess."""
        if inject_proc.poll() is None:
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

    # =========================================================================
    # LOGGING & REPORTING
    # =========================================================================

    def _generate_benchmark_filename(self):
        """Generate benchmark output filename."""
        return (
            f"Benchmark_{self.device}"
            + f"{('_neutron' if self.use_neutron else '_CPU')}"
            + f"{('_voiceID' if self.voice_id else '')}"
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
            f"{'using voiceID' if self.voice_id else ''}"
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

        ttfa_avg = self.ttfa_sum_time / self.ttfa_question_count if self.ttfa_question_count > 0 else 0

        logger.info(
            f"[TTFA] Final Summary: ttfa_sum_time={self.ttfa_sum_time:.4f}s, "
            f"ttfa_question_count={self.ttfa_question_count}, ttfa_avg={ttfa_avg:.4f}s"
        )

        if self.stt and self.error_rate_computer and self.error_rate_computer.scores:
            stt_wer = self.error_rate_computer.summarize("WER")
            file_name = f"WER_model-{self.stt_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            save_path = os.path.join("logs", f"{file_name}.log")
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
                tts_avg_time = self.tts_sum_time / self.questions_processed if self.questions_processed > 0 else 0

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
            self.benchmark_logger.log(f"Questions Processed: {self.questions_processed})")
            self.benchmark_logger.log(f"Questions with Valid TTFA: "
                                      f"{self.ttfa_question_count}/{self.questions_processed}")
        self.benchmark_logger.log(f"NPU: {'ON' if self.use_neutron else 'OFF'}")

        self.benchmark_logger.log("\n--- Component Status ---")
        if self.is_vasr_mode:
            self.benchmark_logger.log(
                f"VIT: Enabled (wake word: '{self.wake_word}')")

        self.benchmark_logger.log(f"STT: {self.stt_model if self.stt else 'OFF'}")

        if self.retriever and self.questions_processed > 0:
            rag_only_answered = self.questions_processed - self.llm_inf_count
            self.benchmark_logger.log(
                f"RAG: {self.retriever.embedding_model.name} | "
                f"Answered Directly: {rag_only_answered}/{self.questions_processed}"
            )
        else:
            self.benchmark_logger.log("RAG: OFF")

        if self.llm:
            if self.questions_processed > 0:
                llm_usage_rate = (self.llm_inf_count / self.questions_processed) * 100
                self.benchmark_logger.log(
                    f"LLM: {self.llm.name} | "
                    f"Answered: {self.llm_inf_count}/{self.questions_processed} ({llm_usage_rate:.1f}%)"
                )
            else:
                self.benchmark_logger.log(f"LLM: {self.llm.name} | Answered: 0/0 (no questions processed)")
        else:
            self.benchmark_logger.log("LLM: OFF")

        self.benchmark_logger.log(
            f"TTS: {'ON' if self.tts else 'OFF'}"
            + (f" | Model: {self.tts.model_name}, Mode: {self.output_mode}" if self.tts else "")
        )

        self.benchmark_logger.log("\n--- Performance Metrics ---")
        self.benchmark_logger.log(
            f"Benchmark Time: Avg = {total_time / self.questions_processed if self.questions_processed > 0 else 0:.2f}s"
            f" | "
            f"Total = {total_time:.2f}s"
        )
        if self.ttfa_question_count > 0:
            self.benchmark_logger.log(
                f"TTFA: Avg = {ttfa_avg:.2f}s (measured for {self.ttfa_question_count}/{self.questions_processed} "
                f"questions)"
            )
        else:
            self.benchmark_logger.log("TTFA: N/A (no valid measurements)")

        self.benchmark_logger.log(f"CPU Usage: Avg = {avg_cpu:.2f}% | Min = {min_cpu:.2f}% | Max = {max_cpu:.2f}%")
        self.benchmark_logger.log(f"Memory Usage: Avg = {avg_memory_used:.2f} MB ({avg_memory_percent:.2f}%)")

        if self.is_vasr_mode and self.vit_sum_time > 0:
            ww_detection_expected = bench_len // 2 + bench_len % 2 if self.voice_id else bench_len
            ww_detection_rate = (self.wake_word_detected / ww_detection_expected) * 100 if bench_len > 0 else 0
            ww_missed = ww_detection_expected - self.wake_word_detected

            self.benchmark_logger.log("\n--- VIT Metrics ---")
            self.benchmark_logger.log(f"Wake Word Detection Rate: {self.wake_word_detected}/{ww_detection_expected} "
                                      f"({ww_detection_rate:.1f}%)")
            self.benchmark_logger.log(f"Wake Words Missed: {ww_missed}")
            self.benchmark_logger.log(
                f"Detection Time (successful only): Avg = "
                f"{(self.vit_sum_time / self.questions_processed) if self.questions_processed > 0 else 0:.2f}s | "
                f"Total = {self.vit_sum_time:.2f}s")

        if self.stt:
            self.benchmark_logger.log("\n--- STT Metrics ---")
            self.benchmark_logger.log(f"Init Time: {self.stt_init_time:.2f}s")
            self.benchmark_logger.log(
                "Processing Time: Avg = "
                f"{self.stt_sum_time / self.questions_processed if self.questions_processed > 0 else 0:.2f}s | "
                f"Total = {self.stt_sum_time:.2f}s"
            )
            self.benchmark_logger.log(
                f"Word Error Rate: {stt_wer:.2f}%" if stt_wer is not None else "Word Error Rate: N/A"
            )
            self.benchmark_logger.log(
                f"Wave file duration: Avg = {self.wav_sum_duration / bench_len if bench_len > 0 else 0:.2f}s"
            )

        if self.voice_id :
            # First audio, the speaker speaks wake word, second audio the speaker speaks only the question,
            # Third time, the speaker identity changes
            speaker_recognized_expected = bench_len // 2
            self.benchmark_logger.log("\n--- Voice ID Metrics ---")
            self.benchmark_logger.log(f"Init Time: {self.voice_id_init_time:.2f}s")
            self.benchmark_logger.log(
                "Processing Time: Avg = "
                f"{self.vID_sum_time / self.questions_processed if self.questions_processed > 0 else 0:.2f}s | "
                f"Total = {self.vID_sum_time:.2f}s"
            )
            self.benchmark_logger.log(
                f"Speaker recognized: {self.speaker_recognized}/{speaker_recognized_expected} "
                f"({(self.speaker_recognized / speaker_recognized_expected) * 100:.2f}%)"
            )
            self.benchmark_logger.log(
                f"Wave file duration: Avg = {self.wav_sum_duration / bench_len if bench_len > 0 else 0:.2f}s"
            )

        if self.retriever:
            rag_avg_time = self.rag_sum_time / self.questions_processed if self.questions_processed > 0 else 0
            self.benchmark_logger.log("\n--- RAG Metrics ---")
            self.benchmark_logger.log(f"Init Time: {self.rag_init_time:.2f}s")
            self.benchmark_logger.log(f"Processing Time: Avg = {rag_avg_time:.2f}s | Total = {self.rag_sum_time:.2f}s")

        if self.llm_inf_count:
            llm_avg_time = self.llm_sum_time / self.questions_processed if self.questions_processed > 0 else 0
            self.benchmark_logger.log("\n--- LLM Metrics ---")
            self.benchmark_logger.log(f"Init Time: {self.llm_init_time:.2f}s")
            self.benchmark_logger.log(f"Processing Time: Avg = {llm_avg_time:.2f}s | Total = {self.llm_sum_time:.2f}s")
            self.benchmark_logger.log(
                f"TTFT: Avg = {llm_avg_ttft:.2f}s | Min = {self.llm_min_ttft:.2f}s | Max = {self.llm_max_ttft:.2f}s"
            )
            self.benchmark_logger.log(
                f"Tokens/sec: Avg = {llm_avg_tps:.2f} | Min = {self.llm_min_tps:.2f} | Max = {self.llm_max_tps:.2f}"
            )

        if self.tts:
            self.benchmark_logger.log("\n--- TTS Metrics ---")
            self.benchmark_logger.log(f"Init Time: {self.tts_init_time:.2f}s")
            self.benchmark_logger.log(f"Processing Time: Avg = {tts_avg_time:.2f}s | Total = {self.tts_sum_time:.2f}s")
            self.benchmark_logger.log(f"Real Time Factor: Avg = {tts_avg_rtf:.2f}")

        # Calculate combined LLM+TTS metric from actual wall-clock measurement
        llm_tts_avg_time = self.llm_tts_sum_time / self.llm_inf_count if self.llm_inf_count > 0 else 0

        if self.tts and self.llm_inf_count:
            self.benchmark_logger.log("\n--- LLM + TTS Combined Metrics ---")
            self.benchmark_logger.log(
                f"Combined Processing Time: Avg = {llm_tts_avg_time:.2f}s | "
                f"Total = {self.llm_tts_sum_time:.2f}s"
            )

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
            avg_time=total_time / self.questions_processed if self.questions_processed > 0 else 0,
            avg_cpu=avg_cpu,
            avg_mem=avg_memory_used,
            stt_init_time=self.stt_init_time,
            stt_avg_time=self.stt_sum_time / self.questions_processed if self.questions_processed > 0 else 0,
            stt_wer=stt_wer,
            rag_avg_time=self.rag_sum_time / self.questions_processed
            if self.questions_processed > 0 and self.retriever else 0,
            rag_init_time=self.rag_init_time,
            llm_init_time=self.llm_init_time,
            llm_avg_time=self.llm_sum_time / self.questions_processed
            if self.questions_processed > 0 and self.llm_inf_count
            else 0,
            llm_tts_avg_time=llm_tts_avg_time,
            llm_avg_ttft=llm_avg_ttft if self.llm_inf_count else 0,
            llm_avg_tps=llm_avg_tps if self.llm_inf_count else 0,
            tts_init_time=self.tts_init_time,
            tts_avg_rtf=tts_avg_rtf if self.tts else 0,
            tts_avg_time=tts_avg_time if self.tts else 0,
        )
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
        if self.is_vasr_mode and self.vit_sum_time > 0:
            print(f"Wake Word Detection Rate: {self.wake_word_detected}/{ww_detection_expected} "
                  f"({ww_detection_rate:.1f}%)")
            print(f"Wake Words Missed: {ww_missed}")
        if self.voice_id:
            print(
                f"Speaker recognized: {self.speaker_recognized}/{speaker_recognized_expected} "
                f"({(self.speaker_recognized / speaker_recognized_expected) * 100:.2f}%)"
            )
        print(f"Results saved to: {filename}.[log/json]")
        print(f"{'=' * 60}\n")


# =============================================================================
# UTILITY CLASSES AND FUNCTIONS
# =============================================================================


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
        Git_SHA is intentionally excluded so the same config under a new commit
        can be compared / updated.
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
            )
        else:
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
    """

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
                            f"Warning: '{target_filename}' content is not a dictionary. "
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

    new_config_git_sha = new_config_entry.get("Configuration", {}).get("Git_SHA")

    temp_generator = MetricGenerator(
        machine=new_config_entry.get("Platform"),
        llm=new_config_entry.get("Configuration", {}).get("LLM"),
        rag=new_config_entry.get("Configuration", {}).get("RAG"),
        tts=new_config_entry.get("Configuration", {}).get("TTS"),
        stt=new_config_entry.get("Configuration", {}).get("STT"),
        use_npu=new_config_entry.get("Configuration", {}).get("NPU") == "ON",
        linux_version=new_config_entry.get("Configuration", {}).get("Linux_Version"),
        git_sha=new_config_git_sha,
    )
    new_entry_comparison_tuple = temp_generator.get_unique_config_identifier()
    full_identifier_for_print = new_entry_comparison_tuple

    if new_platform not in data:
        data[new_platform] = []

    platform_entries = data[new_platform]
    entry_found = False

    for i, entry in enumerate(platform_entries):
        existing_config_git_sha = entry.get("Configuration", {}).get("Git_SHA")

        existing_temp_generator = MetricGenerator(
            machine=entry.get("Platform"),
            llm=entry.get("Configuration", {}).get("LLM"),
            rag=entry.get("Configuration", {}).get("RAG"),
            tts=entry.get("Configuration", {}).get("TTS"),
            stt=entry.get("Configuration", {}).get("STT"),
            use_npu=entry.get("Configuration", {}).get("NPU") == "ON",
            linux_version=entry.get("Configuration", {}).get("Linux_Version"),
            git_sha=existing_config_git_sha,
        )
        existing_entry_comparison_tuple = existing_temp_generator.get_unique_config_identifier()

        if existing_entry_comparison_tuple == new_entry_comparison_tuple:
            entry_found = True

            if action_on_existing == "update":
                new_config_entry["Configuration"]["LLM"] = temp_generator.llm
                new_config_entry["Configuration"]["RAG"] = temp_generator.retriever
                new_config_entry["Configuration"]["TTS"] = temp_generator.tts
                new_config_entry["Configuration"]["STT"] = temp_generator.stt
                new_config_entry["Configuration"]["NPU"] = temp_generator.npu_status
                new_config_entry["Configuration"]["Linux_Version"] = temp_generator.linux_version
                new_config_entry["Configuration"]["Git_SHA"] = temp_generator.git_sha

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
                        units = metric_units.get(metric_name, "unitless")
                        report_lava_test_case(
                            name=metric_name,
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
                    new_config_entry["Configuration"]["LLM"] = temp_generator.llm
                    new_config_entry["Configuration"]["RAG"] = temp_generator.retriever
                    new_config_entry["Configuration"]["TTS"] = temp_generator.tts
                    new_config_entry["Configuration"]["STT"] = temp_generator.stt
                    new_config_entry["Configuration"]["NPU"] = temp_generator.npu_status
                    new_config_entry["Configuration"]["Linux_Version"] = temp_generator.linux_version
                    new_config_entry["Configuration"]["Git_SHA"] = temp_generator.git_sha

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
                            units = metric_units.get(metric_name, "unitless")
                            report_lava_test_case(
                                name=metric_name,
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
                            units = metric_units.get(detail["metric"], "unitless")
                            report_lava_test_case(
                                name=detail["metric"],
                                result="pass" if detail["improved"] else "fail",
                                measurement=detail["new_val"],
                                units=units,
                            )

                        for metric_name, value in new_config_entry.get("Metrics", {}).items():
                            if metric_name not in priority_metrics:
                                units = metric_units.get(metric_name, "unitless")
                                report_lava_test_case(
                                    name=metric_name,
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
        entry_to_add = {
            "Platform": new_config_entry["Platform"],
            "Configuration": {
                "LLM": temp_generator.llm,
                "RAG": temp_generator.retriever,
                "TTS": temp_generator.tts,
                "STT": temp_generator.stt,
                "NPU": temp_generator.npu_status,
                "Linux_Version": temp_generator.linux_version,
                "Git_SHA": temp_generator.git_sha,
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
                units = metric_units.get(metric_name, "unitless")
                report_lava_test_case(
                    name=metric_name,
                    result="pass",
                    measurement=to_float(value),
                    units=units,
                )


def report_lava_test_case(name, result, measurement, units="unitless", extra_args=None):
    """
    Calls the lava-test-case command to report a single metric result.

    Args:
        name: The test case ID (metric name).
        result: "pass" or "fail".
        measurement: The actual measured value.
        units: The units of the measurement.
        extra_args: Ignored (kept for API compatibility).
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

    print(f"Executing LAVA command: {' '.join(command)}")

    try:
        result_obj = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Successfully reported LAVA test case: {name}")
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
    Compares two sets of metrics and prints a detailed report with color-coded status.

    Args:
        old_entry: Dictionary for the old configuration, containing 'Metrics'.
        new_entry: Dictionary for the new configuration, containing 'Metrics'.
        identifier: Tuple representing the unique configuration identifier.
        tolerance: Percentage tolerance (e.g., 0.05 for 5%).
        lava_test_case: If True, reports results using LAVA test case API.
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

        units = metric_units.get(metric_name, "unitless")

        if old_val == 0 and new_val == 0:
            print(f"- {metric_name}: No data or both zero.")
            if lava_test_case:
                report_lava_test_case(name=metric_name, result="pass", measurement=0.0, units=units)
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
                report_lava_test_case(name=metric_name, result=test_result_lava, measurement=new_val, units=units)
        else:
            print(f"- {metric_name}: Old={old_val:.2f}, New={new_val:.2f}, Change: {diff:+.2f} (N/A % - old was zero)")
            if lava_test_case:
                report_lava_test_case(name=metric_name, result="pass", measurement=new_val, units=units)

    print("------------------------------------------\n")
