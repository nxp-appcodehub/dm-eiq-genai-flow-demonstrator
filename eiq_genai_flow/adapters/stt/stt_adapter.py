# Copyright 2026 NXP
# NXP Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.

import time
import queue
import logging
import numpy as np
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional, Generator

from speech_to_text.speech_to_text import SpeechToText
from speech_to_text.vad import VAD
from speech_to_text.utils.utils import load_audio
from audio_manager.audio_manager_base import ReaderConfig
from adapters.base import BaseAdapter

logger = logging.getLogger(__name__)


@dataclass
class STTAdapterConfig:
    """Configuration for STT Adapter."""

    model_name: str
    language: str = "English"
    task: str = "transcribe"
    channel_indices: Optional[list] = None
    stream_print: bool = False
    timer_print: bool = False
    audio_chunk_duration: float = 5.0
    inactivity_timeout: float = 20.0
    vad_threshold: float = 0.3
    vad_min_silence_duration_ms: int = 200


class STTAdapter(BaseAdapter):
    """
    Adapter for SpeechToText that uses AudioManager for audio capture.

    This adapter bridges the SpeechToText class with the centralized AudioManager,
    handling audio capture, VAD processing, and text generation.
    """

    def __init__(
        self,
        config: STTAdapterConfig,
        audio_manager,
        verbose: bool = False,
    ):
        """
        Initialize STT Adapter.

        Args:
            config: STT adapter configuration
            audio_manager: AudioManager instance for capture
            verbose: Enable verbose logging
        """
        super().__init__(config, audio_manager, verbose)

        # Override thread name
        self._thread_name = "STT_Processing"

        # Text output queue with condition variable for event-driven access
        self.text_queue = queue.Queue()
        self._text_queue_lock = threading.Lock()
        self._text_available_cv = threading.Condition(self._text_queue_lock)

        # Timestamps
        self.speech_start_timestamp = None
        self.speech_end_timestamp = None

        # VAD state
        self.speech_detected = False

        logger.info(f"Initializing STT adapter with model: {config.model_name}")

        # Audio processing parameters
        self.window_size = 512
        self.pre_vad_samples = 1536  # samples kept before start of speech
        self.post_vad_samples = -200  # samples removed at the end of speech

        # Initialize VAD
        self.vad = VAD(
            threshold=config.vad_threshold,
            min_silence_duration_ms=config.vad_min_silence_duration_ms,
            pre_vad_samples=self.pre_vad_samples,
            post_vad_samples=self.post_vad_samples
        )

        # Pre-VAD buffer using deque with auto-limiting (stores arrays)
        num_windows = int(np.ceil(self.pre_vad_samples / self.window_size))
        self.pre_vad_buffer = deque(maxlen=num_windows)

        # Speech buffer using deque (stores arrays)
        self.speech_buffer = deque()

        # Initialize SpeechToText
        self.stt = SpeechToText(
            model_name=self.config.model_name,
            language=self.config.language,
            task=self.config.task,
            stream_print=self.config.stream_print,
            timer_print=self.config.timer_print,
            audio_chunk_duration=self.config.audio_chunk_duration,
        )

        self.sample_rate = self.stt.sample_rate
        self.window_size_sec = self.window_size / self.sample_rate

        # Register with AudioManager
        channel_indices = self.config.channel_indices or [0]

        stt_config = ReaderConfig(
            channels=1,
            format="F32LE",
            channel_indices=channel_indices,
        )

        self.audio_reader = self.audio_manager.register_reader(name="STT", config=stt_config)

        logger.info(f"STT registered: {stt_config.channels}ch {stt_config.format} from channels {channel_indices}")
        logger.info(f"STT window size: {self.window_size} samples ({self.window_size_sec * 1000:.1f}ms)")
        logger.info(f"STT pre-VAD buffer: {num_windows} windows ({self.pre_vad_samples} samples)")
        logger.info("STT adapter initialized successfully")

    def shutdown(self):
        """Shutdown the adapter and cleanup resources."""
        logger.info("Shutting down STT adapter...")

        # Disable if running
        if self.is_running:
            self.disable()

        # Unregister from audio manager
        if self.audio_reader:
            self.audio_reader.unregister()

        # Clear text queue and wake any waiting threads
        with self._text_available_cv:
            while not self.text_queue.empty():
                try:
                    self.text_queue.get_nowait()
                except queue.Empty:
                    break
            self._text_available_cv.notify_all()

        logger.info("STT adapter shutdown complete")

    def _worker_loop(self):
        """
        Main worker loop that processes audio and generates text.

        Reads audio from AudioManager, processes through VAD and SpeechToText,
        and stores results in the text queue.
        """
        # Enable the audio reader
        self.audio_reader.enable(sync_to_current=True)

        try:
            while not self._stop_event.is_set():
                # Event-driven blocking read with timeout:
                # - Wakes IMMEDIATELY when data arrives (via condition variable)
                # - Falls back to timeout if no data arrives
                # - Returns None on timeout (loop continues and stop event is checked)
                audio_window = self.audio_reader.read(self.window_size, blocking=True, timeout=self.window_size_sec)
                if audio_window is not None and len(audio_window) == self.window_size:
                    self._process_vad_window(audio_window)

                # Check stop event between reads
                if self._stop_event.is_set():
                    break

        except Exception as e:
            logger.error(f"Error in STT worker loop: {e}", exc_info=True)
        finally:
            self.audio_reader.disable()

    def _process_vad_window(self, audio_window: np.ndarray):
        """Process a window of audio through VAD and SpeechToText."""
        speech_timestamps = self.vad(audio_window)

        # Speech start detected
        if speech_timestamps and not self.speech_detected:
            if "start" in speech_timestamps:
                self.speech_start_timestamp = time.perf_counter()

                # Build speech buffer with pre-VAD context
                # Convert pre-VAD deque to single array and add to new deque
                if self.pre_vad_buffer:
                    pre_samples = np.concatenate(list(self.pre_vad_buffer))
                    # Trim to exact pre_vad_samples if exceeded
                    if len(pre_samples) > self.pre_vad_samples:
                        pre_samples = pre_samples[-self.pre_vad_samples:]
                    # Create new deque with pre-samples and current window
                    self.speech_buffer = deque([pre_samples, audio_window])
                else:
                    # No pre-VAD data available
                    self.speech_buffer = deque([audio_window])

                self.speech_detected = True
                logger.debug(f"Speech started (pre-VAD: {len(pre_samples) if self.pre_vad_buffer else 0} samples)")

        # Speech end detected
        elif speech_timestamps and "end" in speech_timestamps:
            self.speech_end_timestamp = time.perf_counter()

            if self.speech_buffer:
                # Add final window
                self.speech_buffer.append(audio_window)

                # Concatenate all chunks
                speech_data = np.concatenate(self.speech_buffer)

                # Trim to post_vad_samples (keep from start)
                if len(speech_data) > self.post_vad_samples:
                    speech_data = speech_data[: self.post_vad_samples]

                logger.debug(f"Speech ended ({len(speech_data)} samples total)")

                # Process through SpeechToText (handles audio saving internally)
                final_text = ""
                for text in self.stt(speech_data.astype(np.float32), ending=True):
                    if text:
                        final_text = text
                        with self._text_available_cv:
                            self.text_queue.put(text)
                            self._text_available_cv.notify_all()

                # Store final result
                if final_text:
                    info = {
                        "start_timestamp": self.speech_start_timestamp,
                        "end_timestamp": self.speech_end_timestamp,
                        "duration": (self.speech_end_timestamp - self.speech_start_timestamp),
                        "samples": len(speech_data),
                    }
                    self._store_result(final_text, info)
                    logger.info(f"Transcribed: '{final_text}' ({info['duration']:.2f}s)")

            self._reset_vad_state()
            self._stop_event.set()  # Auto-disable after speech

        # Speech ongoing - accumulate audio
        elif self.speech_detected:
            self.speech_buffer.append(audio_window)

            # Process chunk if we have enough samples
            total_samples = sum(len(chunk) for chunk in self.speech_buffer)

            if total_samples >= self.stt.current_chunk_length:
                # Concatenate all chunks
                speech_data = np.concatenate(self.speech_buffer)

                # Extract chunk for processing
                chunk = speech_data[: self.stt.current_chunk_length]

                # Keep remaining data in deque
                remainder = speech_data[self.stt.current_chunk_length :]

                # Update speech_buffer with remainder
                if len(remainder) > 0:
                    self.speech_buffer = deque([remainder])
                else:
                    self.speech_buffer = deque()

                logger.debug(f"Processing chunk: {len(chunk)} samples, remainder: {len(remainder)} samples")

                # Process chunk (streaming mode)
                for text in self.stt(chunk.astype(np.float32)):
                    if text:
                        with self._text_available_cv:
                            self.text_queue.put(text)
                            self._text_available_cv.notify_all()

        # No speech - save to pre-VAD buffer
        else:
            # Deque with maxlen automatically drops oldest when full
            self.pre_vad_buffer.append(audio_window)

    def _reset_vad_state(self):
        """Reset VAD processing state."""
        self.pre_vad_buffer.clear()
        self.speech_buffer.clear()
        self.speech_detected = False
        if self.vad:
            self.vad.reset_states()

    def process(self, input_data: np.ndarray = None) -> Optional[str]:
        """
        Process audio input.

        Note: STT processing happens in _worker_loop.
        This method exists for BaseAdapter compatibility.
        """
        pass

    def mic_to_text(self) -> Generator[str, None, None]:
        """
        Process microphone input to text (generator).

        Yields:
            Transcribed text as it becomes available
        """
        start_time = time.time()

        # Ensure capture is running before enabling STT
        if self.audio_manager and not self.audio_manager.is_capture_running():
            logger.info("Starting capture for STT processing")
            self.audio_manager.start_capture()

        # Enable if not already running
        if not self.is_running:
            self.enable(sync_to_current=True)

        # Loop while thread is alive OR queue has items
        while self.is_thread_alive() or not self.text_queue.empty():
            with self._text_available_cv:
                # Check if text is available
                if not self.text_queue.empty():
                    text = self.text_queue.get_nowait()
                    yield text
                    start_time = time.time()  # Reset timeout on activity
                    continue

                # Check if thread is still alive
                if not self.is_thread_alive():
                    logger.debug("Worker thread stopped, draining queue")
                    break

                # Check inactivity timeout before waiting
                if time.time() - start_time >= self.config.inactivity_timeout:
                    logger.info("STT inactivity timeout reached")
                    break

                # Wait efficiently for text or timeout
                self._text_available_cv.wait(timeout=0.1)

    def file_to_text(self, audio_file: str, use_vad: bool = True) -> str:
        """
        Process audio file to text.

        Args:
            audio_file: Path to audio file
            use_vad: Whether to use VAD preprocessing

        Returns:
            Transcribed text
        """
        import torch

        logger.info(f"Processing file: {audio_file}")

        audio_input, _ = load_audio(audio_file, sample_rate=self.sample_rate)

        # Select channel if multi-channel
        if len(audio_input.shape) > 1:
            channel_idx = self.config.channel_indices[0] if self.config.channel_indices else 0
            audio_input = audio_input[channel_idx]

        # Apply VAD if requested
        if use_vad:
            audio_input = self.vad.process(audio_input)
            if audio_input is None:
                logger.warning("No speech detected in file")
                return ""

        # Process through SpeechToText
        text = ""
        if self.stt.model_type == "whisper":
            input_chunks = self.stt.audio_processor.split(audio_input)
        else:
            input_chunks = torch.split(audio_input, self.stt.current_chunk_length)

        for chunk_idx, chunk in enumerate(input_chunks):
            ending = chunk_idx == len(input_chunks) - 1
            for text in self.stt(chunk, ending=ending):
                pass

        return text

    def get_status(self) -> dict:
        """Get adapter status with STT-specific info."""
        base_status = super().get_status()

        # Calculate buffer sizes
        pre_vad_buffer_samples = sum(len(chunk) for chunk in self.pre_vad_buffer)
        speech_buffer_samples = sum(len(chunk) for chunk in self.speech_buffer)

        base_status.update(
            {
                "model_name": self.stt.model_name if self.stt else self.config.model_name,
                "language": self.stt.language if self.stt else self.config.language,
                "task": self.stt.task if self.stt else self.config.task,
                "weight_format": self.stt.weight_format if self.stt else "unknown",
                "speech_detected": self.speech_detected,
                "text_queue_size": self.text_queue.qsize(),
                "sample_rate": self.sample_rate,
                "pre_vad_buffer_samples": pre_vad_buffer_samples,
                "speech_buffer_samples": speech_buffer_samples,
                "pre_vad_buffer_chunks": len(self.pre_vad_buffer),
                "speech_buffer_chunks": len(self.speech_buffer),
            }
        )
        return base_status
