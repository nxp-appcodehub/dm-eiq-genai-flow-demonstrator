#!/usr/bin/env python3
# Copyright 2026 NXP
# NXP Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.

import os
import re
import time
import queue
import logging
import threading
import numpy as np
import soundfile as sf
from typing import Optional, List
from dataclasses import dataclass
from adapters.base import BaseAdapter
from tts.model import TextToSpeech
from tts.config import TTSConstants

logger = logging.getLogger(__name__)


@dataclass
class TTSAdapterConfig:
    """Configuration for TTS Adapter."""

    tts_config: object  # The TTS model config
    mode: str = "default"
    playback_device: Optional[str] = None
    quiet: bool = False
    lava_test: bool = False


class TTSAdapter(BaseAdapter):
    """
    Adapter for Text-To-Speech (TTS) system.

    Uses TTSGenerator for audio generation and AudioManager for playback.
    """

    def __init__(self, config: TTSAdapterConfig, audio_manager=None, verbose=False):
        """
        Initialize TTS Adapter.

        Args:
            config: TTSAdapterConfig instance
            audio_manager: AudioManager instance for audio playback
            verbose: Enable verbose logging
        """
        super().__init__(config, audio_manager, verbose)

        # Override thread name
        self._thread_name = "TTS_Worker"

        # Text queue with condition variable for event-driven access
        self.text_queue = queue.Queue()
        self._text_queue_lock = threading.Lock()
        self.text_queue_cv = threading.Condition(self._text_queue_lock)

        self.generation_complete = threading.Event()

        # Constants
        self.const = TTSConstants()

        # Regex to split text into tokens (words, punctuation, whitespace)
        self.token_splitter = re.compile(r'\w+|[^\w\s]|\s+')

        # Metrics
        self.timestamp_ttfa = 0
        self.inference_time = 0
        self.start_play = True

        logger.info("Initializing TTS Adapter with TextToSpeech")

        # Create TTS instance
        self.tts = TextToSpeech(
            config=self.config.tts_config,
            mode=self.config.mode,
        )

        logger.info(f"TTS Adapter initialized with model: {self.tts.model_name}")

    def shutdown(self):
        """Shutdown the TTS adapter."""
        # Disable to stop worker thread
        if self.is_running:
            self.disable()

        # Clean up tts
        self.tts = None

    def _worker_loop(self):
        """
        Main worker loop - processes text queue and generates audio.
        """
        segment = ""
        start_time = time.perf_counter()
        self.start_play = True

        # Send initial START token
        with self.text_queue_cv:
            self.text_queue.put(self.const.START_TOKEN)
            self.text_queue_cv.notify_all()

        logger.debug("TTS worker loop started")

        try:
            while not self._stop_event.is_set():
                with self.text_queue_cv:
                    # Wait for text to be available
                    while self.text_queue.empty() and not self._stop_event.is_set():
                        self.text_queue_cv.wait(timeout=0.1)

                    # Check stop event
                    if self._stop_event.is_set():
                        break

                    # Get token if available
                    if not self.text_queue.empty():
                        token = self.text_queue.get_nowait()

                        # Notify if queue just became empty
                        if self.text_queue.empty():
                            self.text_queue_cv.notify_all()
                    else:
                        continue

                # Handle special tokens (outside the lock)
                if token == self.const.EXIT_TOKEN or token is None:
                    logger.debug("Exit token received")
                    break

                elif token == self.const.END_TOKEN:
                    # Generate final segment
                    if segment:
                        self._generate_and_queue_audio(segment)
                    segment = ""
                    self.inference_time = time.perf_counter() - start_time

                    # Signal end of stream to audio_manager
                    if self.audio_manager and hasattr(self.audio_manager, "signal_stream_end"):
                        self.audio_manager.signal_stream_end()

                    logger.debug("End of sequence processed")

                    # Signal that generation is complete
                    self.generation_complete.set()

                    # Reset for next question/generation cycle
                    self.start_play = True

                elif token == self.const.START_TOKEN:
                    self.start_play = True
                    logger.debug("Start token received")
                    start_time = time.perf_counter()

                else:
                    # Clear completion flag only when receiving actual text after completion
                    if self.generation_complete.is_set():
                        logger.debug("Clearing generation_complete - new text received")
                        self.generation_complete.clear()
                        start_time = time.perf_counter()

                    # Accumulate text tokens
                    # Generate at punctuation marks
                    if token[0] == " " and segment and segment[-1] in self.const.PUNCTUATIONS:
                        self._generate_and_queue_audio(segment)
                        segment = ""

                    segment += token

        except Exception as e:
            logger.error(f"Error in TTS worker loop: {e}", exc_info=True)
        finally:
            self.generation_complete.set()  # Ensure it's set on exit
            logger.debug(f"TTS worker loop stopped (total time: {self.inference_time:.2f}s)")

    def _generate_and_queue_audio(self, text: str):
        """
        Generate audio from text and send directly to audio_manager.

        Args:
            text: Text segment to synthesize
        """
        try:
            # Generate audio
            tts_out = self.tts.generate(text)
            # Send directly to audio_manager (no intermediate queue)
            if self.audio_manager:
                if isinstance(tts_out, np.ndarray):
                    self._set_timestamp_ttfa()
                    logger.debug(f"Sending audio chunk to manager: shape={tts_out.shape}")
                    if not self.config.quiet:
                        self.audio_manager.play_audio_async(tts_out, sample_rate=self.config.tts_config.samplerate)
                else:
                    for i, audio_chunk in enumerate(tts_out):
                        if i == 0:
                            self._set_timestamp_ttfa()
                        logger.debug(f"Sending audio chunk to manager: shape={audio_chunk.shape}")
                        if not self.config.quiet:
                            self.audio_manager.play_audio_async(audio_chunk,
                                                                sample_rate=self.config.tts_config.samplerate)
        except Exception as e:
            logger.error(f"Error generating audio for text '{text}': {e}", exc_info=True)

    def _set_timestamp_ttfa(self):
        """Set timestamp for Time To First Audio metric."""
        if self.start_play:
            self.timestamp_ttfa = time.perf_counter()
            self.start_play = False

    def process(self, text: Optional[str] = None, eos: bool = False):
        """
        Process text input for TTS generation.

        Args:
            text: Text to synthesize (can be a single token or full sentence)
            eos: End of sequence flag
        """
        # Auto-enable if not running
        if not self.is_running:
            self.enable()

        with self.text_queue_cv:
            if text:
                tokens = self.token_splitter.findall(text)
                for tok in tokens:
                    self.text_queue.put(tok)
            if eos:
                self.text_queue.put(self.const.END_TOKEN)
            self.text_queue_cv.notify_all()

    def wait_for_completion(self):
        """
        Wait for TTS generation and playback to complete.
        """
        # 1. Wait for text queue to empty
        logger.debug(f"Waiting for text queue to empty... (size: {self.text_queue.qsize()})")
        with self.text_queue_cv:
            while not self.text_queue.empty():
                self.text_queue_cv.wait()

        # 2. Wait for generation to complete
        logger.debug("Waiting for generation to complete...")
        self.generation_complete.wait()

        # 3. Wait for playback to complete
        if self.audio_manager:
            logger.debug("Waiting for audio_manager playback to complete...")
            with self.audio_manager.playback_queue_cv:
                while not self.audio_manager.is_playback_complete():
                    self.audio_manager.playback_queue_cv.wait(timeout=0.5)

        logger.debug("TTS completion wait finished")

    def stop_generation(self):
        """Stop current TTS generation (interrupt)."""
        # Clear text queue
        with self.text_queue_cv:
            while not self.text_queue.empty():
                try:
                    self.text_queue.get_nowait()
                except queue.Empty:
                    break
            self.text_queue_cv.notify_all()

        # Stop audio_manager playback
        if self.audio_manager:
            self.audio_manager.stop_playback()

        logger.info("TTS generation stopped and queues cleared")

    def generate_audio_fillers(self, dir_path: str, sentences: List[str]):
        os.makedirs(dir_path, exist_ok=True)
        for i, sentence in enumerate(sentences):
            filler_path = os.path.join(dir_path, f"{i}.wav")
            if os.path.exists(filler_path):
                os.remove(filler_path)
            audio = self.tts.generate(sentence)
            if not isinstance(audio, np.ndarray):
                audio_list = []
                for chunk in audio:
                    audio_list.append(chunk)
                audio = np.concatenate(audio_list, axis=0).copy()
            sf.write(filler_path, audio, 16000)

    @property
    def model_name(self):
        """Get TTS model name."""
        return self.tts.model_name if self.tts else "Unknown"

    @property
    def metrics(self):
        """Get TTS metrics."""
        return self.tts.metrics if self.tts else {}

    def get_status(self):
        """Get adapter status with TTS-specific info."""
        base_status = super().get_status()
        base_status.update(
            {
                "model_name": self.model_name,
                "text_queue_size": self.text_queue.qsize(),
                "timestamp_ttfa": self.timestamp_ttfa,
                "inference_time": self.inference_time,
            }
        )
        return base_status
