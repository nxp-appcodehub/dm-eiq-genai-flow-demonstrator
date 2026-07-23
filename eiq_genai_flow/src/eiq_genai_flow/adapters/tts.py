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
import numpy as np
import soundfile as sf
from typing import Optional, List
from dataclasses import dataclass
from eiq_genai_flow.adapters.base import EventAdapter
from eiq_genai_flow.adapters.event_manager import EventManager, EventType, Event
from tts.model import TextToSpeech
from tts.config import TTSConstants

logger = logging.getLogger(__name__)


@dataclass
class TTSAdapterConfig:
    """Configuration for TTS Adapter."""

    tts_config: object  # The TTS model config
    lang: str = "English"
    mode: str = "default"
    playback_device: Optional[str] = None
    quiet: bool = False
    lava_test: bool = False


class TTSAdapter(EventAdapter):
    """
    Adapter for Text-To-Speech (TTS) system.

    Uses TTSGenerator for audio generation and AudioManager for playback.
    """

    def __init__(self,
                 audio_manager,
                 event_manager: EventManager,
                 config: TTSAdapterConfig):
        super().__init__(audio_manager=audio_manager, event_manager=event_manager)

        # TTS configuration
        self.config = config

        # Constants
        self.const = TTSConstants()

        # Regex to split text into tokens (words, punctuation, whitespace)
        self.token_splitter = re.compile(r'\w+|[^\w\s]|\s+')

        # Metrics
        self.timestamp_ttfa = 0
        self.inference_time = 0

        # segments concatenation
        self.segment = ""

        # manage concurrent accesses
        self.playing_source = None
        self.pending_queue = queue.Queue()

        # timing measurement
        self.start_time = 0

        # Create TTS instance
        self.tts = TextToSpeech(
            config=self.config.tts_config,
            mode=self.config.mode,
        )

        self.subscribed_events.append(EventType.TTS_PROCESS)

        logger.info("Initialized TTS Adapter")

    def _process_event(self, event: Event):
        """
        Event handling function called from a dedicated thread.
        """
        # Handle only TTS_PROCESS events
        if event.event_type != EventType.TTS_PROCESS:
            logger.debug(f"Ignoring event: {event.event_type}")
            return

        # Retrieve event data
        text = event.data.get("text", "")
        eos = event.data.get("eos", True)
        source = event.source

        if not self.playing_source:
            # nothing is already playing, start now.
            self.playing_source = source
            self.timestamp_ttfa = 0
            self.start_time = time.perf_counter()

            self.publish(EventType.TTS_START_SEGMENT)
        else:
            # another source is currently playing
            if self.playing_source != source:
                self.pending_queue.put(event)
                return

        if text:
            # split text into tokens and concatenate into segments
            tokens = self.token_splitter.findall(text)
        else:
            tokens = []

        for token in tokens:
            # Accumulate text tokens
            # Generate at punctuation marks
            if token and token[0] == " " and self.segment and self.segment[-1] in self.const.PUNCTUATIONS:
                self._generate_and_queue_audio(self.segment)
                self.segment = ""
            else:
                self.segment += token

        # Generate final segment
        if eos:
            if self.segment:
                self._generate_and_queue_audio(self.segment)
            self.segment = ""
            self.inference_time = time.perf_counter() - self.start_time

            logger.debug("End of sequence processed")

            # Generation complete, wait for playback to finish before processing next
            # TODO: could be optimized by starting generation of a potential next text
            #       in parallel while playback is still happening. But this would require
            #       a complex state machine to handle for a marginal use-case.

            if self.audio_manager:
                # Signal end of stream to audio_manager
                logger.debug("Waiting for audio_manager playback to complete...")
                if hasattr(self.audio_manager, "signal_stream_end"):
                    self.audio_manager.signal_stream_end()
                # Wait signal from audio_manager that playback is complete
                with self.audio_manager.playback_queue_cv:
                    while not self.audio_manager.is_playback_complete():
                        self.audio_manager.playback_queue_cv.wait(timeout=0.5)

            logger.debug("TTS completion wait finished")

            self.publish(EventType.TTS_COMPLETE, {"tts_process_source": self.playing_source})

            # Reset for next sequence
            self.playing_source = None

            # re-post any pending events from other sources
            while not self.pending_queue.empty():
                self.event_manager.post_event(self.pending_queue.get())

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
                    for audio_chunk in tts_out:
                        self._set_timestamp_ttfa()
                        logger.debug(f"Sending audio chunk to manager: shape={audio_chunk.shape}")
                        if not self.config.quiet:
                            self.audio_manager.play_audio_async(audio_chunk,
                                                                sample_rate=self.config.tts_config.samplerate)
        except Exception as e:
            logger.error(f"Error generating audio for text '{text}': {e}", exc_info=True)

    def _set_timestamp_ttfa(self):
        """Set timestamp for Time To First Audio metric."""
        if self.playing_source and self.timestamp_ttfa == 0:
            self.timestamp_ttfa = time.perf_counter()

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
