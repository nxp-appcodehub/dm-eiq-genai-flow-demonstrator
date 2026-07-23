# Copyright 2026 NXP
# NXP Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.

import logging
import numpy as np
import threading
from dataclasses import dataclass
from typing import Optional
from vad.vad import VAD
from audio_manager.audio_manager_base import ReaderConfig
from eiq_genai_flow.adapters.base import BaseAdapter
from eiq_genai_flow.adapters.event_manager import EventManager, EventType, Event

logger = logging.getLogger(__name__)


@dataclass
class VADAdapterConfig:
    """Configuration for VAD Adapter."""

    channel_indices: Optional[list] = None
    save_audio: bool = False
    threshold: float = 0.3
    min_silence_duration_ms: int = 200
    pre_vad_samples: int = 1536


class VADAdapter(BaseAdapter):
    """
    Voice Activity Detection adapter.

    Monitors audio stream and emits events when speech starts/ends.
    """

    def __init__(self, config: VADAdapterConfig, audio_manager, event_manager: EventManager):
        """
        Initialize VAD Adapter.

        Args:
            config: VAD adapter configuration
            audio_manager: AudioManager instance
            event_manager: Event bus for publishing speech events
            verbose: Enable verbose logging
        """
        super().__init__(audio_manager, event_manager)
        self.config = config
        self.sample_rate = 16000

        # Initialize VAD
        self.vad = VAD(
            sample_rate=self.sample_rate,
            threshold=self.config.threshold,
            min_silence_duration_ms=self.config.min_silence_duration_ms,
            pre_vad_samples=self.config.pre_vad_samples,
            save_audio=self.config.save_audio
        )

        self.pre_vad_samples = self.vad.pre_vad_samples
        self.window_size = self.vad.required_samples
        self.window_size_sec = self.window_size / self.sample_rate
        self._prev_speech_detected = False

        # Register with AudioManager
        channel_indices = self.config.channel_indices or [0]
        vad_config = ReaderConfig(
            channels=1,
            format="F32LE",
            channel_indices=channel_indices,
        )

        self.audio_reader = self.audio_manager.register_reader(name="VAD", config=vad_config)
        self.wake_event = threading.Event()

        logger.debug(f"VAD registered: {vad_config.channels}ch {vad_config.format} from channels {channel_indices}")
        logger.debug(f"VAD window size: {self.window_size} samples ({self.window_size_sec * 1000:.1f}ms)")
        logger.info("VAD adapter initialized successfully")

    def enable(self, sync_to_current=True):
        super().enable(sync_to_current=sync_to_current)
        self.wake_event.clear()
        self.event_manager.subscribe(
            [EventType.VOICE_ID_USED, EventType.VIT_WAKE, EventType.KEYBOARD_WAKE, EventType.CONTINUOUS_WAKE],
            self._on_wake)

    def disable(self, timeout: float = 5.0):
        super().disable(timeout=timeout)
        self._reset_state()

    def _on_wake(self, event: Event):
        # Unblock worker loop of VAD when event received VOICE_ID_USED, VIT_WAKE or KEYBOARD_WAKE
        logger.info(f"_on_wake event from {event.source}")
        if event.event_type == EventType.VIT_WAKE:
            # Force VAD to restart to avoid case (when voice id is used) when VAD started before VIT wake
            self._reset_state()
            if event.data and "speech_end" in event.data:
                self.audio_reader.read_index = event.data["speech_end"]

        self.wake_event.set()

    def _worker_loop(self):
        while not self._stop_event.is_set():

            if not self.wake_event.wait(timeout=self.window_size_sec):
                # Not process samples before wake event
                self.audio_reader.read_index += self.window_size
                continue

            # Event-driven blocking read

            read_index = self.audio_reader.read_index
            audio_window = self.audio_reader.read(self.window_size, blocking=True, timeout=self.window_size_sec)

            if audio_window is not None:
                self._process_vad_window(audio_window, read_index)

    def _process_vad_window(self, audio_window: np.ndarray, read_index: int):
        """
        Process audio window through VAD.

        Args:
            audio_window: Audio samples to process
        """
        speech_detected, output_chunk, pre_vad_samples = self.vad(audio_window)

        # Speech just started
        if speech_detected and not self._prev_speech_detected:
            logger.info("Speech started")
            pre_vad = pre_vad_samples if pre_vad_samples is not None else self.pre_vad_samples
            logger.debug(f"Speech started at index {read_index}, pre_vad={pre_vad}")
            self.publish(EventType.VAD_SPEECH_START, data={"speech_start": read_index - pre_vad})

        # Speech just ended
        elif not speech_detected and self._prev_speech_detected:
            logger.info("Speech ended")
            logger.debug(f"Speech ended at index {read_index + self.window_size}")
            self.publish(EventType.VAD_SPEECH_END, data={"speech_end": read_index + self.window_size})

        self._prev_speech_detected = speech_detected

    def _reset_state(self):
        """Reset VAD state."""
        self._prev_speech_detected = False
        if self.vad:
            self.vad.flush()
            self.vad.init_streaming_state()

    def shutdown(self):
        """Shutdown the adapter."""
        logger.info("Shutting down VAD adapter...")

        if self.is_running:
            self.disable()

        if self.audio_reader:
            self.audio_reader.unregister()

        logger.info("VAD adapter shutdown complete")

    def get_status(self) -> dict:
        """Get VAD adapter status."""
        base_status = super().get_status()

        base_status.update(
            {
                "speech_detected": self._prev_speech_detected,
                "window_size": self.window_size,
                "sample_rate": self.sample_rate,
            }
        )

        return base_status
