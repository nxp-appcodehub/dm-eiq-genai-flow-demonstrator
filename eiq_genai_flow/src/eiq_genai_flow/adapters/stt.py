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
import queue
import logging
import threading
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional
from shared_utils.utils import get_number_of_cores

from speech_to_text.speech_to_text import SpeechToText
from speech_to_text.utils.utils import consume_buffer
from audio_manager.audio_manager_base import ReaderConfig
from eiq_genai_flow.adapters.base import BaseAdapter
from eiq_genai_flow.adapters.event_manager import EventManager, EventType, Event

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
    use_neutron: int = False


class STTAdapter(BaseAdapter):
    """
    Adapter for SpeechToText that uses AudioManager for audio capture.

    This adapter bridges the SpeechToText class with the centralized AudioManager,
    handling audio capture, VAD processing, and text generation.
    """

    def __init__(self, config: STTAdapterConfig, audio_manager, event_manager: EventManager, vad_window_size_sec: int):
        super().__init__(audio_manager, event_manager)
        self.config = config

        # Text output queue
        self.text_queue = queue.Queue()

        # state management
        self._speech_start_event = threading.Event()
        self._speech_end_event = threading.Event()
        self._state_lock = threading.Lock()

        # Speech processing state
        self.speech_start_index = None
        self.speech_end_index = None

        logger.info(f"Initializing STT adapter with model: {config.model_name}")

        # Leave headroom for gstreamer audio processing on platform whith enough cores
        nb_onnx_cores = get_number_of_cores()
        backend = os.getenv("AUDIO_BACKEND", "auto")
        if backend == "gstbf" and nb_onnx_cores >= 4:
            nb_onnx_cores = nb_onnx_cores - 1

        # Initialize SpeechToText
        self.stt = SpeechToText(
            model_name=self.config.model_name,
            language=self.config.language,
            task=self.config.task,
            stream_print=self.config.stream_print,
            timer_print=self.config.timer_print,
            audio_chunk_duration=self.config.audio_chunk_duration,
            nb_onnx_cores=nb_onnx_cores,
            use_neutron_enc=self.config.use_neutron,
            use_neutron_dec=False,  # Best configuration using neutron on encoder only
        )

        self.vad_window_size_sec = vad_window_size_sec
        self.sample_rate = self.stt.sample_rate
        self.window_size = int(self.vad_window_size_sec * self.stt.sample_rate)
        self.pre_wake_window_size = deque()

        # Register with AudioManager
        channel_indices = self.config.channel_indices or [0]

        stt_config = ReaderConfig(
            channels=1,
            format="F32LE",
            channel_indices=channel_indices,
        )

        self.audio_reader = self.audio_manager.register_reader(name="STT", config=stt_config)

        logger.debug(f"STT registered: {stt_config.channels}ch {stt_config.format} from channels {channel_indices}")
        logger.info("STT adapter initialized successfully")

    def enable(self, sync_to_current=True):
        # clear
        self._speech_start_event.clear()
        self._speech_end_event.clear()
        self.pre_wake_window_size.clear()
        self.speech_start_index = None
        self.speech_end_index = None
        self._inactivity_elapsed = 0.0
        self._wake_received = False  # only start inactivity counter after wake

        # Subscribe to wake events - _on_wake will then subscribe to VAD_SPEECH_START
        self.subscribe([
            EventType.VIT_WAKE,
            EventType.VOICE_ID_WAKE,
            EventType.KEYBOARD_WAKE,
            EventType.CONTINUOUS_WAKE
        ], self._on_wake)

        super().enable(sync_to_current=sync_to_current)

    def disable(self, timeout: float = 5.0):
        super().disable(timeout=timeout)

        # drain text_queue (if stt is force-interrupted)
        self.text_queue = queue.Queue()  # TODO: maybe do more properly?

    def _on_wake(self, event: Event):
        logger.info(f"_on_wake event from {event.source}")
        with self._state_lock:
            self._wake_received = True  # start inactivity counter from now
            if event.event_type == EventType.VIT_WAKE :
                # enable VAD start directly
                self.subscribe(EventType.VAD_SPEECH_START, self._on_vad_speech_start)
                # not allow event VOICE_ID_WAKE if VIT_WAKE
                self.unsubscribe(EventType.VOICE_ID_WAKE, self._on_wake)

            elif event.event_type == EventType.VOICE_ID_WAKE :
                # take only the speech allowed by voice id
                if not self._speech_start_event.is_set():
                    # if speech not already start
                    self.speech_start_index = event.data.get("speech_start")
                    self.audio_reader.read_index = self.speech_start_index
                    window_size = event.data.get("window_size")
                    if window_size is not None :
                        if window_size > self.stt.current_chunk_length:
                            # Split window size into chunk of self.stt.current_chunk_length
                            count, remainder = divmod(window_size, self.stt.current_chunk_length)
                            self.pre_wake_window_size.extend([self.stt.current_chunk_length] * count)
                            if remainder:
                                self.pre_wake_window_size.append(remainder)
                        else :
                            self.pre_wake_window_size.append(window_size)

                    self._speech_start_event.set()

                if event.data.get("is_speech_ended") :
                    # if speech ended
                    self.speech_end_index = event.data.get("speech_end")
                    self._speech_end_event.set()
                    self.unsubscribe(EventType.VOICE_ID_WAKE, self._on_wake)

                # not allow event VIT_WAKE if VOICE_ID_WAKE
                self.unsubscribe(EventType.VIT_WAKE, self._on_wake)
            else :
                # enable VAD start for keyboard or continuous wake
                self.subscribe(EventType.VAD_SPEECH_START, self._on_vad_speech_start)

    def _on_vad_speech_start(self, event: Event):
        logger.info(f"_on_vad_speech_start event from {event.source}")
        with self._state_lock:
            self.speech_start_index = event.data.get("speech_start")
            self.audio_reader.read_index = self.speech_start_index
            self._speech_start_event.set()

            # enable VAD end, avoid another VAD_SPEECH_START while STT processes
            self.subscribe(EventType.VAD_SPEECH_END, self._on_vad_speech_end)
            self.unsubscribe(EventType.VAD_SPEECH_START, self._on_vad_speech_start)

    def _on_vad_speech_end(self, event: Event):
        logger.info(f"_on_vad_speech_end event from {event.source}")
        with self._state_lock:
            self.speech_end_index = event.data.get("speech_end")
            logger.debug("_speech_end_event: set")
            self._speech_end_event.set()

            self.unsubscribe(EventType.VAD_SPEECH_END, self._on_vad_speech_end)

    def _worker_loop(self):
        speech_to_process = deque()
        while not self._stop_event.is_set():
            # Waits for speech start, with inactivity timeout
            if not self._speech_start_event.wait(timeout=self.vad_window_size_sec):
                # Only count inactivity after a wake event has been received
                if self._wake_received:
                    self._inactivity_elapsed += self.vad_window_size_sec
                    if self._inactivity_elapsed >= self.config.inactivity_timeout:
                        logger.warning(
                            f"Inactivity timeout ({self.config.inactivity_timeout}s). No speech detected, try again."
                        )
                        self.publish(EventType.TIMEOUT)
                        self._stop_event.set()
                        return
                continue

            # After VOICE_ID_WAKE, read samples needed by Voice ID for speaker's verification during speech
            if self.pre_wake_window_size:
                window_size = self.pre_wake_window_size.popleft()
                samples = self.audio_reader.read(window_size, blocking=True)
                speech_to_process.extend(samples)

            else:
                samples = self.audio_reader.read(self.window_size, blocking=True)
                speech_to_process.extend(samples)

            if self._speech_end_event.wait(timeout=self.vad_window_size_sec):
                logger.debug("_speech_end_event: consume_buffer")

                # get remaining data until speech_end_index
                window_size = self.speech_end_index - self.audio_reader.read_index
                samples = self.audio_reader.read(window_size, blocking=True)
                speech_to_process.extend(samples)

                chunk = consume_buffer(speech_to_process, len(speech_to_process))
                ending = True

            elif len(speech_to_process) >= self.stt.current_chunk_length:
                logger.debug("full buffer: consume_buffer")
                chunk = consume_buffer(speech_to_process, self.stt.current_chunk_length)
                ending = False

            else:
                continue

            logger.debug(f"processing: {chunk.shape}")

            # process
            for text in self.stt(chunk.astype(np.float32), ending=ending):
                if text:
                    self.publish(EventType.INPUT_TEXT, data=text)

            if ending:
                self._stop_event.set()

                self.publish(
                    EventType.STT_END,
                    data={"speech_start": self.speech_start_index, "speech_end": self.speech_end_index},
                )

                self.publish(EventType.END_OF_INPUT)

    def shutdown(self):
        """Shutdown the adapter and cleanup resources."""
        logger.info("Shutting down STT adapter...")

        # Disable if running
        self.disable()

        # Unregister from audio manager
        if self.audio_reader:
            self.audio_reader.unregister()

        # Clear text queue
        while not self.text_queue.empty():
            try:
                self.text_queue.get_nowait()
            except queue.Empty:
                break

        logger.info("STT adapter shutdown complete")

    def get_status(self) -> dict:
        """Get adapter status."""
        base_status = super().get_status()

        base_status.update(
            {
                "model_name": self.stt.model_name if self.stt else self.config.model_name,
                "language": self.stt.language if self.stt else self.config.language,
                "task": self.stt.task if self.stt else self.config.task,
                "text_queue_size": self.text_queue.qsize(),
            }
        )

        return base_status
