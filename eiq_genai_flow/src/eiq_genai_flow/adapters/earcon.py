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
import os
import random
import soundfile as sf

from typing import Dict

from eiq_genai_flow.adapters.base import EventAdapter
from eiq_genai_flow.adapters.event_manager import EventManager, EventType, Event

logger = logging.getLogger(__name__)


class EarconConfig:
    """Default Configuration for earcon sounds."""
    wakeword_earcon: bool = True
    tts_start_earcon: bool = True
    intent_earcon: bool = True
    tts_filler_earcon: bool = False


class EarconAdapter(EventAdapter):
    """
    Manages loading and playing earcon sounds (earcons).

    Centralizes earcon sound management to avoid code duplication
    and provide a simple interface for playing audio earcons.
    """

    def __init__(self,
                 audio_manager,
                 event_manager: EventManager,
                 assets_path: str,
                 config: EarconConfig = None,
                 ):
        """
        Initialize the earcon manager.
        """
        super().__init__(audio_manager=audio_manager, event_manager=event_manager)
        self.sounds: Dict[str, tuple] = {}

        if config is None:
            config = EarconConfig()

        # Load predifined earcon
        if config.wakeword_earcon:
            self.subscribed_events.append(EventType.VIT_WAKE)
            self._load_wav_earcon("wake_word", os.path.join(assets_path, "ww_earcon.wav"))

        if config.tts_start_earcon:
            self.subscribed_events.append(EventType.TTS_START_SEGMENT)
            self._load_wav_earcon("tts_start", os.path.join(assets_path, "tts_earcon.wav"))

        if config.intent_earcon:
            self.subscribed_events.append(EventType.INTENT_DETECTED)
            self._load_wav_earcon("intent_detected", os.path.join(assets_path, "intent_earcon.wav"))

        if config.tts_filler_earcon:
            self.subscribed_events.append(EventType.AUDIO_FILLER_REGISTER)
            self.subscribed_events.append(EventType.AUDIO_FILLER_PLAY)
            self.nb_audio_fillers = 0

    def _process_event(self, event: Event):
        """
        Event handlig function called from a dedicated thread.
        """
        if event.event_type == EventType.VIT_WAKE:
            self._play_earcon("wake_word")
        elif event.event_type == EventType.TTS_START_SEGMENT:
            self._play_earcon("tts_start")
        elif event.event_type == EventType.INTENT_DETECTED:
            self._play_earcon("intent_detected")
        elif event.event_type == EventType.AUDIO_FILLER_REGISTER:
            self._load_wav_earcon(f"audio_filler_{self.nb_audio_fillers}", event.data)
            self.nb_audio_fillers += 1
        elif event.event_type == EventType.AUDIO_FILLER_PLAY:
            if self.nb_audio_fillers > 0:
                idx = random.randint(0, self.nb_audio_fillers - 1)
                self._play_earcon(f"audio_filler_{idx}")

    def _load_wav_earcon(self, name: str, wav_path: str) -> bool:
        """
        Register and load an earcon sound from a WAV file.

        Args:
            name: Unique identifier for this earcon
            wav_path: Path to the WAV file

        Returns:
            True if successfully loaded, False otherwise
        """
        try:
            if not os.path.exists(wav_path):
                logger.warning(f"earcon sound not found: {wav_path}")
                return False

            audio_data, sample_rate = sf.read(wav_path)
            self.sounds[name] = (audio_data, sample_rate)
            logger.debug(f"Registered earcon '{name}' from {wav_path} (shape={audio_data.shape})")
            return True

        except Exception as e:
            logger.warning(f"Failed to load earcon sound {wav_path}: {e}")
            return False

    def _play_earcon(self, name: str) -> bool:
        """
        Play an earcon sound.

        Args:
            name: Name of earcon to play

        Returns:
            True if playback was initiated, False otherwise
        """
        # Check if earcon exists
        if name not in self.sounds:
            logger.warning(f"earcon '{name}' not registered, cannot play")
            return False

        try:
            audio_data, sample_rate = self.sounds[name]
            duration = len(audio_data) / sample_rate

            logger.debug(
                f"Playing earcon '{name}': shape={audio_data.shape}, sr={sample_rate}, duration={duration:.2f}s"
            )

            # Use async playback to avoid blocking issues with playback state
            self.audio_manager.play_audio_async(audio_data, sample_rate=sample_rate)

            # Signal end of stream to audio_manager
            if self.audio_manager and hasattr(self.audio_manager, "signal_stream_end"):
                self.audio_manager.signal_stream_end()

            return True
        except Exception as e:
            logger.error(f"Failed to play earcon '{name}': {e}")
            return False
