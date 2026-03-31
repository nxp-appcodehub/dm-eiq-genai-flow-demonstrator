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
import logging
from typing import Dict
import soundfile as sf

logger = logging.getLogger(__name__)


class EarconManager:
    """
    Manages loading and playing earcon sounds (earcons).

    Centralizes earcon sound management to avoid code duplication
    and provide a simple interface for playing audio earcons.
    """

    def __init__(self, audio_manager):
        """
        Initialize the earcon manager.

        Args:
            audio_manager: AudioManager instance for playback
        """
        self.audio_manager = audio_manager
        self.sounds: Dict[str, tuple] = {}
        self.enabled: Dict[str, bool] = {}

    def register_earcon(self, name: str, wav_path: str, enabled: bool = False) -> bool:
        """
        Register and load an earcon sound from a WAV file.

        Args:
            name: Unique identifier for this earcon
            wav_path: Path to the WAV file
            enabled: Whether to enable this earcon immediately

        Returns:
            True if successfully loaded, False otherwise
        """
        try:
            if not os.path.exists(wav_path):
                logger.warning(f"earcon sound not found: {wav_path}")
                return False

            audio_data, sample_rate = sf.read(wav_path)
            self.sounds[name] = (audio_data, sample_rate)
            self.enabled[name] = enabled
            logger.debug(f"Registered earcon '{name}' from {wav_path} (enabled={enabled}, shape={audio_data.shape})")
            return True

        except Exception as e:
            logger.warning(f"Failed to load earcon sound {wav_path}: {e}")
            return False

    def enable_earcon(self, name: str, enabled: bool = True) -> None:
        """
        Enable or disable a specific earcon.

        Args:
            name: Name of earcon to enable/disable
            enabled: True to enable, False to disable
        """
        if name not in self.sounds:
            logger.warning(f"Cannot enable unknown earcon: {name}")
            return

        self.enabled[name] = enabled
        status = "enabled" if enabled else "disabled"
        logger.debug(f"earcon '{name}' {status}")

    def play_earcon(self, name: str, force: bool = False) -> bool:
        """
        Play an earcon sound.

        Args:
            name: Name of earcon to play
            force: If True, play even if disabled

        Returns:
            True if playback was initiated, False otherwise
        """
        # Check if earcon exists
        if name not in self.sounds:
            logger.warning(f"earcon '{name}' not registered, cannot play")
            return False

        # Check if enabled (unless forced)
        if not force and not self.enabled.get(name, False):
            logger.debug(f"earcon '{name}' is disabled, skipping")
            return False

        try:
            audio_data, sample_rate = self.sounds[name]
            duration = len(audio_data) / sample_rate

            logger.debug(
                f"Playing earcon '{name}': shape={audio_data.shape}, sr={sample_rate}, duration={duration:.2f}s"
            )

            # Use async playback to avoid blocking issues with playback state
            self.audio_manager.play_audio_async(audio_data, sample_rate=sample_rate)

            return True
        except Exception as e:
            logger.error(f"Failed to play earcon '{name}': {e}")
            return False

    def is_loaded(self, name: str) -> bool:
        """Check if a earcon is loaded."""
        return name in self.sounds

    def is_enabled(self, name: str) -> bool:
        """Check if a earcon is enabled."""
        return self.enabled.get(name, False)

    def unload_earcon(self, name: str) -> None:
        """Unload a earcon sound from memory."""
        if name in self.sounds:
            del self.sounds[name]
            if name in self.enabled:
                del self.enabled[name]
            logger.debug(f"Unloaded earcon '{name}'")

    def clear_all(self) -> None:
        """Clear all loaded earcon sounds."""
        self.sounds.clear()
        self.enabled.clear()
        logger.debug("Cleared all earcon sounds")
