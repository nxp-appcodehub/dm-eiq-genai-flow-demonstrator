# Copyright 2026 NXP
# NXP Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.

"""
ALSA Audio Backend

Provides pure ALSA-based audio capture and playback using pyalsaaudio.
Lightweight alternative to GStreamer for systems without it.
"""

import time
import logging
import threading
import numpy as np
import alsaaudio as aa
from typing import Optional

from audio_manager.audio_manager_base import AudioManager, CaptureConfig, PlaybackConfig

logger = logging.getLogger(__name__)


class AudioManagerALSA(AudioManager):
    """ALSA-based audio manager using pyalsaaudio."""

    def __init__(
        self, capture_config: Optional[CaptureConfig] = None, playback_config: Optional[PlaybackConfig] = None
    ):
        super().__init__(capture_config, playback_config)

        # Convert string formats to ALSA constants for both capture and playback
        # This allows configuration to use human-readable format strings
        self.alsa_capture_format = self._string_to_alsa_format(self.capture_config.format)
        self.alsa_playback_format = self._string_to_alsa_format(self.playback_config.format)

        # ALSA PCM device handles - initialized when devices are opened
        self.capture_pcm: Optional[aa.PCM] = None
        self.playback_pcm: Optional[aa.PCM] = None

        # Track current playback sample rate for device reuse optimization
        self.playback_sample_rate: Optional[int] = None

        logger.info("AudioManagerALSA initialized")

    @staticmethod
    def _string_to_alsa_format(format_str: str):
        """
        Convert string format to ALSA constant.

        Maps human-readable format strings (e.g., "S16LE") to pyalsaaudio constants.
        Defaults to S32LE (32-bit signed little-endian) if format is not recognized.
        """
        format_map = {
            "S32LE": aa.PCM_FORMAT_S32_LE,  # 32-bit signed little-endian
            "S16LE": aa.PCM_FORMAT_S16_LE,  # 16-bit signed little-endian
            "F32LE": aa.PCM_FORMAT_FLOAT_LE,  # 32-bit float little-endian
            "S8": aa.PCM_FORMAT_S8,  # 8-bit signed
            "U8": aa.PCM_FORMAT_U8,  # 8-bit unsigned
        }
        return format_map.get(format_str, aa.PCM_FORMAT_S32_LE)

    # =========================================================================
    # Capture implementation
    # =========================================================================

    def _open_capture_device(self, retries: int = 5) -> aa.PCM:
        """
        Open ALSA capture device with retry logic.

        Attempts to open the capture device multiple times if it's busy.
        This handles cases where the device might be briefly occupied by
        another process or needs time to become available.

        Args:
            retries: Number of attempts to open the device

        Returns:
            Configured ALSA PCM capture object

        Raises:
            RuntimeError: If device cannot be opened after all retries
        """
        # Use configured samples per frame as the ALSA period size
        # Period size determines how many frames are read at once
        periodsize = self.capture_config.samples_per_frame

        for attempt in range(retries):
            try:
                return aa.PCM(
                    type=aa.PCM_CAPTURE,
                    mode=aa.PCM_NORMAL,
                    rate=self.capture_config.sample_rate,
                    channels=self.capture_config.channels,
                    periodsize=periodsize,
                    device=self.capture_device,
                    format=self.alsa_capture_format,
                )
            except aa.ALSAAudioError as e:
                # Retry only if device is busy and we have retries remaining
                if "Device or resource busy" in str(e) and attempt < retries - 1:
                    # Wait for one frame duration before retrying
                    time.sleep(self.capture_config.frame_duration_sec)
                else:
                    raise

        raise RuntimeError("Failed to open capture device after retries")

    def start_capture(self):
        """
        Start ALSA audio capture.

        Creates and starts a background thread that continuously reads audio
        from the ALSA capture device. Initializes capture state and statistics.
        """
        # Prevent starting if already running
        if self.capture_running.is_set():
            return

        logger.info(f"Starting capture: {self.capture_device} ({self.capture_config.sample_rate}Hz, "
                    f"{self.capture_config.channels}ch)")

        # Set flags and reset statistics
        self.capture_running.set()
        self.buffers_dropped = 0  # Track number of dropped buffers
        self.fade_in_done = False  # Track fade-in state for smooth startup

        # Start capture thread as daemon so it doesn't prevent program exit
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

    def stop_capture(self):
        """
        Stop ALSA audio capture.

        Signals the capture thread to stop, waits for it to finish,
        and cleans up the ALSA device handle.
        """
        if not self.capture_running.is_set():
            return

        logger.info("Stopping capture")

        # Signal thread to stop
        self.capture_running.clear()

        # Wait for thread to finish (with timeout to prevent hanging)
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)

        # Close and clean up ALSA device
        if self.capture_pcm:
            try:
                self.capture_pcm.close()
            except Exception:
                pass  # Ignore errors during cleanup
            self.capture_pcm = None

    def _capture_loop(self):
        """
        Capture thread loop.

        Continuously reads audio frames from the ALSA device and processes them.
        Runs in a separate thread until capture_running is cleared.
        """
        try:
            # Open the ALSA capture device
            self.capture_pcm = self._open_capture_device()

            # Start audio recorder if configured
            if self.audio_capture_recorder:
                self.audio_capture_recorder.start()

            # Determine numpy dtype based on ALSA format
            # This ensures correct interpretation of raw bytes from ALSA
            dtype_map = {
                aa.PCM_FORMAT_FLOAT_LE: np.float32,
                aa.PCM_FORMAT_S32_LE: np.int32,
                aa.PCM_FORMAT_S16_LE: np.int16,
                aa.PCM_FORMAT_S8: np.int8,
                aa.PCM_FORMAT_U8: np.uint8,
            }
            dtype = dtype_map.get(self.alsa_capture_format, np.int32)

            # Main capture loop - runs until stop is requested
            while self.capture_running.is_set():
                # Blocking read - waits until data is available
                # Returns (number of frames read, raw audio data)
                length, data = self.capture_pcm.read()

                if length > 0:
                    # Convert raw bytes to numpy array with correct data type
                    # .copy() ensures data is writable and not tied to buffer
                    audio_data = np.frombuffer(data, dtype=dtype).copy()

                    # Reshape multi-channel audio to (samples, channels)
                    if self.capture_config.channels > 1:
                        audio_data = audio_data.reshape(-1, self.capture_config.channels)

                    # Process the captured audio (fade-in, callbacks, recording, etc.)
                    self._process_captured_audio(audio_data)

        except Exception as e:
            logger.error(f"Capture error: {e}", exc_info=True)
        finally:
            # Ensure recorder is stopped even if error occurs
            if self.audio_capture_recorder:
                self.audio_capture_recorder.stop()

    # =========================================================================
    # Playback implementation
    # =========================================================================

    def _open_playback_device(self, sample_rate: Optional[int] = None) -> aa.PCM:
        """
        Open ALSA playback device with configured period size.

        Period size controls how much audio data ALSA expects per write,
        affecting latency and buffer underrun behavior.
        """
        sr = sample_rate or self.playback_config.sample_rate

        # Calculate periodsize based on sample rate and frame duration
        # (must recalculate if sample_rate is different from config)
        if sr != self.playback_config.sample_rate:
            periodsize = int(sr * self.playback_config.frame_duration_sec)
        else:
            periodsize = self.playback_config.samples_per_frame

        logger.info(f"Opening playback: {self.playback_device} at {sr}Hz (periodsize={periodsize})")

        return aa.PCM(
            type=aa.PCM_PLAYBACK,
            mode=aa.PCM_NORMAL,
            rate=sr,
            channels=self.playback_config.channels,
            format=self.alsa_playback_format,
            device=self.playback_device,
            periodsize=periodsize,
        )

    def start_playback(self):
        """
        Start ALSA playback thread.

        Creates a background thread that monitors the playback queue and
        sends audio to the ALSA device. Optionally pre-opens the device
        if keep_device_open is enabled.
        """
        if self.playback_running.is_set():
            return

        logger.info(f"Starting playback: {self.playback_device} ({self.playback_config.sample_rate}Hz)")

        # Pre-open device if keep_device_open is enabled
        # This avoids device open/close overhead for each playback
        if self.playback_config.keep_device_open:
            self.playback_pcm = self._open_playback_device()
            self.playback_sample_rate = self.playback_config.sample_rate

        # Start playback thread
        self.playback_running.set()
        self.playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self.playback_thread.start()

    def stop_playback(self):
        """
        Stop ALSA playback.

        Signals the playback thread to stop, waits for it to finish,
        and cleans up the ALSA device handle.
        """
        if not self.playback_running.is_set():
            return

        logger.info("Stopping playback")

        # Signal thread to stop
        self.playback_running.clear()

        # Wait for thread to finish (with timeout)
        if self.playback_thread:
            self.playback_thread.join(timeout=2.0)

        # Close and clean up ALSA device
        if self.playback_pcm:
            try:
                self.playback_pcm.close()
            except Exception:
                pass  # Ignore cleanup errors
            self.playback_pcm = None

    def _playback_loop(self):
        """
        Playback thread loop.

        Continuously monitors the playback queue and sends audio to the device.
        Handles dynamic sample rate switching and device management based on
        the keep_device_open configuration.
        """
        # Local device handle used when keep_device_open is False
        local_pcm = None
        local_sample_rate = None

        try:
            while self.playback_running.is_set():
                audio_to_play = None
                sr = self.playback_config.sample_rate

                # Thread-safe access to playback queue
                with self.playback_queue_lock:
                    if self.playback_queue:
                        # Dequeue audio and its sample rate
                        audio_to_play, sr = self.playback_queue.popleft()
                    else:
                        # Wait for audio to be queued (with timeout for clean shutdown)
                        # This prevents busy-waiting and allows thread to check running flag
                        self.playback_queue_cv.wait(timeout=1.0)

                        # Check again after wake up (could be shutdown signal)
                        if self.playback_queue:
                            audio_to_play, sr = self.playback_queue.popleft()

                if audio_to_play is not None:
                    # Handle device management based on keep_device_open setting
                    if self.playback_config.keep_device_open:
                        # Reuse persistent device, reopen only if sample rate changes
                        if self.playback_sample_rate != sr:
                            if self.playback_pcm:
                                self.playback_pcm.close()
                            self.playback_pcm = self._open_playback_device(sr)
                            self.playback_sample_rate = sr
                        pcm_to_use = self.playback_pcm
                    else:
                        # Open/close device for each playback (thread-local handle)
                        # Reopen only if sample rate changes
                        if local_sample_rate != sr:
                            if local_pcm:
                                local_pcm.close()
                            local_pcm = self._open_playback_device(sr)
                            local_sample_rate = sr
                        pcm_to_use = local_pcm

                    # Send audio to the device
                    self._push_audio_to_device(audio_to_play, sr, pcm_to_use)

        except Exception as e:
            logger.error(f"Playback loop error: {e}", exc_info=True)
        finally:
            # Clean up local device handle if used
            if local_pcm:
                local_pcm.close()

    def _push_audio_to_device(self, audio_data: np.ndarray, sample_rate: int, pcm_out: aa.PCM):
        """
        Push audio to ALSA device.

        Converts audio data to the appropriate format and writes it to the
        ALSA device. Handles channel adjustment and format conversion.

        Args:
            audio_data: Audio samples to play (float32, normalized to [-1, 1])
            sample_rate: Sample rate of the audio
            pcm_out: ALSA PCM device to write to
        """
        try:
            # Signal that playback is active (for synchronization)
            self.playback_active.set()

            # Apply fade-out, channel adjustment, and format conversion in base class
            playback_data = self._prepare_playback_audio(audio_data)

            # Write audio data to ALSA device
            if pcm_out.write(playback_data.tobytes()) < 0:
                pcm_out.write(playback_data.tobytes())  # force re-write when something went wrong

        except Exception as e:
            logger.error(f"Playback push error: {e}", exc_info=True)
        finally:
            # Signal that playback is complete
            self.playback_active.clear()

            # Notify any threads waiting for playback completion
            with self.playback_queue_cv:
                self.playback_queue_cv.notify_all()
