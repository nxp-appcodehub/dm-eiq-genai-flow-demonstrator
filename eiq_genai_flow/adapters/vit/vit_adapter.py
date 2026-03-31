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
from dataclasses import dataclass, field
from typing import List
from vit.src.vit.vit import VIT
from audio_manager.audio_manager_base import AudioRecorder, ReaderConfig
from adapters.base import BaseAdapter

logger = logging.getLogger(__name__)


@dataclass
class VITConfig:
    """Configuration for VIT adapter."""

    wake_word_model: str
    operating_mode: str = "wakeword"
    noise_floor: float = -80.0
    noise_threshold: float = 10.0
    save_audio_vit: bool = False
    channel_indices: List[int] = field(default_factory=lambda: [0])


class VITAdapter(BaseAdapter):
    """
    Adapter class to integrate VIT wake word detection module with GenAI Flow.
    Handles audio streaming, wake word detection, and result management.
    """

    def __init__(self, config: VITConfig, audio_manager, verbose=False):
        """
        Initialize VIT adapter.

        Args:
            config: VITConfig instance with VIT-specific configuration
            audio_manager: AudioManager instance (ALSA or GStreamer implementation)
            verbose: Enable verbose logging
        """
        super().__init__(config, audio_manager, verbose)

        # Legacy bypass tokens for compatibility with main flow
        self.bypass_vit_wwd = "WWD:bypass_vit"
        self.bypass_vit_asr_wwd = "WWD:bypass_vit_asr"

        # Override thread name
        self._thread_name = "VIT_Detection"

        logger.info(f"Initializing VIT adapter with model: {self.config.wake_word_model}")

        # Initialize the VIT module with new API
        self.vit = VIT(
            model_path=self.config.wake_word_model,
            operating_mode=self.config.operating_mode,
            noise_floor=self.config.noise_floor,
            noise_threshold=self.config.noise_threshold,
        )

        # Get VIT audio parameters from the initialized VIT instance
        self.sample_rate = VIT.SAMPLE_RATE
        self.frame_size = VIT.SAMPLES_PER_FRAME
        self.frame_duration_s = self.frame_size / self.sample_rate

        logger.info(f"VIT audio parameters: {self.sample_rate}Hz, {self.frame_size} samples/frame")

        # Register with VIT's required format: mono S16LE
        vit_config = ReaderConfig(channels=1, format="S16LE", channel_indices=self.config.channel_indices)
        self.audio_reader = self.audio_manager.register_reader(name="VIT", config=vit_config)

        logger.info(
            f"VIT registered: wants {vit_config.channels}ch {vit_config.format} from channels "
            f"{self.config.channel_indices}, will receive converted audio from master stream"
        )

        # Setup VIT audio recorder if enabled
        self.vit_recorder = None
        if self.config.save_audio_vit:
            self.vit_recorder = AudioRecorder(
                save_directory=self.audio_manager.capture_config.audio_save_path,
                sample_rate=self.sample_rate,
                channels=1,
                filename_prefix="vit_int16_",
                format="S16LE",
            )
            logger.info("VIT audio recorder initialized")

        logger.info("VIT adapter initialized successfully")

    def shutdown(self):
        """Shutdown VIT and cleanup resources."""

        # Use base class disable
        self.disable()

        # Stop recorder if still running
        if self.vit_recorder and self.vit_recorder.is_recording:
            self.vit_recorder.stop()

        # Cleanup VIT instance
        if self.vit:
            try:
                self.vit.delete_instance()
            except Exception as e:
                logger.error(f"Error during VIT shutdown: {e}")

        # Unregister from audio manager
        if self.audio_reader:
            self.audio_reader.unregister()

        logger.info("VIT adapter shutdown complete")

    def _worker_loop(self):
        """VIT worker loop with event-driven audio reading."""

        # Start recording if enabled
        if self.vit_recorder:
            self.vit_recorder.start()
            logger.info("VIT audio recording started")

        try:
            while not self._stop_event.is_set():
                # Event-driven blocking read with timeout:
                # - Wakes IMMEDIATELY when data arrives (via condition variable)
                # - Falls back to timeout if no data arrives
                # - Returns None on timeout (loop continues and stop event is checked)
                samples = self.audio_reader.read(
                    self.frame_size,
                    blocking=True,
                    timeout=self.frame_duration_s,
                )

                if samples is not None and len(samples) == self.frame_size:
                    # Samples are already int16 mono - use directly
                    frame_data = samples

                    # Record the audio (already int16 format as sent to VIT)
                    if self.vit_recorder and self.vit_recorder.is_recording:
                        self.vit_recorder.write(frame_data)

                    # Process with VIT (expects int16 mono)
                    detection_type, info = self.vit(frame_data, frame_data)

                    if detection_type == "wakeword":
                        # Calculate ABSOLUTE indices in master buffer
                        with self.audio_manager.master_buffer_lock:
                            current_write_idx = self.audio_manager.write_index

                        # VIT's offsets are relative to current detection frame
                        # Convert to absolute indices
                        ww_end_abs_idx = current_write_idx - info["end_offset"]
                        ww_start_abs_idx = current_write_idx - info["start_offset"]

                        info["ww_end_abs_index"] = ww_end_abs_idx
                        info["ww_start_abs_index"] = ww_start_abs_idx
                        info["write_index_at_detection"] = current_write_idx

                        logger.info(
                            f"Wake word detected: {info['name']} (ID: {info['id']}) "
                            f"Energy: {info['energy']:.1f}dB, "
                            f"Start offset: {info['start_offset']}, "
                            f"End offset: {info['end_offset']}, "
                            f"Wake word absolute indices: "
                            f"start={ww_start_abs_idx}, end={ww_end_abs_idx}"
                        )

                        # Format result in legacy format
                        energy_str = f"{info['energy']:.1f}"
                        ww_string = f"WWD:{info['name']}_ID{info['id']}_E{energy_str}dB"

                        # Store result using base class method
                        self._store_result(ww_string, info)

                        # Auto-disable after detection
                        self._stop_event.set()
                        break

                # Check stop event between reads
                if self._stop_event.is_set():
                    break

        except Exception as e:
            logger.error(f"Error in VIT worker loop: {e}", exc_info=True)

        finally:
            # Stop recording when detection ends
            if self.vit_recorder and self.vit_recorder.is_recording:
                self.vit_recorder.stop()
                logger.info("VIT audio recording stopped")

    def process(self, input_data=None):
        """
        Process wake word detection (called by worker loop).

        Args:
            input_data: Audio data (handled internally via audio_reader)

        Returns:
            Detection result string or None
        """
        # VIT processing happens in _worker_loop
        # This method is here for base class compatibility
        pass

    def enable(self, clear_buffer=True):
        """
        Enable wake word detection (start listening).

        Args:
            clear_buffer: Ignored (kept for backward compatibility)
        """
        # Reset VIT internal state before enabling
        if self.vit:
            self.vit.reset()

        # Use base class enable with sync_to_current=True (real-time)
        super().enable(sync_to_current=True)

    def wait_for_wake_word(self, stop_requested=False):
        """
        Wait for wake word detection (blocking).

        This is a convenience wrapper around wait_for_result() with a more
        specific name for wake word detection.

        Args:
            stop_requested: External signal to stop waiting

        Returns:
            Wake word detection string (e.g., "WWD:HeyNXP_ID1_E45.2dB")
            or empty string if stopped/failed
        """
        return self.wait_for_result(stop_requested=stop_requested)

    def bypass(self, bypass_asr=False):
        """
        Bypass VIT and simulate wake word detection.

        Args:
            bypass_asr: If True, also bypass ASR in the result token
        """
        logger.info(f"VIT bypass triggered (bypass_asr={bypass_asr})")

        if bypass_asr:
            result = self.bypass_vit_asr_wwd
        else:
            result = self.bypass_vit_wwd

        # Create dummy detection info for bypass mode
        info = {
            "id": 0,
            "name": "BYPASS",
            "energy": 0.0,
            "start_offset": 0,
            "end_offset": 0,
            "ww_start_abs_index": 0,
            "ww_end_abs_index": 0,
            "write_index_at_detection": 0,
        }

        # Store result using base class method
        self._store_result(result, info)

        if self.is_running:
            self._stop_event.set()

    def get_status(self):
        """
        Get VIT status information.

        Returns:
            dict: Status information from VIT module and adapter
        """
        # Get base adapter status
        status = super().get_status()

        if self.vit:
            # Get VIT status as dictionary
            vit_status = self.vit.get_status()

            # Merge VIT-specific status
            status.update(
                {
                    "vit": vit_status,
                    "sample_rate": self.sample_rate,
                    "frame_size": self.frame_size,
                    "frame_duration_s": self.frame_duration_s,
                    "recorder_active": (self.vit_recorder.is_recording if self.vit_recorder else False),
                    "operating_mode": self.config.operating_mode,
                    "noise_floor": self.config.noise_floor,
                    "noise_threshold": self.config.noise_threshold,
                }
            )

        return status
