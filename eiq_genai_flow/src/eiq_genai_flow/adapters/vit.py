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
from vit.vit import VIT
from audio_manager.audio_manager_base import AudioRecorder, ReaderConfig
from eiq_genai_flow.adapters.base import BaseAdapter
from eiq_genai_flow.adapters.event_manager import EventManager, EventType

logger = logging.getLogger(__name__)


class VITConfig:
    """Configuration class for VIT adapter."""

    def __init__(
        self,
        wake_word_model,
        operating_mode="wakeword",
        noise_floor=-80.0,
        noise_threshold=10.0,
        save_audio_vit=False,
        channel_indices=None,
    ):
        """
        Initialize VIT configuration.

        Args:
            wake_word_model: Path to VIT model file
            operating_mode: Operating mode ("wakeword" or "wakeword_command")
            noise_floor: Noise floor threshold in dB (default: -80.0)
            noise_threshold: Noise threshold in dB (default: 10.0)
            save_audio_vit: Enable recording of VIT audio (default: False)
            channel_indices: List of channel indices to use (default: [0])
        """
        self.wake_word_model = wake_word_model
        self.operating_mode = operating_mode
        self.noise_floor = noise_floor
        self.noise_threshold = noise_threshold
        self.save_audio_vit = save_audio_vit
        self.channel_indices = channel_indices if channel_indices is not None else [0]


class VITAdapter(BaseAdapter):
    """
    Adapter class to integrate VIT wake word detection module with GenAI Flow.
    Handles audio streaming, wake word detection, and result management.
    """

    def __init__(self, config, audio_manager, event_manager: EventManager):
        """
        Initialize VIT adapter.

        Args:
            config: VITConfig instance with VIT-specific configuration
            audio_manager: AudioManager instance (ALSA or GStreamer implementation)
            verbose: Enable verbose logging
        """
        super().__init__(audio_manager, event_manager)
        self.config = config
        self._continuous_active = False
        self.event_manager.subscribe(EventType.CONTINUOUS_WAKE, self._on_continuous_wake)
        self.event_manager.subscribe(EventType.TIMEOUT, self._on_timeout)

        # Override thread name
        logger.debug(f"Initializing VIT adapter with model: {self.config.wake_word_model}")

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

        logger.debug(f"VIT audio parameters: {self.sample_rate}Hz, {self.frame_size} samples/frame")

        # Register with VIT's required format: mono S16LE
        vit_config = ReaderConfig(channels=1, format="S16LE", channel_indices=self.config.channel_indices)
        self.audio_reader = self.audio_manager.register_reader(name="VIT", config=vit_config)

        logger.debug(
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

                    if self._stop_event.is_set():  # allow early stop if vit_wake has been called
                        logger.debug("early stop _worker_loop")
                        break

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

                        info["speech_end"] = ww_end_abs_idx
                        info["speech_start"] = ww_start_abs_idx
                        info["write_index_at_detection"] = current_write_idx

                        logger.info(f"Wake word detected: {info['name']}")
                        print(f"Wake word detected: {info['name']}")
                        logger.debug(
                            f"Wake word ID: {info['id']}, "
                            f"Energy: {info['energy']:.1f}dB, "
                            f"Start offset: {info['start_offset']}, "
                            f"End offset: {info['end_offset']}, "
                            f"Wake word absolute indices: "
                            f"start={ww_start_abs_idx}, end={ww_end_abs_idx}"
                        )

                        self.publish(EventType.VIT_WAKE, data=info)

                        # Auto-disable after detection
                        self._stop_event.set()

        except Exception as e:
            logger.error(f"Error in VIT worker loop: {e}", exc_info=True)

        finally:
            # Clear running flag
            self._enabled.clear()
            # Stop recording when detection ends
            if self.vit_recorder and self.vit_recorder.is_recording:
                self.vit_recorder.stop()
                logger.info("VIT audio recording stopped")

    def _on_continuous_wake(self, event):
        self._continuous_active = True

    def _on_timeout(self, event):
        self._continuous_active = False

    def enable(self, sync_to_current=True):
        """
        Enable wake word detection (start listening).

        Args:
            sync_to_current
        """
        if self._continuous_active:
            return
        # Reset VIT internal state before enabling
        self.vit.reset()
        super().enable(sync_to_current=sync_to_current)

    def disable(self, timeout: float = 5.0):
        super().disable(timeout=timeout)

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
