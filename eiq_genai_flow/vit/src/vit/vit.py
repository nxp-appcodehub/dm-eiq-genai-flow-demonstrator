# Copyright 2025-2026 NXP
# NXP Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.


import ctypes
import logging
from pathlib import Path

from vit import vit_ops

# Magic numbers that should be at the start of valid VIT models
WATERMARK_1 = 0xABFE34A2
WATERMARK_1_ZWWD = 0xABFE34A3

# Expected model version
EXPECTED_MAJOR_VERSION = 4
EXPECTED_MEDIUM_VERSION = 13

logger = logging.getLogger(__name__)


class VIT:
    # Class-level constants
    SAMPLES_PER_FRAME = vit_ops.VIT_SAMPLES_PER_30MS_FRAME
    SAMPLE_RATE = vit_ops.VIT_SAMPLE_RATE
    NUMBER_OF_CHANNELS = vit_ops.VIT_NUMBER_OF_CHANNEL
    AUDIO_FRAME_SIZE = SAMPLES_PER_FRAME * ctypes.sizeof(ctypes.c_int16)

    def __init__(
        self,
        model_path=None,
        operating_mode="wakeword",
        noise_floor=-80.0,
        noise_threshold=10.0,
    ):
        self.vit = None
        self.model_loaded = False

        # Set operating mode based on parameter
        if operating_mode.lower() == "wakeword_command":
            op_mode = (
                vit_ops.VIT_OperatingMode.VIT_WAKEWORD_ENABLE
                | vit_ops.VIT_OperatingMode.VIT_VOICECMD_ENABLE
            )
        elif operating_mode.lower() == "wakeword":
            op_mode = vit_ops.VIT_OperatingMode.VIT_WAKEWORD_ENABLE
        else:
            raise ValueError(
                f"Invalid operating_mode: {operating_mode}. Must be 'wakeword' or 'wakeword_command'"
            )

        # To force operating mode to be defined as enum type
        self.operating_mode = vit_ops.VIT_OperatingMode(op_mode)

        if not self._print_lib_info():
            raise RuntimeError("Lib info failed")

        if not self._load_model(model_path):
            raise RuntimeError(f"Model loading failed (path={model_path})")

        if not self._print_model_info():
            raise RuntimeError("Model info failed")

        if not self._create_instance():
            raise RuntimeError("Instance creation failed")

        if not self._configure(
            operating_mode=self.operating_mode,
            noise_floor=noise_floor,
            noise_threshold=noise_threshold,
            time_span=3.0,
        ):
            raise RuntimeError("VIT configuration failed")

    def __del__(self):
        """Cleanup when application is destroyed"""
        if self.vit is not None:
            try:
                self.vit.delete_instance()
            except Exception:
                pass

    def _print_lib_info(self):
        """logger.info VIT library information"""
        logger.debug("\n")
        logger.debug("=" * 60)
        logger.debug("VIT Library Information")
        logger.debug("=" * 60)

        # Use module-level function
        status, lib_info = vit_ops.get_lib_info()

        if status != vit_ops.VIT_ReturnStatus.VIT_SUCCESS:
            logger.error(f"Error getting library info: {status}")
            return False

        logger.debug(f"VIT Library Release: 0x{lib_info.VIT_LIB_Release:08X}")

        # Decode features
        features = []
        if lib_info.VIT_Features_Supported & vit_ops.VIT_OperatingMode.VIT_LPVAD_ENABLE:
            features.append("LPVAD")
        if (
            lib_info.VIT_Features_Supported
            & vit_ops.VIT_OperatingMode.VIT_WAKEWORD_ENABLE
        ):
            features.append("WakeWord")
        if (
            lib_info.VIT_Features_Supported
            & vit_ops.VIT_OperatingMode.VIT_VOICECMD_ENABLE
        ):
            features.append("VoiceCommand")
        if (
            lib_info.VIT_Features_Supported
            & vit_ops.VIT_OperatingMode.VIT_SPEECHTOINTENT_ENABLE
        ):
            features.append("SpeechToIntent")

        logger.debug(f"Supported Features: {', '.join(features)}")
        logger.debug(f"Channels Supported: {lib_info.NumberOfChannels_Supported}")
        logger.debug("=" * 60 + "\n")

        return True

    def _load_model(
        self, model_path, location=vit_ops.VIT_Model_Location.VIT_MODEL_IN_FAST_MEM
    ):
        """Load VIT model from file"""

        if model_path is None:
            logger.error("Model path cannot be None")
            return False

        if Path(model_path).suffix.lower() != ".bin":
            logger.error(f"Model file must have .bin extension, got: {model_path}")
            print(f"Model file must have .bin extension, got: {model_path}")
            self.print_model_creation_guidance()
            return False

        logger.debug(f"Loading VIT model from: {model_path}")

        try:
            with open(model_path, "rb") as f:
                model_data = f.read()
        except FileNotFoundError:
            logger.error(f"Error: Model file not found: {model_path}")
            return False
        except IOError as e:
            logger.error(f"Error reading model file: {e}")
            return False

        logger.debug(f"Loaded binary model: {len(model_data)} bytes ({len(model_data) / 1024:.2f} KB)")

        # Use the existing validation function
        is_valid = self._validate_model_data(model_data)

        if is_valid:
            logger.debug("✓ Binary model validation PASSED")
        else:
            logger.error("✗ Binary model validation FAILED")
            return False

        # Create VIT instance if not already created
        if self.vit is None:
            self.vit = vit_ops.VIT()

        status = self.vit.set_model(model_data, location)

        if status != vit_ops.VIT_ReturnStatus.VIT_SUCCESS:
            logger.error(f"Error in the model validation: {status}")
            return False

        logger.debug("Model loaded successfully!")
        self.model_loaded = True

        return True

    def _print_model_info(self):
        """logger.info VIT model information"""
        if not self.model_loaded:
            logger.warning("No model loaded!")
            return False

        logger.debug("\n")
        logger.debug("=" * 60)
        logger.debug("VIT Model Information")
        logger.debug("=" * 60)

        status, model_info = self.vit.get_model_info()

        if status != vit_ops.VIT_ReturnStatus.VIT_SUCCESS:
            logger.error(f"Error getting model info: {status}")
            return False

        logger.debug(f"Model Release: 0x{model_info.VIT_Model_Release:08X}")
        logger.debug(f"Language: {model_info.Language}")
        logger.debug(f"Number of Wake Words: {model_info.NbOfWakeWords}")
        logger.debug(f"Zero Wake Word Delay: {model_info.ZeroWakeWordDelay}")
        logger.debug(f"Wake Word Advance Model: {model_info.WakeWordAdvanceModel}")
        logger.debug(f"WW/VoiceCmds Strings Included: {model_info.WW_VoiceCmds_Strings}")

        ptr = model_info.pWakeWord_List
        n_ww = model_info.NbOfWakeWords  # If this field exists
        if ptr:
            offset = 0
            for i in range(n_ww):
                # Read null-terminated string at current offset
                s = ctypes.string_at(ptr + offset)
                ww = s.decode("utf-8")
                logger.info(f"Wakeword supported #{i + 1}: '{ww}'")
                print(f"Wakeword supported #{i + 1}: '{ww}'")
                # Increment by string length + 1 for null terminator
                offset += len(s) + 1

        if self.operating_mode & vit_ops.VIT_OperatingMode.VIT_VOICECMD_ENABLE:
            logger.debug(f"Number of Voice commands: {model_info.NbOfVoiceCmds}")
            ptr = model_info.pVoiceCmds_List  # Pointer address
            n_commands = model_info.NbOfVoiceCmds  # If this field exists
            if ptr:
                offset = 0
                for i in range(n_commands):
                    # Read null-terminated string at current offset
                    s = ctypes.string_at(ptr + offset)
                    cmd = s.decode("utf-8")
                    logger.info(f"Voice Command {i + 1}: '{cmd}'")
                    # Increment by string length + 1 for null terminator
                    offset += len(s) + 1

        logger.debug("=" * 60 + "\n")

        return True

    def _create_instance(self, device_id=vit_ops.VIT_DeviceId.VIT_IMX_A5X):
        """Create VIT instance"""
        if not self.model_loaded:
            logger.warning("Error: Model must be loaded before creating instance!")
            return False

        logger.debug("\nCreating VIT instance...")

        # Setup instance parameters
        params = vit_ops.VIT_InstanceParams()
        params.SampleRate_Hz = vit_ops.VIT_SAMPLE_RATE
        params.NumberOfChannel = vit_ops.VIT_NUMBER_OF_CHANNEL
        params.SamplesPerFrame = vit_ops.VIT_SAMPLES_PER_30MS_FRAME
        params.DeviceId = device_id
        params.APIVersion = vit_ops.VIT_API_VERSION

        logger.debug(f"Sample Rate: {params.SampleRate_Hz} Hz")
        logger.debug(f"Channels: {params.NumberOfChannel.value}")
        logger.debug(f"Samples per Frame: {params.SamplesPerFrame}")
        logger.debug(f"Device ID: {params.DeviceId}")

        status = self.vit.create_instance(params)

        if status != vit_ops.VIT_ReturnStatus.VIT_SUCCESS:
            logger.error(f"Error creating instance: {status}")
            return False

        logger.debug("VIT instance created successfully!")
        return True

    def delete_instance(self):
        """Delete VIT instance and free all resources"""
        if self.vit is None:
            logger.warning("No VIT instance to delete")
            return True

        if not self.vit.is_instance_created():
            logger.warning("VIT instance not created, nothing to delete")
            return True

        logger.debug("Deleting VIT instance and freeing resources...")
        status = self.vit.delete_instance()

        if status != vit_ops.VIT_ReturnStatus.VIT_SUCCESS:
            logger.error(f"Error deleting instance: {status}")
            return False

        logger.debug("VIT instance deleted successfully!")
        return True

    def _configure(
        self, operating_mode, noise_floor=-90.0, noise_threshold=0.0, time_span=3.0
    ):
        """Configure VIT control parameters"""
        logger.debug("\nConfiguring VIT parameters...")

        # Setup control parameters
        ctrl_params = vit_ops.VIT_ControlParams()
        ctrl_params.OperatingMode = operating_mode
        ctrl_params.Input_Noise_Floor = noise_floor
        ctrl_params.Noise_Floor_Threshold = noise_threshold
        ctrl_params.Command_Time_Span = time_span
        ctrl_params.WakeWordDelayRecovering = False
        ctrl_params.Feature_LowRes = False

        logger.debug(f"Operating Mode: 0x{operating_mode.value:02X}")
        logger.debug(f"Input Noise Floor: {noise_floor} dB")
        logger.debug(f"Noise Floor Threshold: {noise_threshold} dB")
        logger.debug(f"Command Time Span: {time_span} seconds")

        status = self.vit.set_control_params(ctrl_params)

        if status != vit_ops.VIT_ReturnStatus.VIT_SUCCESS:
            logger.error(f"Error setting control parameters: {status}")
            return False

        logger.debug("VIT configured successfully!")
        return True

    def reset(self):
        """Reset VIT instance"""
        logger.debug("Resetting VIT instance...")
        status = self.vit.reset()

        if status != vit_ops.VIT_ReturnStatus.VIT_SUCCESS:
            logger.error(f"Error resetting instance: {status}")
            return False

        logger.debug("VIT instance reset successfully!")
        return True

    def get_status(self):
        """Get and logger.info VIT status"""
        logger.debug("\n")
        logger.debug("=" * 60)
        logger.debug("VIT Status")
        logger.debug("=" * 60)

        status, status_params = self.vit.get_status()

        if status != vit_ops.VIT_ReturnStatus.VIT_SUCCESS:
            logger.error(f"Error getting status: {status}")
            return False

        logger.debug(f"Model Release: 0x{status_params.VIT_MODEL_Release:08X}")
        logger.debug(f"Library Release: 0x{status_params.VIT_LIB_Release:08X}")

        features = []
        if (
            status_params.VIT_Features_Supported
            & vit_ops.VIT_OperatingMode.VIT_LPVAD_ENABLE
        ):
            features.append("LPVAD")
        if (
            status_params.VIT_Features_Supported
            & vit_ops.VIT_OperatingMode.VIT_WAKEWORD_ENABLE
        ):
            features.append("WakeWord")
        if (
            status_params.VIT_Features_Supported
            & vit_ops.VIT_OperatingMode.VIT_VOICECMD_ENABLE
        ):
            features.append("VoiceCommand")
        if (
            status_params.VIT_Features_Supported
            & vit_ops.VIT_OperatingMode.VIT_SPEECHTOINTENT_ENABLE
        ):
            features.append("SpeechToIntent")
        logger.debug(f"Supported Features: {', '.join(features)}")

        logger.debug(f"Channels Supported: {status_params.NumberOfChannels_Supported}")
        logger.debug(f"Device Selected: {status_params.Device_Selected}")
        logger.debug(f"LPVAD Event Detected: {status_params.LPVAD_EventDetected}")
        logger.debug(f"Instance Created: {self.vit.is_instance_created()}")
        logger.debug("=" * 60 + "\n")

        return True

    def __call__(self, mic_ref_buffer, mic_buffer) -> tuple[str | None, dict | None]:
        """Process audio frame and detect wake word or command"""
        # Process frame
        status, detection = self.vit.process(
            mic_ref_buffer,  # mic reference buffer
            mic_buffer,
        )

        # Check for detections
        if detection == vit_ops.VIT_DetectionStatus.VIT_WW_DETECTED:
            return ("wakeword", self._wakeword_detected())
        elif detection == vit_ops.VIT_DetectionStatus.VIT_VC_DETECTED:
            return ("command", self._command_detected())

        return (None, None)

    def _wakeword_detected(self):
        status, ww = self.vit.get_wakeword()
        if status == vit_ops.VIT_ReturnStatus.VIT_SUCCESS:
            return {
                "id": ww.Id,
                "name": ww.Name,
                "energy": ww.dB_Energy,
                "start_offset": ww.StartOffset,
                "end_offset": ww.EndOffset,
            }
        return None

    def _command_detected(self):
        status, cmd = self.vit.get_voice_command()
        if status == vit_ops.VIT_ReturnStatus.VIT_SUCCESS:
            return {
                "id": cmd.Id,
                "name": getattr(cmd, "Name", ""),  # If Name exists
                "start_offset": getattr(cmd, "StartOffset", 0),
                "end_offset": getattr(cmd, "EndOffset", 0),
            }
        return None

    def print_model_creation_guidance(self):
        """Print guidance on how to create a valid VIT model"""
        logger.warning("")
        logger.warning("=" * 80)
        logger.warning("HOW TO CREATE A VALID VIT MODEL")
        logger.warning("=" * 80)
        logger.warning("")
        logger.warning(
            " The VIT library used in this demonstrator is a development version for the upcoming 4.13 release."
        )
        logger.warning(
            " Custom wake-word model generation is currently not available through the public online tool."
        )
        logger.warning(
            " Contact NXP Voice Team to request a custom wake-word model for your specific use case."
        )
        logger.warning("")
        logger.warning("=" * 80)

    def _validate_model_data(self, model_data):
        """Perform basic validation on the model data"""

        if len(model_data) < 1024:  # Assume model should be at least 1KB
            logger.error(f"Model seems very small ({len(model_data)} bytes)")
            self._print_model_creation_guidance()
            return False

        # Check for VIT model magic numbers at the beginning
        # Read first 4 bytes as little-endian uint32
        magic_number = int.from_bytes(model_data[:4], byteorder="little")

        if magic_number == WATERMARK_1:
            logger.debug(
                f"✓ Valid VIT model detected - Magic number: 0x{magic_number:08X} (WATERMARK_1)"
            )
        elif magic_number == WATERMARK_1_ZWWD:
            logger.debug(
                f"✓ Valid VIT model detected - Magic number: 0x{magic_number:08X} (WATERMARK_1_ZWWD)"
            )
        else:
            logger.error(f"✗ Invalid magic number: 0x{magic_number:08X}")
            logger.error(
                f"Expected: 0x{WATERMARK_1:08X} (WATERMARK_1) or 0x{WATERMARK_1_ZWWD:08X} (WATERMARK_1_ZWWD)"
            )
            logger.error("This model may not be a valid VIT binary model.")
            self._print_model_creation_guidance()
            return False

        # Extract and validate model version
        model_minor = model_data[4]
        model_medium = model_data[5]
        model_major = model_data[6]
        logger.debug(
            f"Raw version bytes - Major: {model_major}, Medium: {model_medium}, Minor: {model_minor}"
        )

        expected_major = EXPECTED_MAJOR_VERSION
        expected_medium = EXPECTED_MEDIUM_VERSION

        # Validate model version
        if model_major is None or model_medium is None:
            logger.error("Cannot validate version - version information not available")
            self.print_model_creation_guidance()
            return False

        logger.debug(f"Model version: {model_major}.{model_medium}")
        logger.debug(f"Expected version: {expected_major}.{expected_medium}")

        if model_major == expected_major and model_medium == expected_medium:
            logger.debug("✓ Model version validation PASSED")
        else:
            logger.error("✗ Model version validation FAILED")
            logger.error(
                "Expected version {expected_major}.{expected_medium}, but found {model_major}.{model_medium}"
            )
            self._print_model_creation_guidance()
            return False

        # Check for common patterns that might indicate a valid model
        # Most binary models start with some kind of header or magic bytes
        first_bytes = model_data[:16]
        logger.debug(f"First 16 bytes: {' '.join([f'0x{b:02x}' for b in first_bytes])}")

        last_bytes = model_data[-16:]
        logger.debug(f"Last 16 bytes: {' '.join([f'0x{b:02x}' for b in last_bytes])}")

        return True
