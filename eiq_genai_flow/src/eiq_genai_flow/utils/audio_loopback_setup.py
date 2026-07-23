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
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class AudioLoopbackSetup:
    """Setup virtual audio loopback for benchmark testing."""

    ASOUNDRC_CONTENT = """# ============================================
# Virtual Audio Devices for eIQ GenAI Flow Testing
# ============================================
# Loopback pairs:
#   Play to device 1 → Capture from device 0
# ============================================

# Capture endpoint (device 0) - for eiq_genai_flow
pcm.fake_capture {
    type plug
    slave {
        pcm "hw:Loopback,0,0"
        format S32_LE
        rate 16000
        channels 2
    }
    hint {
        show on
        description "Virtual Loopback Capture (Device 0)"
    }
}

# Injection endpoint (device 1) - for aplay
pcm.fake_input {
    type plug
    slave {
        pcm "hw:Loopback,1,0"
        format S32_LE
        rate 16000
        channels 2
    }
    hint {
        show on
        description "Virtual Loopback Injection (Device 1)"
    }
}

"""

    def __init__(self, config_file=None):
        """
        Initialize audio loopback setup.

        Args:
            config_file: Path to .asoundrc file. Defaults to ~/.asoundrc
        """
        if config_file is None:
            self.config_file = Path.home() / ".asoundrc"
        else:
            self.config_file = Path(config_file)

    def _load_kernel_module(self):
        """Load snd-aloop kernel module."""
        # Check if already loaded
        try:
            result = subprocess.run(["lsmod"], capture_output=True, text=True, check=True)
            if "snd_aloop" in result.stdout:
                logger.debug("snd-aloop module already loaded")
                return
        except subprocess.CalledProcessError:
            pass

        logger.debug("Loading snd-aloop kernel module...")
        try:
            subprocess.run(["sudo", "modprobe", "snd-aloop"], check=True, capture_output=True, text=True)
            logger.debug("snd-aloop module loaded successfully")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to load snd-aloop module: {e.stderr}\n"
                f"Please run manually: sudo modprobe snd-aloop"
            )
        except FileNotFoundError:
            raise RuntimeError("modprobe command not found.")

    def _create_alsa_config(self):
        """Create .asoundrc ALSA configuration file."""
        logger.debug(f"Creating ALSA configuration at: {self.config_file}")

        with open(self.config_file, "w") as f:
            f.write(self.ASOUNDRC_CONTENT)

        logger.debug("Configuration created successfully!")

    def _verify_setup(self):
        """Verify the loopback setup is working."""
        logger.debug("\nVerifying setup:")
        out = subprocess.run("aplay -L | grep -A1 'fake_'", shell=True, capture_output=True, text=True, check=False)
        result = out.stdout.strip()
        if result:
            logger.debug(f"\nAvailable virtual devices:\n{result}")
        else:
            raise RuntimeError("No fake_ devices found in aplay -L output")

    def setup(self):
        """
        Perform complete setup of audio loopback.

        Returns:
            bool: True if setup successful, False otherwise
        """
        logger.debug("Setting up virtual audio loopback for benchmark testing...")

        self._load_kernel_module()
        self._create_alsa_config()
        self._verify_setup()

        logger.debug("Setup complete! You can now use:\n"
                     "  Capture device: plughw:CARD=Loopback\n"
                     "  Injection: aplay -D fake_input <wav_file>")

    def clean(self):
        """Remove loopback configuration (optional cleanup method)."""
        logger.debug("Cleaning up audio loopback configuration...")

        if self.config_file.exists():
            logger.debug(f"Removing configuration file: {self.config_file}")
            self.config_file.unlink()

        logger.debug("Cleanup complete")
