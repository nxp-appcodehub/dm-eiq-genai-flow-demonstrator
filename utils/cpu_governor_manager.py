# -*- coding: utf-8 -*-

# Copyright 2025 NXP
# NXP Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.

import subprocess
import os
import atexit
import logging

logger = logging.getLogger(__name__)


class CpuGovernorManager:
    def __init__(self):
        self.original_governor = None
        self.num_cores = os.cpu_count()
        self._available_governors = None

        # Auto-restore on program exit
        atexit.register(self.restore)

    def get_available_governors(self):
        """Get list of available governors from the system"""
        if self._available_governors is not None:
            return self._available_governors

        try:
            with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors", "r") as f:
                governors = f.read().strip().split()
                self._available_governors = governors
                logger.debug(f"Available governors: {governors}")
                return governors
        except FileNotFoundError:
            logger.warning("scaling_available_governors file not found, using fallback list")
            # Fallback to common governors if file doesn't exist
            fallback_governors = ["performance", "powersave", "userspace", "ondemand", "conservative", "schedutil"]
            self._available_governors = fallback_governors
            return fallback_governors
        except Exception as e:
            logger.error(f"Error reading available governors: {e}")
            return []

    def validate_governor(self, governor):
        """Validate that the governor is available on this system"""
        available = self.get_available_governors()
        if not available:
            logger.error("No available governors found")
            return False

        if governor not in available:
            logger.error(f"Invalid governor: '{governor}'. Available governors: {available}")
            return False

        return True

    def save_and_set(self, new_governor="performance"):
        """Save current governor and set new one"""
        try:
            # Save current governor (just check CPU 0)
            with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor", "r") as f:
                self.original_governor = f.read().strip()

            logger.info(f"Saved original governor: {self.original_governor}")

            # Validate governor input using system's available governors
            if not self.validate_governor(new_governor):
                return False

            # Use cpufreq-set to set governor for all cores
            result = subprocess.run(["sudo", "cpufreq-set", "-g", new_governor], capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                logger.info(f"Set CPU governor to: {new_governor}")
                return True
            else:
                logger.error(f"Failed to set governor: {result.stderr}")
                return False

        except FileNotFoundError:
            logger.error("cpufreq-set command not found. Please install cpufrequtils package.")
            return False
        except subprocess.TimeoutExpired:
            logger.error("Timeout while setting governor")
            return False
        except Exception as e:
            logger.error(f"Error managing governor: {e}")
            return False

    def restore(self):
        """Restore original governor"""
        if self.original_governor:
            try:
                # Validate the original governor using system's available governors
                if not self.validate_governor(self.original_governor):
                    logger.warning(f"Original governor '{self.original_governor}' not available, skipping restore")
                    return

                result = subprocess.run(["sudo", "cpufreq-set", "-g", self.original_governor], capture_output=True, text=True, timeout=10)

                if result.returncode == 0:
                    logger.info(f"Restored governor to: {self.original_governor}")
                    self.original_governor = None
                else:
                    logger.error(f"Failed to restore governor: {result.stderr}")

            except FileNotFoundError:
                logger.error("cpufreq-set command not found during restore")
            except subprocess.TimeoutExpired:
                logger.error("Timeout while restoring governor")
            except Exception as e:
                logger.error(f"Error restoring governor: {e}")

    def get_current_governor(self):
        """Get the current CPU governor"""
        try:
            with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor", "r") as f:
                return f.read().strip()
        except Exception as e:
            logger.error(f"Error reading current governor: {e}")
            return None

    def list_available_governors(self):
        """Public method to list available governors"""
        return self.get_available_governors()


# Global instance
_cpu_governor_manager = None


def setup_cpu_governor(governor="performance"):
    """Simple function to set up governor"""
    global _cpu_governor_manager
    _cpu_governor_manager = CpuGovernorManager()
    return _cpu_governor_manager.save_and_set(governor)


def restore_cpu_governor():
    """Simple function to restore governor"""
    global _cpu_governor_manager
    if _cpu_governor_manager:
        _cpu_governor_manager.restore()


def get_available_governors():
    """Simple function to get available governors"""
    manager = CpuGovernorManager()
    return manager.get_available_governors()


def get_current_governor():
    """Simple function to get current governor"""
    manager = CpuGovernorManager()
    return manager.get_current_governor()
