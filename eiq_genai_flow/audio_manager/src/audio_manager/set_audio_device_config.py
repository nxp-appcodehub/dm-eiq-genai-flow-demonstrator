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
import subprocess


logger = logging.getLogger(__name__)


def run_cmd(cmd: str):
    logger.info(f"Running {cmd}")
    result = subprocess.run(cmd.split(), capture_output=True, text=True)
    if result.stdout:
        logger.debug(result.stdout)
    if result.stderr:
        logger.error(result.stderr)


def set_capture_device_config(capture_device):
    logger.debug(f"Setting capture config: {capture_device}")
    if "wm8962audio" in capture_device:
        run_cmd("amixer -c wm8962audio sset 'Capture' 60")
    elif "wm8960audio" in capture_device:
        run_cmd("amixer -c micfilaudio cset name='MICFIL Quality Select' 'High'")
        run_cmd("amixer -c wm8960audio sset 'Capture' 60")
    else:
        logger.debug(
            f"No specific capture config applied for '{capture_device}' device, "
            f"you can customize it in {os.path.basename(__file__)}"
        )


def set_playback_device_config(playback_device):
    logger.debug(f"Setting playback config: {playback_device}")
    if "wm8962audio" in playback_device:
        run_cmd("amixer -c wm8962audio set 'Headphone' 110 on")
    elif "wm8960audio" in playback_device:
        run_cmd("amixer -c wm8960audio set 'Headphone' 125")
    else:
        logger.debug(
            f"No specific playback config applied for '{playback_device}' device, "
            f"you can customize it in {os.path.basename(__file__)} "
        )
