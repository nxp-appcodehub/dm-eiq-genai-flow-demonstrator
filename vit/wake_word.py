# Copyright 2025 NXP
# NXP Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.

import platform
import subprocess
import sys
import time
import os
import psutil
from utils.utils import create_message_queue, send_message
import logging

from vit.scripts.convert_model import validate_binary_model

logger = logging.getLogger(__name__)


class VIT:
    def __init__(self, capture_device, wake_word_model, py_to_c_queue, c_to_py_queue, verbose):
        self.sig_vit_on = 40
        self.sig_vit_off = 41
        self.sig_c_kill = 44
        self.verbose = verbose
        self.bypass_vit_wwd = "WWD:bypass_vit"
        self.bypass_vit_asr_wwd = "WWD:bypass_vit_asr"
        self.is_running = False

        directory_path = os.path.dirname(__file__)

        if platform.machine() == "x86_64":
            app_path = os.path.join(directory_path, "x86_64/vit")
        else:
            app_path = os.path.join(directory_path, "aarch64/vit")

        self.mq_to_c = create_message_queue(py_to_c_queue, self.verbose)

        # mq from VIT
        self.mq_from_c = create_message_queue(c_to_py_queue, self.verbose)

        self.kill_process()

        if not validate_binary_model(wake_word_model):
            logger.error(f"Invalid VIT wakeword Model: + {wake_word_model}", exc_info=True)
            sys.exit(1)

        cmd_args = [app_path, "-d", str(capture_device), "-m", str(wake_word_model), "-v", str(1 if self.verbose else 0)]
        logger.debug("Command executed: " + " ".join(cmd_args))
        subprocess.Popen(
            cmd_args,
            stdout=subprocess.PIPE,
        )

    def enable(self):
        send_message(self.mq_to_c, self.sig_vit_on, self.verbose)
        self.is_running = True

    def disable(self):
        send_message(self.mq_to_c, self.sig_vit_off, self.verbose)
        self.is_running = False

    def shutdown(self):
        send_message(self.mq_to_c, self.sig_c_kill)
        self.is_running = False

    def get_info(self, stop_threads):
        ww = ""
        while not stop_threads and ww == "":
            try:
                ww, _ = self.mq_from_c.receive()
                ww = ww.decode("utf-8").strip()
            except BaseException:
                pass
        return ww

    def set_wakeword(self, wakeword):
        self.mq_from_c.send(wakeword)

    def bypass(self, bypass_asr=False):
        if bypass_asr:
            self.mq_from_c.send(self.bypass_vit_asr_wwd)
            time.sleep(0.5)
        else:
            self.mq_from_c.send(self.bypass_vit_wwd)

    @staticmethod
    def kill_process():
        for proc in psutil.process_iter(["name"]):
            if proc.info["name"] == "vit":
                proc.kill()
