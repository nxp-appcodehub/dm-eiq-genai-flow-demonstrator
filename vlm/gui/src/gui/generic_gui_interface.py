# Copyright 2025-2026 NXP
# NXP Confidential and Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.

import threading
import posix_ipc
from gui.config import GuiConfig
import logging

logger = logging.getLogger(__name__)

def clear_message_queue(mq):
    try:
        # Set a timeout value for the reception call (in milliseconds)
        timeout_ms = 0
        # Receive and discard all messages in the queue
        while True:
            try:
                message, _ = mq.receive(timeout_ms)
            except posix_ipc.BusyError:
                # The queue is empty
                break
    except Exception:
        logger.error("Can not clear message queue:", exc_info=True)

class GenericGuiInterface(threading.Thread):
    def __init__(self, callback=None, user_config=GuiConfig):
        super().__init__()
        self.user_config = user_config()
        self.verbose = self.user_config.verbose
        self.callback = callback

        # Attempt to open or create the eGF to GUI queue
        try:
            if self.user_config.egf_to_gui_queue_path:
                self.mq_to_gui = posix_ipc.MessageQueue(
                    self.user_config.egf_to_gui_queue_path,
                    flags=posix_ipc.O_CREAT | posix_ipc.O_EXCL,  # O_EXCL ensures creation if it doesn't exist
                    max_messages=self.user_config.max_messages,
                    max_message_size=self.user_config.max_message_size,
                    mode=0o644,
                    write=True,
                    read=True
                )
                clear_message_queue(self.mq_to_gui)
                logger.info(f"{self.user_config.egf_to_gui_queue_path} Queue created successfully.")
            else:
                self.mq_to_gui = None
        except posix_ipc.ExistentialError:
            # If the queue already exists, open it with read and write access
            self.mq_to_gui = posix_ipc.MessageQueue(
                self.user_config.egf_to_gui_queue_path,
                write=True,
                read=True
            )
            logger.info(f"{self.user_config.egf_to_gui_queue_path} Queue opened successfully.")
        except Exception as e:
            raise RuntimeError(f"{self.user_config.egf_to_gui_queue_path} mqueue not created/opened: {e}")

        # Attempt to open or create the GUI to LLMP QUEUE
        try:
            if self.user_config.gui_to_egf_queue_path:
                self.mq_from_gui = posix_ipc.MessageQueue(
                    self.user_config.gui_to_egf_queue_path,
                    flags=posix_ipc.O_CREAT | posix_ipc.O_EXCL,  # O_EXCL ensures creation if it doesn't exist
                    max_messages=self.user_config.max_messages,
                    max_message_size=self.user_config.max_message_size,
                    mode=0o644,
                    write=True,
                    read=True
                )
                clear_message_queue(self.mq_from_gui)
                logger.info(f"{self.user_config.gui_to_egf_queue_path} Queue created successfully.")
            else:
                self.mq_from_gui = None
        except posix_ipc.ExistentialError:
            # If the queue already exists, open it with read and write access
            self.mq_from_gui = posix_ipc.MessageQueue(
                self.user_config.gui_to_egf_queue_path,
                write=True,
                read=True
            )
            logger.info(f"{self.user_config.gui_to_egf_queue_path} Queue opened successfully.")
        except Exception as e:
            raise RuntimeError(f"{self.user_config.gui_to_egf_queue_path} mqueue not created/opened: {e}")

        self.running = True

    def run(self):
        if self.mq_from_gui:
            while self.running:
                logger.info(self.user_config.gui_to_egf_queue_path + " read thread waiting for msg")
                try:
                    message, _ = self.mq_from_gui.receive(timeout=None)
                    message = message.decode()
                    if message:
                        logger.info(self.user_config.gui_to_egf_queue_path + " thread got msg " + message)
                        self.callback(message)
                except posix_ipc.BusyError:
                    # Message queue is busy, try again later
                    continue
                except Exception as e:
                    raise RuntimeError(f"Error: {e}")

    def stop(self):
        if self.mq_from_gui:
            self.running = False

    # Response from LLM to Gui
    def _send_to_gui(self, input_string):
        if self.mq_to_gui:
            logger.debug(f"Sending message to GUI: {input_string}")
            self.mq_to_gui.send(input_string.encode())

    # Response from LLM to Gui
    def send_rsp(self, input_string):
        self._send_to_gui("RSP:" + input_string)

    # Command from LLM to Gui
    def send_cmd(self, input_string):
        self._send_to_gui("CMD:" + input_string)

    def send_perf(self, input_string):
        self._send_to_gui("PERF:" + input_string)

    # WakeWord detection to Gui
    def send_wwd(self, input_string=''):
        self._send_to_gui("WWD:" + input_string)

    # VIT is started info to Gui
    def send_vis(self):
        self._send_to_gui("VIS:")

    # ASR to Gui
    def send_qst(self, input_string):
        self._send_to_gui("QST:" + input_string)

    # Connection info to Gui
    def send_connect(self):
        self._send_to_gui("CON:")

    # Disconnection info to Gui
    def send_disconnect(self):
        self._send_to_gui("DIS:")

    # TTS has finished info to Gui
    def send_thf(self):
        self._send_to_gui("THF:")

    # Speech activity
    def send_speech_activity(self, input_string):
        self._send_to_gui("VAD:" + input_string)

    def send_speaker_activity(self, input_string):
        self._send_to_gui("SPK:" + input_string)

    #  Triggers a class method after a specified delay.
    def send_delayed_cmd(self, method_name, delay, *args, **kwargs):
        if hasattr(self, method_name) and callable(getattr(self, method_name)):
            method = getattr(self, method_name)
            timer = threading.Timer(delay, method, args, kwargs)
            timer.start()
        else:
            raise AttributeError(f"Method '{method_name}' not found in the class.")
