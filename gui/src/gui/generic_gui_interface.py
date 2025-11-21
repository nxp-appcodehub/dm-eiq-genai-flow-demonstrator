# Copyright 2025 NXP
# NXP Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.

import threading
from utils.utils import clear_message_queue
import inspect
import posix_ipc
from gui.config import GuiConfig
import logging

logger = logging.getLogger(__name__)


class GenericGuiInterface(threading.Thread):
    def __init__(self, callback=None, user_config=GuiConfig):
        super().__init__()
        self.user_config = user_config()
        self.verbose = self.user_config.verbose
        self.callback = callback

        # Attempt to open or create the LLMP to GUI queue
        try:
            self.mq_to_gui = posix_ipc.MessageQueue(
                self.user_config.llmp_to_gui_queue_path,
                flags=posix_ipc.O_CREAT | posix_ipc.O_EXCL,  # O_EXCL ensures creation if it doesn't exist
                max_messages=self.user_config.max_messages,
                max_message_size=self.user_config.max_message_size,
                mode=0o644,
                write=True,
                read=True
            )
            clear_message_queue(self.mq_to_gui)
            logger.info(f"{self.user_config.llmp_to_gui_queue_path} Queue created successfully.")
        except posix_ipc.ExistentialError:
            # If the queue already exists, open it with read and write access
            self.mq_to_gui = posix_ipc.MessageQueue(
                self.user_config.llmp_to_gui_queue_path,
                write=True,
                read=True
            )
            logger.info(f"{self.user_config.llmp_to_gui_queue_path} Queue opened successfully.")
        except Exception as e:
            raise RuntimeError(f"{self.user_config.llmp_to_gui_queue_path} mqueue not created/opened: {e}")

        # Attempt to open or create the GUI to LLMP QUEUE
        try:
            self.mq_from_gui = posix_ipc.MessageQueue(
                self.user_config.gui_to_llmp_queue_path,
                flags=posix_ipc.O_CREAT | posix_ipc.O_EXCL,  # O_EXCL ensures creation if it doesn't exist
                max_messages=self.user_config.max_messages,
                max_message_size=self.user_config.max_message_size,
                mode=0o644,
                write=True,
                read=True
            )
            clear_message_queue(self.mq_from_gui)
            logger.info(f"{self.user_config.gui_to_llmp_queue_path} Queue created successfully.")
        except posix_ipc.ExistentialError:
            # If the queue already exists, open it with read and write access
            self.mq_from_gui = posix_ipc.MessageQueue(
                self.user_config.gui_to_llmp_queue_path,
                write=True,
                read=True
            )
            logger.info(f"{self.user_config.gui_to_llmp_queue_path} Queue opened successfully.")
        except Exception as e:
            raise RuntimeError(f"{self.user_config.gui_to_llmp_queue_path} mqueue not created/opened: {e}")

        self.running = True

    def run(self):
        if self.mq_from_gui:
            while self.running:
                logger.info(self.user_config.gui_to_llmp_queue_path + " read thread waiting for msg")
                try:
                    message, _ = self.mq_from_gui.receive(timeout=None)
                    message = message.decode()
                    if message:
                        logger.info(self.user_config.gui_to_llmp_queue_path + " thread got msg " + message)
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
    def send_rsp(self, input_string):
        if self.mq_to_gui:
            input_string = "RSP:" + input_string
            self.mq_to_gui.send(input_string.encode())

    # Command from LLM to Gui
    def send_cmd(self, input_string):
        if self.mq_to_gui:
            input_string = "CMD:" + input_string
            self.mq_to_gui.send(input_string.encode())

    # WakeWord detection to Gui
    def send_wwd(self, input_string):
        if self.mq_to_gui:
            input_string = "WWD:" + input_string
            self.mq_to_gui.send(input_string.encode())

    # VIT is started info to Gui
    def send_vis(self):
        if self.mq_to_gui:
            input_string = "VIS:"
            self.mq_to_gui.send(input_string.encode())

    # ASR to Gui
    def send_qst(self, input_string):
        if self.mq_to_gui:
            input_string = "QST:" + input_string
            self.mq_to_gui.send(input_string.encode())

    # Connection info to Gui
    def send_connect(self):
        if self.mq_to_gui:
            input_string = "CON:"
            self.mq_to_gui.send(input_string.encode())

    # Disconnection info to Gui
    def send_disconnect(self):
        if self.mq_to_gui:
            input_string = "DIS:"
            self.mq_to_gui.send(input_string.encode())

    # TTS has finished info to Gui
    def send_thf(self):
        if self.mq_to_gui:
            input_string = "THF:"
            self.mq_to_gui.send(input_string.encode())

    #  Triggers a class method after a specified delay.
    def send_delayed_cmd(self, method_name, delay, *args, **kwargs):
        if hasattr(self, method_name) and callable(getattr(self, method_name)):
            method = getattr(self, method_name)
            timer = threading.Timer(delay, method, args, kwargs)
            timer.start()
        else:
            raise AttributeError(f"Method '{method_name}' not found in the class.")
