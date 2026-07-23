# Copyright 2025-2026 NXP
# NXP Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.

from eiq_genai_flow.utils.utils import clear_message_queue
import posix_ipc
import logging

logger = logging.getLogger(__name__)

LLMC_TO_TTS_QUEUE = "/llmc_to_tts_queue"
ASR_TO_LLMC_QUEUE = "/asr_to_llmc_queue"


class msg_q:
    def __init__(self, callback=None, verbose=False):
        super().__init__()

        self.mq_to_llmc = None
        self.mq_from_llmc = None
        self.verbose = verbose
        self.callback = callback

        max_message_size = 512
        max_messages = 20

        # Attempt to open or create ASR_TO_LLMC_QUEUE
        try:
            self.mq_to_llmc = posix_ipc.MessageQueue(
                ASR_TO_LLMC_QUEUE,
                flags=posix_ipc.O_CREAT | posix_ipc.O_EXCL,  # O_EXCL ensures creation if it doesn't exist
                max_messages=max_messages,
                max_message_size=max_message_size,
                mode=0o644,
                write=True,
                read=True,
            )
            clear_message_queue(self.mq_to_llmc)
            logger.info(f"{ASR_TO_LLMC_QUEUE} Queue created successfully.")
        except posix_ipc.ExistentialError:
            # If the queue already exists, open it with read and write access
            self.mq_to_llmc = posix_ipc.MessageQueue(ASR_TO_LLMC_QUEUE, write=True, read=True)
            logger.info(f"{ASR_TO_LLMC_QUEUE} Queue opened successfully.")
        except Exception as e:
            logger.warning(f"Warning, {ASR_TO_LLMC_QUEUE} mqueue not created/opened: {e}")

        # Attempt to open or create LLMC_TO_TTS_QUEUE
        try:
            self.mq_from_llmc = posix_ipc.MessageQueue(
                LLMC_TO_TTS_QUEUE,
                flags=posix_ipc.O_CREAT | posix_ipc.O_EXCL,  # O_EXCL ensures creation if it doesn't exist
                max_messages=max_messages,
                max_message_size=max_message_size,
                mode=0o644,
                write=True,
                read=True,
            )
            clear_message_queue(self.mq_from_llmc)
            logger.info(f"{LLMC_TO_TTS_QUEUE} Queue created successfully.")
        except posix_ipc.ExistentialError:
            # If the queue already exists, open it with read and write access
            self.mq_from_llmc = posix_ipc.MessageQueue(LLMC_TO_TTS_QUEUE, write=True, read=True)
            logger.info(f"{LLMC_TO_TTS_QUEUE} Queue opened successfully.")
        except Exception as e:
            logger.warning(f"Warning, {LLMC_TO_TTS_QUEUE} mqueue not created/opened: {e}")

        self.running = True

    # Clear mq from LLMC
    def clear_mq_from_llmc(self):
        if self.mq_to_llmc:
            clear_message_queue(self.mq_to_llmc)
        else:
            logger.error("Error, mq_to_llmc handle is None")

    # Send data to LLM Client
    def send_data(self, input_string):
        if self.mq_to_llmc:
            self.mq_to_llmc.send(input_string.encode())
        else:
            logger.error("Error, mq_to_llmc handle is None")

    # Get data from LLM Client
    def get_data(self, timeout=None):
        message = ""
        if self.mq_from_llmc:
            try:
                message, _ = self.mq_from_llmc.receive(timeout)
                message = message.decode()
                if message:
                    logger.info("got msg from LLMC: " + message)
            except posix_ipc.BusyError:
                logger.warning(
                    f"Timeout occurred while waiting for message, is the client sending data on "
                    f"{self.mq_from_llmc.name}?"
                )
                return "TIMEOUT"
            except Exception as e:
                logger.error(f"Error receiving message: {e}", exc_info=True)
        else:
            logger.error("Error, mq_from_llmc handle is None")

        return message
