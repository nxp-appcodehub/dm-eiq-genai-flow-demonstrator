# Copyright 2026 NXP
# NXP Confidential and Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.

import os
import shutil
import logging
from huggingface_hub import hf_hub_download
from pathlib import Path
from datetime import datetime
from colorama import Fore, Style

logger = logging.getLogger(__name__)


def ensure_local_or_download_hf(models_dir, relative_path, folder=False, config=None):
    """
    Downloads a Nexus folder if not already present locally.
    """
    local_path = os.path.join(models_dir, relative_path)
    if not os.path.exists(local_path):
        if folder:
            len_download_file = len(config.hf_processors_list)
            if len_download_file >= 1:
                i = 0
                while (i < len_download_file):
                    logger.info(local_path)
                    _ = hf_hub_download(
                        config.hf_repo_id,
                        config.hf_processors_list[i],
                        cache_dir=models_dir,
                        local_dir=local_path
                    )
                    i += 1
                if i == len_download_file:
                    return local_path
        else:
            logger.info('Downloading model folder')
            _ = hf_hub_download(
                config.hf_repo_id,
                config.hf_vision_session,
                subfolder="onnx",
                cache_dir=models_dir,
                local_dir=models_dir + config.model_folder
            )
            _ = hf_hub_download(
                config.hf_repo_id,
                config.hf_embedding_session,
                subfolder="onnx",
                cache_dir=models_dir,
                local_dir=models_dir + config.model_folder
            )
            _ = hf_hub_download(
                config.hf_repo_id,
                config.hf_decoder_session,
                subfolder="onnx",
                cache_dir=models_dir,
                local_dir=models_dir + config.model_folder
            )

    return local_path


def remove(path: str | Path, allow_dir: bool = False):
    p = Path(path)
    if p.is_file():
        p.unlink()
    elif p.is_dir():
        if not allow_dir:
            raise ValueError(f"Refusing to delete directory without allow_dir=True: {p}")
        shutil.rmtree(p)
    else:
        raise FileNotFoundError(p)

    logger.info(f"Deleted: {path}")


class PrettyLoggerFormatter(logging.Formatter):
    """Logging Formatter"""

    format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    FORMATS = {
        logging.DEBUG: Fore.LIGHTBLUE_EX + format + Style.RESET_ALL,
        logging.INFO: Fore.LIGHTGREEN_EX + format + Style.RESET_ALL,
        logging.WARNING: Fore.LIGHTYELLOW_EX + format + Style.RESET_ALL,
        logging.ERROR: Fore.LIGHTRED_EX + format + Style.RESET_ALL,
        logging.CRITICAL: Fore.RED + Style.BRIGHT + format + Style.RESET_ALL,
    }

    def format(self, record):
        logger_format = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(fmt=logger_format, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


def setup_logging(level=logging.DEBUG, root_path: str = ".", saved_log_limit: int = 20):
    from logging.handlers import QueueHandler, QueueListener
    import queue
    import atexit

    # Create a custom logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.NOTSET)  # Set the root logger's level

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Create log folder
    log_folder_path = os.path.join(root_path, "logs")
    if not os.path.exists(log_folder_path):
        os.makedirs(log_folder_path)

    # Get list of log files in the directory
    log_files = [f for f in os.listdir(log_folder_path) if os.path.isfile(os.path.join(log_folder_path, f))]

    # Check if there are 20 or more log files
    if len(log_files) >= saved_log_limit:
        # Sort log files by creation time
        log_files.sort(key=lambda x: os.path.getctime(os.path.join(log_folder_path, x)))
        # Delete the oldest log file
        remove(os.path.join(log_folder_path, log_files[0]))

    log_file_path = os.path.join(log_folder_path, datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + ".log")

    # Create file handler (will run in background thread)
    file_handler = logging.FileHandler(filename=log_file_path)
    file_handler.setLevel(logging.DEBUG)

    # Create formatters
    console_formatter = PrettyLoggerFormatter()
    file_formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Set formatters
    console_handler.setFormatter(console_formatter)
    file_handler.setFormatter(file_formatter)

    # Create queue for async file logging
    log_queue = queue.Queue()
    queue_handler = QueueHandler(log_queue)

    # Create queue listener (runs file handler in background thread)
    queue_listener = QueueListener(log_queue, file_handler, respect_handler_level=True)
    queue_listener.start()

    # Ensure queue listener stops on program exit
    atexit.register(queue_listener.stop)

    # Add handlers to the logger
    root_logger.addHandler(queue_handler)  # Async file logging
    root_logger.addHandler(console_handler)  # Direct console logging

    logger.info(f"Log file saved at: {log_file_path}")
