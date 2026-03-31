# Copyright 2024-2026 NXP
# NXP Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.

import os
import json
import pickle
import logging
import numpy as np
import multiprocessing
from datetime import datetime
from colorama import Fore, Style

logger = logging.getLogger(__name__)


def save_json(destination_path: str, data: dict) -> None:
    """
    Save a json file.
    :param destination_path: path of the created json file
    :param data: saved dictionary
    :return: None
    """
    with open(destination_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_json(json_path: str) -> dict:
    """
    Load a json file.
    :param json_path: path of the json file
    :return: content of the json file as a dictionary
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def get_file_list(repo_path: str, extensions: str | list[str] = None) -> list[str]:
    """
    Retrieve all JSON files from a given repository path.

    :param repo_path: Path of the repository to search for JSON files
    :param extensions: Supported file extension(s)
    :return: List of paths to all JSON files found in the repository
    """
    json_files = []

    # Convert list to tuple for endswith
    if isinstance(extensions, list):
        extensions = tuple(extensions)

    # Walk through all directories and files in the given path
    for root, dirs, files in os.walk(repo_path):
        # Filter files ending with .json
        if extensions is None:
            return files

        for file in files:
            if file.endswith(extensions):
                json_files.append(file)
    return json_files


def save_pkl(destination_path: str, data: dict) -> None:
    """
    Save a pickle file.
    :param destination_path: path of the created pickle file
    :param data: saved dictionary
    :return: None
    """
    with open(destination_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_markdown(destination_path: str, content: str) -> None:
    """
    Save a Markdown file.
    :param destination_path: path of the created Markdown file
    :param content: Markdown content as a string
    :return: None
    """
    with open(destination_path, "w", encoding="utf-8") as f:
        f.write(content)


def load_markdown(md_path: str) -> str:
    """
    Load a Markdown file.
    :param md_path: path of the Markdown file
    :return: content of the Markdown file as a string
    """
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()
    return content


def print_dict(dictionary: dict, depth: int = 0):
    """
    Print a dictionary in the terminal
    :param dictionary:
    :param depth:
    :return: None
    """
    for i, (name, value) in enumerate(dictionary.items(), start=1):
        print(f"│  • {name}: {_summarize_value(value)}")


def pretty_print(name: str, result_dictionary: dict) -> None:
    """
    Print RAG results in a structured and cleaner format.
    :param name: The title of the print.
    :param result_dictionary: The dictionary containing all the information to print.
    """
    print(f"╭─ {name}:")
    print_dict(result_dictionary)
    print("╰─")


def check_censored_word_presence(query: str) -> bool:
    """
    Check if a censored word present in a list is contained in a string.
    :param query: The string in which we search for censored words.
    :return: bool: Retrun true if a censored word is found
    """
    words = set(query.split(" "))
    if not bool(words & CENSORED_WORDS):  # If no intersection
        return False
    return True


class PrettyLoggerFormatter(logging.Formatter):
    """Logging Formatter"""
    format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    FORMATS = {
        logging.DEBUG: Fore.LIGHTBLUE_EX + format + Style.RESET_ALL,
        logging.INFO: Fore.LIGHTGREEN_EX + format + Style.RESET_ALL,
        logging.WARNING: Fore.LIGHTYELLOW_EX + format + Style.RESET_ALL,
        logging.ERROR: Fore.LIGHTRED_EX + format + Style.RESET_ALL,
        logging.CRITICAL: Fore.RED + Style.BRIGHT + format + Style.RESET_ALL
    }

    def format(self, record):
        logger_format = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(fmt=logger_format, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)


def setup_logging(level=logging.DEBUG, root_path: str = '.', saved_log_limit: int = 20):
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
        os.remove(os.path.join(log_folder_path, log_files[0]))

    log_file_path = os.path.join(log_folder_path, datetime.now().strftime('%Y-%m-%d_%H:%M:%S') + '.log')

    # Create file handler (will run in background thread)
    file_handler = logging.FileHandler(filename=log_file_path)
    file_handler.setLevel(logging.DEBUG)

    # Create formatters
    console_formatter = PrettyLoggerFormatter()
    file_formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                                       datefmt="%Y-%m-%d %H:%M:%S")

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
    root_logger.addHandler(queue_handler)    # Async file logging
    root_logger.addHandler(console_handler)  # Direct console logging

    logger.info(f"Log file saved at: {log_file_path}")


def get_number_of_cores():
    try:
        # Attempt to get the number of cores using multiprocessing module
        num_cores = multiprocessing.cpu_count()
    except NotImplementedError:
        # If the multiprocessing module is not available, fall back to os module
        num_cores = os.cpu_count()

    return num_cores


def get_leaf_classes(cls):
    """
    Recursively finds all leaf subclasses of a given class.
    A leaf class is one that does not have any subclasses.
    :param cls: The base class to start the search from.
    :return: A list of all leaf subclasses.
    """
    subclasses = cls.__subclasses__()  # Get direct subclasses of the class
    leaf_classes = []

    # Recursively collect leaf classes from each subclass
    for subclass in subclasses:
        leaf_classes.extend(get_leaf_classes(subclass))

    # If no subclasses exist, this is a leaf class
    if not subclasses:
        return [cls]

    return leaf_classes


def _summarize_dict(dictionary: dict) -> dict:
    """
    Summarize all values of the given dictionary using the `_summarize_value` method.
    :param dictionary: The dictionary to be summarized.
    :return: A summarized dictionary.
    """
    return {key: _summarize_value(value) for key, value in dictionary.items()}


def _summarize_value(value):
    """
    Summarize the value for logs by handling dictionaries, lists, torch tensors, and numpy arrays.
    :param value: The value to be summarized.
    :return: A summarized representation of the value.
    """
    from torch import Tensor
    # If the value is a numpy array, summarize its shape and dtype
    if isinstance(value, np.ndarray):
        return f"ndarray(shape={value.shape}, dtype={value.dtype})"
    # If the value is a torch Tensor, summarize its shape and dtype
    if isinstance(value, Tensor):
        return f"tensor(shape={value.shape}, dtype={value.dtype})"
    # If the value is a dictionary, recursively summarize it
    elif isinstance(value, dict):
        return _summarize_dict(value)
    # If the value is a list, recursively summarize each item in the list
    elif isinstance(value, list):
        return [_summarize_value(v) for v in value]
    # If it's neither a numpy array, dictionary, nor list, return it as is
    return value


def _log_dict(dictionary: dict) -> None:
    """
    Logs each key-value pair in a dictionary on a single line using a summarized
    format.

    Each entry is formatted as:

    "│  • key: `_summarize_value(value)`"

    where the value is processed by `_summarize_value` to provide a concise
    representation (e.g., for tensors, arrays, etc.).
    :param dictionary: The dictionary whose contents will be logged.
    :return: None
    """
    for i, (name, value) in enumerate(dictionary.items(), start=1):
        logger.info(f"│  • {name}: {_summarize_value(value)}")


def pretty_log(name: str, result_dictionary: dict) -> None:
    """
    Logs a dictionary in a visually structured and readable format.

    The output is framed with a header and footer using box-drawing characters,
    and each key-value pair is logged in a clean, indented format via `_log_dict`.

    Example output for name="Results" and result_dictionary={accuracy: 0.95, loss: tensor([...]), config: {...}:

    ╭─ Results:

    │ • accuracy: 0.95

    │ • loss: tensor(shape=(1,), dtype=float32)

    │ • config: {'lr': 0.001, 'batch_size': 32}

    ╰─

    :param name: A title or label for the dictionary being logged.
    :param result_dictionary: The dictionary containing the data to log.
    :return: None
    """
    logger.info(f"╭─ {name}:")
    _log_dict(result_dictionary)
    logger.info("╰─")


CENSORED_WORDS = {
    "clunge",
    "seductress",
    "slaughter",
    "hooters",
    "crucified",
    "cannibalism",
    "fuck",
    "honkers",
    "oppai",
    "wincest",
    "arrested",
    "jerk off",
    "fascist",
    "sensual",
    "knob",
    "teratoma",
    " mao zedong",
    "cannibal",
    "crotch",
    "bodily fluids",
    "hentai",
    "labia",
    "coochie",
    "phallus",
    "kill",
    "suicide",
    "skimpy",
    "bondage",
    "gruesome",
    "smut",
    "arse",
    "poop",
    "vivisection",
    "killing",
    "shaft",
    "playboy",
    "tryphophobia",
    "big black",
    "nude",
    "horny",
    "jail",
    "honkey",
    "xi jinping",
    "minge",
    "brothel",
    "heroin",
    "breasts",
    "bruises",
    "sexy female",
    "thick",
    "marijuana",
    "legs spread",
    "khorne",
    "handcuffs",
    "girth",
    "badonkers",
    "seducing",
    "orgy",
    "cutting",
    "nipple",
    "sensored",
    "pleasure",
    "taboo",
    "fentanyl",
    "guts",
    "dick",
    "ballgag",
    "bulging",
    "pleasures",
    "thot",
    "hitler",
    "big ass",
    "engorged",
    "erotic seductive",
    "sadist",
    "nasty",
    "flesh",
    "infested",
    "hardcore",
    "bosom",
    "hemoglobin",
    "making love",
    "voluptuous",
    "bimbo",
    "coon",
    "visceral",
    "veiny",
    "shag",
    "dominatrix",
    "ass",
    "incest",
    "bunghole",
    "mammaries",
    "ovaries",
    "surgery",
    "naughty",
    "crucifixion",
    "sultry",
    "prophet mohammed",
    "nazi",
    "busty",
    "sperm",
    "decapitate",
    "crack",
    "female body parts",
    "bloodbath",
    "censored",
    "bloody",
    "ahegao",
    "cocaine",
    "indecent",
    "cronenberg",
    "penis",
    "mommy milker",
    "shibari",
    "meth",
    "bloodshot",
    "seductive",
    "human centipede",
    "weed",
    "cussing",
    "vagina",
    "organs",
    "corpse",
    "sexy",
    "slave",
    "gory",
    "slavegirl",
    "somit",
    "torture",
    "bdsm",
    "twerk",
    "errect",
    "succubus",
    "stripped",
    "naked",
    "massacre",
    "kinbaku",
    "pinup",
    "massive chests",
    "booty",
    "shit",
    "infected",
    "flashy",
    "drugs",
    "staline",
    "porn",
}
