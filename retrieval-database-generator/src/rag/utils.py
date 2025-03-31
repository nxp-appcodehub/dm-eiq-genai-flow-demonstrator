# Copyright 2024-2025 NXP
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
import textwrap
import numpy as np
from colorama import Fore, Style
from pprint import pformat
from torch import Tensor


def exit_program(message):
    """
    Prints a message and forcefully exits the program.
    :param message: The message to be displayed before exiting.
    :return: None
    """

    print(message)
    os._exit(1)


def save_json(destination_path: str, data: dict) -> None:
    """
    Save a json file.
    :param destination_path: path of the created json file
    :param data: saved dictionary
    :return: None
    """

    with open(destination_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_json(json_path: str) -> dict:
    """
    Load a json file.
    :param json_path: path of the json file
    :return: content of the json file as a dictionary
    """

    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def get_child_folder_names(directory: str) -> list[str]:
    """
    Get a list of all child folder names in the given directory.
    :param directory: Path to the parent directory
    :return: List of child folder names
    """

    if not os.path.isdir(directory):
        raise NotADirectoryError(f"{directory} is not a valid directory path.")

    # List only child folder names (not files, no full paths)
    child_folders = [
        name for name in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, name))
    ]
    return child_folders


def get_file_list(repo_path: str, extensions: str | list[str] = None) -> list[str]:
    """
    Retrieve all files from a given repository path with certain extensions.
    :param repo_path: Path of the repository to search for files
    :param extensions: List of supported file extension to find
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


def load_pkl(pkl_path: str) -> dict:
    """
    Load a pickle file.
    :param pkl_path: path of the pickle file
    :return: content of the pickle file as a dictionary
    """

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data


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


def summarize_value(value: list | dict | Tensor) -> list | dict | str:
    """
    Summarize the value for print by handling numpy arrays, dictionaries, and lists.
    :param value: The value to be summarized, which can be a numpy array, a dictionary, or a list.
    :return: A summarized representation of the value.
    """

    # If the value is a numpy array, summarize its shape and dtype
    if isinstance(value, np.ndarray):
        return f"ndarray(shape={value.shape}, dtype={value.dtype})"
    # If the value is a torch Tensor, summarize its shape and dtype
    if isinstance(value, Tensor):
        return f"tensor(shape={value.shape}, dtype={value.dtype})"
    # If the value is a dictionary, recursively summarize it
    elif isinstance(value, dict):
        return summarize_dict(value)
    # If the value is a list, recursively summarize each item in the list
    elif isinstance(value, list):
        return [summarize_value(v) for v in value]
    # If it's neither a numpy array, dictionary, nor list, return it as is
    return value


def summarize_dict(dictionary: dict | list) -> dict | list:
    """
    Recursively summarize a dictionary (or a list of dictionaries) by replacing numpy arrays
    with summarized representations and recursively handling nested dictionaries or lists.
    :param dictionary: The dictionary (or list of dictionaries) to be summarized.
    :return: A recursively summarized dictionary or list of dictionaries.
    """

    # If the input is a list, apply summarize_dict recursively to each item in the list
    if isinstance(dictionary, list):
        return [summarize_dict(item) if isinstance(item, dict) else summarize_value(item) for item in dictionary]

    # If the input is a dictionary, apply summarize_value to each key-value pair
    return {key: summarize_value(value) for key, value in dictionary.items()}


def pretty_print(name: str, result_dictionary: dict) -> None:
    """
    Print RAG results in a structured and cleaner format.
    :param name: The title of the print.
    :param result_dictionary: The dictionary containing all the information to print.
    """

    separator = "=" * 70
    padding = (len(separator) - len(name) - 2) // 2  # Subtract 2 for spacing around the text

    # Wrap long lists for readability
    def format_list(data_list, max_length=300, depth:int = 0):
        prefix = depth * "    " + "   "
        formatted = pformat(data_list, compact=True, width=max_length)
        return textwrap.indent(formatted, prefix=prefix)  # Indent for readability

    # Print Boxed Results
    print(Fore.BLUE, Style.BRIGHT, f"{'=' * padding} {name} {'=' * padding}", Style.RESET_ALL)

    def print_dict(dictionary: dict, depth: int = 0):
        tabulation = "    " * depth
        for name, value in dictionary.items():
            if isinstance(value, list) or isinstance(value, np.ndarray):
                print(Fore.BLUE, f"{tabulation}- {name}: ", Fore.RESET)
                print(format_list(summarize_dict(value), depth=depth))
            elif isinstance(value, dict):
                print(Fore.BLUE, f"{tabulation}- {name}: ", Fore.RESET)
                print_dict(value, depth=depth+1)
            else:
                print(Fore.BLUE, f"{tabulation}- {name}: ", Fore.RESET, value)
    print_dict(result_dictionary)
    print()


def check_censored_word_presence(query: str) -> bool:
    """
    Check if a censored word present in a list is contained in a string.
    :param query: The string in which we search for censored words.
    :return: bool: Retrun true if a censored word is found
    """

    words = set(query.split(' '))
    if not bool(words & CENSORED_WORDS):  # If no intersection
        return False
    return True


def get_leaf_classes(cls):
    """
    Recursively retrieves all leaf subclasses of a given class.
    A leaf class is a subclass that does not have any further subclasses.
    :param cls: The base class to search for leaf subclasses.
    :return: A list of all leaf subclasses of the given class.
    """

    subclasses = cls.__subclasses__()
    leaf_classes = []

    for subclass in subclasses:
        leaf_classes.extend(get_leaf_classes(subclass))

    if not subclasses:
        return [cls]
    return leaf_classes


def get_number_of_cores() -> int | None:
    """
    Get the number of cores using multiprocessing module, or using os module
    (if the multiprocessing module is not available)
    :return: int | None: Return the number of available CPU cores
    """

    import multiprocessing
    try:
        # Attempt to get the number of cores using multiprocessing module
        num_cores = multiprocessing.cpu_count()
    except NotImplementedError:
        # If the multiprocessing module is not available, fall back to os module
        num_cores = os.cpu_count()

    return num_cores


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
    "porn"
}
