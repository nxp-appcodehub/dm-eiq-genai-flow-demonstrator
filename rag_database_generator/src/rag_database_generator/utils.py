# Copyright 2024-2026 NXP
# NXP Confidential and Proprietary.
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
from shared_utils.utils import _summarize_value

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
