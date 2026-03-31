# -*- coding: utf-8 -*-

# Copyright 2023-2026 NXP
# NXP Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.

import importlib.util
import inspect
import subprocess
import sys
import hashlib
import re
import os
import struct
import logging
import posix_ipc
from dataclasses import fields, replace

from shared_utils.utils import pretty_log

logger = logging.getLogger(__name__)


screen_text_dict = {
    1: [
        "Here is the navigation system.",
        "Where would you like to go?",
        "Please choose a destination.",
        "Where are we going?",
    ],  # Map
    2: ["Would you like some music?", "What would you like to hear?", "Welcome to media."],  # Media
    20: ["Would you like some music?", "What would you like to hear?", "Your music.", "Here is your music!"],  # Outrun
    21: [
        "Would you like some podcasts?",
        "What is your favorite station?",
        "Select your radio station.",
    ],  # Radio
    22: [
        "Would you like to watch something?",
        "Do you want to watch a series?",
        "Please select your video.",
    ],  # Video
    23: [
        "You want to play a game?",
        "Here are your games.",
        "Please select your game.",
    ],  # Games
    3: ["Here is your car interface.", "Manage your car.", "Your car dashboard is ready."],  # My Car
    31: ["Energy dashboard.", "Here is your consumption data.", "You can check your energy consumption here."],
    # Energy Consumption
    32: [
        "Here is your tire interface.",
        "Your can monitor the tire pressure here.",
        "Tire interface.",
        "Tire pressure.",
    ],  # Tire Pressure
    33: [
        "Here is your lights and doors screen.",
        "Manage your lights, doors, and wipers here.",
        "Lights, doors, and wipers.",
        "You can manage lights, doors, and wipers here.",
    ],  # Lights and doors
    30: [
        "You can change your drive mode here.",
        "Which drive mode would you like?",
        "Please choose your driving mode.",
    ],  # Drive mode
    300: ["Drive mode switch to Normal."],  # Drive mode normal
    301: ["Drive mode switch to Comfort."],  # Drive mode comfort
    302: ["Drive mode switch to Sport."],  # Drive mode sport
    303: ["Drive mode switch to Eco."],  # Drive mode eco
    4: ["Here are your apps.", "Your apps are here."],  # Apps
    40: ["App opened."],
    41: ["App opened."],
    42: ["App opened."],
    43: ["App opened."],
    44: ["App opened."],
    45: ["App opened."],
    46: ["App opened."],
    47: ["App opened."],
    48: ["App opened."],
    49: ["App opened."],
    5: ["Interface settings.", "Manage your interface settings.", "Settings."],  # Settings
    6: ["You are home.", "Here is the main menu.", "Main menu."],  # Home
}


def validate_submodule(dir_path: str, path: str) -> bool:
    """
    Validates a path, checking that it corresponds to a non-empty folder.
    """

    complete_path = os.path.join(dir_path, path)
    # Check if the path exists and is a non-empty directory
    if os.path.isdir(complete_path) and os.listdir(complete_path):
        return True
    else:
        return False


def get_soc_id():
    soc_id_path = "/sys/devices/soc0/soc_id"
    try:
        with open(soc_id_path, "r") as file:
            soc_id = file.read().strip()
            return soc_id
    except FileNotFoundError:
        logger.error(f"The file '{soc_id_path}' was not found.", exc_info=True)
        raise FileNotFoundError(f"The file '{soc_id_path}' was not found.")
    except Exception as e:
        logger.error(f"Can not get SOC id {str(e)}", exc_info=True)
        raise


def get_machine():
    machine_path = "/sys/devices/soc0/machine"
    try:
        with open(machine_path, "r") as file:
            machine = file.read().strip()
            return machine
    except FileNotFoundError:
        logger.error(f"The file '{machine_path}' was not found.", exc_info=True)
        raise FileNotFoundError(f"The file '{machine_path}' was not found.")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise


def get_revision():
    revision_path = "/sys/devices/soc0/revision"
    try:
        with open(revision_path, "r") as file:
            revision = file.read().strip()
            return revision
    except FileNotFoundError:
        logger.error(f"The file '{revision_path}' was not found.", exc_info=True)
        raise FileNotFoundError(f"The file '{revision_path}' was not found.")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise


def is_service_running(service_name: str) -> bool:
    """
    Check if a systemd service is running.
    Returns True if active, False otherwise.
    """
    try:
        result = subprocess.run(
            ["systemctl", "is-active", service_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        return result.stdout.strip() == "active"
    except Exception as e:
        print(f"Error checking service {service_name}: {e}")
        return False


def exec_time(string, start, end):
    text = string + f"{end - start:0.3f}s"
    logger.info(text)


def get_linux_version():
    return os.uname().release


def get_neutron_info():
    neutron_file_path = "/dev/neutron0"
    if os.path.exists(neutron_file_path):
        try:
            buf = read_file(neutron_file_path)
        except Exception:
            return "Cannot read"
        return buf if buf != "" else "Empty"
    else:
        return "Not available"


def get_git_commit_sha():
    try:
        # Get the current commit SHA, suppressing error messages
        sha = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
            .strip()
            .decode("utf-8")
        )
        # Check if the working tree is dirty (any changes not committed)
        is_dirty = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL).strip()

        # If there are any changes, the tree is dirty
        if is_dirty:
            return f"{sha} (dirty)"
        else:
            return sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        # If not a git repository, try to read from git_sha.txt
        try:
            # Get the absolute path of the main script being executed
            # sys.argv[0] is the path to the script itself
            main_script_path = os.path.abspath(sys.argv[0])
            main_script_dir = os.path.dirname(main_script_path)

            # Construct the path to git_sha.txt relative to the main script's directory
            sha_file_path = os.path.join(main_script_dir, "git_sha.txt")

            if os.path.exists(sha_file_path):
                with open(sha_file_path, "r") as f:
                    return f.read().strip()
            else:
                return "Not a git repository and git_sha.txt not found"
        except Exception as e:
            return f"Not a git repository and failed to read git_sha.txt: {e}"


def read_file(file_path):
    with open(file_path, "r") as file:
        content = file.read()
    return content


def get_sha256(file_path):
    """Calculate SHA-256 hash of a file"""
    if os.path.exists(file_path):
        with open(file_path, "rb") as file:
            sha256_hash = hashlib.sha256()
            while chunk := file.read(8192):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    else:
        return f"{file_path} is not available"


def get_installed_versions(packages):
    versions = {}
    for package in packages:
        try:
            # Retrieve package version
            version = importlib.metadata.version(package)
            versions[package] = version
        except importlib.metadata.PackageNotFoundError:
            # Package is not installed
            versions[package] = "Not installed"
    return versions


def print_system_info(config):
    import onnxruntime as ort

    sys_info = {
        "Linux Kernel": get_linux_version(),
        "Neutron FW sha256": get_sha256(config.neutron_fw_path),
        "Neutron Info": get_neutron_info(),
        "ORT Build Info": ort.get_build_info().replace("ORT Build Info: ", ""),
        "ORT so sha256": get_sha256(config.ort_lib_path),
        "Python packages": get_installed_versions(config.python_packages_versions_to_display),
        "Commit sha": get_git_commit_sha(),
    }
    pretty_log(name="System Info", result_dictionary=sys_info)


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


def create_message_queue(name, verbose):
    max_message_size = 1024
    max_messages = 10
    try:
        # Create the message queue with appropriate flags
        mq = posix_ipc.MessageQueue(
            name, flags=posix_ipc.O_CREAT, max_messages=max_messages, max_message_size=max_message_size, mode=0o666
        )
        logger.debug(f"Message queue {name} successfully created.")
        clear_message_queue(mq)
        return mq
    except posix_ipc.ExistentialError:
        logger.error(f"Message queue {name} already exists.", exc_info=True)
        raise
    except Exception:
        logger.error(f"Error occurred while creating the message queue {name}", exc_info=True)
        raise


def send_message(mq, message, verbose=False):
    if mq is not None:
        try:
            mq.send(struct.pack("i", message))
            # Send a message to the message queue
            logger.debug("sending " + str(message))
        except Exception:
            logger.error("Error occurred while sending message", exc_info=True)
            raise
    else:
        logger.warning("Message queue is closed. Cannot send message.")


def discover_daughter_classes(base_path: str, parent_class: object):
    """
    Discover the daughters classes inherited from the class_type in the base_path.
    """
    daughter_classes = {}
    pattern = re.compile(rf"class\s+(\w+)\s*\(\s*{parent_class.__name__}\s*\)\s*:")

    search_path = base_path

    if not os.path.exists(search_path):
        return daughter_classes

    # Walk through the directory tree starting from base_path
    for root, _, files in os.walk(search_path):
        for file in files:
            if file.endswith(".py") and not file.startswith("__"):
                file_path = os.path.join(root, file)
                module_name = os.path.splitext(os.path.basename(file))[0]

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    matches = pattern.findall(content)
                    if not matches:
                        continue

                    # Only load if pattern found
                    rel_module = os.path.relpath(file_path, search_path)
                    module_name = rel_module.replace(os.sep, ".").rsplit(".py", 1)[0]
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if issubclass(obj, parent_class) and obj is not parent_class:
                            relative = os.path.relpath(root, search_path)
                            top_module = relative.split(os.sep)[0] if relative != "." else module_name
                            daughter_classes[top_module] = obj
                except (ModuleNotFoundError, AttributeError, ImportError):
                    logger.error(f"Can not load module: {module_name}", exc_info=True)

    return daughter_classes


def overwrite_config(default_cls, overwrite_cls=None, model_name=None):
    """Apply config.py overrides to model configuration and log results

    Args:
        overwrite_cls: Instance of the Config class from config.py
        default_cls: Instance of a model config class
        model_name: Name of the model
    """

    if not overwrite_cls:
        return default_cls

    overwrite_conf = {
        f.name: getattr(overwrite_cls, f.name) for f in fields(default_cls) if hasattr(overwrite_cls, f.name)
    }

    # Logging
    logger.info(f"=== {model_name} Configuration ===")
    for f in fields(overwrite_cls):
        display_name = f.name.replace("_", " ").replace("-", " ").title()
        if getattr(overwrite_cls, f.name, None) is not None:
            logger.info(
                f"  {display_name}: {overwrite_conf[f.name]} "
                f"(from config.py, model default was: {getattr(default_cls, f.name)})"
            )
        else:
            logger.info(f"  {display_name}: {getattr(default_cls, f.name, 'Not defined')} (default config)")
    logger.info("========================\n")

    return replace(default_cls, **overwrite_conf)
