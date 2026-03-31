# Copyright 2025-2026 NXP
# NXP Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.

import os
import typer
import logging
from enum import Enum
from utils import validate_submodule, discover_daughter_classes
from shared_utils.utils import (
    get_capture_devices,
    get_playback_devices,
    get_default_playback_device,
    get_default_capture_device,
)

logger = logging.getLogger(__name__)


class ArgumentManager:
    def __init__(self, config, directory_path):
        self.config = config
        self.directory_path = directory_path
        self.module_configs = self._setup_module_configurations()
        self._initialize_module_configs()
        self.arguments = self._create_argument_enums()
        self.options = self._create_cli_options()

    def _setup_module_configurations(self):
        """Setup module availability and configurations."""
        return {
            "gui": {
                "available": validate_submodule(self.directory_path, os.path.join("gui", "modules")),
                "modules": [],
                "config_classes": {},
            },
            "stt": {
                "available": validate_submodule(self.directory_path, "speech_to_text"),
                "models": [],
                "capture_devices": [],
            },
            "llm": {"available": validate_submodule(self.directory_path, "llm"), "models": []},
            "rag": {"available": validate_submodule(self.directory_path, "rag")},
            "tts": {
                "available": validate_submodule(self.directory_path, "tts"),
                "output_modes": [],
                "playback_devices": [],
            },
        }

    def _initialize_module_configs(self):
        """Initialize module-specific configurations."""
        # GUI configuration
        if self.module_configs["gui"]["available"]:
            from gui.config import GuiConfig

            gui_config_classes = discover_daughter_classes(
                base_path=os.path.join(self.directory_path, "gui", "modules"), parent_class=GuiConfig
            )
            self.module_configs["gui"]["config_classes"] = gui_config_classes
            self.module_configs["gui"]["modules"] = list(gui_config_classes.keys())

        # STT configuration
        if self.module_configs["stt"]["available"]:
            try:
                self.module_configs["stt"]["models"] = ["moonshine-base", "moonshine-tiny", "whisper-small.en"]
            except ImportError as e:
                logger.warning(f"Could not import stt models: {e}")

            # Handle capture devices
            capture_devices = get_capture_devices()
            if capture_devices is None:
                logger.warning("No audio capture device found, stt is disabled")
                self.module_configs["stt"]["available"] = False
            else:
                self.module_configs["stt"]["capture_devices"] = capture_devices

        # LLM configuration
        if self.module_configs["llm"]["available"]:
            self.module_configs["llm"]["models"] = ["no_llm", "danube-500M-q8", "danube-500M-q4"]

        # TTS configuration
        if self.module_configs["tts"]["available"]:
            playback_devices = get_playback_devices()
            if playback_devices is None:
                logger.warning("No audio playback device found, TTS is disabled")
                self.module_configs["tts"]["available"] = False
            else:
                self.module_configs["tts"]["output_modes"] = ["tts"]
                self.module_configs["tts"]["playback_devices"] = playback_devices

    def _create_argument_enums(self):
        """Create Typer enums for command line arguments."""
        enums = {}

        # stt enums
        if self.module_configs["stt"]["available"] and self.module_configs["stt"]["models"]:
            enums["STTArgs"] = Enum("STTArgs", {model: model for model in self.module_configs["stt"]["models"]})
        else:
            enums["STTArgs"] = Enum("STTArgs", {"none": "none"})

        if self.module_configs["stt"]["available"] and self.module_configs["stt"]["capture_devices"]:
            enums["CaptureDeviceArgs"] = Enum(
                "CaptureDeviceArgs", {device: device for device in self.module_configs["stt"]["capture_devices"]}
            )
        else:
            enums["CaptureDeviceArgs"] = Enum("CaptureDeviceArgs", {"none": "none"})

        # Input modes
        input_modes = ["keyb"]
        if self.module_configs["stt"]["available"]:
            input_modes.extend(["kasr", "vasr"])
        input_modes.extend(self.module_configs["gui"]["modules"])
        enums["InputModesArgs"] = Enum("InputModesArgs", {mode: mode for mode in input_modes})

        # TTS enums
        output_modes = ["text"] + self.module_configs["tts"]["output_modes"]
        enums["OutputModesArgs"] = Enum("OutputModesArgs", {mode: mode for mode in output_modes})

        if self.module_configs["tts"]["available"] and self.module_configs["tts"]["playback_devices"]:
            enums["PlaybackDeviceArgs"] = Enum(
                "PlaybackDeviceArgs", {device: device for device in self.module_configs["tts"]["playback_devices"]}
            )
        else:
            enums["PlaybackDeviceArgs"] = Enum("PlaybackDeviceArgs", {"none": "none"})

        # LLM enum
        if self.module_configs["llm"]["available"] and self.module_configs["llm"]["models"]:
            enums["LlmArgs"] = Enum("Llm", {llm: llm for llm in self.module_configs["llm"]["models"]})
        else:
            enums["LlmArgs"] = Enum("Llm", {"none": "none"})

        # Logging enum
        enums["LoggingLevel"] = Enum("LoggingLevel", {level: level for level in logging._nameToLevel.keys()})

        return enums

    def _create_cli_options(self):
        """Create CLI options based on available modules."""
        options = {}

        # Input mode option
        help_text = "The input mode: 'keyb' for Keyboard entry"
        if self.module_configs["stt"]["available"]:
            help_text += ", 'kasr' for keyboard trigger + stt, 'vasr' for VIT wakeWord + stt"
        if self.module_configs["gui"]["modules"]:
            if len(self.module_configs["gui"]["modules"]) == 1:
                help_text += f", or the '{self.module_configs['gui']['modules'][0]}' GUI module"
            else:
                help_text += f", or a GUI module in {self.module_configs['gui']['modules']}"
        help_text += "."
        options["input_mode"] = typer.Option("keyb", "--input-mode", "-i", help=help_text, show_default=True)

        # Capture device option
        options["capture_device"] = typer.Option(
            get_default_capture_device() if self.module_configs["stt"]["available"] else None,
            "--capture-device",
            hidden=not self.module_configs["stt"]["available"],
            help="The alsa audio capture device.",
            show_default=self.module_configs["stt"]["available"],
        )

        # LLM model option
        options["llm_model"] = typer.Option(
            "danube-500M-q8" if self.module_configs["llm"]["available"] else None,
            "--llm-model",
            "-m",
            help="The LLM used.",
            show_default=True,
            hidden=not self.module_configs["llm"]["available"],
            expose_value=self.module_configs["llm"]["available"],
        )

        # System prompt option
        options["system_prompt"] = typer.Option(
            self.config.default_system_prompt if self.module_configs["llm"]["available"] else None,
            "--system-prompt",
            "-p",
            help="System prompt for the LLM.",
            hidden=not self.module_configs["llm"]["available"],
            expose_value=self.module_configs["llm"]["available"],
        )

        # Output mode option
        if self.module_configs["tts"]["available"]:
            default_mode = "tts"
            help_text = f"The output mode: 'text' for textual response or {self.module_configs['tts']['output_modes']} "
            "for audio response."
        else:
            default_mode = "text"
            help_text = "The output mode: 'text' for textual response."
        options["output_mode"] = typer.Option(default_mode, "--output-mode", "-o", help=help_text, show_default=True)

        # Playback device option
        options["playback_device"] = typer.Option(
            get_default_playback_device() if self.module_configs["tts"]["available"] else None,
            "--playback-device",
            hidden=not self.module_configs["tts"]["available"],
            help="The alsa audio playback device.",
            show_default=self.module_configs["tts"]["available"],
        )

        # stt model option
        default_stt = (
            "moonshine-base"
            if (self.module_configs["stt"]["available"] and self.module_configs["stt"]["models"])
            else "none"
        )
        options["stt_model"] = typer.Option(
            default_stt,
            "--stt",
            "-a",
            help="stt model used.",
            show_default=self.module_configs["stt"]["available"],
            hidden=not self.module_configs["stt"]["available"],
            expose_value=self.module_configs["stt"]["available"],
        )

        # RAG option
        options["use_rag"] = typer.Option(
            False,
            "--use-rag",
            "-r",
            help="Activate Retrieval-Augmented Generation to classify queries and customize LLM outputs.",
            show_default=self.module_configs["rag"]["available"],
            hidden=not self.module_configs["rag"]["available"],
            expose_value=self.module_configs["rag"]["available"],
        )

        # Wake word model option
        options["wake_word_model"] = typer.Option(
            self.config.wake_model_path if self.module_configs["stt"]["available"] else None,
            "--wake-word-model",
            "-w",
            help=(
                "Path to the VIT wake-word binary model file (.bin). "
                "The demonstrator uses VIT Library version 4.13 (development version). "
                "For custom wake-word models, contact the NXP Voice Team via NXP Community Forum "
                "or your technical support contact. Models are delivered in binary format, "
                "no conversion required."
            ),
            show_default=self.module_configs["stt"]["available"],
            hidden=not self.module_configs["stt"]["available"],
            expose_value=self.module_configs["stt"]["available"],
        )

        return options

    def get_module_configs(self):
        return self.module_configs

    def get_arguments_and_options(self):
        return self.arguments, self.options
