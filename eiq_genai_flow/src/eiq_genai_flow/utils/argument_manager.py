# Copyright 2025-2026 NXP
# NXP Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.

import typer
import logging
from enum import Enum
import importlib
import pkgutil
import inspect
from eiq_genai_flow.gui.config import GuiConfig
from shared_utils.utils import (
    get_capture_devices,
    get_playback_devices,
    get_default_playback_device,
    get_default_capture_device,
)
from eiq_genai_flow.llm_backends.ara_dnpu_client.ara_detector import AraDetector

logger = logging.getLogger(__name__)


def discover_gui_classes(package_names: list[str]) -> dict[str, type]:
    gui_class = GuiConfig()
    daughter_classes = {}

    for package_name in package_names:
        try:
            package = importlib.import_module(package_name)
        except ImportError:
            logger.exception("Cannot import package '%s'", package_name)
            continue

        modules = [package]
        # Import every submodule of the package
        if hasattr(package, "__path__"):
            for _, module_name, _ in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
                try:
                    modules.append(importlib.import_module(module_name))
                except ImportError:
                    logger.exception("Cannot import module '%s'", module_name)
        # Look for subclasses
        for module in modules:
            for _, obj in inspect.getmembers(module, inspect.isclass):
                if (gui_class.validate_subclass(obj) and obj is not gui_class):
                    daughter_classes[package_name] = obj
                    break
            if package_name in daughter_classes:
                break

    return daughter_classes


class ArgumentManager:
    def __init__(self, config):
        self.config = config
        self.ara_detector = AraDetector(config)
        self.module_configs = self._setup_module_configurations()
        self._initialize_module_configs()
        self.arguments = self._create_argument_enums()
        self.options = self._create_cli_options()

    def _setup_module_configurations(self):
        """Setup module availability and configurations."""
        available_gui = [module for module in self.config.available_gui_list if self._is_package_available(module)]
        return {
            "gui": {
                "available": bool(available_gui),
                "list": available_gui
            },
            "stt": {
                "available": self._is_package_available("speech_to_text"),
                "models": [],  # Will be populated dynamically
                "capture_devices": [],
            },
            "llm": {"available": self._is_package_available("llm"), "models": []},
            "rag": {"available": self._is_package_available("retriever")},
            "tts": {"available": self._is_package_available("tts"), "output_modes": [], "playback_devices": []},
            "voice_id": {"available": self._is_package_available("voice_id")},
            "vit": {"available": self._is_package_available("vit")},
            "vad": {"available": self._is_package_available("vad")},
        }

    def _initialize_module_configs(self):
        """Initialize module-specific configurations."""
        # GUI configuration
        guis = discover_gui_classes(self.module_configs["gui"]["list"])
        self.module_configs["gui"]["config_classes"] = guis
        self.module_configs["gui"]["modules"] = list(guis.keys())

        # STT configuration
        if self.module_configs["stt"]["available"]:
            try:
                from speech_to_text.models.model_config import ModelConfig
                from shared_utils.utils import get_leaf_classes

                # Get available stt models dynamically
                stt = get_leaf_classes(ModelConfig)
                stt_models = [model.name for model in stt]
                self.module_configs["stt"]["models"] = stt_models  # Use discovered models

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
            llms = ["no_llm"]

            # Check for GUI-based discrete NPU (renamed from discrete-npu-client)
            try:
                from eiq_genai_flow.utils.utils import is_service_running

                if is_service_running(self.config.discrete_npu_service):
                    llms += ["gui-dnpu-client"]
                    logger.debug("GUI DNPU client available")
            except Exception as e:
                logger.warning(f"Could not check GUI DNPU service: {e}")

            # Check for ARA discrete NPU
            try:
                logger.debug("Checking ARA DNPU availability...")
                ara_available, ara_llms = self.ara_detector.is_ara_available(auto_start_connector=True)

                if ara_available and ara_llms:
                    logger.info(f"ARA DNPU detected with {len(ara_llms)} model(s): {', '.join(ara_llms)}")
                    llms += ara_llms
                else:
                    logger.debug("ARA DNPU not available")

            except Exception as e:
                logger.warning(f"ARA DNPU detection failed: {e}", exc_info=True)

            # Add models from LLM module
            try:
                from llm.config.models_config import get_model_names

                llms += get_model_names()
            except ImportError as e:
                logger.warning(f"Could not import LLM models: {e}")

            self.module_configs["llm"]["models"] = llms

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
        if self.module_configs["stt"]["available"] and self.module_configs["vad"]["available"]:
            input_modes.extend(["kasr"])
            if self.module_configs["vit"]["available"]:
                input_modes.extend(["vasr"])
                input_modes.extend(self.module_configs["gui"]["modules"])
            elif self.module_configs["voice_id"]["available"]:
                logger.warning("Voice ID mode not available without VIT module")
                self.module_configs["voice_id"]["available"] = False
        else :
            # Desactivate vit and voice id in case of stt is not available
            self.module_configs["voice_id"]["available"] = False
            self.module_configs["vit"]["available"] = False

        enums["InputModesArgs"] = Enum("InputModesArgs", {mode: mode for mode in input_modes})

        # TTS enums
        output_modes = ["text"]
        enums["PlaybackDeviceArgs"] = Enum("PlaybackDeviceArgs", {"none": "none"})
        if self.module_configs["tts"]["available"] :
            output_modes.extend(self.module_configs["tts"]["output_modes"])
            if self.module_configs["tts"]["playback_devices"]:
                enums["PlaybackDeviceArgs"] = Enum(
                    "PlaybackDeviceArgs", {device: device for device in self.module_configs["tts"]["playback_devices"]}
                )
        enums["OutputModesArgs"] = Enum("OutputModesArgs", {mode: mode for mode in output_modes})

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

        # Voice ID option available
        options["voice_id"] = typer.Option(
            False,
            "--voice-id",
            help="Voice ID mode.",
            show_default=self.module_configs["voice_id"]["available"],
            hidden=not self.module_configs["voice_id"]["available"],
            expose_value=self.module_configs["voice_id"]["available"],
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
        vit_model_path = None
        if self.module_configs["vit"]["available"]:
            if self.config.wake_model_path:
                vit_model_path = self.config.wake_model_path
            else:
                from vit.vit import VIT
                vit_model_path = VIT.get_default_model_path()
        options["wake_word_model"] = typer.Option(
            vit_model_path,
            "--wake-word-model",
            "-w",
            help=(
                "Path to the VIT wake-word binary model file (.bin). "
                "The demonstrator uses VIT Library version 4.13. "
                "For custom wake-word models, use the VIT Model Generation Tool (https://vit.nxp.com/#/). "
                "Models are delivered in binary format, no conversion required."
            ),
            show_default=self.module_configs["vit"]["available"],
            hidden=not self.module_configs["vit"]["available"],
            expose_value=self.module_configs["vit"]["available"],
        )
        return options

    def get_module_configs(self):
        return self.module_configs

    def get_arguments_and_options(self):
        return self.arguments, self.options

    @staticmethod
    def _is_package_available(package_name: str) -> bool:
        """Check if a package is available for import (e.g. installed via wheel)."""
        return importlib.util.find_spec(package_name) is not None
