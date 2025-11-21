# Copyright 2024-2025 NXP
# NXP Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.

import sys
from dataclasses import dataclass


def get_onnxruntime_lib_path():
    """Get ONNX Runtime library path with dynamic Python version"""
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    return f"/usr/lib/python{python_version}/site-packages/onnxruntime/capi/onnxruntime_pybind11_state.so"


@dataclass
class Config:
    # Interfaces
    py_to_c_queue: str = "/py_to_c"
    c_to_py_queue: str = "/c_to_py"
    asr_queue: str = "/asr"
    neutron_fw_path: str = "/usr/lib/firmware/NeutronFirmware.elf"
    neutron_fw_llm_path: str = "/usr/lib/firmware/NeutronFwllm.elf"

    ort_lib_path: str = get_onnxruntime_lib_path()
    tests_data_path: str = "tests/data"
    benchmark_questions_file: str = "tests/data/questions.txt"
    update_global_benchmark_json: bool = False

    # Messages
    prompt = "Please type your question:"
    start_kasr = "Press Enter to start ASR"
    start_vasr = "Speak the wakeword to start"
    start_kasr_info = f"\n##### {start_kasr} #####"
    start_vasr_info = f"\n##### {start_vasr} #####"
    start_kasr_startup = "#### Press Enter when ASR is ready to start #####"
    listening_info = "I'm listening!"
    tts_start_text: str = "Hello, I am your eIQ Gen AI Flow assistant, how can I help you?"
    out_of_domain_response_list = ["I'm sorry, but I can't help with that request.",
                                   "I'm unable to assist you with this topic.",
                                   "I cannot provide details on that subject."]
    ambiguous_response_list = ["Can you please reformulate?",
                               "Could you clarify your question?",
                               "Could you make your question clearer?"]

    # Parameters
    asr_timeout_sec: int = 20
    similarity_threshold: float = 0.65  # Minimum value of similarity for not considering the question ambiguous

    # Notification when tts is going to talk
    play_tts_sound: bool = True

    # Default args
    default_system_prompt: str = "Helpful assistant."

    # Debug
    python_packages_versions_to_display = ["onnxruntime", "onnx", "transformers", "numpy", "torch", "accelerate",
                                           "optimum", "sentence-transformers", "posix-ipc", "silero-vad"]


    # CPU settings
    set_cpu_governor: bool = True
    cpu_governor: str = "performance"
    restore_cpu_governor_on_exit: bool = True

    # VIT wake-word model
    wake_model_path: str = "vit/models/VIT_Model_en.bin"
    play_wake_word_sound: bool = True
