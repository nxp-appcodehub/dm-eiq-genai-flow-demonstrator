# Copyright 2025 NXP
# NXP Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.

from datetime import datetime
import glob
import json
import os
import subprocess
import sys
import threading
import time
import wave
import contextlib
import onnxruntime as ort
import psutil
from eiq_genai_flow import eIQGenAIFlow
from tests.benchmark.questions_to_wav import questions_to_wav
from utils.utils import get_git_commit_sha, get_installed_versions, get_linux_version, get_sha256, get_neutron_info


class Benchmark(eIQGenAIFlow):
    def __init__(
        self,
        config,
        input_mode,
        capture_device,
        wake_word_model,
        gui_config_class,
        llm_model,
        use_rag,
        system_prompt,
        output_mode,
        playback_device,
        continuous,
        asr_model,
        use_neutron,
        benchmark,
        verbose,
    ):
        # Call parent constructor first
        super().__init__(
            config,
            input_mode,
            capture_device,
            wake_word_model,
            gui_config_class,
            llm_model,
            use_rag,
            system_prompt,
            output_mode,
            playback_device,
            continuous,
            asr_model,
            use_neutron,
            benchmark,
            verbose,
        )

        # Initialize benchmark-specific attributes
        self._init_benchmark_attributes()

    def _init_benchmark_attributes(self):
        """Initialize benchmark-specific tracking variables"""
        self.rag_sum_time = 0
        self.llm_sum_ttft = 0
        self.llm_min_ttft = 1000
        self.llm_max_ttft = 0
        self.llm_sum_tps = 0
        self.llm_min_tps = 1000
        self.llm_max_tps = 0
        self.llm_inf_count = 0
        self.llm_sum_time = 0
        self.benchmark_logger = BenchmarkLogger()

    def run(self):
        # Initialize stats storage
        stats = {
            "cpu_usage": [],
            "memory_used": [],
            "memory_percent": [],
            "running": True,  # Control flag for the monitoring thread
        }

        filename = (
            f"Benchmark_{self.device}"
            + f"{('_neutron' if self.use_neutron else '_CPU')}"
            + f"{('_' + self.asr_model + '-asr' if self.asr else '')}"
            + f"{('_rag' if self.retriever else '')}"
            + f"{('_' + self.llm.name + '-llm' if self.llm else '_no_llm')}"
            + f"{('_' + self.output_mode if self.tts else '')}"
            + datetime.now().strftime("_%Y%m%d_%H%M%S_%f")
        )

        self.linux_version = get_linux_version()
        self.benchmark_logger.set_log_file(filename + ".log")
        self.benchmark_logger.clear_log_file()
        self.benchmark_logger.log(f"Benchmarking: {(self.asr_model if self.asr else '')}{(self.llm.name if self.llm else '_no_llm')}{(' with RAG' if self.retriever else '')}{(', with TTS' if self.tts else '')} on {self.full_machine} {'using neutron' if self.use_neutron else 'using CPU'}\n")
        print_benchmark_system_info(self.benchmark_logger, self.config)
        if self.llm:
            self.benchmark_logger.log(f"Actual ORT Execution Providers being used: {self.llm.actual_providers}\n")

        # Read question file:
        with open(self.config.benchmark_questions_file) as file:
            lines = [line.rstrip() for line in file]
        bench_len = len(lines)

        if self.asr:
            from asr.utils import ErrorRateComputer

            error_rate_computer = ErrorRateComputer()

            wav_files = generate_wav_files(wav_dir=self.config.tests_data_path, text_file_path=self.config.benchmark_questions_file, text_file_len=bench_len)
            self.benchmark_logger.log(f"{len(wav_files)} audio files found")
            self.verbose_mode = False

        handle_question_time = 0
        wav_sum_duration = 0
        asr_sum_time, asr_wer = 0, 0
        rag_avg_time = 0
        llm_avg_time, llm_avg_ttft, llm_avg_tps = 0, 0, 0
        tts_avg_time, tts_sum_time, tts_avg_rtf = 0, 0, 0
        ttfa_sum_time, ttfa_avg = 0, 0

        # Start monitoring in a separate thread
        monitor_thread = threading.Thread(target=monitor_system, args=(stats,))
        monitor_thread.start()

        # Run LLM benchmark
        start_time = time.perf_counter()
        if self.asr:
            for idx, (line, wav_file) in enumerate(zip(lines, wav_files)):
                # wav inputs
                asr_start_time = time.perf_counter()
                # Add VAD latency for ttfa computation
                ttfa_start_time = asr_start_time - self.asr.min_silence_duration_ms / 1000
                wav_sum_duration += get_wav_duration(wav_file)
                question = self.asr.file_to_text(audio_file=wav_file)
                asr_sum_time += time.perf_counter() - asr_start_time
                self.egf_print(f"text: {line}")

                normalized_question = self.asr.text_normalizer(question).split(" ")
                line = self.asr.text_normalizer(line).split(" ")

                error_rate_computer.append(ids=[idx], predict=[normalized_question], target=[line])

                self.egf_print(f"ASR: {question}")

                handle_question_start_time = time.perf_counter()
                self.handle_question(question)
                if self.tts:
                    ttfa_sum_time += self.tts.timestamp_ttfa - ttfa_start_time
                    tts_sum_time += self.tts.inference_time
                handle_question_time += time.perf_counter() - handle_question_start_time
        else:
            for question in lines:
                self.egf_print(question)
                handle_question_start_time = time.perf_counter()
                self.handle_question(question)
                handle_question_time += time.perf_counter() - handle_question_start_time
                if self.tts:
                    ttfa_sum_time += self.tts.timestamp_ttfa - handle_question_start_time
                    tts_sum_time += self.tts.inference_time

        # Stop monitoring
        stats["running"] = False
        monitor_thread.join()

        total_time = time.perf_counter() - start_time
        avg_cpu = sum(stats["cpu_usage"]) / len(stats["cpu_usage"]) if stats["cpu_usage"] else 0
        min_cpu = min(stats["cpu_usage"]) if stats["cpu_usage"] else 0
        max_cpu = max(stats["cpu_usage"]) if stats["cpu_usage"] else 0

        if self.asr:
            asr_wer = error_rate_computer.summarize("error_rate")
            from asr.utils import get_timestamp

            file_name = f"WER_model-{self.asr_model}_{get_timestamp()}"
            save_path = os.path.join(os.path.dirname(__file__), "tests", "results", f"{file_name}.txt")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w") as w:
                error_rate_computer.write_stats(w)
            print(f"ASR WER saved in {save_path}")

        if self.llm_inf_count:
            llm_avg_ttft = self.llm_sum_ttft / self.llm_inf_count
            llm_avg_tps = self.llm_sum_tps / self.llm_inf_count

        if self.tts:
            tts_metrics = self.tts.metrics
            rtf_list = tts_metrics["rtf"]
            ttfa_avg = ttfa_sum_time / bench_len
            if len(rtf_list) > 0:
                tts_avg_rtf = sum(rtf_list) / len(rtf_list)
                tts_avg_time = tts_sum_time / bench_len

        avg_memory_used = sum(stats["memory_used"]) / len(stats["memory_used"]) if stats["memory_used"] else 0
        avg_memory_percent = sum(stats["memory_percent"]) / len(stats["memory_percent"]) if stats["memory_percent"] else 0

        # Summarize results
        self.benchmark_logger.log("\n=== Benchmark Summary ===")
        self.benchmark_logger.log(f"Platform: {self.full_machine}")
        self.benchmark_logger.log(f"Linux Kernel: {self.linux_version}")
        self.benchmark_logger.log(f"Total Benchmark Questions: {bench_len}")
        self.benchmark_logger.log(f"NPU: {'ON' if self.use_neutron else 'OFF'}")

        # Component statuses
        self.benchmark_logger.log("\n--- Component Status ---")
        self.benchmark_logger.log(f"ASR: {self.asr_model if self.asr else 'OFF'}")
        self.benchmark_logger.log(f"RAG: {self.retriever.embedding_model.name if self.retriever else 'OFF'}" + (f" | Answered: {bench_len - self.llm_inf_count}" if self.retriever else ""))
        self.benchmark_logger.log(f"LLM: {self.llm.name if self.llm else 'OFF'}" + (f" | Answered: {self.llm_inf_count}" if self.llm else ""))
        self.benchmark_logger.log(f"TTS: {'ON' if self.tts else 'OFF'}" + (f" | Model: {self.tts.model_name}, Mode: {self.output_mode}" if self.tts else ""))

        # Timing and performance
        self.benchmark_logger.log("\n--- Performance Metrics ---")

        self.benchmark_logger.log(f"Benchmark Time: Avg = {total_time / bench_len:.2f}s | Total = {total_time:.2f}s")
        self.benchmark_logger.log(f"TTFA : Avg = {ttfa_avg:.2f}s")
        self.benchmark_logger.log(f"CPU Usage: Avg = {avg_cpu:.2f}% | Min = {min_cpu:.2f}% | Max = {max_cpu:.2f}%")
        self.benchmark_logger.log(f"Memory Usage: Avg = {avg_memory_used:.2f} MB ({avg_memory_percent:.2f}%)")

        # ASR metrics
        if self.asr:
            self.benchmark_logger.log("\n--- ASR Metrics ---")
            self.benchmark_logger.log(f"Init Time: {self.asr_init_time:.2f}s")
            self.benchmark_logger.log(f"Processing Time: Avg = {asr_sum_time / bench_len:.2f}s | Total = {asr_sum_time:.2f}s")
            self.benchmark_logger.log(f"Word Error Rate: {asr_wer:.2f}")
            self.benchmark_logger.log(f"Wave file duration: Avg = {wav_sum_duration / bench_len:.2f}s")

        # RAG metrics
        if self.retriever:
            rag_avg_time = self.rag_sum_time / bench_len
            self.benchmark_logger.log("\n--- RAG Metrics ---")
            self.benchmark_logger.log(f"Init Time: {self.rag_init_time:.2f}s")
            self.benchmark_logger.log(f"Processing Time: Avg = {rag_avg_time:.2f}s | Total = {self.rag_sum_time:.2f}s")

        # LLM + TTS processing
        handle_question_avg_time = handle_question_time / bench_len
        shared_llm_tts_processing = self.tts and self.llm_inf_count

        handle_question_time_string = "LLM+TTS" if shared_llm_tts_processing else "LLM" if self.llm_inf_count else "TTS" if self.tts else "UNKNOWN"
        self.benchmark_logger.log(f"\n--- {handle_question_time_string} Processing ---")
        self.benchmark_logger.log(f"Time: Avg = {handle_question_avg_time - rag_avg_time:.2f}s | Total = {handle_question_time - self.rag_sum_time:.2f}s")

        # LLM-specific metrics
        if self.llm_inf_count:
            llm_avg_time = self.llm_sum_time / bench_len
            self.benchmark_logger.log("\n--- LLM Metrics ---")
            self.benchmark_logger.log(f"Init Time: {self.llm_init_time:.2f}s")
            self.benchmark_logger.log(f"Processing Time: Avg = {llm_avg_time:.2f}s | Total = {self.llm_sum_time:.2f}s")
            self.benchmark_logger.log(f"TTFT: Avg = {llm_avg_ttft:.2f}s | Min = {self.llm_min_ttft:.2f}s | Max = {self.llm_max_ttft:.2f}s")
            self.benchmark_logger.log(f"Tokens/sec: Avg = {llm_avg_tps:.2f} | Min = {self.llm_min_tps:.2f} | Max = {self.llm_max_tps:.2f}")

        # TTS-specific metrics
        if self.tts:
            self.benchmark_logger.log("\n--- TTS Metrics ---")
            self.benchmark_logger.log(f"Init Time: {self.tts_init_time:.2f}s")
            self.benchmark_logger.log(f"Processing Time: Avg = {tts_avg_time:.2f}s | Total = {tts_sum_time:.2f}s")
            self.benchmark_logger.log(f"Real Time Factor: Avg = {tts_avg_rtf:.2f}")

        generator1 = MetricGenerator(
            machine=self.full_machine,
            llm=self.llm.name if self.llm else "",
            rag=self.retriever.embedding_model.name if self.retriever else "",
            tts=self.tts.model_name if self.tts else "",
            asr=self.asr_model if self.asr else "",
            linux_version=self.linux_version,
            use_npu=self.use_neutron,
        )
        metrics_data1 = generator1.get_full_config_entry(
            ttfa_avg=ttfa_avg,
            avg_time=total_time / bench_len,
            avg_cpu=avg_cpu,
            avg_mem=avg_memory_used,
            asr_init_time=self.asr_init_time,
            asr_avg_time=asr_sum_time / bench_len,
            asr_wer=asr_wer,
            rag_avg_time=rag_avg_time,
            rag_init_time=self.rag_init_time,
            llm_init_time=self.llm_init_time,
            llm_avg_time=llm_avg_time,
            llm_tts_avg_time=handle_question_time / bench_len - self.rag_sum_time / bench_len,
            llm_avg_ttft=llm_avg_ttft,
            llm_avg_tps=llm_avg_tps,
            tts_init_time=self.tts_init_time,
            tts_avg_rtf=tts_avg_rtf,
            tts_avg_time=tts_avg_time,
        )

        # create the single entry for the json file
        grouped_metrics_data = {metrics_data1["Platform"]: [{k: v for k, v in metrics_data1.items() if k != "Platform"}]}

        save_to_json_file(grouped_metrics_data, f"{filename}.json")

        if self.config.update_global_benchmark_json:
            update_json_file(
                f"{filename}.json",
                "metrics.json",
                tolerance=0.05,
                action_on_existing="update_if_improved",
            )

        print(f"Results in {filename}.[log/json]")

        # Clean up threads before exiting
        if hasattr(self, "monitor_thread") and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)

        # Call parent cleanup
        self.clean_up()
        os._exit(0)

    def add_benchmark_stats(self, rag_time=0, llm_ttft=0, llm_tps=0, llm_time=0):
        if self.llm and llm_ttft and llm_tps and llm_time:
            self.llm_inf_count += 1
            self.llm_sum_ttft += llm_ttft
            self.llm_min_ttft = min(self.llm_min_ttft, llm_ttft)
            self.llm_max_ttft = max(self.llm_max_ttft, llm_ttft)
            self.llm_sum_tps += llm_tps
            self.llm_min_tps = min(self.llm_min_tps, llm_tps)
            self.llm_max_tps = max(self.llm_max_tps, llm_tps)
            self.llm_sum_time += llm_time
        if self.retriever and rag_time:
            self.rag_sum_time += rag_time


class BenchmarkLogger:
    def __init__(self, log_file_path="benchmark_log.txt"):
        self.log_file_path = log_file_path

    # Method to set the log file path
    def set_log_file(self, log_file_path):
        self.log_file_path = log_file_path
        print(f"Log file path set to '{self.log_file_path}'.")

    def clear_log_file(self):
        if os.path.exists(self.log_file_path):
            os.remove(self.log_file_path)
            print(f"Log file '{self.log_file_path}' has been deleted.")

    # Print function that conditionally logs to a file
    def log(self, message, color="RESET", style="NORMAL"):
        # Write to the file if benchmark mode is enabled
        with open(self.log_file_path, "a") as log_file:
            print(message, file=log_file)

    def append_print(self, msg):
        # Write to the file if benchmark mode is enabled
        with open(self.log_file_path, "a") as log_file:
            log_file.write(msg)  # No newline added


def print_benchmark_system_info(logger, config):
    if logger:
        logger.log("System Info:")
        logger.log(f"Linux Kernel: {get_linux_version()}")
        logger.log(f"Neutron FW sha256: {get_sha256(config.neutron_fw_path)}")
        logger.log(f"Neutron FW LLM sha256: {get_sha256(config.neutron_fw_llm_path)}")
        logger.log(f"Neutron Info: {get_neutron_info()}")
        logger.log(f"ORT build info:  {ort.get_build_info()}")
        logger.log(f"ORT so sha256: {get_sha256(config.ort_lib_path)}")
        logger.log(f"Python packages: {get_installed_versions(config.python_packages_versions_to_display)}")
        logger.log(f"LLMP commit sha: {get_git_commit_sha()}")
        logger.log("\n")


def generate_wav_files(wav_dir, text_file_path, text_file_len):
    # Use glob to find all files matching the pattern question_*.wav
    wav_files = glob.glob(os.path.join(wav_dir, "question_*.wav"))
    if len(wav_files) != text_file_len:
        print(f"question_*.wav files count does not match {text_file_len} length, regenerating")

        for wav_file in wav_files:
            try:
                os.remove(wav_file)
                print(f"Deleted: {wav_file}")
            except OSError as e:
                print(f"Error deleting {wav_file}: {e}")
        wav_files = glob.glob(os.path.join(wav_dir, "question_*.wav"))
        if not len(wav_files):
            print(f"No question_*.wav files found in {wav_dir}.Trying to generate some from the questions.txt files\n")
            questions_to_wav(text_file_path, wav_dir)
            wav_files = glob.glob(os.path.join(wav_dir, "question_*.wav"))
            if not len(wav_files):
                print(f"Error: no question_*.wav files found for ASR in {wav_dir}.Check the {text_file_path} file or run the questions_to_wave.py script manually")
                exit(1)

    return sorted(wav_files)


def monitor_system(stats, interval=0.5):
    # Continuously log CPU and memory usage to `stats` every `interval` seconds
    while stats["running"]:
        cpu_percent = psutil.cpu_percent(interval=None)
        memory_info = psutil.virtual_memory()

        stats["cpu_usage"].append(cpu_percent)
        stats["memory_used"].append(memory_info.used / (1024**2))  # in MB
        stats["memory_percent"].append(memory_info.percent)

        time.sleep(interval)


def get_wav_duration(file_path):
    with contextlib.closing(wave.open(file_path, "r")) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        return duration


def _get_metric_definitions():
    """
    Returns a dictionary containing definitions for metrics comparison and units.
    """
    return {
        "metrics_to_compare": ["time_avg", "llm_avg_ttft", "llm_avg_tps", "cpu_avg", "mem_avg", "asr_wer", "asr_avg_time"],
        "higher_is_better_metrics": {"llm_avg_tps"},
        "lower_is_better_metrics": {"ttfa_avg", "time_avg", "llm_avg_time", "llm_avg_ttft", "cpu_avg", "mem_avg", "asr_wer", "asr_avg_time", "tts_avg_rtf", "tts_avg_time"},
        "metric_units": {
            "ttfa_avg": "seconds",
            "time_avg": "seconds",
            "cpu_avg": "%",
            "mem_avg": "MB",
            "asr_init_time": "seconds",
            "asr_avg_time": "seconds",
            "asr_wer": "%",
            "rag_init_time": "seconds",
            "rag_avg_time": "seconds",
            "llm_init_time": "seconds",
            "llm_avg_time": "seconds",
            "llm_tts_avg_time": "seconds",
            "llm_avg_ttft": "seconds",
            "llm_avg_tps": "tokens/second",
            "tts_init_time": "seconds",
            "tts_avg_time": "seconds",
            "tts_avg_rtf": "seconds",
        },
    }


def _get_base_model_name(full_name):
    """
    Extracts a simpler base name from a potentially full file path or complex model string.
    Removes path and file extension.
    Handles None, empty string, and "OFF".
    """
    if not full_name or full_name == "OFF":
        return "OFF"
    return os.path.splitext(os.path.basename(str(full_name)))[0]


class MetricGenerator:
    def __init__(
        self,
        machine,
        llm=None,
        rag=None,
        tts=None,
        asr=None,
        use_npu=False,
        linux_version=None,
        git_sha=None,  # This is provided
    ):
        self.machine = machine
        # Normalize model names immediately upon initialization
        self.llm = _get_base_model_name(llm)
        self.retriever = _get_base_model_name(rag)
        self.tts = _get_base_model_name(tts)
        self.asr = _get_base_model_name(asr)
        self.npu_status = "ON" if use_npu else "OFF"
        self.linux_version = linux_version
        self.git_sha = git_sha if git_sha is not None else get_git_commit_sha()

    def get_full_config_entry(
        self,
        ttfa_avg,
        avg_time,
        avg_cpu,
        avg_mem,
        asr_init_time,
        asr_avg_time,
        asr_wer,
        rag_init_time,
        rag_avg_time,
        llm_init_time,
        llm_avg_time,
        llm_tts_avg_time,
        llm_avg_ttft,
        llm_avg_tps,
        tts_init_time,
        tts_avg_rtf,
        tts_avg_time,
    ):
        """
        Generates a dictionary representing a single configuration entry
        with nested configuration parameters and performance metrics.
        The Configuration models will be stored as their base names.
        """
        return {
            "Platform": self.machine,
            "Configuration": {
                "LLM": self.llm,
                "RAG": self.retriever,
                "TTS": self.tts,
                "ASR": self.asr,
                "NPU": self.npu_status,
                "Linux_Version": self.linux_version,
                "Git_SHA": self.git_sha,
            },
            "Metrics": {
                "ttfa_avg": f"{ttfa_avg:.2f}",
                "time_avg": f"{avg_time:.2f}",
                "cpu_avg": f"{avg_cpu:.2f}",
                "mem_avg": f"{avg_mem:.2f}",
                "asr_init_time": f"{asr_init_time:.2f}",
                "asr_avg_time": f"{asr_avg_time:.2f}",
                "asr_wer": f"{asr_wer:.2f}",
                "rag_init_time": f"{rag_init_time:.2f}",
                "rag_avg_time": f"{rag_avg_time:.2f}",
                "llm_init_time": f"{llm_init_time:.2f}",
                "llm_avg_time": f"{llm_avg_time:.2f}",
                "llm_tts_avg_time": f"{llm_tts_avg_time:.2f}",
                "llm_avg_ttft": f"{llm_avg_ttft:.2f}",
                "llm_avg_tps": f"{llm_avg_tps:.2f}",
                "tts_init_time": f"{tts_init_time:.2f}",
                "tts_avg_time": f"{tts_avg_time:.2f}",
                "tts_avg_rtf": f"{tts_avg_rtf:.2f}",
            },
        }

    def get_unique_config_identifier(self, new_config_entry=None):
        """
        Generates a tuple representing the unique identity of a configuration.
        This is used to check if a configuration already exists in the list.
        If new_config_entry is provided, it extracts details from there;
        otherwise, it uses the instance's attributes (which are already base names).
        """
        if new_config_entry:
            config_details = new_config_entry.get("Configuration", {})
            return (
                new_config_entry.get("Platform"),
                _get_base_model_name(config_details.get("LLM")),
                _get_base_model_name(config_details.get("RAG")),
                _get_base_model_name(config_details.get("TTS")),
                _get_base_model_name(config_details.get("ASR")),
                config_details.get("NPU"),
                config_details.get("Linux_Version"),
                # Git_SHA is NOT part of the unique identifier for comparison,
                # as you might want to update metrics for the same config
                # but under a new git SHA. If you want it part of the identifier, uncomment:
                # config_details.get("Git_SHA"),
            )
        else:
            # Attributes are already base names due to __init__ normalization
            return (
                self.machine,
                self.llm,
                self.retriever,
                self.tts,
                self.asr,
                self.npu_status,
                self.linux_version
            )


def report_lava_test_case(name, result, measurement, units="unitless", extra_args=None):
    """
    Calls the lava-test-case command to report a single metric result in the specified format:
    lava-test-case TEST_CASE_ID --result RESULT --units UNITS --measurement MEASUREMENT

    Args:
        name (str): The name of the test case (TEST_CASE_ID), which is now just the metric name.
        result (str): The result of the test case ("pass" or "fail").
        measurement (float/str): The actual measured value (MEASUREMENT).
        units (str): The units of the measurement (UNITS).
        extra_args (list): This parameter is ignored as per the specified lava-test-case format.
    """
    command = ["lava-test-case", str(name), "--result", str(result), "--units", str(units), "--measurement", str(measurement)]

    # Print the command being executed for debugging/traceability
    print(f"Executing LAVA command: {' '.join(command)}")

    try:
        # Keep capture_output=True to catch potential errors in stdout/stderr
        # check=True will raise CalledProcessError if the command returns a non-zero exit code
        result_obj = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Successfully reported LAVA test case: {name}")
        # Optionally, print stdout/stderr even on success if needed for verbose logging
        if result_obj.stdout:
            print(f"  LAVA command STDOUT: {result_obj.stdout.strip()}")
        if result_obj.stderr:
            print(f"  LAVA command STDERR: {result_obj.stderr.strip()}")

    except subprocess.CalledProcessError as e:
        print(f"Error reporting LAVA test case {name}: {e}")
        print(f"  STDOUT: {e.stdout.strip()}")
        print(f"  STDERR: {e.stderr.strip()}")
    except FileNotFoundError:
        print("Error: 'lava-test-case' command not found. Please ensure LAVA tools are installed and in your PATH.")


def compare_metrics(old_entry, new_entry, identifier, tolerance=0.05, lava_test_case=False):
    """
    Compares two sets of metrics from nested entries and prints a detailed report,
    indicating whether the change is good (green) or bad (red) based on metric type.
    It expects 'old_entry' and 'new_entry' to be dictionaries containing a 'Metrics'
    sub-dictionary.

    Args:
        old_entry (dict): The dictionary for the old configuration, containing 'Metrics'.
        new_entry (dict): The dictionary for the new configuration, containing 'Metrics'.
        identifier (tuple): A tuple representing the unique identifier of the configuration.
        tolerance (float): The percentage tolerance (e.g., 0.01 for 1%). If the absolute
                           percentage difference is within this tolerance, it's considered
                           "NO SIGNIFICANT CHANGE".
        lava_test_case (bool): If True, reports results using LAVA test case API format.
    """
    print(f"\n--- Comparison for Configuration: {identifier} ---")

    metric_defs = _get_metric_definitions()
    metrics_to_compare = metric_defs["metrics_to_compare"]
    higher_is_better_metrics = metric_defs["higher_is_better_metrics"]
    lower_is_better_metrics = metric_defs["lower_is_better_metrics"]
    metric_units = metric_defs["metric_units"]

    def to_float(val):
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0.0

    old_metrics = old_entry.get("Metrics", {})
    new_metrics = new_entry.get("Metrics", {})

    for metric_name in metrics_to_compare:
        old_val = to_float(old_metrics.get(metric_name, 0))
        new_val = to_float(new_metrics.get(metric_name, 0))

        lava_test_name = metric_name  # As per the new requirement

        units = metric_units.get(metric_name, "unitless")

        if old_val == 0 and new_val == 0:
            print(f"- {metric_name}: No data or both zero.")
            if lava_test_case:
                report_lava_test_case(
                    name=lava_test_name,
                    result="pass",
                    measurement=0.0,
                    units=units,
                )
            continue

        diff = new_val - old_val
        status = "NO CHANGE"
        color_start = "\033[0m"
        percentage_diff = 0.0

        if old_val != 0:
            percentage_diff = (diff / old_val) * 100
        elif diff != 0:
            percentage_diff = float("inf")

        test_result_lava = "fail"

        if abs(percentage_diff) <= (tolerance * 100):
            status = "NO SIGNIFICANT CHANGE"
            color_start = "\033[0m"
            test_result_lava = "pass"

        elif diff > 0:
            if metric_name in higher_is_better_metrics:
                status = "IMPROVED"
                color_start = "\033[92m"
                test_result_lava = "pass"
            elif metric_name in lower_is_better_metrics:
                status = "REGRESSED"
                color_start = "\033[91m"
                test_result_lava = "fail"
            else:
                status = "CHANGED"
                color_start = "\033[0m"
                test_result_lava = "pass"

        elif diff < 0:
            if metric_name in higher_is_better_metrics:
                status = "REGRESSED"
                color_start = "\033[91m"
                test_result_lava = "fail"
            elif metric_name in lower_is_better_metrics:
                status = "IMPROVED"
                color_start = "\033[92m"
                test_result_lava = "pass"
            else:
                status = "CHANGED"
                color_start = "\033[0m"
                test_result_lava = "pass"
        else:
            test_result_lava = "pass"

        if old_val != 0:
            print(f"- {metric_name}: Old={old_val:.2f}, New={new_val:.2f}, {status}: {color_start}{diff:+.2f} ({percentage_diff:+.2f}%)\033[0m")
            if lava_test_case:
                report_lava_test_case(
                    name=lava_test_name,
                    result=test_result_lava,
                    measurement=new_val,
                    units=units,
                )
        else:
            # Handle cases where old_val is 0 but new_val is not
            print(f"- {metric_name}: Old={old_val:.2f}, New={new_val:.2f}, Change: {diff:+.2f} (N/A % - old was zero)")
            if lava_test_case:
                report_lava_test_case(
                    name=lava_test_name,
                    result="pass",
                    measurement=new_val,
                    units=units,
                )

    print("------------------------------------------\n")


def update_json_file(
    source_json_filepath,
    target_filename="metrics.json",
    action_on_existing="compare",
    tolerance=0.05,
    lava_test_case=False,
):
    """
    Loads data from an existing JSON file (source_json_filepath), extracts its first entry
    ('Platform'), and then uses that entry to update another JSON file (target_filename).

    If a configuration already exists in target_filename:
        - If action_on_existing is "compare", it prints a comparison.
        - If action_on_existing is "update", it updates the existing entry.
        - If action_on_existing is "update_if_improved", it updates if all priority metrics are better.
    If the configuration is new, it adds it and saves the file.

    Args:
        source_json_filepath (str): The path to the JSON file containing the new
                                      configuration entry (e.g., a benchmark.json).
                                      It's expected to be in the format:
                                      {"Platform1": [entry1, entry2], "Platform2": [entry3]}
                                      We will take the *first* entry found in this file
                                      for the update.
        target_filename (str): The path to the JSON file to be updated.
        action_on_existing (str): Specifies behavior if config exists.
                                  Can be "compare" (default), "update", or "update_if_improved".
        tolerance (float): The percentage tolerance to pass to compare_metrics.
        lava_test_case (bool): If True, reports results using LAVA test case API format for comparisons.
    """

    # Helper to safely convert values to float for comparison
    def to_float(val):
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0.0

    metric_defs = _get_metric_definitions()
    metric_units = metric_defs["metric_units"]

    new_config_entry = {}
    if os.path.exists(source_json_filepath):
        try:
            with open(source_json_filepath, "r") as f:
                content = f.read().strip()
                if content:
                    loaded_source_data = json.loads(content)
                    if isinstance(loaded_source_data, dict):
                        found_entry = False
                        for platform_key, entries_list in loaded_source_data.items():
                            if entries_list:
                                # Ensure we get a deep copy to avoid modifying source_json_filepath's data
                                new_config_entry = entries_list[0].copy()
                                new_config_entry["Platform"] = platform_key
                                found_entry = True
                                print(f"Using the first entry from '{platform_key}' in '{source_json_filepath}' for update.")
                                break
                        if not found_entry:
                            print(f"Warning: No valid configuration entries found in '{source_json_filepath}'. No update performed.")
                            return
                    else:
                        print(f"Error: '{source_json_filepath}' content is not in the expected grouped dictionary format. No update performed.")
                        return
                else:
                    print(f"Warning: '{source_json_filepath}' is empty. No update performed.")
                    return
        except json.JSONDecodeError as e:
            print(f"Error: Could not decode JSON from {source_json_filepath}. No update performed. Error: {e}")
            return
    else:
        print(f"Error: Source file '{source_json_filepath}' not found. No update performed.")
        return

    data = {}

    if os.path.exists(target_filename):
        try:
            with open(target_filename, "r") as f:
                content = f.read().strip()
                if content:
                    loaded_data = json.loads(content)
                    if isinstance(loaded_data, dict):
                        data = loaded_data
                    else:
                        print(f"Warning: '{target_filename}' content is not a dictionary. Resetting to an empty dictionary.")
                        data = {}
                else:
                    data = {}
        except json.JSONDecodeError as e:
            print(f"Warning: Could not decode JSON from {target_filename}. Starting with empty data. Error: {e}")
            data = {}

    new_platform = new_config_entry.get("Platform")
    if not new_platform:
        print("Error: New configuration entry from source file is missing 'Platform' key. Cannot process.")
        return

    # Extract Git_SHA from new_config_entry directly for use in MetricGenerator
    new_config_git_sha = new_config_entry.get("Configuration", {}).get("Git_SHA")

    # Create MetricGenerator for the new entry, so its internal attributes are normalized
    # and it captures the Git_SHA from the new config entry
    temp_generator = MetricGenerator(
        machine=new_config_entry.get("Platform"),
        llm=new_config_entry.get("Configuration", {}).get("LLM"),
        rag=new_config_entry.get("Configuration", {}).get("RAG"),
        tts=new_config_entry.get("Configuration", {}).get("TTS"),
        asr=new_config_entry.get("Configuration", {}).get("ASR"),
        use_npu=new_config_entry.get("Configuration", {}).get("NPU") == "ON",
        linux_version=new_config_entry.get("Configuration", {}).get("Linux_Version"),
        git_sha=new_config_git_sha,  # Pass the Git_SHA from the new config file
    )
    # Get the unique identifier from the temp_generator directly (it has normalized names now)
    new_entry_comparison_tuple = temp_generator.get_unique_config_identifier()

    full_identifier_for_print = new_entry_comparison_tuple

    if new_platform not in data:
        data[new_platform] = []

    platform_entries = data[new_platform]
    entry_found = False

    for i, entry in enumerate(platform_entries):
        # Extract Git_SHA from existing entry for MetricGenerator
        existing_config_git_sha = entry.get("Configuration", {}).get("Git_SHA")

        # Create MetricGenerator for the existing entry. Its internal attributes will also be normalized.
        existing_temp_generator = MetricGenerator(
            machine=entry.get("Platform"),
            llm=entry.get("Configuration", {}).get("LLM"),
            rag=entry.get("Configuration", {}).get("RAG"),
            tts=entry.get("Configuration", {}).get("TTS"),
            asr=entry.get("Configuration", {}).get("ASR"),
            use_npu=entry.get("Configuration", {}).get("NPU") == "ON",
            linux_version=entry.get("Configuration", {}).get("Linux_Version"),
            git_sha=existing_config_git_sha,  # Pass the Git_SHA from the existing config
        )
        # Get the unique identifier from existing_temp_generator directly
        existing_entry_comparison_tuple = existing_temp_generator.get_unique_config_identifier()

        if existing_entry_comparison_tuple == new_entry_comparison_tuple:
            entry_found = True

            if action_on_existing == "update":
                # Ensure the new_config_entry's Configuration reflects the normalized model names
                # and the correct Git_SHA before updating.
                new_config_entry["Configuration"]["LLM"] = temp_generator.llm
                new_config_entry["Configuration"]["RAG"] = temp_generator.retriever
                new_config_entry["Configuration"]["TTS"] = temp_generator.tts
                new_config_entry["Configuration"]["ASR"] = temp_generator.asr
                new_config_entry["Configuration"]["NPU"] = temp_generator.npu_status
                new_config_entry["Configuration"]["Linux_Version"] = temp_generator.linux_version
                new_config_entry["Configuration"]["Git_SHA"] = temp_generator.git_sha  # Update Git_SHA here

                if "Configuration" in new_config_entry and "Configuration" in platform_entries[i]:
                    platform_entries[i]["Configuration"].update(new_config_entry["Configuration"])
                elif "Configuration" in new_config_entry:
                    platform_entries[i]["Configuration"] = new_config_entry["Configuration"]

                if "Metrics" in new_config_entry and "Metrics" in platform_entries[i]:
                    platform_entries[i]["Metrics"].update(new_config_entry["Metrics"])
                elif "Metrics" in new_config_entry:
                    platform_entries[i]["Metrics"] = new_config_entry["Metrics"]

                print(f"Configuration already exists. Updated metrics for: {full_identifier_for_print}")
                with open(target_filename, "w") as f:
                    json.dump(data, f, indent=4)
                print(f"File {target_filename} has been updated.")

                if lava_test_case:
                    for metric_name, value in new_config_entry.get("Metrics", {}).items():
                        lava_test_name = metric_name
                        units = metric_units.get(metric_name, "unitless")
                        report_lava_test_case(
                            name=lava_test_name,
                            result="pass",
                            measurement=to_float(value),
                            units=units,
                        )

            elif action_on_existing == "compare":
                compare_metrics(platform_entries[i], new_config_entry, full_identifier_for_print, tolerance=tolerance, lava_test_case=lava_test_case)
                print(f"File {target_filename} was NOT modified as a comparison was performed.")
            elif action_on_existing == "update_if_improved":
                metric_defs = _get_metric_definitions()
                priority_metrics = ["asr_wer", "asr_avg_time", "llm_avg_ttft", "llm_avg_tps", "tts_avg_time", "tts_avg_rtf", "ttfa_avg"]
                higher_is_better_metrics_priority = metric_defs["higher_is_better_metrics"]
                lower_is_better_metrics_priority = metric_defs["lower_is_better_metrics"]

                all_priority_metrics_improved_or_equal = True
                improvement_messages = []
                comparison_details = []

                for metric_name in priority_metrics:
                    old_val = to_float(entry.get("Metrics", {}).get(metric_name, 0))
                    new_val = to_float(new_config_entry.get("Metrics", {}).get(metric_name, 0))

                    current_metric_improved = False
                    message = ""
                    status = ""

                    if old_val == 0:
                        if new_val != 0:
                            current_metric_improved = True
                            message = f"New metric or old data missing, new value: {new_val:.2f}"
                            status = "NEW/IMPROVED"
                        else:
                            message = "No data or both zero."
                            status = "NO_DATA"
                    else:
                        diff = new_val - old_val
                        percentage_diff = (diff / old_val) * 100

                        if metric_name in higher_is_better_metrics_priority:
                            if new_val >= old_val * (1 - tolerance):
                                current_metric_improved = True
                                status = "IMPROVED" if percentage_diff > (tolerance * 100) else "STABLE"
                                message = f"Improved/Stable (+{percentage_diff:+.2f}%)"
                            else:
                                status = "REGRESSED"
                                message = f"Regressed ({percentage_diff:+.2f}%)"

                        elif metric_name in lower_is_better_metrics_priority:
                            if new_val <= old_val * (1 + tolerance):
                                current_metric_improved = True
                                status = "IMPROVED" if percentage_diff < (-tolerance * 100) else "STABLE"
                                message = f"Improved/Stable ({percentage_diff:+.2f}%)"
                            else:
                                status = "REGRESSED"
                                message = f"Regressed (+{percentage_diff:+.2f}%)"
                        else:
                            if abs(percentage_diff) <= (tolerance * 100):
                                current_metric_improved = True
                                status = "STABLE"
                                message = f"No significant change ({percentage_diff:+.2f}%)"
                            else:
                                status = "CHANGED_OUTSIDE_TOLERANCE"
                                message = f"Significant change outside tolerance ({percentage_diff:+.2f}%)"

                    if not current_metric_improved:
                        all_priority_metrics_improved_or_equal = False

                    improvement_messages.append(f"{metric_name}: Old={old_val:.2f}, New={new_val:.2f}, Status: {status} ({message})")
                    comparison_details.append({"metric": metric_name, "old_val": old_val, "new_val": new_val, "status": status, "message": message, "improved": current_metric_improved})

                if all_priority_metrics_improved_or_equal:
                    # Before updating, ensure the stored config values are normalized base names
                    # and that Git_SHA is updated.
                    new_config_entry["Configuration"]["LLM"] = temp_generator.llm
                    new_config_entry["Configuration"]["RAG"] = temp_generator.retriever
                    new_config_entry["Configuration"]["TTS"] = temp_generator.tts
                    new_config_entry["Configuration"]["ASR"] = temp_generator.asr
                    new_config_entry["Configuration"]["NPU"] = temp_generator.npu_status
                    new_config_entry["Configuration"]["Linux_Version"] = temp_generator.linux_version
                    new_config_entry["Configuration"]["Git_SHA"] = temp_generator.git_sha  # Update Git_SHA here

                    if "Configuration" in new_config_entry and "Configuration" in platform_entries[i]:
                        platform_entries[i]["Configuration"].update(new_config_entry["Configuration"])
                    elif "Configuration" in new_config_entry:
                        platform_entries[i]["Configuration"] = new_config_entry["Configuration"]

                    if "Metrics" in new_config_entry and "Metrics" in platform_entries[i]:
                        platform_entries[i]["Metrics"].update(new_config_entry["Metrics"])
                    elif "Metrics" in new_config_entry:
                        platform_entries[i]["Metrics"] = new_config_entry["Metrics"]

                    print(f"Configuration already exists. Updated metrics for: {full_identifier_for_print} (All priority metrics improved or stable).")
                    with open(target_filename, "w") as f:
                        json.dump(data, f, indent=4)
                    print(f"File {target_filename} has been updated.")

                    if lava_test_case:
                        for metric_name, value in new_config_entry.get("Metrics", {}).items():
                            lava_test_name = metric_name
                            units = metric_units.get(metric_name, "unitless")
                            report_lava_test_case(
                                name=lava_test_name,
                                result="pass",
                                measurement=to_float(value),
                                units=units,
                            )

                else:
                    print(f"Configuration already exists. Metrics for {full_identifier_for_print} not updated as not all priority metrics improved or were stable.")
                    for msg in improvement_messages:
                        print(f"  - {msg}")
                    print(f"File {target_filename} was NOT modified.")

                    if lava_test_case:
                        for detail in comparison_details:
                            metric_name = detail["metric"]
                            lava_test_name = metric_name
                            units = metric_units.get(metric_name, "unitless")
                            report_lava_test_case(
                                name=lava_test_name,
                                result="pass" if detail["improved"] else "fail",
                                measurement=detail["new_val"],
                                units=units,
                            )

                        for metric_name, value in new_config_entry.get("Metrics", {}).items():
                            if metric_name not in priority_metrics:
                                lava_test_name = metric_name
                                units = metric_units.get(metric_name, "unitless")
                                report_lava_test_case(
                                    name=lava_test_name,
                                    result="pass",
                                    measurement=to_float(value),
                                    units=units,
                                )

            else:
                print(f"Invalid action_on_existing: '{action_on_existing}'. Must be 'compare', 'update', or 'update_if_improved'. No action taken on existing entry.")
            break

    if not entry_found:
        # Before adding, ensure the stored config values are normalized base names
        # and include the Git_SHA from temp_generator.
        entry_to_add = {
            "Platform": new_config_entry["Platform"],
            "Configuration": {
                "LLM": temp_generator.llm,
                "RAG": temp_generator.retriever,
                "TTS": temp_generator.tts,
                "ASR": temp_generator.asr,
                "NPU": temp_generator.npu_status,
                "Linux_Version": temp_generator.linux_version,
                "Git_SHA": temp_generator.git_sha,  # Include Git_SHA from temp_generator
            },
            "Metrics": new_config_entry.get("Metrics", {}),
        }
        data[new_platform].append(entry_to_add)
        print(f"Added new configuration for platform '{new_platform}': {full_identifier_for_print}")
        with open(target_filename, "w") as f:
            json.dump(data, f, indent=4)
        print(f"File {target_filename} has been updated with a new entry.")
        if lava_test_case:
            for metric_name, value in new_config_entry.get("Metrics", {}).items():
                lava_test_name = metric_name
                units = metric_units.get(metric_name, "unitless")
                report_lava_test_case(
                    name=lava_test_name,
                    result="pass",
                    measurement=to_float(value),
                    units=units,
                )
    sys.exit()


def save_to_json_file(data, filename="metrics.json"):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
