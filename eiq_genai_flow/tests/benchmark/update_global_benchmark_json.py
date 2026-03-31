# Copyright 2025-2026 NXP
# NXP Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.

import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tests.benchmark.benchmark import update_json_file  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Update a global metrics JSON file with a new configuration entry.")
    parser.add_argument(
        "new_config_file",
        type=str,
        help="Path to the JSON file containing the new, single configuration entry "
        "(e.g., Benchmark_PC_CPU_rag_danube-onnx-llm_tts_20250526_164144_910920.json).",
    )
    parser.add_argument(
        "global_metrics_file",
        type=str,
        nargs="?",
        default="metrics.json",
        help="Path to the global metrics JSON file (e.g., metrics.json). Defaults to 'metrics.json'.",
    )
    parser.add_argument(
        "--action",
        type=str,
        default="update_if_improved",
        choices=["compare", "update", "update_if_improved"],
        help="Action to perform if configuration exists: 'compare', 'update', or 'update_if_improved'. "
        "Defaults to 'update_if_improved'.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="Tolerance for metric comparison (as a decimal, e.g., 0.05 for 5%%). Defaults to 0.05.",
    )
    parser.add_argument(
        "--lava-test-case",
        action="store_true",
        help="Report results using LAVA test case API format.",
    )

    args = parser.parse_args()

    update_json_file(
        source_json_filepath=args.new_config_file,
        target_filename=args.global_metrics_file,
        action_on_existing=args.action,
        tolerance=args.tolerance,
        lava_test_case=args.lava_test_case,
    )


if __name__ == "__main__":
    main()
