# Copyright 2026 NXP
# NXP Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.

import os
import sys
import platform
import subprocess
from setuptools import setup
from setuptools.command.build_py import build_py


def get_package_version():
    with open(os.path.join(os.path.dirname(__file__), "VERSION"), "r") as f:
        version = f.readline().strip()
    return version


def get_local_version_suffix():
    """
    Returns a local version suffix:
    - In release builds: no suffix         -> e.g. 3.0.0
    - With git available: commit sha       -> e.g. 3.0.0+abc1234f.dirty
    - Without git (local build fallback):  -> e.g. 3.0.0+local
    """
    if RELEASE_BUILD:
        return ""
    try:
        sha = (
            subprocess.check_output(
                ["git", "rev-parse", "--short=7", "HEAD"],
                cwd=os.path.dirname(__file__) or ".",
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        dirty = (
            subprocess.call(
                ["git", "diff", "--quiet"],
                cwd=os.path.dirname(__file__) or ".",
                stderr=subprocess.DEVNULL,
            )
            != 0
        )
        return f"+g{sha}{'.dirty' if dirty else ''}"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "+local"


# variables
RELEASE_BUILD = os.environ.get("RELEASE_BUILD", "0") == "1"
PKG_VERSION = get_package_version()
HOST_ARCH = platform.machine()
TARGET_ARCH = os.environ.get("TARGET_ARCH", HOST_ARCH)

LOCAL_VERSION_SUFFIX = get_local_version_suffix()
FULL_VERSION = f"{PKG_VERSION}{LOCAL_VERSION_SUFFIX}"

# The .so suffix to keep: e.g. "cpython-313-aarch64-linux-gnu.so"
PY_TAG = f"cpython-{sys.version_info.major}{sys.version_info.minor}-{TARGET_ARCH}-linux-gnu.so"

print(f"[SETUP] version={FULL_VERSION}  release={RELEASE_BUILD}")
print(f"[SETUP] host={HOST_ARCH}  target={TARGET_ARCH}")
print(f"[SETUP] keeping .so matching: {PY_TAG}")

# Patterns to exclude from the wheel
EXCLUDE_PATTERNS = (
    ".cpp",  # C++ source files (vit_binding.cpp)
    ".h",  # C header files
    ".a",  # static libraries
)


def _should_exclude(filename: str) -> bool:
    """Return True if the file should be excluded from the wheel."""
    # Exclude C/C++ source and static libs
    if any(filename.endswith(ext) for ext in EXCLUDE_PATTERNS):
        return True
    # Exclude .so files that don't match the target arch + Python version
    if filename.endswith(".so") and not filename.endswith(PY_TAG):
        return True
    return False


class FilteredBuildPy(build_py):
    """
    Custom build_py that:
    - Excludes vit_binding.cpp
    - Excludes .h and .a files (libs/ headers and static libraries)
    - Keeps only the .so matching the target arch and current Python version
    - Includes .bin model files
    """

    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        filtered = [(pkg, mod, path) for pkg, mod, path in modules if not _should_exclude(os.path.basename(path))]
        for _, _, path in set(modules) - set(filtered):
            print(f"[SETUP] excluding module: {path}")
        return filtered

    def find_data_files(self, package, src_dir):
        """
        find_data_files returns a flat list of file path strings.
        Filter out unwanted files from the list.
        """
        files = super().find_data_files(package, src_dir)
        kept = [f for f in files if not _should_exclude(os.path.basename(f))]
        for f in set(files) - set(kept):
            print(f"[SETUP] excluding data file: {f}")
        return kept


ext_modules = None

# wheel options: to modify the wheel name
python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
wheel_opts = {"bdist_wheel": {"python_tag": python_tag}}
wheel_opts["bdist_wheel"]["plat_name"] = f"linux_{TARGET_ARCH}"

setup(
    author="NXP AI Software Team",
    ext_modules=ext_modules,
    options=wheel_opts,
    cmdclass={"build_py": FilteredBuildPy},
    version=FULL_VERSION,
    package_data={
        "eiq_genai_flow": [
            "assets/*.wav",  # earcon *.wav files
            "benchmark/data/*.pkl",  # rag db files
            "benchmark/data/*.txt",  # question list files
        ],
    },
)
