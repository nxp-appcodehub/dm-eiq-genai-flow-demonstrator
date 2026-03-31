#!/bin/bash

# Copyright 2025-2026 NXP
# NXP Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.

GUI_NAME="chat_interface"
GUI_NAME_PY="$GUI_NAME.py"
EGF_NAME="eiq_genai_flow.py"
EGF_EXEC="python3"

DIR_PATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# Find EGF_PATH by locating the EGF_NAME file
PROJECT_ROOT="$(cd -- "${DIR_PATH}/../../../" &>/dev/null && pwd)"
EGF_PATH="$(find "$PROJECT_ROOT" -name "$EGF_NAME" 2>/dev/null | head -1)"

GUI_PATH="$PROJECT_ROOT/gui/modules/$GUI_NAME"
GUI_SRC_PATH="$GUI_PATH/src/$GUI_NAME"

EGF_INPUT_MODE="-i $GUI_NAME"
EGF_OUTPUT_MODE="-o tts"
EGF_USE_RAG=true
EGF_MODEL="-m danube-500M-q8"
EGF_ASR_TYPE="-a moonshine-base"
EGF_VERBOSE_MODE=false
EGF_USE_NEUTRON=true

# ALSA devices. Set to "plughw:CARD= + Card name" (i,e: "plughw:CARD=Talk"). See run "arecord -l" or "aplay -l" to see card names, or leave "" for auto-detection
EGF_CAPTURE_DEVICE=""
EGF_PLAYBACK_DEVICE=""

# Queue names as defined in the config.py file
QUEUE_EGF_TO_GUI="/dev/mqueue/egf_to_gui"
QUEUE_GUI_TO_EGF="/dev/mqueue/gui_to_egf"
# VITapp name
VIT_EXE="vit"
# VIT Model Path
VIT_MODEL="-w vit/models/VIT_Model_en.bin"

# Default flag value
kill_flag=false

while getopts "k" opt; do
  case ${opt} in
    k )
      echo "The -k flag is set. Will only kill the processes and clean the queues..."
      kill_flag=true
      ;;
    \? )
      echo "Usage: $(basename "$0") [-k]"
      echo "  -k  Kill only the processes and exit."
      exit 1
      ;;
  esac
done

# Shift positional arguments
shift $((OPTIND -1))

queues=(
    ${QUEUE_EGF_TO_GUI}
    ${QUEUE_GUI_TO_EGF}
)

# Loop through the list and delete if the file exists
for queue in "${queues[@]}"; do
    if [ -e "$queue" ]; then
        rm "$queue"
        echo "$queue deleted."
    else
        echo "$queue does not exist."
    fi
done

# Check and install GUI dependencies
echo "Checking $GUI_NAME dependencies..."
if [ -f "$GUI_PATH/pyproject.toml" ]; then
    # Check if dependencies are installed
    DEPS_MISSING=false

    # Check if GUI itself is importable
    if ! python3 -c "import $GUI_NAME" 2>/dev/null; then
        echo "Missing package: $GUI_NAME"
        DEPS_MISSING=true
    fi

    # Install only if dependencies are missing
    if [ "$DEPS_MISSING" = true ]; then
        echo "Installing/updating $GUI_NAME package..."
        pip install -e "$GUI_PATH"

        if [ $? -ne 0 ]; then
            echo "Failed to install $GUI_NAME package. Please check your environment."
            exit 1
        fi
    else
        echo "$GUI_NAME dependencies seem installed."
    fi
else
    echo "Error: pyproject.toml not found at $GUI_PATH"
    exit 1
fi

# Check for running instances of the vit executable using pgrep
PROCESS_IDS=$(pgrep -f "$VIT_EXE")
if [ -n "$PROCESS_IDS" ]; then
    echo "Killing existing instances of $VIT_EXE..."
    kill -9 $PROCESS_IDS
fi

# Kill the GUI Python script
GUI_PROCESS_IDS=$(pgrep -f "$GUI_NAME")
if [ -n "$GUI_PROCESS_IDS" ]; then
    echo "Killing existing instances of $GUI_NAME..."
    kill -9 $GUI_PROCESS_IDS
fi

# Kill the EGF Python script
EGF_PROCESS_IDS=$(pgrep -f "$EGF_NAME")
if [ -n "$EGF_PROCESS_IDS" ]; then
    echo "Killing existing instances of $EGF_NAME..."
    kill -9 $EGF_PROCESS_IDS
fi

if [ "$kill_flag" = true ]; then
    exit 0
fi

if [[ "$EGF_USE_NEUTRON" == "true" ]]; then
    echo "Neutron NPU will be used if the BSP allows it..."
    USE_NEUTRON=-n
fi

if [[ "$EGF_USE_RAG" == "true" ]]; then
    USE_RAG=-r
fi

if [[ "$EGF_VERBOSE_MODE" == "true" ]]; then
    VERBOSE_MODE=-v
fi

if [[ "$EGF_CAPTURE_DEVICE" == "" ]]; then
    CAPTURE_DEVICE=""
else
    CAPTURE_DEVICE="--capture-device $CAPTURE_DEVICE"
fi

if [[ "$EGF_PLAYBACK_DEVICE" == "" ]]; then
    PLAYBACK_DEVICE=""
else
    PLAYBACK_DEVICE="--playback-device $PLAYBACK_DEVICE"
fi

cd $PROJECT_ROOT

echo "Launching $GUI_NAME..."
python3 $GUI_SRC_PATH/$GUI_NAME_PY &

echo "Launching $EGF_NAME..."
$EGF_EXEC $EGF_NAME $EGF_INPUT_MODE $USE_RAG $EGF_OUTPUT_MODE $EGF_MODEL $EGF_ASR_TYPE $VERBOSE_MODE $USE_NEUTRON $PLAYBACK_DEVICE $CAPTURE_DEVICE $VIT_MODEL
