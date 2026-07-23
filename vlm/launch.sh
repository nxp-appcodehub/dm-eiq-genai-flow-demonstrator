#!/usr/bin/env bash
# Copyright 2026 NXP
# NXP Confidential and Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.

GUI_PROCESS_IDS=$(pgrep -f "chat_interface")
if [ -n "$GUI_PROCESS_IDS" ]; then
    echo "Killing existing instances of chat_interface..."
    kill -9 $GUI_PROCESS_IDS
fi

echo "Launching GUI..."
python3 -m chat_interface -l WARNING & #gui/modules/chat_interface/src/chat_interface/chat_interface.py &

echo "Launching VLM..."
python3 -m vlm  "$@" -g
