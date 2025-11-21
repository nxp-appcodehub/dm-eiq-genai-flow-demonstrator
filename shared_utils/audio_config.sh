#!/bin/bash

# Copyright 2025 NXP
# NXP Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.

audio_setup() {
  mode=$1
  device=$2
  if [[ $mode == "capture" ]]; then
    if [[ $device == "wm8962audio" ]]; then
      amixer -c wm8962audio sset 'Capture' 60
    elif [[ $device == "wm8960audio" ]]; then
      amixer -c micfilaudio cset name='MICFIL Quality Select' 'High'
      amixer -c wm8960audio sset 'Capture' 60
    else
      echo "No specific capture config applied for $device, you can customize it in ${BASH_SOURCE[0]} "
    fi
  elif [[ $mode == "playback" ]]; then
    if [[ $device == "wm8962audio" ]]; then
      amixer -c wm8962audio set 'Headphone' 110 on
    elif [[ $device == "wm8960audio" ]]; then
      amixer -c wm8960audio set 'Headphone' 125
    else
      echo "No specific playback config applied for $device, you can customize it in ${BASH_SOURCE[0]} "
    fi
  else
    echo "Error: '$mode' mode doesn't exist, use 'capture' or 'playback'"
    exit 1
  fi
}

if [ $# -eq 2 ]; then
  audio_setup "$1" "$2"
else
  echo "ERROR!"
  echo "setup audio for capture or playback"
  echo "usage: audio_config.sh {capture,playback} device_name"
fi
