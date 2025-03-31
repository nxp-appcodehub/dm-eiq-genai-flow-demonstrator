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


audio_card_list() {
  cards=$(awk -F '[][]' '/\[.*\]/ {print $2}' /proc/asound/cards)
  echo "$cards"
}

get_audio_card() {
  supported_cards=$1
  device_cards=$2
  for s_card in $supported_cards; do
    for d_card in $device_cards; do
      if [[ "$s_card" == "$d_card" ]]; then
        echo "$d_card"
        return 1
      fi
    done
  done
  echo "default"
}

get_device() {
  # get architecture type
  arch=$(uname -m)
  soc_id=/sys/devices/soc0/soc_id
  if [ "$arch" == "x86_64" ]; then
    echo "PC"
  elif [ -f "$soc_id" ]; then
    cat $soc_id
  else
    echo "ERROR"
  fi
}

audio_setup() {
  mode=$1
  device=$(get_device)
  supported_cards="wm8962audio wm8960audio"
  cards=$(audio_card_list)
  cur_card=$(get_audio_card "$supported_cards" "$cards")
  if [[ "$device" != "PC" ]]; then
    echo "sysdefault:CARD=$cur_card"
    if [[ $mode == "capture" ]]; then
      if [[ $cur_card == "wm8962audio" ]]; then
        amixer -c wm8962audio sset 'Capture' 60
      elif [[ $cur_card == "wm8960audio" ]]; then
        amixer -c micfilaudio cset name='MICFIL Quality Select' 'High'
        amixer -c wm8960audio sset 'Capture' 60
      else
        echo "'$cur_card' audio card doesn't exist"
      fi
    elif [[ $mode == "playback" ]]; then
      if [[ $cur_card == "wm8962audio" ]]; then
        amixer -c wm8962audio set 'Headphone' 110 on
      elif [[ $cur_card == "wm8960audio" ]]; then
        amixer -c wm8960audio set 'Headphone' 125
      else
        echo "'$cur_card' audio card doesn't exist"
      fi
    else
      echo "'$mode' mode doesn't exist use 'capture' or 'playback'"
    fi
  else # PC
    echo "$cur_card"
  fi
}

if [ $# -eq 1 ]; then
  audio_setup "$1"
else
  echo "ERROR!"
  echo "setup audio for capture or playback"
  echo "usage: audio_config.sh {capture,playback}"
fi
