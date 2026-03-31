# -*- coding: utf-8 -*-

# Copyright 2024-2026 NXP
# NXP Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.

import os
import logging
import argparse
import numpy as np
import alsaaudio as aa
from tts.model import TextToSpeech
from shared_utils.utils import setup_logging, get_default_playback_device
from tts.config import MultiSpeakerTTS16kHzConfig, MultiSpeakerTTS16kHzQuantConfig


if __name__ == '__main__':
    LOG_LEVELS = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO
    }
    parser = argparse.ArgumentParser(description='TTS Demo')
    parser.add_argument('-t', '--text', type=str, default="Hello world!", help='text to synthesize')
    parser.add_argument('-m', '--mode', type=str, default=TextToSpeech.get_default_mode(),
                        choices=TextToSpeech.get_mode_list(), help='TTS mode')
    parser.add_argument('-q', '--quantized', action='store_true', help='quantized model')
    parser.add_argument('-p', '--playback-device', type=str, default=get_default_playback_device(),
                        help='playback device (ex: "default", "plughw:CARD=name", "sysdefault:CARD=name", etc)')
    parser.add_argument('-s', '--speaker_id', type=int, default=24, help='speaker id between 1 and 904')
    parser.add_argument('--log-level', type=str, default="INFO", choices=LOG_LEVELS.keys(),
                        help='set the logging level')
    args = parser.parse_args()

    if args.quantized:
        config = MultiSpeakerTTS16kHzQuantConfig(speaker_id=args.speaker_id)
    else:
        config = MultiSpeakerTTS16kHzConfig(speaker_id=args.speaker_id)

    setup_logging(level=LOG_LEVELS[args.log_level])
    tts = TextToSpeech(config, mode=args.mode)
    audio_data = tts.generate(args.text)

    # audio playback using ALSA
    pcm = aa.PCM(
        type=aa.PCM_PLAYBACK,
        mode=aa.PCM_NORMAL,
        channels=2,
        periodsize=8192,
        rate=config.samplerate,
        device=args.playback_device,
        format=aa.PCM_FORMAT_FLOAT_LE
    )

    if isinstance(audio_data, np.ndarray):
        stereo_audio_data = np.repeat(audio_data, 2)
        pcm.write(stereo_audio_data.tobytes())
    else:  # streamed audio
        for data_chunk in audio_data:
            stereo_audio_data = np.repeat(data_chunk, 2)
            written_frames = pcm.write(stereo_audio_data.tobytes())
            if written_frames < 0:  # avoid problem when streaming
                pcm.write(stereo_audio_data.tobytes())
    pcm.close()

os._exit(0)  # exit the program and avoid waiting for the timeout to end
