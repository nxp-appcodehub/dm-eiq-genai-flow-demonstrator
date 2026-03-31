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
import torch
import logging
import argparse
import numpy as np
import alsaaudio as aa
from pathlib import Path
from collections import deque
from speech_to_text.vad import VAD
from speech_to_text.speech_to_text import SpeechToText
from speech_to_text.models.model_config import ModelConfig
from shared_utils.utils import parent_dir, get_leaf_classes, setup_logging, get_default_playback_device
from speech_to_text.utils.utils import consume_buffer, load_audio

logger = logging.getLogger(__name__)


def mic_to_text(stt: SpeechToText, vad: VAD, timeout_s=20):
    assert stt.sample_rate == vad.sample_rate
    sample_rate = stt.sample_rate
    window_size_s = vad.required_samples / sample_rate

    # setup buffers:
    pre_vad_buffer = deque(maxlen=vad.pre_vad_samples)
    speech_to_process = deque()

    # setup alsa:
    capture_device = get_default_playback_device()
    pcm = aa.PCM(
        type=aa.PCM_CAPTURE,
        rate=sample_rate,
        channels=1,
        device=capture_device,
        format=aa.PCM_FORMAT_FLOAT_LE,
        periodsize=vad.required_samples
    )

    logger.info("Recording from microphone...\n")

    speech_detected = False
    total_frames = int(timeout_s / window_size_s)
    for frame_idx in range(total_frames):
        l, data = pcm.read()
        if l > 0:
            audio_data = np.frombuffer(data, dtype=np.float32).copy()
            speech_timestamps = vad(audio_data)

            # VAD is either triggered or ending
            if speech_timestamps:
                # vad is triggered
                if not speech_detected:
                    if 'start' in speech_timestamps:
                        speech_detected = True
                        pre_vad = consume_buffer(pre_vad_buffer, len(pre_vad_buffer))
                        speech_to_process.extend(pre_vad)
                        speech_to_process.extend(audio_data)

                # vad is ending
                elif 'end' in speech_timestamps:
                    speech_detected = False
                    chunk = consume_buffer(speech_to_process, len(speech_to_process))
                    chunk = chunk[:vad.post_vad_samples]
                    yield from stt(chunk, ending=True)

            # vad is up
            elif speech_detected:
                speech_to_process.extend(audio_data)

            # vad is down
            else:
                pre_vad_buffer.extend(audio_data)  # audio_data to prepend on first speech detection

            # enough sample in self.speech_to_process to send to self.streamer
            if len(speech_to_process) >= stt.current_chunk_length:
                chunk = consume_buffer(speech_to_process, stt.current_chunk_length)

                yield from stt(chunk)

    logger.info('Timeout!')


def file_to_text(stt: SpeechToText, vad: VAD, audio_file: str | Path):
    audio_input, _ = load_audio(audio_file, sample_rate=stt.sample_rate)
    audio_input = audio_input[stt.audio_channel_index]

    audio_input = vad.process(audio_input)
    # no speech detected
    if audio_input is None:
        logger.warning(f'Audio {audio_file} is empty!')
        return ''

    if stt.model_type == 'whisper':
        input_chunks = stt.audio_processor.split(audio_input)
    else:
        input_chunks = torch.split(audio_input, stt.audio_chunk_length)

    for chunk_idx, chunk in enumerate(input_chunks):
        ending = chunk_idx == len(input_chunks) - 1

        yield from stt(chunk, ending=ending)


def parse_args():
    parser = argparse.ArgumentParser(description="Audio processing options")

    supported_models = [cls.name for cls in get_leaf_classes(ModelConfig)]
    logging_levels = list(logging._levelToName.values())

    parser.add_argument(
        '-m', '--model',
        type=str.lower,
        choices=supported_models,
        default='whisper-small.en',
        help="Speech-To-Text model to use."
    )
    parser.add_argument(
        '-f', '--file',
        nargs='?',
        const=os.path.join(parent_dir(__file__, level=3), 'tests/data/sample_en.wav'),
        metavar='path/to/the/input_file',
        help="Optional: path to the input audio file. "
             "If given without a value, a default test file will be used. "
             "If omitted, the microphone will be used."
    )
    parser.add_argument(
        '-v', '--verbose',
        type=str.upper,
        choices=logging_levels,
        default='INFO',
        help="Specify the verbose mode."
    )
    args = parser.parse_args()

    # Determine source automatically
    if args.file:
        args.source = 'file'
    else:
        args.source = 'mic'

    # Verify the used model is compatible with the selected language
    return args


def main():
    args = parse_args()
    setup_logging(level=logging._nameToLevel[args.verbose], root_path=parent_dir(__file__, level=3))

    stt = SpeechToText(
        args.model,
        language='English',
        task='transcribe',
    )

    vad = VAD()

    match args.source:
        case 'mic':
            for text in mic_to_text(stt, vad, timeout_s=20):
                pass

        case 'file':
            logger.info(f'Using file: {args.file}\n')
            for text in file_to_text(stt, vad, audio_file=args.file):
                pass


if __name__ == '__main__':
    main()
