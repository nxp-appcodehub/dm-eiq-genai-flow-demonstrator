# Copyright 2025-2026 NXP
# NXP Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.

"""
VIT Example Application
"""

import os
import argparse
import logging
import sys
import wave
from pathlib import Path

import alsaaudio as aa
import numpy as np

from shared_utils.utils import get_default_capture_device, setup_logging, parent_dir
from vit.vit import VIT

logger = logging.getLogger(__name__)


def process_audio_file(vit, audio_file_path):
    """Process audio from WAV file"""
    logger.info(f"Processing audio file: {audio_file_path}")

    try:
        with wave.open(audio_file_path, "rb") as wav_file:
            # Check audio format
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            framerate = wav_file.getframerate()
            n_frames = wav_file.getnframes()

            logger.info(f"  Channels: {channels}")
            logger.info(f"  Sample Width: {sample_width} bytes")
            logger.info(f"  Frame Rate: {framerate} Hz")
            logger.info(f"  Total Frames: {n_frames}")
            logger.info(f"  Duration: {n_frames / framerate:.2f} seconds")

            if framerate != VIT.SAMPLE_RATE:
                logger.warning(f"Sample rate mismatch! Expected {VIT.VIT_SAMPLE_RATE} Hz")

            if sample_width != 2:
                logger.warning("Expected 16-bit audio (2 bytes per sample)")

            # Read all audio data
            audio_data = wav_file.readframes(n_frames)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)

            # Process in frames
            frame_size = VIT.SAMPLES_PER_FRAME
            total_frames = len(audio_array) // frame_size

            logger.info(f"Processing {total_frames} frames...")
            logger.info("-" * 60)

            detections_ww = 0
            detections_vc = 0
            input_buffer = np.zeros(frame_size, dtype=np.int16)

            for frame_idx in range(total_frames):
                start_idx = frame_idx * frame_size
                end_idx = start_idx + frame_size

                input_buffer = audio_array[start_idx:end_idx]

                detection_type, info = vit(
                    input_buffer,  # mic reference
                    input_buffer,  # input data (after AFE)
                )

                if detection_type == "wakeword":
                    detections_ww += 1
                    logger.info("  WAKE WORD DETECTED!")
                    logger.info(f"  ID: {info['id']}")
                    logger.info(f"  Name: {info['name']}")
                    logger.info(f"  Energy: {info['energy']:.2f} dB")
                    logger.info(f"  Start_offset: {info['start_offset']}")
                    logger.info(f"  End_offset: {info['end_offset']}\n")

                elif detection_type == "command":
                    detections_vc += info["name"] != "UNKNOWN"
                    logger.info("  COMMAND DETECTED!")
                    logger.info(f"  ID: {info['id']}")
                    logger.info(f"  Name: {info['name']}")

                # Progress indicator
                if (frame_idx + 1) % 100 == 0:
                    progress = (frame_idx + 1) / total_frames * 100
                    logger.info(f"Progress: {progress:.1f}% ({frame_idx + 1}/{total_frames} frames)")

            logger.info("-" * 60)
            logger.info(f"Processing complete! Total detections: WW - {detections_ww}, VC - {detections_vc}")

            return

    except FileNotFoundError:
        logger.info(f"Error: Audio file not found: {audio_file_path}")
        return None
    except Exception as e:
        logger.info(f"Error processing audio file: {e}")
        import traceback

        traceback.logger.info_exc()
        return None


def process_realtime_audio(vit, duration_seconds=10):
    """Process real-time audio from microphone"""

    logger.info(f"\nProcessing real-time audio from microphone for {duration_seconds} seconds...")

    frame_size = VIT.SAMPLES_PER_FRAME
    sample_rate = VIT.SAMPLE_RATE
    frame_duration_s = frame_size / sample_rate  # 30ms per frame

    # setup alsa
    capture_device = get_default_capture_device()
    pcm = aa.PCM(
        type=aa.PCM_CAPTURE,
        rate=sample_rate,
        channels=VIT.NUMBER_OF_CHANNELS,
        periodsize=int(sample_rate * frame_duration_s),
        device=capture_device,
        format=aa.PCM_FORMAT_FLOAT_LE,
    )

    logger.info("Recording from microphone...")
    logger.info(f"Sample rate: {sample_rate} Hz")
    logger.info(f"Frame size: {frame_size} samples")
    logger.info(f"Frame duration: {frame_duration_s} seconds")
    logger.info("-" * 60)

    detections_ww = 0
    detections_vc = 0

    total_frames = int(duration_seconds / frame_duration_s)

    try:
        for frame_idx in range(total_frames):
            # Read audio frame from microphone
            l, data = pcm.read()
            if l > 0:
                frame_data = np.frombuffer(data, dtype=np.float32)
                frame_data = (frame_data * 32767).astype(np.int16)

            # Process frame
            detection_type, info = vit(
                frame_data,  # data from mic reference
                frame_data,  # input data (after AFE)
            )

            if detection_type == "wakeword":
                detections_ww += 1
                logger.info("\n")
                print("  WAKE WORD DETECTED!")
                logger.info("  WAKE WORD DETECTED!")
                logger.info(f"  ID: {info['id']}")
                logger.info(f"  Name: {info['name']}")
                logger.info(f"  Energy: {info['energy']:.2f} dB")
                logger.info(f"  Start_offset: {info['start_offset']}")
                logger.info(f"  End_offset: {info['end_offset']}\n")

            elif detection_type == "command":
                detections_vc += info["name"] != "UNKNOWN"
                logger.info("\n")
                logger.info("  COMMAND DETECTED!")
                logger.info(f"  ID: {info['id']}")
                logger.info(f"  Name: {info['name']}")

            # Progress indicator
            if (frame_idx + 1) % 100 == 0:
                elapsed = (frame_idx + 1) * 0.03
                logger.info(f"Time: {elapsed:.1f}s / {duration_seconds}s")

    except KeyboardInterrupt:
        logger.info("Recording interrupted")
        raise  # propagate to main()

    finally:
        # Cleanup: Close the PCM device to release audio hardware
        if pcm is not None:
            try:
                pcm.close()
                logger.info("Audio input device closed successfully")
            except Exception as e:
                logger.info(f"Error closing audio device: {e}")

    logger.info("-" * 60)
    logger.info(f"Processing complete! Total detections: WW - {detections_ww}, VC - {detections_vc}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        prog="python -m vit",
        description="VIT (Voice Intelligent Technology) Example Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
    # Process default audio file
    python -m vit

    # Process a specific WAV file
    python -m vit -f ./tests/HeyNXP_en.wav

    # Record from microphone for 30 seconds (default)
    python -m vit -m ./src/vit/models/VIT_Model_en.bin

    # Record from microphone for 60 seconds
    python -m vit -r 60

    # Set logging level
    python -m vit -v warning

AUDIO REQUIREMENTS:
    - Format: WAV (16-bit PCM)
    - Sample Rate: 16000 Hz
    - Channels: 1 (mono)

DETECTION TYPES:
    - Wake Word (WW): Detects configured wake words (e.g., "Hey NXP")

CONTROLS:
    - Press Ctrl+C to stop real-time recording
        """,
    )

    parser.add_argument(
        "--file",
        "-f",
        type=str,
        default=os.path.join(parent_dir(__file__, level=3), "tests/HeyNXP_en.wav"),
        metavar="<audio_file>",
        help="Process a specific audio file (WAV format)",
    )

    parser.add_argument(
        "--realtime",
        "-r",
        nargs="?",
        type=int,
        const=30,
        metavar="<duration>",
        help="Process real-time audio from microphone (default duration: 30 seconds)",
    )

    model_path = os.path.join(os.path.dirname(__file__), "models/VIT_Model_en.bin")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=model_path,
        metavar="<model_path>",
        help=f"Path to VIT model file (default: {model_path})",
    )

    parser.add_argument(
        "--operating-mode",
        type=str,
        choices=["wakeword", "wakeword_command"],
        default="wakeword",
        metavar="<mode>",
        help="Set the operating mode. Choose between: wakeword, wakeword_command (default: wakeword)",
    )

    parser.add_argument(
        "--noise-floor",
        type=float,
        default=-80.0,
        metavar="<dB>",
        help="Input noise floor in dB (default: -80.0)",
    )

    parser.add_argument(
        "--noise-threshold",
        type=float,
        default=10.0,
        metavar="<dB>",
        help="Noise floor threshold in dB (default: 10.0)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        type=str,
        choices=["critical", "error", "warning", "info", "debug", "notset"],
        default="info",
        metavar="<level>",
        help="Set the verbose level. Choose between: critical, error, warning, info, debug, notset (default: info)",
    )

    args = parser.parse_args()

    # Determine mode based on arguments
    if args.realtime is not None:
        args.mode = "realtime"
        args.duration = args.realtime
    else:
        args.mode = "file"

    return args


def main():
    """Main application entry point"""

    # Parse arguments
    args = parse_args()

    # Map verbose level string to logging level
    log_level_map = {
        "critical": logging.CRITICAL,
        "error": logging.ERROR,
        "warning": logging.WARNING,
        "info": logging.INFO,
        "debug": logging.DEBUG,
        "notset": logging.NOTSET,
    }

    log_level = log_level_map[args.verbose]
    setup_logging(level=log_level)

    logger.info("\n")
    logger.info("=" * 60)
    logger.info("VIT Python Example Application")
    logger.info("=" * 60)

    # Create application
    app = VIT(
        args.model,
        operating_mode=args.operating_mode,
        noise_floor=args.noise_floor,
        noise_threshold=args.noise_threshold,
    )

    try:
        # Get status
        app.get_status()

        # Execute based on mode
        if args.mode == "realtime":
            # Real-time microphone processing
            process_realtime_audio(app, duration_seconds=args.duration)
        elif args.mode == "file":
            # Process audio file
            if Path(args.file).exists():
                process_audio_file(app, args.file)
            else:
                logger.error(f"Error: Audio file not found: {args.file}")
                sys.exit(1)

        logger.info("VIT Example Application completed successfully!")

    except KeyboardInterrupt:
        logger.info("\n\nInterrupted by user")
    except Exception:
        logger.exception("An error occurred")  # Automatically logs error + traceback
        sys.exit(1)
    finally:
        logger.info("Cleaning up...")
        app.delete_instance()


if __name__ == "__main__":
    main()
