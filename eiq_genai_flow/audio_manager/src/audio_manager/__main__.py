#!/usr/bin/env python3
# Copyright 2026 NXP
# NXP Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.

"""
Audio Manager - Command Line Interface

Run with: python -m audio_manager [command] [options]
"""

import sys
import argparse
import time
import numpy as np
from audio_manager.audio_factory import (
    create_audio_manager,
    get_available_backends,
    print_backend_info,
)
from audio_manager.audio_manager_base import (
    CaptureConfig,
    PlaybackConfig,
    ReaderConfig,
)


def generate_tone(frequency, duration, sample_rate=16000):
    """Generate a simple sine wave tone."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    # Apply fade in/out to avoid clicks
    fade_samples = int(0.01 * sample_rate)  # 10ms fade
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    tone[:fade_samples] *= fade_in
    tone[-fade_samples:] *= fade_out
    return tone


def cmd_info(args):
    """Display information about available backends."""
    print("\n" + "=" * 60)
    print("Audio Manager - Backend Information")
    print("=" * 60)
    print_backend_info()

    backends = get_available_backends()
    print("\nDetailed Status:")
    for backend, available in backends.items():
        if available:
            try:
                audio_manager = create_audio_manager(backend=backend)
                print(f"  {backend}: {audio_manager.__class__.__name__}")
                audio_manager.shutdown()
            except Exception as e:
                print(f"  {backend}: Error - {e}")
        else:
            print(f"  {backend}: Not available")
    print()


def cmd_test(args):
    """Test audio capture and playback."""
    print("\n" + "=" * 60)
    print("Audio Manager - Capture & Playback Test")
    print("=" * 60)

    capture_config = CaptureConfig(
        capture_device=args.capture_device,
        sample_rate=args.sample_rate,
        channels=args.channels,
        format=args.format,
    )

    playback_config = PlaybackConfig(
        playback_device=args.playback_device,
        sample_rate=args.sample_rate,
        channels=1,
        format=args.format,
    )

    audio_manager = create_audio_manager(
        backend=args.backend,
        capture_config=capture_config,
        playback_config=playback_config,
    )

    print(f"\nBackend: {audio_manager.__class__.__name__}")
    print(f"Capture: {args.capture_device} ({args.sample_rate}Hz, {args.channels}ch)")
    print(f"Playback: {args.playback_device} ({args.sample_rate}Hz)")

    # Start capture and playback
    audio_manager.start_capture()
    audio_manager.start_playback()

    # Register reader
    reader_config = ReaderConfig(
        channels=1,
        format="F32LE",
        channel_indices=[0],
    )
    reader = audio_manager.register_reader("test_reader", config=reader_config)
    reader.enable(sync_to_current=True)

    # Record
    duration = args.duration
    print(f"\nRecording for {duration} seconds...")
    for i in range(duration, 0, -1):
        print(f"  {i}...", end="\r", flush=True)
        time.sleep(1)
    print("\nRecording complete!")

    # Read captured audio
    samples_to_read = args.sample_rate * duration
    audio_data = reader.read(samples_to_read)

    if audio_data is not None:
        print(f"\nCaptured {len(audio_data)} samples")
        print(f"  Shape: {audio_data.shape}")
        print(f"  Dtype: {audio_data.dtype}")
        print(f"  Range: [{audio_data.min():.3f}, {audio_data.max():.3f}]")

        # Play back asynchronously and wait for completion
        print("\nPlaying back captured audio...")
        audio_manager.play_audio_async(audio_data, sample_rate=args.sample_rate)

        # Wait for playback to complete
        while not audio_manager.is_playback_complete():
            time.sleep(0.1)

        print("Playback complete!")
    else:
        print("\nNo audio data captured")

    audio_manager.shutdown()
    print("\nTest complete\n")


def cmd_record(args):
    """Record audio to a WAV file."""
    import os

    print("\n" + "=" * 60)
    print("Audio Manager - Record to File")
    print("=" * 60)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    capture_config = CaptureConfig(
        capture_device=args.capture_device,
        sample_rate=args.sample_rate,
        channels=args.channels,
        format=args.format,
        save_audio=True,
        audio_save_path=output_dir,
    )

    audio_manager = create_audio_manager(
        backend=args.backend,
        capture_config=capture_config,
    )

    print(f"\nBackend: {audio_manager.__class__.__name__}")
    print(f"Capture: {args.capture_device} ({args.sample_rate}Hz, {args.channels}ch)")
    print(f"Output: {output_dir}")

    audio_manager.start_capture()

    duration = args.duration
    print(f"\nRecording for {duration} seconds...")
    for i in range(duration, 0, -1):
        print(f"  {i}...", end="\r", flush=True)
        time.sleep(1)

    print("\nRecording complete!")
    audio_manager.stop_capture()
    audio_manager.shutdown()

    print(f"\nWAV file saved to: {output_dir}")
    print()


def cmd_play_tone(args):
    """Play a test tone."""
    print("\n" + "=" * 60)
    print("Audio Manager - Play Test Tone")
    print("=" * 60)

    playback_config = PlaybackConfig(
        playback_device=args.playback_device,
        sample_rate=args.sample_rate,
        channels=1,
        format=args.format,
    )

    audio_manager = create_audio_manager(
        backend=args.backend,
        playback_config=playback_config,
    )

    print(f"\nBackend: {audio_manager.__class__.__name__}")
    print(f"Playback: {args.playback_device} ({args.sample_rate}Hz)")

    audio_manager.start_playback()

    # Generate and play tones
    frequencies = [262, 294, 330, 349, 392, 440, 494, 523]  # C major scale
    note_names = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"]

    print(f"\nPlaying {len(frequencies)} notes...")
    for freq, name in zip(frequencies, note_names):
        print(f"  Playing {name} ({freq}Hz)...", end="\r", flush=True)
        tone = generate_tone(freq, 0.3, args.sample_rate)
        audio_manager.play_audio_async(tone, sample_rate=args.sample_rate)
        # Wait for playback to complete
        while not audio_manager.is_playback_complete():
            time.sleep(0.1)

    print("\nPlayback complete!          ")
    audio_manager.shutdown()
    print()


def cmd_multi_reader(args):
    """Demonstrate multiple readers."""
    print("\n" + "=" * 60)
    print("Audio Manager - Multi-Reader Demo")
    print("=" * 60)

    capture_config = CaptureConfig(
        capture_device=args.capture_device,
        sample_rate=args.sample_rate,
        channels=2,
        format=args.format,
    )

    audio_manager = create_audio_manager(
        backend=args.backend,
        capture_config=capture_config,
    )

    print(f"\nBackend: {audio_manager.__class__.__name__}")
    print(f"Capture: {args.capture_device} ({args.sample_rate}Hz, 2ch)")

    audio_manager.start_capture()

    # Register multiple readers
    reader1 = audio_manager.register_reader(
        "reader1_ch0", config=ReaderConfig(channels=1, format="F32LE", channel_indices=[0])
    )
    reader1.enable(sync_to_current=True)

    reader2 = audio_manager.register_reader(
        "reader2_ch1", config=ReaderConfig(channels=1, format="F32LE", channel_indices=[1])
    )
    reader2.enable(sync_to_current=True)

    reader3 = audio_manager.register_reader(
        "reader3_stereo", config=ReaderConfig(channels=2, format="S16LE", channel_indices=[0, 1])
    )
    reader3.enable(sync_to_current=True)

    print("\nRegistered 3 readers:")
    print("  - Reader 1: Mono (ch0), F32LE")
    print("  - Reader 2: Mono (ch1), F32LE")
    print("  - Reader 3: Stereo, S16LE")

    # Capture
    duration = args.duration
    print(f"\nCapturing for {duration} seconds...")
    time.sleep(duration)

    # Read from all readers
    samples_to_read = args.sample_rate
    print(f"\nReading {samples_to_read} samples from each reader:")

    data1 = reader1.read(samples_to_read)
    if data1 is not None:
        print(f"  Reader 1: shape={data1.shape}, dtype={data1.dtype}, range=[{data1.min():.3f}, {data1.max():.3f}]")

    data2 = reader2.read(samples_to_read)
    if data2 is not None:
        print(f"  Reader 2: shape={data2.shape}, dtype={data2.dtype}, range=[{data2.min():.3f}, {data2.max():.3f}]")

    data3 = reader3.read(samples_to_read)
    if data3 is not None:
        print(f"  Reader 3: shape={data3.shape}, dtype={data3.dtype}, range=[{data3.min()}, {data3.max()}]")

    audio_manager.shutdown()
    print("\nDemo complete\n")


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Audio Manager - Unified audio capture and playback",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m audio_manager info
  python -m audio_manager test --duration 3
  python -m audio_manager record --duration 5 --output-dir ./recordings
  python -m audio_manager play-tone
  python -m audio_manager multi-reader --duration 2
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Info command
    parser_info = subparsers.add_parser("info", help="Show backend information")
    parser_info.set_defaults(func=cmd_info)

    # Test command
    parser_test = subparsers.add_parser("test", help="Test capture and playback")
    parser_test.add_argument(
        "--backend", default="auto", choices=["auto", "alsa", "gstreamer"], help="Audio backend to use (default: auto)"
    )
    parser_test.add_argument("--capture-device", default="default", help="Capture device name (default: default)")
    parser_test.add_argument("--playback-device", default="default", help="Playback device name (default: default)")
    parser_test.add_argument("--sample-rate", type=int, default=16000, help="Sample rate in Hz (default: 16000)")
    parser_test.add_argument("--channels", type=int, default=2, help="Number of channels (default: 2)")
    parser_test.add_argument("--format", default="S32LE", help="Audio format (default: S32LE)")
    parser_test.add_argument("--duration", type=int, default=3, help="Recording duration in seconds (default: 3)")
    parser_test.set_defaults(func=cmd_test)

    # Record command
    parser_record = subparsers.add_parser("record", help="Record audio to WAV file")
    parser_record.add_argument(
        "--backend", default="auto", choices=["auto", "alsa", "gstreamer"], help="Audio backend to use (default: auto)"
    )
    parser_record.add_argument("--capture-device", default="default", help="Capture device name (default: default)")
    parser_record.add_argument("--sample-rate", type=int, default=16000, help="Sample rate in Hz (default: 16000)")
    parser_record.add_argument("--channels", type=int, default=2, help="Number of channels (default: 2)")
    parser_record.add_argument("--format", default="S32LE", help="Audio format (default: S32LE)")
    parser_record.add_argument("--duration", type=int, default=5, help="Recording duration in seconds (default: 5)")
    parser_record.add_argument("--output-dir", default="./recordings", help="Output directory (default: ./recordings)")
    parser_record.set_defaults(func=cmd_record)

    # Play tone command
    parser_tone = subparsers.add_parser("play-tone", help="Play test tone")
    parser_tone.add_argument(
        "--backend", default="auto", choices=["auto", "alsa", "gstreamer"], help="Audio backend to use (default: auto)"
    )
    parser_tone.add_argument("--playback-device", default="default", help="Playback device name (default: default)")
    parser_tone.add_argument("--sample-rate", type=int, default=16000, help="Sample rate in Hz (default: 16000)")
    parser_tone.add_argument("--format", default="S32LE", help="Audio format (default: S32LE)")
    parser_tone.set_defaults(func=cmd_play_tone)

    # Multi-reader command
    parser_multi = subparsers.add_parser("multi-reader", help="Demonstrate multiple readers")
    parser_multi.add_argument(
        "--backend", default="auto", choices=["auto", "alsa", "gstreamer"], help="Audio backend to use (default: auto)"
    )
    parser_multi.add_argument("--capture-device", default="default", help="Capture device name (default: default)")
    parser_multi.add_argument("--sample-rate", type=int, default=16000, help="Sample rate in Hz (default: 16000)")
    parser_multi.add_argument("--format", default="S32LE", help="Audio format (default: S32LE)")
    parser_multi.add_argument("--duration", type=int, default=2, help="Capture duration in seconds (default: 2)")
    parser_multi.set_defaults(func=cmd_multi_reader)

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Execute command
    try:
        args.func(args)
        return 0
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
