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
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def add_noise_to_audio(audio_data, snr_db=30, noise_type="white"):
    """
    Add noise to audio signal at specified SNR.

    Args:
        audio_data (np.ndarray): Clean audio signal
        snr_db (float): Signal-to-Noise Ratio in dB (higher = less noise)
                       20dB = moderate noise (realistic)
                       10dB = noisy environment
                       30dB = quiet background noise
        noise_type (str): Type of noise ('white', 'pink', 'brown')

    Returns:
        np.ndarray: Audio with added noise

    Example SNR values:
        - 30 dB: Very quiet background
        - 20 dB: Moderate noise (office/café)
        - 10 dB: Noisy environment (street/crowd)
        - 5 dB: Very noisy (loud machinery)
    """
    # Calculate signal power
    signal_power = np.mean(audio_data**2)

    # Calculate noise power based on desired SNR
    # SNR(dB) = 10 * log10(signal_power / noise_power)
    # noise_power = signal_power / (10 ^ (SNR/10))
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    # Generate noise based on type
    if noise_type == "white":
        # White noise (flat spectrum)
        noise = np.random.randn(len(audio_data))
    elif noise_type == "pink":
        # Pink noise (1/f spectrum - more natural sounding)
        noise = _generate_pink_noise(len(audio_data))
    elif noise_type == "brown":
        # Brown noise (1/f^2 spectrum - deeper sound)
        noise = _generate_brown_noise(len(audio_data))
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    # Scale noise to desired power
    noise = noise / np.sqrt(np.mean(noise**2))  # Normalize
    noise = noise * np.sqrt(noise_power)  # Scale to target power

    # Mix signal with noise
    noisy_audio = audio_data + noise

    # Prevent clipping by normalizing if needed
    max_val = np.max(np.abs(noisy_audio))
    if max_val > 1.0:
        noisy_audio = noisy_audio / max_val * 0.95  # Leave some headroom

    return noisy_audio.astype(audio_data.dtype)


def generate_noise(duration_sec, sample_rate, noise_type="white", amplitude=0.1):
    """
    Generate pure noise for a given duration.

    Args:
        duration_sec (float): Duration in seconds
        sample_rate (int): Sample rate in Hz
        noise_type (str): Type of noise ('white', 'pink', 'brown')
        amplitude (float): Noise amplitude (0.0 to 1.0, default 0.1 for moderate level)

    Returns:
        np.ndarray: Generated noise signal
    """
    num_samples = int(duration_sec * sample_rate)

    if noise_type == "white":
        noise = np.random.randn(num_samples)
    elif noise_type == "pink":
        noise = _generate_pink_noise(num_samples)
    elif noise_type == "brown":
        noise = _generate_brown_noise(num_samples)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    # Normalize and scale to desired amplitude
    noise = noise / np.max(np.abs(noise))  # Normalize to [-1, 1]
    noise = noise * amplitude  # Scale to target amplitude

    return noise.astype(np.float32)


def _generate_pink_noise(length):
    """Generate pink noise (1/f spectrum)."""
    # Simple pink noise using the Voss-McCartney algorithm
    white = np.random.randn(length)
    # Apply simple low-pass filtering for pink effect
    pink = np.zeros(length)
    pink[0] = white[0]
    for i in range(1, length):
        pink[i] = 0.99 * pink[i - 1] + 0.01 * white[i]
    return pink


def _generate_brown_noise(length):
    """Generate brown noise (1/f^2 spectrum)."""
    white = np.random.randn(length)
    brown = np.cumsum(white)  # Integrate white noise
    return brown / np.max(np.abs(brown))  # Normalize


def questions_to_wav(
    questions_file: str,
    output_path: str,
    wake_word: str = None,
    wake_word_once_by_speaker: bool = False,
    add_noise: bool = False,
    snr_db: float = 20.0,
    noise_type: str = "white",
    noise_prefix_duration: float = 0.5,
):
    """
    Reads questions from a file, generates speech for each question using a TextToSpeech model,
    optionally adds noise prefix and background noise, and saves the audio to WAV files.

    Args:
        questions_file (str): Path to the text file containing questions (one question per line).
        output_path (str): Directory where the generated WAV files will be saved.
        wake_word (str): Optional wake word to prefix each question (e.g., "Hey NXP").
        add_noise (bool): Whether to add background noise to the audio.
        snr_db (float): Signal-to-Noise Ratio in dB (only used if add_noise=True).
                       Lower = more noise. Typical: 20dB (moderate), 10dB (noisy).
        noise_type (str): Type of noise ('white', 'pink', 'brown').
        noise_prefix_duration (float): Duration of pure noise before audio starts (seconds).
                                       Default 0.5s. Use 0.0 to disable.
    """
    from tts.model import TextToSpeech
    from tts.config import MultiSpeakerTTS16kHzConfig
    import soundfile as sf

    # Read question file
    try:
        with open(questions_file, "r", encoding="utf-8") as file:
            lines = [line.rstrip() for line in file]
    except FileNotFoundError:
        print(f"Error: Questions file '{questions_file}' not found.")
        return
    except Exception as e:
        print(f"Error reading questions file: {e}")
        return

    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    print(f"Output directory '{output_path}' ensured.")

    if add_noise:
        print(f"Adding {noise_type} noise at SNR={snr_db}dB")
    if noise_prefix_duration > 0:
        print(f"Adding {noise_prefix_duration}s noise prefix before each audio")

    # Initialize TTS with first speaker
    current_speaker_id = 1
    config = MultiSpeakerTTS16kHzConfig(speaker_id=current_speaker_id)
    tts = TextToSpeech(config)

    # Generate audio for each question
    for i, question in enumerate(lines):
        if i > 0:
            if wake_word_once_by_speaker:
                # Increment speaker_id every 2 questions
                if i % 2 == 0 :
                    current_speaker_id += 1
            else :
                # Increment speaker_id for each question (to vary voices)
                current_speaker_id += 1

        # Prefix question with wake word if provided
        if wake_word_once_by_speaker:
            if i % 2 == 0:
                # Add wake word only for new speaker
                combined_text = f"{wake_word}. {question}"
                file_prefix = "ww_for_new_speaker_question"
            else:
                combined_text = question
                file_prefix = "ww_for_new_speaker_question"
        elif wake_word :
            # Add natural pause after wake word (comma or period)
            combined_text = f"{wake_word}. {question}"
            file_prefix = "ww_question"
        else:
            combined_text = question
            file_prefix = "question"

        print(f"Generating speech for question {i + 1}/{len(lines)} with speaker_id: {current_speaker_id}")

        try:
            # Generate clean audio
            audio_data = tts.generate(combined_text, speaker_id=current_speaker_id)

            # ===================================================================
            # NOISE HANDLING: Generate continuous noise at consistent power level
            # ===================================================================
            if add_noise or noise_prefix_duration > 0:
                # Calculate target noise power based on signal and SNR
                signal_power = np.mean(audio_data**2)
                snr_linear = 10 ** (snr_db / 10)
                noise_power = signal_power / snr_linear

                # Generate continuous noise for entire duration (prefix + audio)
                total_samples = int(noise_prefix_duration * config.samplerate) + len(audio_data)

                if noise_type == "white":
                    continuous_noise = np.random.randn(total_samples)
                elif noise_type == "pink":
                    continuous_noise = _generate_pink_noise(total_samples)
                elif noise_type == "brown":
                    continuous_noise = _generate_brown_noise(total_samples)
                else:
                    raise ValueError(f"Unknown noise type: {noise_type}")

                # Scale noise to target power (same calculation as add_noise_to_audio)
                continuous_noise = continuous_noise / np.sqrt(np.mean(continuous_noise**2))  # Normalize
                continuous_noise = continuous_noise * np.sqrt(noise_power)  # Scale to target power

                # Split noise into prefix and audio portions
                prefix_samples = int(noise_prefix_duration * config.samplerate)
                noise_prefix = continuous_noise[:prefix_samples]
                noise_for_audio = continuous_noise[prefix_samples:]

                # Mix noise with audio if requested
                if add_noise:
                    audio_data = audio_data + noise_for_audio

                    # Prevent clipping
                    max_val = np.max(np.abs(audio_data))
                    if max_val > 1.0:
                        audio_data = audio_data / max_val * 0.95

                # Add noise prefix if requested
                if noise_prefix_duration > 0:
                    audio_data = np.concatenate([noise_prefix, audio_data])

            # Save to file
            output_file = os.path.join(output_path, f"{file_prefix}_{i:03d}.wav")
            sf.write(output_file, audio_data, config.samplerate)

            duration = len(audio_data) / config.samplerate
            noise_info = f", {noise_type} noise SNR={snr_db}dB" if add_noise else ""
            prefix_info = f", {noise_prefix_duration}s noise prefix" if noise_prefix_duration > 0 else ""
            print(f"  → {output_file} ({duration:.2f}s{noise_info}{prefix_info})")
            if wake_word:
                print(f"     Text: '{combined_text}'")

        except Exception as e:
            print(f"Error generating speech for question '{question}': {e}")
            continue

    print(f"\nFinished generating {len(lines)} WAV files in '{output_path}'.")


def generate_wav_files(
    wav_dir,
    text_file_path,
    text_file_len,
    wake_word=None,
    wake_word_once_by_speaker=False,
    add_noise=False,
    snr_db=20.0,
    noise_type="white",
    noise_prefix_duration=0.8,
):
    """
    Generate WAV files from questions, optionally with wake word prefix, noise, and noise prefix.

    Args:
        wav_dir (str): Directory to save WAV files
        text_file_path (str): Path to questions text file
        text_file_len (int): Expected number of questions
        wake_word (str): Optional wake word to prefix each question
        add_noise (bool): Whether to add background noise
        snr_db (float): Signal-to-Noise Ratio in dB (lower = more noise)
        noise_type (str): Type of noise ('white', 'pink', 'brown')
        noise_prefix_duration (float): Duration of noise prefix in seconds (default 0.5s)

    Returns:
        list: Sorted list of generated WAV file paths
    """
    import glob

    # Determine file pattern based on wake word presence
    if wake_word_once_by_speaker:
        pattern = "ww_for_new_speaker_question_*.wav"
    elif wake_word:
        pattern = "ww_question_*.wav"
    else:
        pattern = "question_*.wav"

    # Check if files already exist
    wav_files = glob.glob(os.path.join(wav_dir, pattern))

    if len(wav_files) != text_file_len:
        print(f"{pattern} files count ({len(wav_files)}) does not match expected {text_file_len}, regenerating...")

        # Delete existing files of this pattern
        for wav_file in wav_files:
            try:
                os.remove(wav_file)
                print(f"Deleted: {wav_file}")
            except OSError as e:
                print(f"Error deleting {wav_file}: {e}")

        # Generate new WAV files
        if wake_word:
            print(f"\nGenerating WAV files with wake word prefix: '{wake_word}'")
        else:
            print("\nGenerating WAV files without wake word")

        if add_noise:
            print(f"Noise will be added: {noise_type} at SNR={snr_db}dB")
        if noise_prefix_duration > 0:
            print(f"Noise prefix: {noise_prefix_duration}s of {noise_type} noise before each audio")

        questions_to_wav(
            questions_file=text_file_path,
            output_path=wav_dir,
            wake_word=wake_word,
            wake_word_once_by_speaker=wake_word_once_by_speaker,
            add_noise=add_noise,
            snr_db=snr_db,
            noise_type=noise_type,
            noise_prefix_duration=noise_prefix_duration,
        )

        # Re-scan for generated files
        wav_files = glob.glob(os.path.join(wav_dir, pattern))

        if not wav_files:
            print(
                f"\nError: No {pattern} files found in {wav_dir} after generation. "
                f"Check the {text_file_path} file or TTS configuration."
            )
            exit(1)

        print(f"\n✓ Successfully generated {len(wav_files)} files")
    else:
        print(f"✓ Found {len(wav_files)} existing {pattern} files")

    return sorted(wav_files)


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate WAV audio files from a list of questions using a TTS model.")
    parser.add_argument(
        "--questions_file",
        type=str,
        required=True,
        help="Path to the text file containing questions (one question per line).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Directory where the generated WAV files will be saved.",
    )
    parser.add_argument(
        "--wake_word",
        type=str,
        default=None,
        help="Optional wake word to prefix each question (e.g., 'Hey NXP').",
    )
    parser.add_argument(
        "--add_noise",
        action="store_true",
        help="Add background noise to the generated audio.",
    )
    parser.add_argument(
        "--snr_db",
        type=float,
        default=20.0,
        help="Signal-to-Noise Ratio in dB (default: 20). Lower = more noise. "
        "Examples: 30dB=quiet, 20dB=moderate, 10dB=noisy, 5dB=very noisy.",
    )
    parser.add_argument(
        "--noise_type",
        type=str,
        default="white",
        choices=["white", "pink", "brown"],
        help="Type of noise to add (default: white).",
    )
    parser.add_argument(
        "--noise_prefix_duration",
        type=float,
        default=0.5,
        help="Duration of pure noise before audio starts, in seconds (default: 0.5). Use 0 to disable.",
    )

    # Parse arguments from the command line
    args = parser.parse_args()

    # Call the main function with parsed arguments
    questions_to_wav(
        questions_file=args.questions_file,
        output_path=args.output_path,
        wake_word=args.wake_word,
        add_noise=args.add_noise,
        snr_db=args.snr_db,
        noise_type=args.noise_type,
        noise_prefix_duration=args.noise_prefix_duration,
    )
