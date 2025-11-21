# Copyright 2025 NXP
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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def questions_to_wav(
    questions_file: str,  # Now required as it comes from CLI
    output_path: str,  # Now required as it comes from CLI
):
    """
    Reads questions from a file, generates speech for each question using a TextToSpeech model,
    and saves the audio to WAV files.

    Args:
        questions_file (str): Path to the text file containing questions (one question per line).
        output_path (str): Directory where the generated WAV files will be saved.
    """
    from tts.inference import TTSGenerator
    from tts.config import MultiSpeakerTTS16kHzConfig
    import soundfile as sf

    # Read question file:
    try:
        with open(questions_file, "r", encoding="utf-8") as file:
            lines = [line.rstrip() for line in file]
    except FileNotFoundError:
        print(f"Error: Questions file '{questions_file}' not found.")
        return
    except Exception as e:
        print(f"Error reading questions file: {e}")
        return

    # Initialize speaker_id
    current_speaker_id = 1

    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    print(f"Output directory '{output_path}' ensured.")

    config = MultiSpeakerTTS16kHzConfig(speaker_id=current_speaker_id)
    tts = TTSGenerator(config)
    for i, question in enumerate(lines):
        # Increment speaker_id for each question.
        if i > 0:
            current_speaker_id += 1

        print(f"Generating speech for question {i + 1} of {len(lines)} with speaker_id: {current_speaker_id}")
        try:
            data = tts.generate(question, speaker_id=current_speaker_id)
            sf.write(f"{output_path}/question_{i:03d}.wav", data, config.samplerate)
        except Exception as e:
            print(f"Error generating speech for question '{question}': {e}")
            continue

    print(f"Finished generating {len(lines)} WAV files in '{output_path}'.")


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate WAV audio files from a list of questions using a TTS model.")
    parser.add_argument("--questions_file", type=str, required=True, help="Path to the text file containing questions (one question per line).")
    parser.add_argument("--output_path", type=str, required=True, help="Directory where the generated WAV files will be saved.")

    # Parse arguments from the command line
    args = parser.parse_args()

    # Call the main function with parsed arguments
    questions_to_wav(questions_file=args.questions_file, output_path=args.output_path)
