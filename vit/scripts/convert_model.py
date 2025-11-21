#!/usr/bin/env python3

# Copyright 2024-2025 NXP
# NXP Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.


"""
Convert VIT_Model_*.h to binary model file
"""

import re
import sys
import os
import logging

# Magic numbers that should be at the start of valid VIT models
WATERMARK_1 = 0xABFE34A2
WATERMARK_1_ZWWD = 0xABFE34A3

# Expected model version
EXPECTED_MAJOR_VERSION = 4
EXPECTED_MEDIUM_VERSION = 9

logger = logging.getLogger(__name__)


def extract_model_data(header_file):
    """Extract the model data array from the header file"""

    with open(header_file, "r") as f:
        content = f.read()

    # Look for the model array definition - handle different possible formats with variable model names
    patterns = [
        r"const\s+PL_MEM_ALIGN\s*\(\s*PL_UINT8\s+VIT_Model_\w+\s*\[\s*\]\s*,\s*[^)]+\)\s*=\s*\{",
        r"const\s+PL_UINT8\s+VIT_Model_\w+\s*\[\s*\]\s*=\s*\{",
        r"PL_UINT8\s+VIT_Model_\w+\s*\[\s*\]\s*=\s*\{",
    ]

    match = None
    for i, pattern in enumerate(patterns):
        match = re.search(pattern, content, re.MULTILINE | re.DOTALL)
        if match:
            # Extract the model name from the match
            model_name_match = re.search(r"VIT_Model_(\w+)", match.group(0))
            if model_name_match:
                model_name = f"VIT_Model_{model_name_match.group(1)}"
                logger.info(f"Found model definition '{model_name}' using pattern {i + 1}")
            break

    if not match:
        logger.error("Could not find VIT_Model_* array definition")
        logger.error("Looking for patterns like: VIT_Model_en, VIT_Model_de, VIT_Model_fr, etc.")
        return None

    # Find the start of the data (after the opening brace)
    start_pos = match.end()

    # Find the matching closing brace
    brace_count = 1
    pos = start_pos

    while pos < len(content) and brace_count > 0:
        if content[pos] == "{":
            brace_count += 1
        elif content[pos] == "}":
            brace_count -= 1
        pos += 1

    if brace_count != 0:
        logger.error("Could not find matching closing brace")
        return None

    # Extract the data between braces
    data_str = content[start_pos : pos - 1]

    # Clean up the data string - remove comments and unnecessary whitespace
    # Remove C-style comments /* ... */
    data_str = re.sub(r"/\*.*?\*/", "", data_str, flags=re.DOTALL)
    # Remove C++ style comments // ...
    data_str = re.sub(r"//.*?$", "", data_str, flags=re.MULTILINE)

    # Extract numeric values more carefully
    # Split by common delimiters and process each token
    tokens = re.split(r"[,\s\n\r]+", data_str)

    model_bytes = bytearray()
    valid_values = 0

    for token in tokens:
        token = token.strip()
        if not token:
            continue

        # Skip non-numeric tokens
        if token in ["{", "}", ";"]:
            continue

        try:
            # Handle hex values
            if token.startswith("0x") or token.startswith("0X"):
                byte_val = int(token, 16)
            # Handle decimal values
            elif token.isdigit():
                byte_val = int(token, 10)
            else:
                # Skip invalid tokens
                logger.debug(f"Skipping invalid token: '{token}'")
                continue

            # Validate byte range
            if byte_val < 0 or byte_val > 255:
                logger.warning(f"Value {byte_val} (from '{token}') is out of byte range, skipping")
                continue

            model_bytes.append(byte_val)
            valid_values += 1

        except ValueError:
            logger.warning(f"Could not parse token: '{token}'")
            continue

    logger.info(f"Extracted {valid_values} valid byte values")

    if len(model_bytes) == 0:
        logger.error("No valid byte values found")
        return None

    return model_bytes


def extract_model_version(model_data):
    """Extract model version information from the model data"""

    if len(model_data) < 7:
        logger.error("Model data too small to contain version information")
        return None, None, None

    model_minor = model_data[4]
    model_medium = model_data[5]
    model_major = model_data[6]

    logger.debug(f"Raw version bytes - Major: {model_major}, Medium: {model_medium}, Minor: {model_minor}")

    return model_major, model_medium, model_minor


def validate_model_version(model_major, model_medium, expected_major=EXPECTED_MAJOR_VERSION, expected_medium=EXPECTED_MEDIUM_VERSION):
    """Validate that the model version matches expected version"""

    if model_major is None or model_medium is None:
        logger.error("Cannot validate version - version information not available")
        return False

    logger.info(f"Model version: {model_major}.{model_medium}")
    logger.info(f"Expected version: {expected_major}.{expected_medium}")

    if model_major == expected_major and model_medium == expected_medium:
        logger.info(f"✓ Model version validation PASSED")
        return True
    else:
        logger.error(f"✗ Model version validation FAILED")
        logger.error(f"Expected version {expected_major}.{expected_medium}, but found {model_major}.{model_medium}")
        return False


def print_model_creation_guidance():
    """Print guidance on how to create a valid VIT model"""
    logger.error("")
    logger.error("=" * 80)
    logger.error("HOW TO CREATE A VALID VIT MODEL")
    logger.error("=" * 80)
    logger.error("")
    logger.error("To create a valid VIT model with version 4.9, please follow these steps:")
    logger.error("")
    logger.error("1. Go to the VIT Model Generation Tool:")
    logger.error("   https://vit.nxp.com/#/")
    logger.error("")
    logger.error("2. Configure the following settings:")
    logger.error("   • SW platform & version: Select 'Linux BSP'")
    logger.error("   • Linux BSP version: Select 'LF6.1.55_2.2.0 - LF6.6.3_1.0.0'")
    logger.error("   • Device: Any device (your choice)")
    logger.error("   • VIT Library version: This will automatically be 4.9 with the above BSP")
    logger.error("")
    logger.error("3. Define up to 3 wake words. The voice commands are not used, but keep at least one.")
    logger.error("")
    logger.error("4. Generate and download the model")
    logger.error("")
    logger.error("5. Extract the VIT_Model_*.h file from the downloaded package")
    logger.error("")
    logger.error("6. Use this script to convert the .h file to .bin format")
    logger.error("")
    logger.error("Note: Only models generated with VIT Library 4.9 are compatible")
    logger.error("      with this conversion script and the current VIT application.")
    logger.error("")
    logger.error("=" * 80)


def validate_model_data(model_data):
    """Perform basic validation on the model data"""

    if len(model_data) < 1024:  # Assume model should be at least 1KB
        logger.warning(f"Model seems very small ({len(model_data)} bytes)")
        return False

    # Check for VIT model magic numbers at the beginning
    if len(model_data) >= 4:
        # Read first 4 bytes as little-endian uint32
        magic_number = int.from_bytes(model_data[:4], byteorder="little")

        if magic_number == WATERMARK_1:
            logger.info(f"✓ Valid VIT model detected - Magic number: 0x{magic_number:08X} (WATERMARK_1)")
        elif magic_number == WATERMARK_1_ZWWD:
            logger.info(f"✓ Valid VIT model detected - Magic number: 0x{magic_number:08X} (WATERMARK_1_ZWWD)")
        else:
            logger.error(f"✗ Invalid magic number: 0x{magic_number:08X}")
            logger.error(f"Expected: 0x{WATERMARK_1:08X} (WATERMARK_1) or 0x{WATERMARK_1_ZWWD:08X} (WATERMARK_1_ZWWD)")
            logger.error("This model may not be a valid VIT binary model.")
            print_model_creation_guidance()
            return False
    else:
        logger.error("Model too small to contain magic number")
        print_model_creation_guidance()
        return False

    # Extract and validate model version
    model_major, model_medium, model_minor = extract_model_version(model_data)
    if not validate_model_version(model_major, model_medium):
        print_model_creation_guidance()
        return False

    # Check for common patterns that might indicate a valid model
    # Most binary models start with some kind of header or magic bytes
    first_bytes = model_data[:16]
    logger.debug(f"First 16 bytes: {' '.join([f'0x{b:02x}' for b in first_bytes])}")

    last_bytes = model_data[-16:]
    logger.debug(f"Last 16 bytes: {' '.join([f'0x{b:02x}' for b in last_bytes])}")

    # Check if last byte looks suspicious (like ASCII '+' = 0x2B)
    if model_data[-1] == 0x2B:  # ASCII '+'
        logger.warning("Last byte is 0x2B (ASCII '+'), this might indicate a parsing error")
        return False

    # Check for reasonable byte distribution
    zero_count = model_data.count(0)
    if zero_count > len(model_data) * 0.9:
        logger.warning("Model contains too many zero bytes")
        return False

    return True


def validate_binary_model(binary_file):
    """Validate a binary VIT model file"""

    if not os.path.exists(binary_file):
        logger.error(f"Binary file '{binary_file}' does not exist")
        return False

    logger.info(f"Validating binary VIT model: {binary_file}")

    try:
        with open(binary_file, "rb") as f:
            model_data = f.read()

        logger.info(f"Loaded binary model: {len(model_data)} bytes ({len(model_data) / 1024:.2f} KB)")

        # Use the existing validation function
        is_valid = validate_model_data(model_data)

        if is_valid:
            logger.info("✓ Binary model validation PASSED")
            return True
        else:
            logger.error("✗ Binary model validation FAILED")
            return False

    except IOError as e:
        logger.error(f"Error reading binary file: {e}")
        return False


def setup_logging(verbose=False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Convert: python3 convert_model.py <input_header_file> <output_binary_file> [--verbose]")
        print("  Validate: python3 convert_model.py --validate <binary_file> [--verbose]")
        print("")
        print("Examples:")
        print("  python3 convert_model.py VIT_Model_en.h VIT_Model_en.bin")
        print("  python3 convert_model.py --validate VIT_Model_en.bin")
        print("  python3 convert_model.py --validate VIT_Model_en.bin --verbose")
        sys.exit(1)

    # Check for verbose flag
    verbose = "--verbose" in sys.argv
    if verbose:
        sys.argv.remove("--verbose")

    # Setup logging
    setup_logging(verbose)

    # Check if this is a validation request
    if sys.argv[1] == "--validate":
        if len(sys.argv) != 3:
            logger.error("--validate requires exactly one binary file argument")
            logger.error("Usage: python3 convert_model.py --validate <binary_file> [--verbose]")
            sys.exit(1)

        binary_file = sys.argv[2]
        is_valid = validate_binary_model(binary_file)
        sys.exit(0 if is_valid else 1)

    # Original conversion functionality
    if len(sys.argv) != 3:
        logger.error("Usage: python3 convert_model.py <input_header_file> <output_binary_file> [--verbose]")
        logger.error("Example: python3 convert_model.py VIT_Model_en.h VIT_Model_en.bin")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    if not os.path.exists(input_file):
        logger.error(f"Input file '{input_file}' does not exist")
        sys.exit(1)

    logger.info(f"Converting {input_file} to {output_file}...")

    model_data = extract_model_data(input_file)
    if model_data is None:
        logger.error("Failed to extract model data")
        sys.exit(1)

    # Validate the extracted data
    if not validate_model_data(model_data):
        response = input("Validation warnings detected. Continue anyway? (y/N): ")
        if response.lower() != "y":
            logger.info("Conversion aborted")
            sys.exit(1)

    # Write binary data to output file
    try:
        with open(output_file, "wb") as f:
            f.write(model_data)

        logger.info(f"Successfully converted model to {output_file}")
        logger.info(f"Model size: {len(model_data)} bytes ({len(model_data) / 1024:.2f} KB)")

    except IOError as e:
        logger.error(f"Error writing output file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
