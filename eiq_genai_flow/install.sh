#!/bin/bash

# Copyright 2024-2025 NXP
# NXP Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.

# Function to display manual date setting instructions
manual_date_instructions() {
    echo "---------------------------------------------"
    echo "The date and time cannot be set automatically."
    echo "To set the date manually, use the following command:"
    echo "sudo date -s 'YYYY-MM-DD HH:MM:SS'"
    echo "For example: date -s '2025-03-31 12:34:56'"
    echo "---------------------------------------------"
}

# Check for internet connectivity by pinging a reliable server
if ! ping -c 1 -W 2 8.8.8.8 > /dev/null 2>&1; then
    echo "ERROR: No internet connection detected."
    echo "Please connect the device to install required packages."
    exit 1
fi

# Set a default date: To retrieve the date from the internet, the local date must not be more than three months behind."
echo "Set a temporary date"
date -s '2025-03-31 12:34:56'

# Fetch the current date and time from httpbin.org
internet_date=$(curl -sI https://httpbin.org/get | grep -i '^date:' | cut -d' ' -f3-)

# Check if the date was fetched successfully
if [ -z "$internet_date" ]; then
    echo "Failed to fetch the date from the internet."
    manual_date_instructions
    exit 1
else
    # Format the date to a suitable format for the date command
    formatted_date=$(date -d "$internet_date" +"%Y-%m-%d %H:%M:%S")

    # Set the system date
    echo "Set actual date retreived from Internet"
    date -s "$formatted_date"
fi

# Check if espeak-ng is installed
if command -v espeak-ng &> /dev/null; then
    exit 0
fi

echo "espeak-ng is not installed. Installing from source..."
# Set variables
VERSION="1.51"
TARBALL="${VERSION}.tar.gz"
URL="https://github.com/espeak-ng/espeak-ng/archive/refs/tags/${TARBALL}"
EXTRACTED_DIR="espeak-ng-${VERSION}"

# Download and extract
wget "$URL" -O "$TARBALL"
if [ $? -ne 0 ]; then
    echo "Failed to download espeak-ng source. Exiting."
    exit 1
fi

tar xf "$TARBALL"
if [ $? -ne 0 ]; then
    echo "Failed to extract espeak-ng source. Exiting."
    exit 1
fi

cd "$EXTRACTED_DIR" || exit 1

# Ensure build system is properly set up
autoreconf -fi

# Build and install
./autogen.sh
./configure --prefix=/usr \
            --with-klatt=no \
            --with-speechplayer=no \
            --with-mbrola=no \
            --with-extdict-ru=no \
            --with-extdict-cmn=no \
            --with-extdict-yue=no

make en -j `nproc`
if [ $? -ne 0 ]; then
    echo "Build failed. Exiting."
    exit 1
fi

sudo make install
if [ $? -ne 0 ]; then
    echo "Installation failed. Exiting."
    exit 1
fi

# Cleanup
cd ..
rm -rf "$EXTRACTED_DIR" "$TARBALL"

echo "espeak-ng installed successfully."
