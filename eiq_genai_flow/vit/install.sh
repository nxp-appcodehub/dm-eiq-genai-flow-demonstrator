#!/bin/bash
# Copyright 2025-2026 NXP
# NXP Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  --dev         Install with development dependencies"
    echo "  --skip-date   Skip setting the system date"
    echo "  -h, --help    Show this help message and exit"
    echo
    echo "Description:"
    echo "  This script installs a package with either runtime dependencies (default)"
    echo "  or with development dependencies when the --dev option is provided."
    echo "  It automatically synchronizes the system date unless --skip-date is used."
    exit 1
}

# Function to display manual date setting instructions
manual_date_instructions() {
    echo "---------------------------------------------"
    echo "The date and time cannot be set automatically."
    echo "To set the date manually, use the following command:"
    echo "sudo date -s 'YYYY-MM-DD HH:MM:SS'"
    echo "For example: date -s '2025-10-09 12:34:56'"
    echo "---------------------------------------------"
}

# Default mode
DEV_MODE=false
SKIP_DATE_SETTING=false

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dev) DEV_MODE=true ;;
        --skip-date) SKIP_DATE_SETTING=true ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
    shift
done

# Check for internet connectivity by pinging a reliable server
if ! ping -c 1 -W 2 8.8.8.8 > /dev/null 2>&1; then
    echo "ERROR: No internet connection detected."
    echo "Please connect the device to install required packages."
    exit 1
fi

# Set the date unless --skip-date is provided
if [ "$SKIP_DATE_SETTING" = false ]; then
    echo "Attempting to synchronize date with NTP servers..."

    # Try with ntpdate, a simple utility for one-time sync
    # We use a combined list of global and China-specific pools for reliability
    if command -v ntpdate &> /dev/null; then
        echo "Using ntpdate to set the date."
        if ! sudo ntpdate -u pool.ntp.org cn.pool.ntp.org > /dev/null 2>&1; then
            echo "Failed to set date using ntpdate."
            manual_date_instructions
        else
            echo "Date set successfully via NTP. Current date: $(date)"
        fi
    elif command -v timedatectl &> /dev/null; then
        echo "Using timedatectl to enable NTP synchronization."
        # Enable NTP sync which will set the date automatically
        sudo timedatectl set-ntp on
        echo "NTP synchronization enabled. The date will be set automatically."
    else
        echo "Neither ntpdate nor timedatectl found. Cannot set date automatically."
        manual_date_instructions
    fi
fi

# install alsa
if [ -f /usr/include/alsa/asoundlib.h ]; then
  echo "alsa is already installed. Skip it..."
else
  VERSION="1.2.13"
  TARBALL="alsa-lib-${VERSION}.tar.bz2"
  URL="https://www.alsa-project.org/files/pub/lib/alsa-lib-${VERSION}.tar.bz2"
  EXTRACTED_DIR="alsa-lib-${VERSION}"
  wget "$URL" -O "$TARBALL"
  tar xf "$TARBALL"
  cd "$EXTRACTED_DIR" || exit 1
  ./configure
  make
  sudo make install
  cd ..
  rm -rf "$EXTRACTED_DIR" "$TARBALL"
fi

# Upgrade pip and install the required Python packages
echo "Upgrading pip..."
pip3 install --upgrade pip --trusted-host pypi.org

# Install the package
if $DEV_MODE; then
    echo "Installing dev required Python packages..."
    pip install -e ".[dev]"
else
    echo "Installing required Python packages..."
    pip install -e .
fi
