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

# Function to display usage
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

# Function to get installed ALSA version
get_alsa_version() {
    if command -v aplay &> /dev/null; then
        local version=$(aplay --version 2>/dev/null | grep -oE "[0-9]+\.[0-9]+\.[0-9]+" | head -1)
        if [ -n "$version" ]; then
            echo "$version"
            return 0
        fi
    fi

    # Fallback: return empty string
    echo ""
    return 1
}

# Default settings
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
    if command -v timedatectl &> /dev/null; then
        echo "Attempting to set the date..."
        echo "Enabling NTP..."
        sudo timedatectl set-ntp true
        sudo systemctl restart systemd-timesyncd

        # Wait a few seconds for initial sync
        echo "Waiting for NTP synchronization..."
        sleep 10

        # Force update if still not synchronized
        SYNCED=$(timedatectl show -p NTPSynchronized --value)
        if [ "$SYNCED" != "yes" ]; then
            echo "NTP not yet synchronized. Attempting immediate sync..."
            sudo systemctl restart systemd-timesyncd
            sleep 5
        fi

        echo "Current system date/time:"
        timedatectl
    else
        echo "timedatectl not found. Cannot set date automatically."
        manual_date_instructions
        echo "or use --skip-date to skip the date setting."
        exit 1
    fi
fi

# install alsa dev package
if [ -f /usr/include/alsa/asoundlib.h ]; then
    echo "ALSA development headers are already installed. Skip it..."
else
    echo "ALSA development headers not found. Checking installed ALSA version..."
    INSTALLED_VERSION=$(get_alsa_version)

    if [ -n "$INSTALLED_VERSION" ]; then
        echo "Found ALSA runtime version: $INSTALLED_VERSION"
        VERSION="$INSTALLED_VERSION"
    else
        VERSION="1.2.13"
        echo "ALSA runtime version not found, falling back to v$VERSION"
    fi

    echo "Downloading ALSA lib version $VERSION..."
    TARBALL="alsa-lib-${VERSION}.tar.bz2"
    URL="https://www.alsa-project.org/files/pub/lib/alsa-lib-${VERSION}.tar.bz2"
    EXTRACTED_DIR="alsa-lib-${VERSION}"
    curl -L -k -o "$TARBALL" "$URL"
    tar xf "$TARBALL"
    echo "Installing ALSA development package..."
    cd "$EXTRACTED_DIR" || exit 1
    ./configure
    make
    sudo make install
    cd ..
    rm -rf "$EXTRACTED_DIR" "$TARBALL"
fi

# install espeak phonetizer
if command -v espeak-ng &> /dev/null; then
    echo "espeak-ng is already installed. Skip it..."
else
    echo "espeak-ng is not installed. Installing from source..."
    # Set variables
    VERSION="1.51"
    TARBALL="${VERSION}.tar.gz"
    URL="https://github.com/espeak-ng/espeak-ng/archive/refs/tags/${TARBALL}"
    EXTRACTED_DIR="espeak-ng-${VERSION}"

    # Download and extract
    curl -L -k -o "$TARBALL" "$URL"
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
    if [ -f ../ltmain.sh ]; then
        mv ../ltmain.sh ./ltmain.sh
    fi

    # Build
    ./autogen.sh
    ./configure --prefix=/usr \
                --with-klatt=no \
                --with-speechplayer=no \
                --with-mbrola=no \
                --with-extdict-ru=no \
                --with-extdict-cmn=yes \
                --with-extdict-yue=no

    # Install
    make -j `nproc`
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
fi

# Upgrade pip and install the required Python packages
echo "Upgrading pip..."
pip3 install --upgrade pip --trusted-host pypi.org
pip3 uninstall eiq_genai_flow -y

# Install the package
if $DEV_MODE; then
    echo "Installing dev required Python packages..."
    pip install -e ".[dev]"
else
    echo "Installing required Python packages..."
    pip install -e .
fi
