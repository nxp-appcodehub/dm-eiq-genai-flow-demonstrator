#!/bin/bash

# Copyright 2024-2026 NXP
# NXP Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.


if [ "$EUID" -eq 0 ]; then
    echo "WARNING: Running as root. Packages will be owned by root."
fi

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  --dev               Install with development dependencies"
    echo "  --gui               Install the optional chat interface GUI (pulls PySide6)"
    echo "  --skip-date         Skip setting the system date"
    echo "  --venv              Use virtual environment instead of system-wide installation"
    echo "  --python VERSION    Specify Python version (e.g., 3.13, 3.14). Default: python3"
    echo "  --recreate-venv     Remove and recreate the virtual environment (only with --venv)"
    echo "  --no-auto-activate  Skip adding venv activation to shell startup (only with --venv)"
    echo "  --no-fix-record     Skip fixing system packages with missing RECORD files"
    echo "  -h, --help          Show this help message and exit"
    echo
    echo "Description:"
    echo "  This script installs a package with either runtime dependencies (default)"
    echo "  or with development dependencies when the --dev option is provided."
    echo "  It automatically synchronizes the system date unless --skip-date is used."
    echo "  By default, packages are installed system-wide (requires sudo for pip)."
    echo "  Use --venv to install in a virtual environment with system site packages in ./venv"
    echo "  By default, missing RECORD files in .dist-info directories are fixed automatically."
    echo "  Use --no-fix-record to skip this step."
    echo
    echo "Examples:"
    echo "  $0 --venv --python 3.14"
    echo "  $0 --dev --python 3.13"
    echo "  $0 --venv --python 3.14 --recreate-venv"
    echo "  $0 --gui --venv --python 3.14"
    echo "  $0 --no-fix-record"
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

# Function to fix missing RECORD files in dist-info directories
fix_record_files() {
    echo "Fixing system packages with missing RECORD files..."

    # Search only in known Python package locations, not the entire filesystem
    SEARCH_PATHS=(
        "/usr/lib/python3"
        "/usr/lib/python3.*"
        "/usr/local/lib/python3.*"
        "/usr/lib64/python3.*"
    )

    for SEARCH_PATH in "${SEARCH_PATHS[@]}"; do
        for DIST_INFO in $(find $SEARCH_PATH -maxdepth 4 -name "*.dist-info" -type d 2>/dev/null); do
            if [ ! -f "$DIST_INFO/RECORD" ]; then
                echo "Creating missing RECORD file in: $DIST_INFO"
                # Use sudo only for system paths, not user paths
                if [[ "$DIST_INFO" == /usr/* ]]; then
                    sudo touch "$DIST_INFO/RECORD" 2>/dev/null || \
                        echo "Warning: Could not create RECORD in $DIST_INFO (permission denied, skipping)"
                else
                    touch "$DIST_INFO/RECORD" 2>/dev/null || true
                fi
            fi
        done
    done

    echo "Done fixing RECORD files."
}

# Function to get installed ALSA version
get_alsa_version() {
    if command -v aplay &> /dev/null; then
        local version
        version=$(aplay --version 2>/dev/null | grep -oE "[0-9]+\.[0-9]+\.[0-9]+" | head -1)
        if [ -n "$version" ]; then
            echo "$version"
            return 0
        fi
    fi

    # Fallback: return empty string
    echo ""
    return 1
}

install_torch_cpu() {
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PYPROJECT_FILE="${SCRIPT_DIR}/pyproject.toml"
    TORCH_VERSION=$(grep -oP '"torch==\K[0-9]+\.[0-9]+\.[0-9]+' "${PYPROJECT_FILE}" | head -1)
    TORCHAUDIO_VERSION=$(grep -oP '"torchaudio==\K[0-9]+\.[0-9]+\.[0-9]+' "${PYPROJECT_FILE}" | head -1)
    ARCH=$(uname -m)
    TORCH_PACKAGES=""

    if [ -n "${TORCH_VERSION}" ]; then
        echo "INFO: torch==${TORCH_VERSION} found, adding to install list"
        TORCH_PACKAGES="${TORCH_PACKAGES} torch==${TORCH_VERSION}"
    else
        echo "INFO: torch not defined in ${PYPROJECT_FILE}, skipping"
    fi

    if [ -n "${TORCHAUDIO_VERSION}" ]; then
        echo "INFO: torchaudio==${TORCHAUDIO_VERSION} found, adding to install list"
        TORCH_PACKAGES="${TORCH_PACKAGES} torchaudio==${TORCHAUDIO_VERSION}"
    else
        echo "INFO: torchaudio not defined in ${PYPROJECT_FILE}, skipping"
    fi

    if [ -n "${TORCH_PACKAGES}" ]; then
        echo "Installing CPU-only ${TORCH_PACKAGES} for ${ARCH}..."

        # Use $PIP_CMD to ensure correct Python version is used
        $PIP_CMD install ${TORCH_PACKAGES} \
            --extra-index-url https://download.pytorch.org/whl/cpu \
            --trusted-host download.pytorch.org
    fi
}

# Function to add venv activation to shell startup
add_venv_to_startup() {
    local venv_path="$1"
    local shell_rc=""

    # Determine which shell config file to use based on $SHELL
    case "$SHELL" in
        */bash)
            shell_rc="$HOME/.bashrc"
            ;;
        */zsh)
            shell_rc="$HOME/.zshrc"
            ;;
        */sh)
            shell_rc="$HOME/.profile"
            ;;
        *)
            # Fallback to .profile for unknown shells
            shell_rc="$HOME/.profile"
            echo "Warning: Unknown shell ($SHELL), using .profile"
            ;;
    esac

    echo "Detected shell: $SHELL"
    echo "Using config file: $shell_rc"

    # Create activation snippet
    local activation_snippet="# Auto-activate eiq_genai_flow venv
if [ -f \"$venv_path/bin/activate\" ]; then
    . \"$venv_path/bin/activate\"
fi"

    # Check if already added
    if grep -q "Auto-activate eiq_genai_flow venv" "$shell_rc" 2>/dev/null; then
        echo "Virtual environment activation already configured in $shell_rc"
        return 0
    fi

    # Add to shell config
    echo "" >> "$shell_rc"
    echo "$activation_snippet" >> "$shell_rc"
    echo "✓ Added venv auto-activation to $shell_rc"
    echo "  Run 'source $shell_rc' or restart your shell to activate"
}

# Default settings
DEV_MODE=false
GUI_MODE=false
SKIP_DATE_SETTING=false
RECREATE_VENV=false
AUTO_ACTIVATE=false
SYSTEM_WIDE=true
VENV_DIR="venv"
FIX_RECORD_MODE=true
PYTHON_CMD="python3"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dev) DEV_MODE=true ;;
        --gui) GUI_MODE=true ;;
        --skip-date) SKIP_DATE_SETTING=true ;;
        --venv) SYSTEM_WIDE=false ;;
        --python)
            if [[ -n "$2" && "$2" != --* ]]; then
                PYTHON_CMD="python$2"
                shift
            else
                echo "Error: --python requires a version argument (e.g., 3.14)"
                usage
            fi
            ;;
        --recreate-venv) RECREATE_VENV=true ;;
        --no-auto-activate) AUTO_ACTIVATE=false ;;
        --no-fix-record) FIX_RECORD_MODE=false ;;
        --fix-record) FIX_RECORD_MODE=true ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
    shift
done

# Verify Python version exists
if ! command -v "$PYTHON_CMD" &> /dev/null; then
    echo "Error: $PYTHON_CMD not found on this system"
    echo "Available Python versions:"
    ls /usr/bin/python* 2>/dev/null || echo "  No Python installations found in /usr/bin/"
    echo ""
    echo "Please install $PYTHON_CMD or specify a different version with --python"
    exit 1
fi

# Display Python version being used
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
echo "Using: $PYTHON_VERSION"
echo ""

# Set pip command based on Python version
if [ "$SYSTEM_WIDE" = true ]; then
    PYTHON_FULL_PATH=$(which "$PYTHON_CMD")
    if [ -z "$PYTHON_FULL_PATH" ]; then
        echo "Error: Could not find full path for $PYTHON_CMD"
        exit 1
    fi
    echo "Python full path: $PYTHON_FULL_PATH"
    PIP_CMD="sudo $PYTHON_FULL_PATH -m pip"
else
    PIP_CMD="$PYTHON_CMD -m pip"
fi

# Run fix-record if requested
if [ "$FIX_RECORD_MODE" = true ]; then
    fix_record_files
fi

# Detect a local "wheels" folder (produced by tools/make_release_package.py for
# i.MX release packages). When present, pip resolves the nxp_eiq_* wheels from
# this folder instead of Nexus, making the package self-contained/offline.
#
# Wheels are organized into per-python-version subfolders (e.g. wheels/py3.13,
# wheels/py3.14) so a single release package can carry wheels for several python
# versions. We pick the subfolder that matches the selected python version and
# fall back to the top-level wheels folder for older/flat layouts.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHEELS_DIR="${SCRIPT_DIR}/wheels"
FIND_LINKS_ARGS=()
if [ -d "${WHEELS_DIR}" ]; then
    # Determine the python version (major.minor) of the selected interpreter.
    PY_VER=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
    PY_WHEELS_DIR="${WHEELS_DIR}/py${PY_VER}"

    if [ -n "${PY_VER}" ] && [ -d "${PY_WHEELS_DIR}" ]; then
        echo "Local wheels folder detected at ${PY_WHEELS_DIR}; installing wheels from it."
        FIND_LINKS_ARGS+=(--find-links "${PY_WHEELS_DIR}")
    else
        echo "Local wheels folder detected at ${WHEELS_DIR}; installing wheels from it."
        FIND_LINKS_ARGS+=(--find-links "${WHEELS_DIR}")
    fi
fi



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
        VERSION="1.2.14"
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
    make -j "$(nproc)"
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

# Handle virtual environment or system-wide installation
if [ "$SYSTEM_WIDE" = true ]; then
    echo "============================================"
    echo "Installing system-wide (no virtual environment)"
    echo "============================================"
    echo ""

    # Check if NXP custom onnxruntime is available system-wide
    ONNXRUNTIME_INSTALLED=false
    if $PYTHON_CMD -c "import onnxruntime" 2>/dev/null; then
        ONNXRUNTIME_VERSION=$($PYTHON_CMD -c "import onnxruntime; print(onnxruntime.__version__)" 2>/dev/null)
        echo "onnxruntime version $ONNXRUNTIME_VERSION is already installed system-wide."
        ONNXRUNTIME_INSTALLED=true
    else
        echo "onnxruntime not found system-wide. It will be installed."
    fi

    # Uninstall previous version
    $PIP_CMD uninstall eiq_genai_flow -y

    install_torch_cpu

    # Build the optional-dependency extras list based on the selected flags.
    # and is therefore only installed when the user explicitly passes --gui.
    # Keeping it opt-in keeps the default install footprint minimal.
    EXTRAS=()
    if $DEV_MODE; then EXTRAS+=("dev"); fi
    if [ "$ONNXRUNTIME_INSTALLED" = false ]; then EXTRAS+=("onnxruntime"); fi
    if $GUI_MODE; then
        EXTRAS+=("gui")
        echo "NOTE: --gui specified. Installing the optional chat interface (PySide6)."
    else
        echo "NOTE: chat interface GUI not requested (use --gui to install it)."
    fi
    EXTRAS_STR=$(IFS=,; echo "${EXTRAS[*]}")
    if [ -n "$EXTRAS_STR" ]; then
        INSTALL_TARGET=".[$EXTRAS_STR]"
    else
        INSTALL_TARGET="."
    fi

    echo "Installing required Python packages system-wide (target: $INSTALL_TARGET)..."
    $PIP_CMD install \
        "${FIND_LINKS_ARGS[@]}" \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        --trusted-host download.pytorch.org \
        -e "$INSTALL_TARGET"


    if [ $? -eq 0 ]; then
        echo ""
        echo "============================================"
        echo "System-wide installation completed successfully!"
        echo "============================================"
        echo ""
        echo "The package is now available globally."
        echo ""
    else
        echo "System-wide installation failed. Please check the errors above."
        exit 1
    fi

else
    # Virtual environment installation
    # Warn if --recreate-venv or --no-auto-activate used without --venv
    if [ "$RECREATE_VENV" = true ]; then
        echo "Note: --recreate-venv only applies when using --venv"
    fi

    echo "============================================"
    echo "Installing with virtual environment"
    echo "============================================"
    echo ""

    # Create or recreate virtual environment
    if [ "$RECREATE_VENV" = true ] && [ -d "$VENV_DIR" ]; then
        echo "Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
    fi

    if [ ! -d "$VENV_DIR" ]; then
        echo "Creating virtual environment with system site packages using $PYTHON_CMD..."
        $PYTHON_CMD -m venv --system-site-packages "$VENV_DIR"
        if [ $? -ne 0 ]; then
            echo "Failed to create virtual environment. Exiting."
            exit 1
        fi
        echo "Virtual environment created at ./$VENV_DIR"
    else
        echo "Virtual environment already exists at ./$VENV_DIR"
    fi

    # Activate virtual environment
    echo "Activating virtual environment..."
    source "$VENV_DIR/bin/activate"

    if [ $? -ne 0 ]; then
        echo "Failed to activate virtual environment. Exiting."
        exit 1
    fi

    echo "Virtual environment activated."

    # Check if NXP custom onnxruntime is available system-wide
    ONNXRUNTIME_INSTALLED=false
    if $PYTHON_CMD -c "import onnxruntime" 2>/dev/null; then
        ONNXRUNTIME_VERSION=$($PYTHON_CMD -c "import onnxruntime; print(onnxruntime.__version__)" 2>/dev/null)
        export PYTHON_VERSION
        PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        echo "onnxruntime version $ONNXRUNTIME_VERSION is already installed system-wide."
        ONNXRUNTIME_INSTALLED=true
    else
        echo "onnxruntime not found system-wide. It will be installed in the virtual environment."
    fi

    # Fix permissions for egg-info directory if it exists and is owned by root
    if [ -d "eiq_genai_flow.egg-info" ]; then
        OWNER=$(stat -c '%U' eiq_genai_flow.egg-info 2>/dev/null || stat -f '%Su' eiq_genai_flow.egg-info 2>/dev/null)
        if [ "$OWNER" = "root" ]; then
            echo "Fixing permissions for eiq_genai_flow.egg-info..."
            sudo chown -R $USER:$USER eiq_genai_flow.egg-info
        fi
    fi

    # Uninstall previous version
    $PIP_CMD uninstall eiq_genai_flow -y

    install_torch_cpu

    # Build the optional-dependency extras list based on the selected flags.
    # and is therefore only installed when the user explicitly passes --gui.
    # Keeping it opt-in keeps the default install footprint minimal.
    EXTRAS=()
    if $DEV_MODE; then EXTRAS+=("dev"); fi
    if [ "$ONNXRUNTIME_INSTALLED" = false ]; then EXTRAS+=("onnxruntime"); fi
    if $GUI_MODE; then
        EXTRAS+=("gui")
        echo "NOTE: --gui specified. Installing the optional chat interface (PySide6)."
    else
        echo "NOTE: chat interface GUI not requested (use --gui to install it)."
    fi
    EXTRAS_STR=$(IFS=,; echo "${EXTRAS[*]}")
    if [ -n "$EXTRAS_STR" ]; then
        INSTALL_TARGET=".[$EXTRAS_STR]"
    else
        INSTALL_TARGET="."
    fi

    echo "Installing required Python packages (target: $INSTALL_TARGET)..."
    pip install --ignore-installed \
        "${FIND_LINKS_ARGS[@]}" \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        --trusted-host download.pytorch.org \
        -e "$INSTALL_TARGET"
    PIP_EXIT_CODE=$?


    if [ $PIP_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "============================================"
        echo "Installation completed successfully!"
        echo "============================================"
        echo ""

        # Add auto-activation if requested
        if [ "$AUTO_ACTIVATE" = true ]; then
            VENV_ABS_PATH="$(cd "$(dirname "$VENV_DIR")" && pwd)/$(basename "$VENV_DIR")"
            add_venv_to_startup "$VENV_ABS_PATH"
            echo ""
        fi

        echo "To activate the virtual environment in your current shell, run:"
        echo "  source $VENV_DIR/bin/activate"
        echo ""
        echo "Or copy and paste this command:"
        echo "  eval \"\$(echo 'source $VENV_DIR/bin/activate')\""
        echo ""
        echo "To deactivate the virtual environment, run:"
        echo "  deactivate"
        echo ""

        if [ "$AUTO_ACTIVATE" = false ]; then
            echo "To enable auto-activation on login, run:"
            echo "  ./install.sh --venv --python $PYTHON_CMD"
            echo ""
        fi

        # Try to activate in current shell (only works if script is sourced)
        if [ "$0" = "${BASH_SOURCE[0]}" ]; then
            echo "Note: Run 'source ./install.sh --venv' instead of './install.sh --venv' to auto-activate the venv."
        else
            echo "Activating virtual environment in current shell..."
            source "$VENV_DIR/bin/activate"
        fi
    else
        echo "Installation failed. Please check the errors above."
        exit 1
    fi
fi
