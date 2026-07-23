#!/bin/bash

# Copyright 2026 NXP
# NXP Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.

install_torch_cpu() {
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PYPROJECT_FILE="${SCRIPT_DIR}/pyproject.toml"
    TORCH_VERSION=$(grep -oP '"torch==\K[0-9]+\.[0-9]+\.[0-9]+' "${PYPROJECT_FILE}" | head -1)
    TORCHVISION_VERSION=$(grep -oP '"torchvision==\K[0-9]+\.[0-9]+\.[0-9]+' "${PYPROJECT_FILE}" | head -1)
    ARCH=$(uname -m)
    TORCH_PACKAGES=""

    if [ -n "${TORCH_VERSION}" ]; then
        echo "INFO: torch==${TORCH_VERSION} found, adding to install list"
        TORCH_PACKAGES="${TORCH_PACKAGES} torch==${TORCH_VERSION}"
    else
        echo "INFO: torch not defined in ${PYPROJECT_FILE}, skipping"
    fi

    if [ -n "${TORCHVISION_VERSION}" ]; then
        echo "INFO: torchvision==${TORCHVISION_VERSION} found, adding to install list"
        TORCH_PACKAGES="${TORCH_PACKAGES} torchvision==${TORCHVISION_VERSION}"
    else
        echo "INFO: torchvision not defined in ${PYPROJECT_FILE}, skipping"
    fi

    if [ -n "${TORCH_PACKAGES}" ]; then
        echo "Installing CPU-only ${TORCH_PACKAGES} for ${ARCH}..."

        pip3 install ${TORCH_PACKAGES} \
            --extra-index-url https://download.pytorch.org/whl/cpu \
            --trusted-host download.pytorch.org
    fi
}

# Check for internet connectivity by pinging a reliable server
if ! ping -c 1 -W 2 8.8.8.8 > /dev/null 2>&1; then
    echo "ERROR: No internet connection detected."
    echo "Please connect the device to install required packages."
    exit 1
fi


# Set the date unless --skip-date is provided

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

if [ -z "$VIRTUAL_ENV" ]; then
  # Handle virtual environment or system-wide installation
  echo "============================================"
  echo "Installing system-wide (no virtual environment)"
  echo "============================================"
  echo ""
  # Check if NXP custom onnxruntime is available system-wide
  ONNXRUNTIME_INSTALLED=false
  if python3 -c "import onnxruntime" 2>/dev/null; then
      ONNXRUNTIME_VERSION=$(python3 -c "import onnxruntime; print(onnxruntime.__version__)" 2>/dev/null)
      echo "onnxruntime version $ONNXRUNTIME_VERSION is already installed system-wide."
      ONNXRUNTIME_INSTALLED=true
  else
      echo "onnxruntime not found system-wide. It will be installed."
  fi

  # Upgrade pip and install the required Python packages
  echo "Uninstall previous version of VLM if any"
  sudo pip3 uninstall vlm -y

  install_torch_cpu

  echo "Installing required Python packages system-wide..."
  PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
  if [ "$ONNXRUNTIME_INSTALLED" = false ]; then
      sudo pip3 install -e ".[onnxruntime,gui]" --find-links=../eiq_genai_flow/wheels/py$PYTHON_VERSION
  else
      sudo pip3 install -e ".[gui]" --find-links=../eiq_genai_flow/wheels/py$PYTHON_VERSION
  fi

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
  # Handle virtual environment
  echo "============================================"
  echo "Installing inside your current venv"
  echo "============================================"
  echo ""
  echo $VIRTUAL_ENV

  # Check if NXP custom onnxruntime is available system-wide
  ONNXRUNTIME_INSTALLED=false
  if python3 -c "import onnxruntime" 2>/dev/null; then
      ONNXRUNTIME_VERSION=$(python3 -c "import onnxruntime; print(onnxruntime.__version__)" 2>/dev/null)
      echo "onnxruntime version $ONNXRUNTIME_VERSION is already installed system-wide."
      ONNXRUNTIME_INSTALLED=true
  else
      echo "onnxruntime not found system-wide. It will be installed."
  fi

  echo "Uninstall previous version of VLM if any"
  python3 -m pip uninstall vlm -y

  install_torch_cpu

  echo "Installing required Python packages inside env"
  PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
  if [ "$ONNXRUNTIME_INSTALLED" = false ]; then
      python3 -m pip install -e ".[onnxruntime,gui]"  --find-links=../eiq_genai_flow/wheels/py$PYTHON_VERSION
  else
      python3 -m pip install -e ".[gui]"  --find-links=../eiq_genai_flow/wheels/py$PYTHON_VERSION
  fi

  if [ $? -eq 0 ]; then
      echo ""
      echo "============================================"
      echo "VLM installation completed successfully inside your env!"
      echo "============================================"
      echo ""
      echo "The package is now available inside your env."
      echo ""
  else
      echo "Env-based installation failed. Please check the errors above."
      exit 1
  fi

fi
