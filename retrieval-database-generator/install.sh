# Copyright 2024-2025 NXP
# NXP Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.

#!/bin/bash

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --trusted-host pypi.org

# Install the package
echo "Installing dependencies ..."
pip install -e .

#Installing Hugging Face Cli
echo "Installing Hugging Face Cli ..."
pip install -U "huggingface_hub[cli]"

echo "Installation complete."
