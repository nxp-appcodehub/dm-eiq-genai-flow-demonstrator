# Copyright 2025-2026 NXP
# NXP Confidential and Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.

#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  --install           Install runtime dependencies (default)"
    echo "  --install-notebooks Install the optional Jupyter notebook dependencies"
    echo "  --notebook          Run the example notebooks"
    echo "  --see-models    See the list of available embedding models"
    echo "  -h, --help      Show this help message and exit"
    echo
    echo "Note: It is highly encouraged to create and use a virtual environment for installation."
    exit 1
}

# Show help if no arguments are provided
if [ $# -eq 0 ]; then
    usage
fi

# Default mode
INSTALL_MODE=false
NOTEBOOK_INSTALL_MODE=false
NOTEBOOK_MODE=false
SEE_EMBEDDING_MODELS=false
PYPI_URL="https://pypi.org/simple"
WHEELS_PATH="./wheels"
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)


# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --install) INSTALL_MODE=true ;;
        --install-notebooks) NOTEBOOK_INSTALL_MODE=true ;;
        --see-models) SEE_EMBEDDING_MODELS=true ;;
        --notebook) NOTEBOOK_MODE=true ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
    shift
done

# Upgrade pip
if $INSTALL_MODE || $NOTEBOOK_INSTALL_MODE; then
    set -e
    echo "Using local index: ${WHEELS_PATH}/py${PYTHON_VERSION}"

    # Install the package. The optional "notebooks" extra pulls in jupyter/nbmake
    # (and their transitive dependencies) and is only installed when explicitly
    # requested via --install-notebooks.
    if $NOTEBOOK_INSTALL_MODE; then
        echo "Installing with Jupyter notebook dependencies..."
        pip install --upgrade pip --index-url "${PYPI_URL}" --find-links="${WHEELS_PATH}/py${PYTHON_VERSION}" -e ".[notebooks]"
    else
        echo "Installing runtime dependencies only..."
        pip install --upgrade pip --index-url "${PYPI_URL}" --find-links="${WHEELS_PATH}/py${PYTHON_VERSION}" -e .
    fi

    echo "Installation complete."
fi

# Run the Jupyter notebooks
if $NOTEBOOK_MODE; then
    if ! command -v jupyter >/dev/null 2>&1; then
        echo "Jupyter is not installed. Install the optional notebook dependencies first:"
        echo "  ./run.sh --install-notebooks"
        exit 1
    fi
    echo "Running the notebooks..."
    jupyter notebook notebooks/
fi

if $SEE_EMBEDDING_MODELS; then
    python -c "from rag_database_generator.embed import EmbeddingModels; print([e.value for e in EmbeddingModels])"
fi