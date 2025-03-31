# Copyright 2025 NXP
# NXP Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.

import os
from typing import List


def get_pdf_file_list(repo_path: str) -> List[str]:
    """
    Retrieve all PDF files from a given repository path.

    :param repo_path: Path of the repository to search for PDF files
    :return: List of absolute paths to all PDF files found in the repository
    """
    pdf_files = []
    # Walk through all directories and files in the given path
    for root, dirs, files in os.walk(repo_path):
        # Filter files ending with .pdf
        for file in files:
            if file.endswith(".pdf"):
                pdf_files.append(file)  # Append full path
    return pdf_files
