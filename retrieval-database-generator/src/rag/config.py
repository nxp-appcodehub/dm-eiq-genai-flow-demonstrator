# Copyright 2024-2025 NXP
# NXP Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.

from dataclasses import dataclass


@dataclass
class Config:
    ########################################## Retrieval inference parameters ##########################################

    top_k: int = 3  # The number of element pre-selected in the database.
    reranking: bool = True  # Activate the reranking option.
    best_k: int = 1  # The number of element among --top-k added top the prompt (must be <= top_k).

    ################################################ Chunking parameters ###############################################

    # (Only used if chunking_method != "HiRAG")
    chunk_size: int = 128  # We recommend 128
    chunk_overlap: int = 64  # We recommend half of `chunk_size`

    ################################################ Embedding parameters ##############################################

    database_description: str = ""
