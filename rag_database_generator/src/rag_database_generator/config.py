# Copyright 2024-2026 NXP
# NXP Confidential and Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.

from dataclasses import dataclass
from retriever import Config as RetrievalConfig


@dataclass
class Config(RetrievalConfig):
    # Parse
    output_format: str = ".md"  # Must be in [".md", ".json"]

    # Chunk
    chunking_method: str = "HiRAG"  # Must be in ["HiRAG", "SpaCy", "NLTK", "fixed"]
    # (Only used if chunking_method != "HiRAG")
    chunk_size: int = 128  # We recommend 128
    chunk_overlap: int = 64  # We recommend half of `chunk_size`

    # Embedding Model
    # Must be in ["all-MiniLM-L6-v2", "GIST-all-MiniLM-L6-v2", "gte-small-zh"]
    embedding_model: str = "all-MiniLM-L6-v2"
    use_onnx_model: bool = True  # Use the ONNX embedding model instead of torch model.
    use_quant_model: bool = False  # Use the embedding model quantized in QInt8 (only available for ONNX model).

    # Database parameters
    database_name: str = "rag_database"
    database_description: str = "Database description (to be overwrited)"
