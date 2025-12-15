# Copyright 2024-2025 NXP
# NXP Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.

import os
import typer
import nltk
import spacy
import logging
from enum import Enum
from tqdm import tqdm
from rag.config import Config as RAGConfig
from hirag.config import Config as HiRAGConfig
from rag.utils import save_json, get_file_list, load_json, load_markdown, setup_logging
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter, NLTKTextSplitter, SpacyTextSplitter

logger = logging.getLogger(__name__)


class AvailableChunkingStrategies(str, Enum):
    """Chunking strategies supported."""
    HIRAG = "HiRAG"
    SPACY = "SpaCy"
    NLTK = "NLTK"
    RECURSIVE = "recursive"
    FIXED = "fixed"


def merge_dicts_with_new_keys(dict1: dict, dict2: dict) -> dict:
    """
    Function to merge dictionaries with new keys
    :param dict1:
    :param dict2:
    :return: dict1 + dict2 with incremented ids
    """

    if dict1 == {}:
        return dict2

    result = dict1.copy()  # Start with a copy of the first dictionary
    offset = len(dict1)  # Offset for new keys

    for i, (key, value) in enumerate(dict2.items(), start=1):
        new_key = str(offset + i)
        result[new_key] = value

    return result


def init_text_splitter(chunking_method: str, chunk_size: int, chunk_overlap: int) -> tuple:
    """
    Initialize a text splitter based on the specified chunking method.
    :param chunking_method: The chunking method used.
    :param chunk_size: The chunk size.
    :param chunk_overlap: The chunk overlap.
    :return: tuple[Union[RecursiveCharacterTextSplitter, CharacterTextSplitter, NLTKTextSplitter, SpacyTextSplitter, HiRAGTextSplitter], str]
    :raise: ValueError: If the `chunking_method` is not one of the recognized methods.
    """

    if chunking_method == "recursive":
        return RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                              chunk_overlap=chunk_overlap,
                                              separators=["\n\n", "\n", " "]), chunking_method
    elif chunking_method == "fixed":
        return CharacterTextSplitter(separator="",
                                     chunk_size=chunk_size,
                                     chunk_overlap=chunk_overlap,
                                     length_function=len), chunking_method
    elif chunking_method == "NLTK":
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except:
            logger.warning("'punkt_tab' not found. Downloading...")
            nltk.download("punkt_tab")
        return NLTKTextSplitter(chunk_size=chunk_size,
                                chunk_overlap=chunk_overlap), chunking_method

    elif chunking_method == "SpaCy":
        try:
            spacy.load("en_core_web_sm")
        except:
            logger.warning("'en_core_web_sm' not found. Downloading...")
            spacy.cli.download("en_core_web_sm")
        return SpacyTextSplitter(chunk_size=chunk_size,
                                 chunk_overlap=chunk_overlap), chunking_method

    elif chunking_method == "HiRAG":
        from hirag.hirag_text_splitters import HiRAGTextSplitter
        try:
            config = HiRAGConfig()
            return HiRAGTextSplitter(config=config), chunking_method
        except Exception:
            logger.error(f"The LLM specified in rag/src/rag/preprocessing/hirag/config.py: {HiRAGConfig().llm_name} can't be loaded.", exc_info=True)
            logger.warning("Possible reasons:")
            logger.warning("   • Invalid/missing Hugging Face access token. Please check your environment variables.")
            logger.warning("   • GPU overloaded")
            logger.warning("The SpaCy chunking strategy is used instead.")
            return init_text_splitter(chunking_method="SpaCy", chunk_size=128, chunk_overlap=64)

    else:
        error_text = ("Unknown chunking_method. Must be one of: " +
                      str([strategy.value for strategy in AvailableChunkingStrategies]))
        logger.error(error_text)
        raise ValueError(error_text)


def generate_chunks(config: RAGConfig,
                    files_to_keep: list[str],
                    chunking_method: AvailableChunkingStrategies,
                    input_folder: str = None,
                    output_folder: str = None) -> dict:
    """
    Create chunks from JSON or Markdown files saved in src/parsed_files.
    """

    src_dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    origin_folder = os.path.join(src_dir_path, "data", "parsed_files") if input_folder is None else input_folder
    saving_folder = os.path.join(src_dir_path, "data", "chunked_files") if output_folder is None else output_folder

    text_splitter, splitter_name = init_text_splitter(chunking_method.value, config.chunk_size, config.chunk_overlap)

    all_chunks = {}

    if files_to_keep == ["all"]:
        files_to_keep = get_file_list(repo_path=origin_folder, extensions=[".json", ".md"])
        if not files_to_keep:
            error_text = f"You must have at least one file in {origin_folder} folder."
            logger.error(error_text)
            raise FileNotFoundError(error_text)

    for file_name in files_to_keep:

        file_path = os.path.join(origin_folder, file_name)
        if os.path.isfile(file_path):
            extension = os.path.splitext(file_path)[1]
            if extension == ".json":
                data = load_json(file_path)
            elif extension == ".md":
                data = load_markdown(file_path)
            else:
                error_text = f"Unsupported extension: {extension}"
                logger.error(error_text)
                raise ValueError(error_text)
        else:
            error_text = f"There is no {file_name} in {origin_folder}."
            logger.error(error_text)
            raise ValueError(error_text)

        if splitter_name == "HiRAG":
            chunks = text_splitter.generate_hirag_chunks(data)

        else:
            chunks = {}

            if isinstance(data, dict):
                id = 0
                for key, value in tqdm(data.items(), desc=f"Chunking text with {splitter_name}"):
                    text = value.pop('text')
                    chunks_entry = {'chunks': text_splitter.split_text(text)}
                    # Add metadata
                    for k, v in value.items():
                        chunks_entry[k] = v
                    chunks[id] = chunks_entry  # Assign to chunks
                    id += 1
            else:
                splited_chunks = text_splitter.split_text(data)
                chunks = {}
                for id, chunk in tqdm(enumerate(splited_chunks), desc=f"Chunking text with {splitter_name}"):
                    chunks[id] = {
                        "chunks": [chunk],
                        "source": file_name
                    }

        if not os.path.exists(saving_folder):
            os.makedirs(saving_folder)

        saved_file_name = f"{os.path.splitext(file_name)[0]}_{splitter_name}_chunks.json"
        destination_path = os.path.join(saving_folder, saved_file_name)
        save_json(destination_path=destination_path, data=chunks)
        print(f"Successfully saved {destination_path}")
        logger.info(f"Successfully saved {destination_path}")

        all_chunks = merge_dicts_with_new_keys(all_chunks, chunks)

    text_splitter = None
    return all_chunks


def main():
    app = typer.Typer(
        name="Document chunking",
        add_completion=False,
        context_settings={"help_option_names": ["-h", "--help"]},
    )

    config = RAGConfig()

    @app.command()
    def parse_args(
            parsed_files_to_chunk: list[str] = typer.Option(
                ["all"],
                "--file-to-chunk", "-f",
                help=f"File name in data{os.sep}parsed_files to chunks "
                     "('-f all' for all files and '-f file1 -f file2 ...' for a list of files)",
                show_default=True
            ),
            chunking_method: AvailableChunkingStrategies = typer.Option(
                "HiRAG",
                "--chunking-method", "-c",
                help="Method used to chunk text.",
                show_default=True
            ),
            use_traceback: bool = typer.Option(
                False,
                "--use-traceback", "-t",
                help="Activate Typer traceback of exceptions and errors.",
            )
    ):
        """
        Generate chunk file(s).
        """
        app.pretty_exceptions_enable = use_traceback

        setup_logging(level=logging.INFO, root_path=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

        generate_chunks(config=config,
                        files_to_keep=parsed_files_to_chunk,
                        chunking_method=chunking_method)

    app()


if __name__ == '__main__':
    main()
