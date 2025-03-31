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
import warnings
import logging
from tqdm import tqdm
from colorama import Fore
from enum import Enum
from rag.utils import save_json, get_file_list, load_json, load_markdown


logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


class AvailableChunkingStrategies(str, Enum):
    """Chunking strategies supported."""
    HIRAG = "HiRAG"
    SPACY = "SpaCy"
    NLTK = "NLTK"
    RECURSIVE = "recursive"
    FIXED = "fixed"


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
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        return RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                              chunk_overlap=chunk_overlap,
                                              separators=["\n\n", "\n", " "]), chunking_method
    elif chunking_method == "fixed":
        from langchain_text_splitters import CharacterTextSplitter
        return CharacterTextSplitter(separator="",
                                     chunk_size=chunk_size,
                                     chunk_overlap=chunk_overlap,
                                     length_function=len), chunking_method
    elif chunking_method == "NLTK":
        from langchain_text_splitters import NLTKTextSplitter
        import nltk
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except:
            print("'punkt_tab' not found. Downloading...")
            nltk.download("punkt_tab")
        return NLTKTextSplitter(chunk_size=chunk_size,
                                chunk_overlap=chunk_overlap), chunking_method

    elif chunking_method == "SpaCy":
        from langchain_text_splitters import SpacyTextSplitter
        import spacy
        try:
            spacy.load("en_core_web_sm")
        except:
            print("'en_core_web_sm' not found. Downloading...")
            spacy.cli.download("en_core_web_sm")
        return SpacyTextSplitter(chunk_size=chunk_size,
                                 chunk_overlap=chunk_overlap), chunking_method

    elif chunking_method == "HiRAG":
        from hirag.hirag_text_splitters import HiRAGTextSplitter
        try:
            from hirag.hirag_text_splitters import HiRAGTextSplitter
            return HiRAGTextSplitter(), chunking_method
        except:
            print(Fore.RED, "\rWarning: The LLM used by HiRAG can't be loaded from Hugging Face. \nInvalid or missing "
                            f"Hugging Face access token. Please check your environment variables.\n"
                            "The SpaCy chunking strategy is used instead.", Fore.RESET)
            return init_text_splitter(chunking_method="SpaCy", chunk_size=128, chunk_overlap=64)

    else:
        raise ValueError("Unknown chunking_method. Must be one of: 'recursive', 'fixed', 'NLTK', 'SpaCy', or 'HiRAG'.")


def generate_chunks(files_to_keep: list[str],
                    chunk_size: int,
                    chunk_overlap: int,
                    chunking_method: AvailableChunkingStrategies) -> None:
    """
    Create chunks from markdown files.
    """

    src_dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    origin_folder = os.path.join(src_dir_path, "data", "parsed_files")
    saving_folder = os.path.join(src_dir_path, "data", "chunked_files")

    text_splitter, splitter_name = init_text_splitter(chunking_method, chunk_size, chunk_overlap)

    if files_to_keep == ["all"]:
        files_to_keep = get_file_list(repo_path=origin_folder, extensions=[".md"])
        if not files_to_keep:
            raise FileNotFoundError(f"You must have at least one file in {origin_folder} folder.")

    for file_name in tqdm(files_to_keep, desc=f"Chunking with {splitter_name}"):

        file_path = os.path.join(origin_folder, file_name)
        if os.path.isfile(file_path):
            extension = os.path.splitext(file_path)[1]
            if extension == ".json":
                data = load_json(file_path)
            elif extension == ".md":
                data = load_markdown(file_path)
            else:
                raise ValueError(f"Unsupported extension: {extension}")
        else:
            raise ValueError(f"There is no {file_name} in {origin_folder}.")

        if splitter_name == "HiRAG":
            chunks = text_splitter.generate_hirag_chunks(data)

        else:
            all_chunks = text_splitter.split_text(data)
            chunks = {}
            for id, chunk in enumerate(all_chunks):
                chunks[id] = {
                    "chunks": [chunk],
                    "source": file_name
                }

        if not os.path.exists(saving_folder):
            os.makedirs(saving_folder)

        saved_file_name = f"{os.path.splitext(file_name)[0]}_{splitter_name}_chunks.json"
        destination_path = os.path.join(saving_folder, saved_file_name)
        save_json(destination_path=destination_path, data=chunks)
        print(Fore.LIGHTGREEN_EX, f"\rSuccessfully saved {saved_file_name} at: ", destination_path, Fore.RESET)


def main():
    app = typer.Typer(
        name="Document chunking",
        add_completion=False,
        context_settings={"help_option_names": ["-h", "--help"]},
    )

    @app.command()
    def parse_args(
            parsed_files_to_chunk: list[str] = typer.Option(
                ["all"],
                "--file-to-chunk", "-f",
                help=f"File name in data{os.sep}parsed_files to chunks "
                     "('-f all' for all files and '-f file1 -f file2 ...' for a list of files)",
                show_default=True
            ),
            chunk_size: int = typer.Option(
                128,
                "--chunk-size", "-s",
                help="Length (in characters) of each chunk. "
                     "Larger chunks may improve response quality but increase processing time.",
                show_default=True
            ),
            chunk_overlap: int = typer.Option(
                64,
                "--chunk-overlap", "-o",
                help="Overlap between consecutive chunks, typically set to half the chunk_size to preserve context "
                     "across chunks.",
                show_default=True
            ),
            chunking_method: AvailableChunkingStrategies = typer.Option(
                "HiRAG",
                "--chunking-method", "-c",
                help="Method used to chunk text. (eIQ GenAI Flow uses HiRAG)",
                show_default=True
            )
    ):
        generate_chunks(files_to_keep=parsed_files_to_chunk,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        chunking_method=chunking_method)

    app()


if __name__ == '__main__':
    main()
