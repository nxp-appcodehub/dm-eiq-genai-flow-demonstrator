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
import torch
from tqdm import tqdm
from colorama import Fore
from rag.utils import load_json, save_pkl, get_file_list
from rag.models.embedding_models.embedding_models import EmbeddingModel


def generate_embeddings(files_to_keep: list[str],) -> None:
    """
    Create embeddings from the chunks files saved in --origin-folder. If --files-to-keep is left to the default value
    all files will be used.
    """
    src_dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    origin_folder = os.path.join(src_dir_path, "data", "chunked_files")
    saving_folder = os.path.join(src_dir_path, "data")

    data = {}

    embedding_model = EmbeddingModel(name="all-MiniLM-L6-v2")

    data["embedding_model"] = embedding_model.embedding_model_version
    data["database_description"] = input(
        "Enter a brief description of your database content. It will be displayed when loading. "
        "(Press Enter to confirm): "
    )
    data["database_generator_files"] = []
    index = 0

    if files_to_keep == ["all"]:
        files_to_keep = get_file_list(repo_path=origin_folder, extensions=".json")
        if not files_to_keep:
            raise FileNotFoundError(f"You must have at least one json file in {origin_folder} folder.")

    for file_name in files_to_keep:
        data["database_generator_files"].append(file_name)
        file_path = os.path.join(origin_folder, file_name)
        if os.path.isfile(file_path):
            chunks = load_json(file_path)
        else:
            raise ValueError(f"There is no {file_name} in {origin_folder}.")

        # generate embeddings from documentation
        for id, item in tqdm(chunks.items(), desc=f"Generating embeddings for {file_name} file"):
            data[index] = {}
            if not ("chunks" in item):
                raise ValueError(f"Every item must contain at least a chunks attribute.\n {id}: {item}")
            chunks = item.pop("chunks")
            if "complete_chunks" in item:
                embeddings = embedding_model.encode(item["complete_chunks"])
                embeddings = torch.tile(embeddings, (len(chunks), 1))
                reranking_embedding = embedding_model.encode(item["complete_chunks"].split(';')[0])  # Question only
            else:
                embeddings = embedding_model.encode(chunks)
                reranking_embedding = torch.mean(embeddings, dim=0)
            data[index]["embeddings"] = embeddings
            data[index]["reranking_embedding"] = reranking_embedding
            data[index]["chunks"] = chunks
            data[index]["chunked_file_id"] = id
            for key, value in item.items():
                data[index][key] = value

            if "source" not in data[index]:
                data[index]["source"] = file_name

            index += 1

    # save in pkl
    destination_path = os.path.join(saving_folder, "rag_database.pkl")
    save_pkl(destination_path=destination_path, data=data)
    print(Fore.LIGHTGREEN_EX, "\rSuccessfully saved rag_database.pkl at: ", destination_path, Fore.RESET)


def main():
    app = typer.Typer(
        name="Document Embedding",
        add_completion=False,
        context_settings={"help_option_names": ["-h", "--help"]},
    )

    @app.command()
    def parse_args(
            chunked_files_to_embed: list[str] = typer.Option(
                ["all"],
                "--file-to-embed", "-f",
                help=f"File name in data{os.sep}chunked_files to embed in database "
                     "('-f all' for all files and '-f file1 -f file2 ...' for a list of files)",
                show_default=True
            )
    ):

        generate_embeddings(files_to_keep=chunked_files_to_embed)

    app()


if __name__ == '__main__':
    main()
