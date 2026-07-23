# Copyright 2024-2026 NXP
# NXP Confidential and Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.

import os
import re
import typer
import torch
from enum import Enum
from tqdm import tqdm
from typing import Annotated, Optional
from datasets import load_dataset
from shared_utils.utils import setup_logging, get_leaf_classes
from rag_database_generator import ROOT_DIR
from rag_database_generator.utils import load_json, save_pkl
from retriever import EmbeddingModel
from rag_database_generator.config import Config
import logging

logger = logging.getLogger(__name__)


class AdditionalDatasets(str, Enum):
    """Additional datasets supported."""

    GENERAL_KNOWLEDGE = "general_knowledge"
    HEALTH = "health"


EmbeddingModels = Enum("EmbeddingModels", {re.sub(r"[^A-Za-z0-9]+", "_", cls_.name).upper(): cls_.name
                                           for cls_ in get_leaf_classes(EmbeddingModel)})


def generate_additional_knowledge_chunks(knowledge_domains: list[str]) -> list[dict]:
    """
    Generate a chunks json file from external knowledge datasets on selected domains.
    :param knowledge_domains: list of domains the user wishes to give the model access to
    :return list[dict]: List of external chunk dictionaries
    """
    additional_knowledge_list = []

    for domain in knowledge_domains:
        knowledge_chunks = {}
        if domain == "general_knowledge":
            ds = load_dataset("MuskumPillerum/General-Knowledge", split="train")
            for id, row in enumerate(ds):
                knowledge_chunks[id] = {}
                chunk = (row["Question"] or "") + ";" + (row["Answer"] or "")
                knowledge_chunks[id]["chunks"] = [row["Answer"] or ""]
                knowledge_chunks[id]["complete_chunks"] = chunk
                knowledge_chunks[id]["source"] = "general_knowledge_db"
                knowledge_chunks[id]["domain"] = domain

        elif domain == "health":
            ds = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train")
            for id, row in enumerate(ds):
                knowledge_chunks[id] = {}
                chunk = (row["Question"] or "") + ";" + (row["Answer"] or "")
                knowledge_chunks[id]["chunks"] = [row["Answer"] or ""]
                knowledge_chunks[id]["complete_chunks"] = chunk
                knowledge_chunks[id]["source"] = "health_db"
                knowledge_chunks[id]["domain"] = domain
        else:
            error_text = f"Unknown additional database domain: {domain}"
            logger.error(error_text)
            raise ValueError(error_text)

        additional_knowledge_list.append(knowledge_chunks)

    return additional_knowledge_list


def generate_embeddings(
    config: Config,
    files_to_keep: list[str] | dict,
    embedding_model: str | EmbeddingModel,
    additional_knowledge_domains: list[str] = None,
    saving_folder: str = None,
) -> dict:
    """
    Create embeddings from the chunks files saved in --origin-folder. If --files-to-keep is left to the default value
    all files will be used.
    """
    os.makedirs(saving_folder, exist_ok=True)
    data = {}
    index = 0

    if isinstance(embedding_model, str):
        embedding_model = EmbeddingModel(name=embedding_model, config=config)

    data["embedding_model"] = embedding_model.embedding_model_version
    data["database_description"] = config.database_description
    data["database_generator_files"] = []

    for source in files_to_keep:
        if isinstance(source, dict):
            chunks = source
            file_name = "Generated from chunks directly."
            data["database_generator_files"].append("Generated from chunks directly.")

        elif isinstance(source, str):
            file_name = os.path.splitext(os.path.basename(source))[0]
            if not os.path.isfile(source):
                raise ValueError(f"There is no {source}.")
            chunks = load_json(source)
            data["database_generator_files"].append(file_name)

        else:
            raise TypeError(f"Expected str or dict, got {type(source)}")

        for id, item in tqdm(chunks.items(), desc=f"Generating embeddings for {file_name}"):
            data[index] = {}
            if not ("chunks" in item):
                error_text = f"Every item must contain at least a chunks attribute.\n {id}: {item}"
                logger.error(error_text)
                raise ValueError(error_text)
            chunks = item.pop("chunks")
            if "complete_chunks" in item:
                embeddings = embedding_model.encode(item["complete_chunks"])
                embeddings = torch.tile(embeddings, (len(chunks), 1))
                reranking_embedding = embedding_model.encode(item["complete_chunks"].split(";")[0])  # Question only
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

    # add additional knowledge chunks
    if additional_knowledge_domains:
        additional_knowledge_list = generate_additional_knowledge_chunks(knowledge_domains=additional_knowledge_domains)
        for additional_knowledge, domain in zip(additional_knowledge_list, additional_knowledge_domains):
            for id, item in tqdm(
                additional_knowledge.items(), desc=f"Generating embeddings from {domain} knowledge database"
            ):
                data[index] = {}
                chunks = item.pop("chunks")
                if "complete_chunks" in item:
                    embeddings = embedding_model.encode(item["complete_chunks"])
                    embeddings = torch.tile(embeddings, (len(chunks), 1))
                    reranking_embedding = embedding_model.encode(item["complete_chunks"].split(";")[0])  # question
                else:
                    embeddings = embedding_model.encode(chunks)
                    reranking_embedding = torch.mean(embeddings, dim=0)
                data[index]["embeddings"] = embeddings
                data[index]["chunks"] = chunks
                data[index]["reranking_embedding"] = reranking_embedding
                data[index]["chunked_file_id"] = id
                for key, value in item.items():
                    data[index][key] = value
                index += 1

    # save in pkl
    destination_path = os.path.join(saving_folder, f"{config.database_name}.pkl")
    save_pkl(destination_path=destination_path, data=data)
    print(f"Successfully saved {destination_path}")
    logger.info(f"Successfully saved {destination_path}")

    return data


def main():
    app = typer.Typer(
        name="Document Embedding",
        add_completion=False,
        context_settings={"help_option_names": ["-h", "--help"]},
    )
    LoggingLevel = Enum("LoggingLevel", {level: level for level in logging._nameToLevel.keys()})

    @app.command()
    def parse_args(
        file_path_list: Annotated[list[str], typer.Argument(
            help="Files to process. Can be repeated: file1.json file2.json..."
        )],
        additional_knowledge_domains: Optional[list[AdditionalDatasets]] = typer.Option(
            None,
            "--additional-knowledge",
            "-a",
            help="List of domain you want to add to your rag_database.pkl file.",
            show_default=True,
        ),
        logging_level: LoggingLevel = typer.Option(
            "INFO",
            "--logging-level",
            "-l",
            help="Level of displayed information.",
        ),
        use_traceback: bool = typer.Option(
            False,
            "--use-traceback",
            "-t",
            help="Activate Typer traceback of exceptions and errors.",
        ),
    ):
        """
        Generate RAG database from chunk file(s).
        """
        app.pretty_exceptions_enable = use_traceback
        setup_logging(level=logging._nameToLevel[logging_level.value], root_path=ROOT_DIR)

        config = Config()
        embedding_model = EmbeddingModel(name=config.embedding_model, config=config)
        logger.debug(f"Embedding model: {config.embedding_model}")

        saving_folder = os.path.join(ROOT_DIR, "data", "databases")

        if additional_knowledge_domains:
            additional_knowledge_domains = [domain.value for domain in additional_knowledge_domains]

        generate_embeddings(
            config=config,
            files_to_keep=file_path_list,
            additional_knowledge_domains=additional_knowledge_domains,
            embedding_model=embedding_model,
            saving_folder=saving_folder
        )

    app()


if __name__ == "__main__":
    main()
