# Copyright 2025-2026 NXP
# NXP Confidential and Proprietary.
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
import logging
from enum import Enum
from typing import Annotated, Optional
from shared_utils.utils import setup_logging
from rag_database_generator import ROOT_DIR
from rag_database_generator.config import Config
from rag_database_generator.parse import ParsingOutputFormat, DoclingParser
from rag_database_generator.chunk import ChunkingStrategies, generate_chunks
from rag_database_generator.embed import EmbeddingModel, AdditionalDatasets, generate_embeddings

logger = logging.getLogger(__name__)


def main():
    app = typer.Typer(
        name="Document Parsing",
        no_args_is_help=True,
        add_completion=False,
        context_settings={"help_option_names": ["-h", "--help"]},
    )
    LoggingLevel = Enum("LoggingLevel", {level: level for level in logging._nameToLevel.keys()})

    @app.command()
    def parse_args(
        file_path_list: Annotated[list[str], typer.Argument(
            help="Files to process. Can be repeated: file1.pdf file2.pdf..."
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
        Parse the PDF file into a manageable Markdown file. (Tables are not yet supported)
        """
        app.pretty_exceptions_enable = use_traceback
        setup_logging(level=logging._nameToLevel[logging_level.value], root_path=ROOT_DIR)

        config = Config()
        output_format = ParsingOutputFormat(config.output_format)

        data_dir = os.path.join(ROOT_DIR, "data")

        # Parsing
        parser = DoclingParser()
        parsed_files = []
        for file_path in file_path_list:
            logger.info(f"Loaded: {file_path}")
            if os.path.isfile(file_path):
                parsed_files.append(parser.parse_file(file=file_path,
                                                      saving_folder="",
                                                      output_format=output_format)
                                    )

            else:
                error_text = f"There is no {file_path}."
                logger.error(error_text)
                raise ValueError(error_text)

        # Free GPU memory
        parser = None
        torch.cuda.empty_cache()

        # Chunking
        chunking_method = ChunkingStrategies(config.chunking_method)
        chunks = generate_chunks(config=config,
                                 files_to_keep=parsed_files,
                                 chunking_method=chunking_method,
                                 saving_folder="")

        # Embedding
        embedding_model = EmbeddingModel(name=config.embedding_model, config=config)
        if additional_knowledge_domains:
            additional_knowledge_domains = [domain.value for domain in additional_knowledge_domains]
        generate_embeddings(config=config,
                            files_to_keep=[chunks],
                            additional_knowledge_domains=additional_knowledge_domains,
                            embedding_model=embedding_model,
                            saving_folder=os.path.join(data_dir, "databases"))

    app()


if __name__ == "__main__":
    main()
