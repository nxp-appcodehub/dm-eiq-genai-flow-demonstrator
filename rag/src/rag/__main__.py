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
import logging
from enum import Enum
from rag.retrieval import Retriever
from rag.config import Config as RAGConfig
from rag.utils import setup_logging, get_leaf_classes
from rag.models.embedding_models.embedding_models import EmbeddingModel

logger = logging.getLogger(__name__)


def main():
    app = typer.Typer(
        name="Retriever",
        no_args_is_help=True,
        add_completion=False,
        context_settings={"help_option_names": ["-h", "--help"]},
    )

    src_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    rag_config = RAGConfig()

    LoggingLevel = Enum("LoggingLevel", {level: level for level in logging._nameToLevel.keys()})

    @app.command()
    def parse_args(
        rag_db_name: str = typer.Option(
            "rag_database.pkl",
            "--rag-database", "-d",
            help=f"RAG database file name in data{os.sep}.",
        ),
        logging_level: LoggingLevel = typer.Option(
            "INFO",
            '--logging-level', "-l",
            help="Level of displayed information.",
        ),
        use_traceback: bool = typer.Option(
            False,
            "--use-traceback", "-t",
            help="Activate Typer traceback of exceptions and errors.",
        )
    ):
        """
        Retrieve chunk(s) from RAG database.
        """
        app.pretty_exceptions_enable = use_traceback

        logging_level = logging._nameToLevel[logging_level.value]
        setup_logging(level=logging_level, root_path=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

        embedding_model = EmbeddingModel(config=rag_config,
                                         name="all-MiniLM-L6-v2")

        rag_db_path = os.path.join(src_dir_path, "data", rag_db_name)

        retriever = Retriever(config=rag_config,
                              embedding_model=embedding_model,
                              rag_db=rag_db_path)

        # contextual information is retrieved based on the user query
        while True:
            # Ask the user for a question
            user_input = input("Ask a question (or type 'q' to quit): ")

            # Check if the user pressed only Enter (empty input)
            if user_input == '':
                continue  # Do nothing and continue the loop

            # Check if the user wants to quit
            if user_input == 'q':
                logger.warning("Exiting the program.")
                break

            best_chunk_list, _, _ = retriever(query=user_input)

    app()


if __name__ == '__main__':
    main()
