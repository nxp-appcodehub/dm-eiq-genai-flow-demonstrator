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
from colorama import Fore
from rag.retrieval import Retriever
from shared_utils.utils import setup_logging, pretty_log
from rag.config import Config as RAGConfig
from rag.models.llms.huggingface_llm import Danube, AvailableLLMs
from rag.models.embedding_models.embedding_models import EmbeddingModel
import logging

logger = logging.getLogger(__name__)


def main():
    app = typer.Typer(
        name="Run LLM with RAG",
        no_args_is_help=True,
        add_completion=False,
        context_settings={"help_option_names": ["-h", "--help"]},
    )

    src_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    @app.command()
    def parse_args(
        generic_prompt: str = typer.Option(
            "You are an assistant, short answer using the following information: ",
            "--prompt", "-p",
            help="Generic guideline given to the LLM.",
            show_default=True
        ),
        llm: AvailableLLMs = typer.Option(
            AvailableLLMs.DANUBE_500M,
            "--model", "-m",
            help="LLM model to be used.",
            show_default=True
        ),
        rag_db_name: str = typer.Option(
            "rag_database.pkl",
            "--rag-database", "-d",
            help=f"RAG database file name in data{os.sep}.",
        ),
        use_traceback: bool = typer.Option(
            False,
            "--use-traceback", "-t",
            help="Activate Typer traceback of exceptions and errors.",
        )
    ):
        """
        Run the LLM specified in --model, fine-tuned with RAG.
        """
        app.pretty_exceptions_enable = use_traceback

        setup_logging(level=logging.INFO, root_path=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

        # define the LLM
        llm = llm.initialize()
        logger.info(f"LLM model used: {llm.model_config.name}")

        # RAG's retriever
        rag_config = RAGConfig()
        rag_db_path = os.path.join(src_dir_path, "data", rag_db_name)
        retriever = Retriever(config=rag_config,
                              embedding_model="all-MiniLM-L6-v2",
                              rag_db=rag_db_path)

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

            # contextual information is retrieved based on the user query
            chunk_list, similarity_list, metadata_list = retriever(query=user_input)

            # Command detection
            if "intent" in metadata_list[0]:
                logger.info(f">>>> Command detected: {metadata_list[0]['intent']}")
                print(Fore.GREEN, '\r', metadata_list[0]['intent'], Fore.RESET)
                logger.info("LLM is bypassed...")
                continue

            rag_prompt = generic_prompt
            if len(chunk_list) > 0:
                rag_prompt += ' '.join(chunk_list)

            llm_input, llm_output = llm(rag_prompt=rag_prompt, query=user_input)

            # Print the retrieved results
            pretty_log(name="LLM", result_dictionary={
                "Prompt": llm_input,
                "Answer": llm_output,
            })
            print(Fore.GREEN, '\r', llm_output, Fore.RESET)

    app()


if __name__ == '__main__':
    main()
