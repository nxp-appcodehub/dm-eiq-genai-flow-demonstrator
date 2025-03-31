# Copyright 2024-2025 NXP
# NXP Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.

import typer
from enum import Enum
from colorama import Fore
from rag.retrieval import Retriever
from rag.utils import pretty_print, get_leaf_classes
from rag.config import Config as RAGConfig
from rag.models.llms.huggingface_llm import Danube, AvailableLLMs
from rag.models.embedding_models.embedding_models import EmbeddingModel



def main():
    app = typer.Typer(
        name="Run LLM with RAG",
        no_args_is_help=True,
        add_completion=False,
        context_settings={"help_option_names": ["-h", "--help"]},
    )

    AvailableEmbeddingModels = Enum('AvailableEmbeddingModels',
                                    {cls_.name: cls_.name for cls_ in get_leaf_classes(EmbeddingModel)})

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
        embedding_model: AvailableEmbeddingModels = typer.Option(
            "all-MiniLM-L6-v2",
            "--embedding-model", "-e",
            help="List of domain you want to add to your rag_database.pkl file.",
            show_default=True
        ),
        verbose: bool = typer.Option(
            False,
            "--verbose", "-v",
            help="Display more information.",
            show_default=True
        )
    ):
        """
        Run the LLM specified in --model, fine-tuned with RAG.
        """

        # define the LLM
        llm = llm.initialize()
        print(Fore.LIGHTGREEN_EX, f"\rLLM model used: {llm.model_config.name}", Fore.RESET)

        embedding_model = embedding_model.value

        # RAG's retriever
        rag_config = RAGConfig()
        retriever = Retriever(config=rag_config,
                              embedding_model_name=embedding_model,
                              verbose=verbose)

        while True:
            # Ask the user for a question
            user_input = input("Ask a question (or type 'q' to quit): ")

            # Check if the user pressed only Enter (empty input)
            if user_input == '':
                continue  # Do nothing and continue the loop

            # Check if the user wants to quit
            if user_input == 'q':
                print("Exiting the program.")
                break

            # contextual information is retrieved based on the user query
            chunk_list, similarity_list, metadata_list = retriever(query=user_input)

            # Command detection
            if "intent" in metadata_list[0]:
                print("\033[91m", f"\r >>>> Command n°{metadata_list[0]['intent']} detected.\nLLM is bypassed.", "\033[0m")
                continue

            rag_prompt = generic_prompt
            if len(chunk_list) > 0:
                rag_prompt += ' '.join(chunk_list)

            # Special prompting for Danube
            if isinstance(llm, Danube):
                rag_prompt = (chunk_list, metadata_list)

            llm_input, llm_output = llm(rag_prompt=rag_prompt, query=user_input)

            if verbose:
                # Print the retrieved results
                pretty_print(name="LLM", result_dictionary={
                    "Prompt": llm_input,
                    "Answer": llm_output,
                })
            else:
                print(Fore.GREEN, llm_output, Fore.RESET)

    app()


if __name__ == '__main__':
    main()
