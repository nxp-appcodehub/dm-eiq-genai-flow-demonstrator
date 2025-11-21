# Copyright 2025 NXP
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
from shared_utils.utils import setup_logging
from document_parsing.docling_parser import DoclingParser, AvailableOutputFormat

logger = logging.getLogger(__name__)

def get_pdf_file_list(repo_path: str) -> list[str]:
    """
    Retrieve all PDF files from a given repository path.

    :param repo_path: Path of the repository to search for PDF files
    :return: List of absolute paths to all PDF files found in the repository
    """
    pdf_files = []
    # Walk through all directories and files in the given path
    for root, dirs, files in os.walk(repo_path):
        # Filter files ending with .pdf
        for file in files:
            if file.endswith(".pdf"):
                pdf_files.append(file)  # Append full path
    return pdf_files


def main():
    app = typer.Typer(
        name="Document Parsing",
        no_args_is_help=True,
        add_completion=False,
        context_settings={"help_option_names": ["-h", "--help"]},
    )

    @app.command()
    def parse_args(
        pdf_files_to_parse: list[str] = typer.Option(
            ["all"],
            "--file-to-parse", "-f",
            help=f"File name in data{os.sep}input_files to parse "
                 "('-f all' for all files and '-f file1 -f file2 ...' for a list of files)",
            show_default=True
        ),
        output_format: AvailableOutputFormat = typer.Option(
            ".md",
            "--output-format", "-o",
            help="Format of the output file after parsing.",
            show_default=True
        ),
        use_traceback: bool = typer.Option(
            False,
            "--use-traceback", "-t",
            help="Activate Typer traceback of exceptions and errors.",
        )
    ):
        """
        Parse the PDF file into a manageable Markdown file. (Tables are not yet supported)
        """
        app.pretty_exceptions_enable = use_traceback

        setup_logging(level=logging.INFO, root_path=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

        output_format = output_format.value
        logger.debug(f"Parsing output format: {output_format}")

        src_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        origin_folder = os.path.join(src_dir_path, "data", "input_files")
        saving_folder = os.path.join(src_dir_path, "data", "parsed_files")

        if pdf_files_to_parse == ["all"]:
            pdf_files_to_parse = get_pdf_file_list(origin_folder)
            if not pdf_files_to_parse:
                error_text = f"You must have at least one PDF file in {origin_folder} folder."
                logger.error(error_text)
                raise FileNotFoundError(error_text)

        parser = DoclingParser()

        for file_name in pdf_files_to_parse:
            file_path = os.path.join(origin_folder, file_name)
            logger.info(f"Loaded: {file_path}")
            if os.path.isfile(file_path):
                parser.parse(input_file=file_path,
                             destination_path=os.path.join(saving_folder, os.path.splitext(file_name)[0] + output_format),
                             output_format=output_format)
            else:
                error_text = f"There is no {file_name} in {origin_folder}."
                logger.error(error_text)
                raise ValueError(error_text)

    app()


if __name__ == '__main__':
    main()
