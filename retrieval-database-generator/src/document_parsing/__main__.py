# Copyright 2025 NXP
# NXP Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.

import typer
import os
from colorama import Fore
from document_parsing.docling_parser import DoclingParser
from document_parsing.utils import get_pdf_file_list

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
        max_num_pages: int = typer.Option(
            1000,
            "--max-num-page", "-m",
            help="Maximum number of pages to parse.",
            show_default=True
        )
    ):
        """
        Parse the PDF file into a manageable Markdown file. (Images and tables are not yet supported)
        """

        src_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        origin_folder = os.path.join(src_dir_path, "data", "input_files")
        saving_folder = os.path.join(src_dir_path, "data", "parsed_files")

        if pdf_files_to_parse == ["all"]:
            pdf_files_to_parse = get_pdf_file_list(origin_folder)
            if not pdf_files_to_parse:
                raise FileNotFoundError(f"You must have at least one PDF file in {origin_folder} folder.")

        parser = DoclingParser(max_num_pages=max_num_pages)

        for file_name in pdf_files_to_parse:
            file_path = os.path.join(origin_folder, file_name)
            print(Fore.LIGHTGREEN_EX, f"\rLoaded: {file_path}", Fore.RESET)
            if os.path.isfile(file_path):
                parser.parse(input_file=file_path,
                             destination_path=os.path.join(saving_folder, (os.path.splitext(file_name)[0])))
            else:
                raise ValueError(f"There is no {file_name} in {origin_folder}.")

    app()


if __name__ == '__main__':
    main()
