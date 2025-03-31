# Copyright 2025 NXP
# NXP Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.

import re
from colorama import Fore
from docling.document_converter import DocumentConverter
from rag.utils import save_json, save_markdown


class DoclingParser:
    def __init__(self, max_num_pages: int = 1000):
        self.max_num_pages = max_num_pages
        self.converter = DocumentConverter()

    @staticmethod
    def clean_tables_from_markdown(markdown_file: str) -> str:
        """
        Removes all Markdown tables from the given text.
        A Markdown table is detected as:
        - Lines containing the '|' character (table rows)
        - Separator lines consisting of '-' and '|'
        Any detected table content will be removed, while preserving other text.
        :param markdown_file: The input Markdown text
        :return: The cleaned Markdown text without tables
        """

        lines = markdown_file.split("\n")
        new_lines = []
        inside_table = False
        for line in lines:
            if "|" in line:  # Detect table row
                inside_table = True
                continue
            elif inside_table and re.match(r"^\s*-+\s*(\|-+)*$", line.strip()):  # Detect table header
                continue
            else:
                inside_table = False
                new_lines.append(line)
        return "\n".join(new_lines)

    @staticmethod
    def clean_images_from_markdown(markdown_file: str) -> str:
        """
        Removes all Markdown images from the given text.
        A Markdown image is detected as:
        <!-- image -->
        Any detected image will be removed, while preserving other text.
        :param markdown_file: The input Markdown text
        :return: The cleaned Markdown text without images
        """

        new_markdown_file = markdown_file.replace("<!-- image -->", "")
        return new_markdown_file

    def parse(self, input_file: str, destination_path: str) -> None:
        result = self.converter.convert(source=input_file, max_num_pages=self.max_num_pages)
        content = result.document.export_to_markdown()
        cleaned_content = self.clean_tables_from_markdown(content)
        cleaned_content = self.clean_images_from_markdown(cleaned_content)
        save_markdown(destination_path=destination_path + '.md', content=cleaned_content)
        print(Fore.LIGHTGREEN_EX, "\rSaved: ", destination_path + '.md', Fore.RESET)
