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
import re
import typer
import logging
from enum import Enum
from typing import Annotated
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ImageRefMode, PictureItem
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, PictureDescriptionVlmOptions, AcceleratorOptions
from shared_utils.utils import setup_logging
from rag_database_generator.utils import save_json, save_markdown
from rag_database_generator import ROOT_DIR
from rag_database_generator.config import Config

logger = logging.getLogger(__name__)


class ParsingOutputFormat(str, Enum):
    """Chunking strategies supported."""
    MARKDOWN = ".md"
    JSON = ".json"


class DoclingParser:
    def __init__(self):
        pipeline_options = PdfPipelineOptions(
            images_scale=2.0,
            generate_page_images=True,
            generate_picture_images=True,
            do_picture_description=True,
            accelerator_options=AcceleratorOptions(cuda_use_flash_attention2=False),
            picture_description_options=PictureDescriptionVlmOptions(
                # Add here a VLM Hugging Face repo_id, like "ibm-granite/granite-vision-3.1-2b-preview"
                repo_id="llava-hf/llava-v1.6-mistral-7b-hf",
                prompt="Describe this image in a few sentences. Be concise and accurate.",
                # trust_remote_code = True
            ),
        )

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            }
        )

    @staticmethod
    def _convert_json_format(parsed_dict: dict) -> dict:
        converted_dict = {}
        id = 0
        section_header = None

        # Manage text
        for elem in parsed_dict["texts"]:
            if elem["label"] == "section_header":
                section_header = elem["text"]
            else:
                if ("pictures" in elem["parent"]["$ref"]) or ("tables" in elem["parent"]["$ref"]):
                    continue
                else:
                    converted_dict[id] = {
                        "text": elem["text"],
                        "self_ref": elem["self_ref"],
                        "page": elem["prov"][0]["page_no"],
                        "parent": elem["parent"]["$ref"],
                        "origin_file": parsed_dict["name"] + ".pdf",
                    }

                    # Add section_header if it's not None
                    if section_header is not None:
                        converted_dict[id]["section_header"] = section_header

                    # Add children if it's not an empty list
                    if elem.get("children"):
                        converted_dict[id]["children"] = elem["children"]
                    id += 1

        # Manage images
        for elem in parsed_dict["pictures"]:
            captions = []
            for refs in elem["captions"]:
                caption_dict = parsed_dict["texts"][int(refs["$ref"].split("/")[-1])]
                print(caption_dict)
                if caption_dict["self_ref"] == refs["$ref"]:
                    captions.append(caption_dict["text"])

            annotations = []
            for annotation in elem["annotations"]:
                annotations.append(annotation["text"])

            if annotations:
                enriched_description = f"![{'AI generated alt text: ' + ' '.join(annotations)}](image.png)\n"
            else:
                enriched_description = "![](image.png)\n"
            if captions:
                enriched_description += " ".join(captions)

            converted_dict[id] = {
                "captions": elem["captions"],
                "annotations": elem["annotations"],
                "text": enriched_description,
                "self_ref": elem["self_ref"],
                "page": elem["prov"][0]["page_no"],
                "parent": elem["parent"]["$ref"],
                "origin_file": parsed_dict["name"] + ".pdf",
            }
            id += 1

        # Manage tables
        # TODO: support tables

        return converted_dict

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
    def annotate_picture_in_markdown(doc, md_content: str, placeholder: str):
        # Get annotations
        picture_counter = 0
        annotations_list = []
        for element, _level in doc.iterate_items():
            if isinstance(element, PictureItem):
                picture_counter += 1
                if element.annotations:
                    alt_text = 'AI generated alt text: '
                    for ann in element.annotations:
                        alt_text += ' ' + ann.text
                        md_content = md_content.replace(ann.text, "")
                    annotation = f"![{alt_text}](image{picture_counter}.png)"
                else:
                    annotation = f"![](image{picture_counter}.png)"
                annotations_list.append(annotation)

        # Replace placeholders sequentially with the corresponding image annotations.
        # The '1' in replace() ensures that each placeholder is replaced in order.
        for ann in annotations_list:
            md_content = md_content.replace(placeholder, ann, 1)

        return md_content

    def parse_file(self, file: str, saving_folder: str, output_format: str) -> tuple[str | dict, str] :
        destination_path = ""
        file_name = os.path.splitext(os.path.basename(file))[0]
        if saving_folder:
            os.makedirs(saving_folder, exist_ok=True)
            destination_path = os.path.join(saving_folder, f"{file_name}{output_format}")
        result = self.converter.convert(source=file)
        if output_format == ".json":
            parsed_dict = result.document.export_to_dict()
            final_content = self._convert_json_format(parsed_dict=parsed_dict)
            if destination_path:
                save_json(destination_path=destination_path, data=final_content)
        elif output_format == ".md":
            placeholder = "%%ANNOTATION%%"
            content = result.document.export_to_markdown(
                image_mode=ImageRefMode.PLACEHOLDER, image_placeholder=placeholder
            )
            enriched_content = self.annotate_picture_in_markdown(
                doc=result.document, md_content=content, placeholder=placeholder
            )
            final_content = self.clean_tables_from_markdown(enriched_content)  # TODO: support tables and remove
            if destination_path:
                save_markdown(destination_path=destination_path, content=final_content)
        else:
            error_text = f"This output_type: {output_format} is not supported."
            logger.error(error_text)
            raise ValueError(error_text)

        if destination_path:
            print(f"Saved: {destination_path}")
            logger.info(f"Saved: {destination_path}")

        return final_content, file_name

    def parse(self, files_to_keep: list[str], saving_folder: str, output_format: ParsingOutputFormat) -> list:
        output = []
        for file_path in files_to_keep:
            logger.info(f"Loaded: {file_path}")
            if os.path.isfile(file_path):
                res = self.parse_file(
                    file=file_path,
                    saving_folder=saving_folder,
                    output_format=output_format.value,
                )
                output.append(res)
            else:
                error_text = f"There is no {file_path}."
                logger.error(error_text)
                raise ValueError(error_text)
        return output


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
        logger.debug(f"Parsing output format: {output_format}")

        saving_folder = os.path.join(ROOT_DIR, "data", "parsed_files")

        parser = DoclingParser()

        parser.parse(
            files_to_keep=file_path_list,
            saving_folder=saving_folder,
            output_format=output_format,
        )

    app()


if __name__ == "__main__":
    main()
