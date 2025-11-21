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
import torch
from enum import Enum
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ImageRefMode, PictureItem
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, PictureDescriptionVlmOptions, AcceleratorOptions
from rag.utils import save_json, save_markdown
import logging

logger = logging.getLogger(__name__)

class AvailableOutputFormat(str, Enum):
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
            accelerator_options=AcceleratorOptions(cuda_use_flash_attention2=True),
            picture_description_options = PictureDescriptionVlmOptions(
                repo_id="llava-hf/llava-v1.6-mistral-7b-hf",  # <-- add here a VLM Hugging Face repo_id, like "ibm-granite/granite-vision-3.1-2b-preview"
                prompt="Describe this image in a few sentences. Be concise and accurate.",
                # trust_remote_code = True
            )
        )
        pipeline_options.accelerator_options.cuda_use_flash_attention2 = True

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
                        "origin_file": parsed_dict["name"] + ".pdf"
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
                caption_dict = parsed_dict["texts"][int(refs["$ref"].split('/')[-1])]
                print(caption_dict)
                if caption_dict["self_ref"] == refs["$ref"]:
                    captions.append(caption_dict["text"])

            annotations = []
            for annotation in elem["annotations"]:
                annotations.append(annotation["text"])

            if annotations:
                enriched_description = f"![{'AI generated alt text: ' + ' '.join(annotations)}](image.png)\n"
            else:
                enriched_description = f"![](image.png)\n"
            if captions:
                enriched_description += ' '.join(captions)

            converted_dict[id] = {
                "captions": elem["captions"],
                "annotations": elem["annotations"],
                "text": enriched_description,
                "self_ref": elem["self_ref"],
                "page": elem["prov"][0]["page_no"],
                "parent": elem["parent"]["$ref"],
                "origin_file": parsed_dict["name"] + ".pdf"
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
                    annotation = f"![{'AI generated alt text: ' + ' '.join(ann.text for ann in element.annotations)}](image{picture_counter}.png)"
                else:
                    annotation = f"![](image{picture_counter}.png)"
                annotations_list.append(annotation)

        # Replace placeholders sequentially with the corresponding image annotations.
        # The '1' in replace() ensures that each placeholder is replaced in order.
        for ann in annotations_list:
            md_content = md_content.replace(placeholder, ann, 1)

        return md_content

    def parse(self, input_file: str, destination_path: str, output_format: str) -> str | dict:
        result = self.converter.convert(source=input_file)
        if output_format == ".json":
            parsed_dict = result.document.export_to_dict()
            final_content = self._convert_json_format(parsed_dict=parsed_dict)
            if destination_path:
                save_json(destination_path=destination_path, data=final_content)
        elif output_format == ".md":
            placeholder = "%%ANNOTATION%%"
            content = result.document.export_to_markdown(image_mode=ImageRefMode.PLACEHOLDER,
                                                         image_placeholder=placeholder)
            enriched_content = self.annotate_picture_in_markdown(doc=result.document,
                                                                md_content=content,
                                                                placeholder=placeholder)
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

        return final_content
