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
from document_parsing.docling_parser import DoclingParser
from document_parsing.config import Config as DocumentParsingConfig


if __name__ == "__main__":
    config = DocumentParsingConfig()
    parser = DoclingParser(config=config)
    examples_path = os.path.dirname(os.path.abspath(__file__))
    result = parser.converter.convert(source=os.path.join(examples_path, "resources", "Medical.pdf"),
                                      max_num_pages=config.max_num_pages)
    markdown = result.document.export_to_markdown()
    print(markdown)