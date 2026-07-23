# Copyright 2025-2026 NXP
# NXP Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.

import logging

logger = logging.getLogger(__name__)


class LLMClientBase:
    def __init__(self, device_name=None, prompt=None):
        self.device_name = device_name
        self.prompt = prompt
        self.long_token = "[...]"
        self.llm_end_token = "</s>"
        self.name = "llm_client_base"
        self.actual_providers = "NA"
        self.llm_input_size = None

    def shutdown(self):
        raise NotImplementedError

    def send(self, data):
        raise NotImplementedError

    def get(self):
        raise NotImplementedError

    def generate(self, question):
        raise NotImplementedError

    def format_prompt(self, rag_prompt):
        final_prompt = self.prompt
        if rag_prompt:
            final_prompt += " " + rag_prompt
        return final_prompt

    def process_question(self, prompt, question):
        raise NotImplementedError
