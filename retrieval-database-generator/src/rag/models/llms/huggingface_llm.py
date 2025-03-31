# Copyright 2024-2025 NXP
# NXP Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.

from enum import Enum
from typing import Type
from transformers import pipeline
from dataclasses import dataclass
from rag.models.llms.config import Llama3_8B_config


class AvailableLLMs(str, Enum):
    """LLMs supported."""
    LLAMA3_8B = "llama3.1-8B"

    @property
    def config(self):
        """Returns the corresponding LLM configuration."""
        return {
            AvailableLLMs.LLAMA3_8B: Llama3_8B_config,
        }[self]

    @property
    def model_class(self) -> Type:
        """Returns the corresponding LLM class."""
        return {
            AvailableLLMs.LLAMA3_8B: Llama3,
        }[self]

    def initialize(self, **kwargs):
        """Initializes the LLM with its configuration."""
        return self.model_class(self.config, **kwargs)


class HuggingFaceLLM:
    def __init__(self, model_config: dataclass()):
        self.model_config = model_config

        self.pipe = pipeline(task="text-generation",
                             model=self.model_config.model_id,
                             torch_dtype=self.model_config.torch_dtype,
                             device_map=self.model_config.device,
                             return_full_text=False)

    def generate_input_prompt(self, prompt: str, query: str):
        pass

    def __call__(self, rag_prompt: str, query: str) -> tuple:
        """
        Generate answer from LLM.
        :param prompt: contextual information given as input to the TinyLlama model
        :param query: query given as input to the TinyLlama model
        :return: generated answer from the TinyLlama model
        """

        pipe_input = self.generate_input_prompt(rag_prompt, query)

        pipe_output = self.pipe(pipe_input,
                                max_new_tokens=self.model_config.max_tokens_to_keep,
                                do_sample=True,
                                temperature=self.model_config.temperature,
                                top_k=self.model_config.top_k,
                                top_p=self.model_config.top_p,
                                sequence_bias=self.model_config.sequence_bias,
                                pad_token_id=self.pipe.tokenizer.eos_token_id)

        return pipe_input, pipe_output[0]["generated_text"]


class Llama3(HuggingFaceLLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_input_prompt(self, prompt: str, query: str) -> list:
        return [
            {"role": "system", "content": prompt},
            {"role": "user", "content": query},
        ]
