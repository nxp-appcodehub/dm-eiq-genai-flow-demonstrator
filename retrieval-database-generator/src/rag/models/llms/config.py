# Copyright 2024-2025 NXP
# NXP Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.

from dataclasses import dataclass
import torch


@dataclass
class LLM_config:
    torch_dtype = torch.bfloat16
    min_tokens_to_keep: int = 1
    temperature: float = 0.25
    top_k: int = 10000
    top_p: float = 0.95
    repetition_penalty: float = 1.2
    long_token: str = "[...]"


@dataclass
class Llama3_8B_config(LLM_config):
    name = "Llama3.1-8B"
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    type: str = "Instruct"
    max_tokens_to_keep: int = 1000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_key_transposed: bool = True
    sequence_bias = {(2,): 3.}
