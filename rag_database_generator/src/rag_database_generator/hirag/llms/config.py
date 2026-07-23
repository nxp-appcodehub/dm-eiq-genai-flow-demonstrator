# Copyright 2024-2026 NXP
# NXP Confidential and Proprietary.
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
class LLM_Config:
    torch_dtype = torch.bfloat16
    min_tokens_to_keep: int = 1
    temperature: float = 0.25
    top_k: int = 10000
    top_p: float = 0.95
    repetition_penalty: float = 1.2
    long_token: str = "[...]"


@dataclass
class Tinyllama_Config(LLM_Config):
    name: str = "TinyLlama-1B"
    model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    type: str = "Chat"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_key_transposed: bool = False
    max_tokens_to_keep: int = 128
    sequence_bias = {(2,): 3.0}


@dataclass
class Danube_Config(LLM_Config):
    name: str = "Danube3-500M"
    model_id: str = "h2oai/h2o-danube3-500m-chat"
    type: str = "Chat"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_key_transposed: bool = True
    max_tokens_to_keep: int = 96
    sequence_bias = {(2,): 3.0, (28736,): -10.0, (348,): -10.0, (28703,): -10.0, (619,): -10.0}


@dataclass
class Gemma_Config(LLM_Config):
    name: str = "Gemma-2B"
    model_id: str = "google/gemma-2b-it"
    type: str = "Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_key_transposed: bool = False
    max_tokens_to_keep: int = 128
    sequence_bias = {(1,): 3.0}


@dataclass
class Llama2_Config(LLM_Config):
    name: str = "Llama2-7B"
    model_id: str = "meta-llama/Llama-2-7b-chat-hf"
    type: str = "Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_key_transposed: bool = True
    max_tokens_to_keep: int = 1000
    sequence_bias = {(2,): 3.0}


@dataclass
class _Llama3_Config(LLM_Config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_key_transposed: bool = True
    sequence_bias = {(2,): 3.0}


@dataclass
class Llama3_1B_Config(_Llama3_Config):
    name: str = "Llama3.2-1B"
    model_id: str = "meta-llama/Llama-3.2-1B-Instruct"
    type: str = "Instruct"
    max_tokens_to_keep: int = 128


@dataclass
class Llama3_8B_Config(_Llama3_Config):
    name = "Llama3.1-8B"
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    type: str = "Instruct"
    max_tokens_to_keep: int = 1000
