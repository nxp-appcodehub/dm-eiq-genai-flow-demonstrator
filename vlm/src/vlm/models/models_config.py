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
from dataclasses import dataclass, field

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(BASE_DIR, 'models/')


class VLM_config:
    name: str
    config_id: str

    vision_session: str
    embedding_session: str
    decoder_session: str


@dataclass
class SmolVLM2_256M_config(VLM_config):
    hf_repo_id: str = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
    hf_subfolder = "onnx/"
    hf_processors_list: list[str] = field(default_factory=lambda: [
        'added_tokens.json',
        'chat_template.json',
        'config.json',
        'generation_config.json',
        'merges.txt',
        'preprocessor_config.json',
        'processor_config.json',
        'special_tokens_map.json',
        'tokenizer.json',
        'tokenizer_config.json',
        'vocab.json'
    ])

    hf_vision_session = "vision_encoder.onnx"
    hf_embedding_session = "embed_tokens.onnx"
    hf_decoder_session = "decoder_model_merged.onnx"

    config_id: str = 'smolvlm/256M/processor/'
    processor_id: str = 'smolvlm/256M/processor/'

    model_folder = "smolvlm/256M/float/"

    vision_session: str = model_folder + hf_subfolder + hf_vision_session
    embedding_session: str = model_folder + hf_subfolder + hf_embedding_session
    decoder_session: str = model_folder + hf_subfolder + hf_decoder_session

    eou_token_id = 49279
    temperature = 0.2
    top_p = 0.9


@dataclass
class SmolVLM2_256M_q8_config(SmolVLM2_256M_config):
    model_folder = "smolvlm/256M/int8/"
    hf_subfolder = "onnx/"

    hf_vision_session = "vision_encoder_quantized.onnx"
    hf_embedding_session = "embed_tokens_int8.onnx"
    hf_decoder_session = "decoder_model_merged_int8.onnx"

    vision_session: str = model_folder + hf_subfolder + hf_vision_session
    embedding_session: str = model_folder + hf_subfolder + hf_embedding_session
    decoder_session: str = model_folder + hf_subfolder + hf_decoder_session


@dataclass
class SmolVLM2_500M_config(VLM_config):
    hf_repo_id: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    hf_subfolder = "onnx/"
    hf_processors_list: list[str] = field(default_factory=lambda: [
        'added_tokens.json',
        'chat_template.json',
        'config.json',
        'generation_config.json',
        'merges.txt',
        'preprocessor_config.json',
        'processor_config.json',
        'special_tokens_map.json',
        'tokenizer.json',
        'tokenizer_config.json',
        'vocab.json'
    ])

    hf_vision_session = "vision_encoder.onnx"
    hf_embedding_session = "embed_tokens.onnx"
    hf_decoder_session = "decoder_model_merged.onnx"

    config_id: str = 'smolvlm/500M/processor/'
    processor_id: str = 'smolvlm/500M/processor/'

    model_folder = "smolvlm/500M/float/"

    vision_session: str = model_folder + hf_subfolder + hf_vision_session
    embedding_session: str = model_folder + hf_subfolder + hf_embedding_session
    decoder_session: str = model_folder + hf_subfolder + hf_decoder_session

    eou_token_id = 49279
    temperature = 0.2
    top_p = 0.9


@dataclass
class SmolVLM2_500M_q8_config(SmolVLM2_500M_config):
    model_folder = "smolvlm/500M/int8/"
    hf_subfolder = "onnx/"

    hf_vision_session = "vision_encoder_quantized.onnx"
    hf_embedding_session = "embed_tokens_int8.onnx"
    hf_decoder_session = "decoder_model_merged_int8.onnx"

    vision_session: str = model_folder + hf_subfolder + hf_vision_session
    embedding_session: str = model_folder + hf_subfolder + hf_embedding_session
    decoder_session: str = model_folder + hf_subfolder + hf_decoder_session


@dataclass
class SmolVLM2_500M_q8_config_neutron(SmolVLM2_500M_q8_config):

    hf_vision_session = "vision_encoder_quantized.onnx"
    hf_embedding_session = "embed_tokens_int8.onnx"
    hf_decoder_session = "decoder_model_merged_int8.onnx"
