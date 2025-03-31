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
import sys
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer
from colorama import Fore
from rag.utils import get_number_of_cores
import onnxruntime as ort


class EmbeddingModel:
    name = None
    saving_folder = os.path.dirname(__file__)
    if hasattr(sys, '_MEIPASS'):
        # Update path for Pyinstaller package
        saving_folder = saving_folder.replace("_internal", "rag/src")

    def __new__(cls, name: str):
        """Dynamically instantiate the correct subclass based on `name`."""
        name_to_class_dict = {subclass.name: subclass for subclass in cls.__subclasses__()}
        if name in name_to_class_dict:
            return super().__new__(name_to_class_dict[name])  # Instantiate the correct subclass
        raise ValueError(f"Unknown model name: {name}")

    def __init__(self, name: str):
        """Initialize common attributes for all embedding models."""
        self.name = name
        self.onnx_model_path = os.path.join(self.saving_folder, self.name, 'saved_models', f'{self.name}.onnx')
        self.tokenizer_path = os.path.join(self.saving_folder, self.name, 'tokenizer')
        self.config_path = os.path.join(self.saving_folder, self.name)

        self.embedding_model_version = self.name + ".onnx"
        print(Fore.LIGHTGREEN_EX, f"\rEmbedding model used: {self.embedding_model_version}", Fore.RESET)

    def encode(self, texts):
        pass  # This method is intended to be overridden in child classes


class AllMiniL6V2(EmbeddingModel):
    name = "all-MiniLM-L6-v2"
    hf_path = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, name: str, use_onnx: bool = False, use_quant: bool = False):
        """Ensure the correct initialization via parent class."""
        super().__init__(name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = get_number_of_cores()
        # session_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self._embedding_model = ort.InferenceSession(self.onnx_model_path,
                                                     providers=['CPUExecutionProvider'],
                                                     sess_options=session_options)

    def encode(self, texts):
        if isinstance(texts, str):
            # Convert single text to a list for consistent handling
            texts = [texts]

        # Tokenize all texts into input IDs, attention masks, and token type IDs
        input_ids = self._tokenizer(texts, padding=True, truncation=True, return_tensors='pt')['input_ids']

        input_feed = {"input_ids": np.array(input_ids, dtype=np.int64),
                      "token_type_ids": np.ones_like(input_ids),
                      "attention_mask": np.zeros_like(input_ids)}
        last_hidden_state, _ = self._embedding_model.run(output_names=["last_hidden_state", "pooler_output"],
                                                         input_feed=input_feed)
        last_hidden_state = torch.tensor(last_hidden_state, dtype=torch.float32)

        attention_mask = torch.ones_like(input_ids)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1),
                                                                                         min=1e-9)
        return F.normalize(embeddings, p=2, dim=1)
