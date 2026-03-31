# Copyright 2026 NXP
# NXP Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.

import time
import torch
import os.path
import logging
import onnxruntime
import numpy as np
from torch import nn
from abc import abstractmethod
from transformers import AutoProcessor
from vlm.models.models_config import models_dir
from vlm.utils import ensure_local_or_download_hf, remove
from vlm.models.models_config import (SmolVLM2_256M_q8_config,
                                      SmolVLM2_500M_q8_config,
                                      SmolVLM2_256M_config,
                                      SmolVLM2_500M_config
                                      )
from transformers.image_utils import load_image
from transformers.feature_extraction_utils import BatchFeature
from vlm.models.smolvlm.processing_smolvlm import SmolVLMProcessor
from transformers.models.smolvlm.processing_smolvlm import get_image_prompt_string
from transformers import AutoConfig, LogitsProcessorList, TemperatureLogitsWarper, TopPLogitsWarper

logger = logging.getLogger(__name__)


class VLM:
    def __init__(self, model_config, user_params, fixed_image, processor_cls=AutoProcessor):
        self.model_config = model_config()
        self.ensure_models_local()
        self.config = AutoConfig.from_pretrained(self.model_config.config_id)
        self.processor = processor_cls.from_pretrained(self.model_config.processor_id)
        self.processor.image_processor.do_image_splitting = False

        session_options = onnxruntime.SessionOptions()
        session_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        session_options.intra_op_num_threads = user_params.n_threads
        # avail_providers = onnxruntime.get_available_providers()
        # avail_providers is disabled as Neutron is not ready yet for this model  # providers = avail_providers
        self.vision_session = self._initialize_onnx_sessions(self.model_config.vision_session, None)
        self.embed_session = self._initialize_onnx_sessions(
            self.model_config.embedding_session,
            sess_options=session_options
        )
        self.decoder_session = self._initialize_onnx_sessions(
            self.model_config.decoder_session,
            sess_options=session_options
        )

        # Set config values
        self.max_new_tokens = user_params.max_new_tokens
        self.image_features = None
        self.generated_tokens = np.array([[]], dtype=np.int64)

        # For now fixed image
        self.image = load_image(fixed_image)
        self.image_inputs = self.processor.image_processor([[self.image]],
                                                           **{'return_row_col_info': True, 'return_tensors': 'np'})

        self.logits_warper = LogitsProcessorList([
            TemperatureLogitsWarper(self.model_config.temperature),
            TopPLogitsWarper(top_p=self.model_config.top_p),
        ])

        self.perf = dict.fromkeys(["embed", "vision", "decoder_ttft", "decoder"])

    def _handle_load_failure(self, func, cleanup, sess_options, retry):
        """Handle model load failure with retry and cleanup."""
        remove(cleanup)
        new_path = ensure_local_or_download_hf(models_dir, os.path.relpath(cleanup, models_dir))
        return func(new_path, sess_options, retry)

    def _initialize_onnx_sessions(self, model_path, sess_options, retry=True):
        try:
            return onnxruntime.InferenceSession(
                model_path,
                sess_options=sess_options
            )
        except Exception:
            if not retry:
                raise RuntimeError("Failed loading model and tried to re-download it")
            return self._handle_load_failure(self._initialize_onnx_sessions,
                                             cleanup=model_path,
                                             sess_options=sess_options,
                                             retry=False
                                             )

    def ensure_models_local(self):
        self.model_config.config_id = ensure_local_or_download_hf(
            models_dir=models_dir,
            relative_path=self.model_config.config_id,
            folder=True,
            config=self.model_config)
        self.model_config.processor_id = ensure_local_or_download_hf(
            models_dir,
            self.model_config.processor_id,
            folder=True,
            config=self.model_config
        )
        self.model_config.vision_session = ensure_local_or_download_hf(
            models_dir,
            self.model_config.vision_session,
            config=self.model_config
        )
        self.model_config.embedding_session = ensure_local_or_download_hf(
            models_dir,
            self.model_config.embedding_session,
            config=self.model_config
        )
        self.model_config.decoder_session = ensure_local_or_download_hf(
            models_dir,
            self.model_config.decoder_session,
            config=self.model_config
        )

    def str_perf(self):
        return (
            f"Vision: {self.perf['vision']:.2f}s "
            f"| TTFT: {self.perf['decoder_ttft'] + self.perf['vision']:.2f}s "
            f"(Decoder {self.perf['decoder_ttft']:.2f}s) "
            f"| Current decode speed: {1 / self.perf['decoder']:.2f}tok/s"
        )

    @abstractmethod
    def get_position_ids(self, inputs):
        pass

    @abstractmethod
    def run_vision(self, inputs):
        pass

    @abstractmethod
    def process_text(self, text):
        pass

    def token_decode(self, output_ids):
        decoded_token = self.processor.tokenizer._convert_id_to_token(output_ids)
        decoded_token = decoded_token.replace('▁', ' ')
        decoded_token = decoded_token.replace('<0x0A>', ' ')
        decoded_token = decoded_token.replace('Ġ', ' ')
        decoded_token = decoded_token.replace('Ċ', '\n')
        return decoded_token

    def process_message(self, text):
        inputs = {}
        # As image is fixed
        inputs.update(self.image_inputs)

        """
        [
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions about images."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image"},
                ],
            },
        ]
        """

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": text}
                ]
            },
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        input_text = self.process_text(text=[prompt])
        input_text_tokenized = self.processor.tokenizer(input_text,
                                                        **{'add_special_tokens': True, 'is_split_into_words': False,
                                                           'padding': False})
        inputs.update(input_text_tokenized)
        inputs = BatchFeature(inputs, tensor_type="np")

        batch_size = inputs['input_ids'].shape[0]
        first_token = True
        past_key_values = {
            f'past_key_values.{layer}.{kv}': np.zeros(
                [batch_size, self.config.text_config.num_key_value_heads, 0, self.config.text_config.head_dim],
                dtype=np.float32)
            for layer in range(self.config.text_config.num_hidden_layers)
            for kv in ('key', 'value')
        }

        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        position_ids = self.get_position_ids(inputs)
        generated = torch.tensor(input_ids)

        for i in range(self.max_new_tokens):
            timer = time.time()
            inputs_embeds = self.embed_session.run(None, {'input_ids': input_ids})[0]
            if first_token:
                embedding_time = time.time() - timer
                self.perf['embed'] = embedding_time
                logger.debug(f"Embedding session: {embedding_time}s")
                timer = time.time()
            if self.image_features is None:
                # Only compute vision features if not already computed
                vision_time = time.time() - timer
                self.image_features = self.run_vision(inputs)
                self.perf['vision'] = vision_time
                logger.debug(f"Vision session: {embedding_time}s")
                timer = time.time()

            if first_token:
                # Merge text and vision embeddings only if first token
                inputs_embeds[inputs['input_ids'] == self.config.image_token_id] = (
                    self.image_features.reshape(-1, self.image_features.shape[-1])
                )

            logits, *present_key_values = self.decoder_session.run(None, dict(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **past_key_values,
            ))
            decoder_time = time.time() - timer
            if first_token:
                self.perf['decoder_ttft'] = decoder_time
                logger.debug(f"TTFT: {decoder_time}s")
            else:
                self.perf['decoder'] = decoder_time
                logger.debug(f"Decode step: {decoder_time}s")
            first_token = False
            timer = time.time()

            input_ids = torch.tensor(logits[:, -1])
            input_ids = self.logits_warper(generated, input_ids)

            if self.do_sample:
                probs = nn.functional.softmax(input_ids, dim=-1)
                input_ids = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                input_ids = input_ids.argmax(-1, keepdims=True)
            logger.debug(f"Logits: {time.time() - timer}s")
            attention_mask = np.ones_like(input_ids)
            position_ids = position_ids[:, -1:] + 1

            generated = torch.cat([generated, input_ids], dim=-1)
            input_ids = input_ids.numpy()

            for j, key in enumerate(past_key_values):
                past_key_values[key] = present_key_values[j]

            if ((input_ids == self.config.text_config.eos_token_id).all()
                    or (input_ids == self.model_config.eou_token_id).all()):
                break
            i += 1
            yield self.token_decode(input_ids[0, 0])
            if i == self.max_new_tokens:
                yield "[...]"
        return


class SmolVLM(VLM):
    def __init__(self, config, **kwargs):
        super().__init__(config, processor_cls=SmolVLMProcessor, **kwargs)
        self.do_sample = False

    def get_position_ids(self, inputs):
        return np.cumsum(inputs['attention_mask'], axis=-1)

    def run_vision(self, inputs):
        dat_time = time.time()
        vision_features = self.vision_session.run(
            ['image_features'],
            {
                'pixel_values': inputs['pixel_values'],
                'pixel_attention_mask': inputs['pixel_attention_mask'].astype(np.bool_)
            }
        )

        self.perf['vision'] = time.time() - dat_time
        return vision_features[0]

    def process_text(self, text):
        timer = time.time()
        image_rows = [[0]]
        image_cols = [[0]]

        prompt_strings = []
        for sample, sample_rows, sample_cols in zip(text, image_rows, image_cols):
            # Replace image token with fake tokens around the expanded image token sequence of length `image_seq_len`
            image_prompt_strings = []
            for n_rows, n_cols in zip(sample_rows, sample_cols):
                image_prompt_string = get_image_prompt_string(
                    n_rows,
                    n_cols,
                    self.processor.image_seq_len,
                    image_token=self.processor.image_token,
                    fake_token_around_image=self.processor.fake_image_token,
                    global_image_token=self.processor.global_image_token,
                )
                image_prompt_strings.append(image_prompt_string)

            split_sample = sample.split(self.processor.image_token)
            if len(split_sample) == 0:
                raise ValueError("The image token should be present in the text.")

            # Place in the image prompt strings where the image tokens are
            sample = split_sample[0]
            for i, image_prompt_string in enumerate(image_prompt_strings):
                sample += image_prompt_string + split_sample[i + 1]
            prompt_strings.append(sample)
        logger.debug(f"Process text: {time.time() - timer}s")
        return prompt_strings


def make_VLM(name, precision, **kwargs):
    _name_mapping = {
        'smolvlm-256M':
            {'subclass': SmolVLM,
             'config': SmolVLM2_256M_q8_config if precision == "q8" else SmolVLM2_256M_config
             },
        'smolvlm-500M':
            {'subclass' : SmolVLM,
             'config' : SmolVLM2_500M_q8_config if precision == "q8" else SmolVLM2_500M_config
             },
    }

    return _name_mapping[name]["subclass"](_name_mapping[name]["config"], **kwargs)
