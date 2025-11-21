# Copyright 2024-2025 NXP
# NXP Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.

import torch
import re
import warnings
from tqdm import tqdm
from transformers import AutoTokenizer, LogitsProcessorList, TopKLogitsWarper, TopPLogitsWarper, LogitNormalization, \
    TemperatureLogitsWarper, SequenceBiasLogitsProcessor, AutoModelForCausalLM, RepetitionPenaltyLogitsProcessor, \
    DynamicCache
from rag.models.llms.huggingface_llm import AvailableLLMs
from rag.preprocessing.generate_chunks import init_text_splitter
from hirag.config import Config as HiRAGConfig
import logging

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")


class HiRAGTextSplitter:
    def __init__(self, config: HiRAGConfig):
        self.hirag_config = config
        self.model_config = AvailableLLMs(self.hirag_config.llm_name).config()
        self.generic_prompt = self.hirag_config.prompt
        self.pattern = re.compile(self.hirag_config.parsing_QA_pattern, re.DOTALL)
        logger.debug("Loading HiRAG LLM Tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.model_id,
                                                       torch_dtype=self.model_config.torch_dtype)
        logger.debug("Loading HiRAG LLM")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_config.model_id,
                                                      torch_dtype=self.model_config.torch_dtype,
                                                      device_map=self.model_config.device)

        self.logits_warper = LogitsProcessorList([
            TemperatureLogitsWarper(self.model_config.temperature),
            TopKLogitsWarper(top_k=self.model_config.top_k, min_tokens_to_keep=self.model_config.min_tokens_to_keep),
            TopPLogitsWarper(top_p=self.model_config.top_p, min_tokens_to_keep=self.model_config.min_tokens_to_keep),
            RepetitionPenaltyLogitsProcessor(self.model_config.repetition_penalty),
            LogitNormalization()
        ])
        self.logits_processor = LogitsProcessorList([
            SequenceBiasLogitsProcessor(sequence_bias=self.model_config.sequence_bias),
            LogitNormalization()
        ])
        self.eos_token_id = self.tokenizer.eos_token_id
        self._running = True

        self.local_text_splitter, _ = init_text_splitter(self.hirag_config.chunking_method,
                                                   self.hirag_config.chunk_size,
                                                   self.hirag_config.chunk_overlap)
        self.global_text_splitter, _ = init_text_splitter(self.hirag_config.chunking_method,
                                                          self.hirag_config.global_chunk_size,
                                                          self.hirag_config.global_chunk_overlap)
        logger.info(f"HiRAG LLM ({self.model_config.model_id}) correctly loaded")

    def _stop(self):
        self._running = False

    def _apply_system_prompt(self, question):
        system_prompt = [
            {"role": "system",
             "content": self.generic_prompt},
            {"role": "user",
             "content": question},
        ]
        return system_prompt

    def _process_context(self, context):
        system_prompt = self._apply_system_prompt(context)
        input_text = self.tokenizer.apply_chat_template(system_prompt, tokenize=False, add_generation_prompt=True)
        return input_text

    def _token_decode(self, output_ids):
        if self.model_config.name == "Llama3.1-8B":
            decoded_token = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        elif self.model_config.name == "Llama2-7B":
            decoded_token = self.tokenizer.convert_ids_to_tokens(output_ids)
            decoded_token = decoded_token.replace('▁', ' ')
            decoded_token = decoded_token.replace('<0x0A>', '\n')
        else:
            error_text = f"You need to specify how to decode the generated tokens for the {self.model_config.name} model"
            logger.error(error_text)
            raise NotImplementedError(error_text)
        return decoded_token

    def _generate(self, text):

        self._running = True
        model_inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.model_config.device)

        input_ids = model_inputs['input_ids']
        attention_mask = model_inputs['attention_mask']

        generated = input_ids
        self.llm_input_size = generated.shape[1]

        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        model_inputs = {
            'input_ids': input_ids,
            'past_key_values': None,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
        }

        i = 0

        with torch.no_grad():
            while self._running:
                outputs = self.model(
                    **model_inputs,
                    use_cache=True,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                )

                next_token_logits = outputs.logits[:, -1, :]
                next_token_logits = self.logits_processor(generated, next_token_logits)
                next_token_logits = self.logits_warper(generated, next_token_logits)
                i += 1

                input_ids = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                output_ids = input_ids.item()

                if output_ids == self.eos_token_id:
                    break

                if i == self.model_config.max_tokens_to_keep:  # Limit to avoid infinite loop
                    yield self.model_config.long_token
                    break

                decoded_token = self._token_decode(output_ids)

                if i == 1:
                    decoded_token = decoded_token.lstrip()

                yield decoded_token

                generated = torch.cat([generated, input_ids], dim=-1)

                attention_mask = torch.cat([attention_mask, torch.tensor([[1.]], device=self.model_config.device)], 1)
                model_inputs.update({
                    'input_ids': input_ids,
                    'past_key_values': DynamicCache.from_legacy_cache(outputs.past_key_values),
                    'attention_mask': attention_mask,
                    'position_ids': torch.tensor([[generated.size(-1) - 1]]).to(self.model_config.device)
                })

    def split_text(self, context: str) -> list:
        qa_list = []
        chunk_list = self.local_text_splitter.split_text(context)
        for chunk in chunk_list:
            count = 0
            complete_answer = ""
            llm_input = self._process_context(context=chunk)
            for i, decoded_token in enumerate(self._generate(llm_input)):
                if "Q:" in decoded_token:
                    count += 1
                    if count == self.hirag_config.max_QA_pair_limit + 1:
                        self._stop()
                complete_answer += decoded_token
            # Extract question-answer pairs
            pairs = self.pattern.findall(complete_answer)
            qa_list.extend(pairs)

        return qa_list

    @staticmethod
    def stop_at_n_qa(text: str, n: int) -> tuple[str, bool]:
        matches = [m.start() for m in re.finditer(r'\bQ\s*:', text)]  # Find all occurrences of "Q:" with optional space
        if len(matches) > n:
            return text[:matches[-1]], True  # Keep only text before the 4th occurrence
        return text, False  # Return full text if "Q:" appears less than 4 times

    def local_qa_convertion(self, context: str) -> list:
        self.generic_prompt = self.hirag_config.local_understanding_prompt
        qa_list = []
        chunk_list = self.local_text_splitter.split_text(context)

        for chunk in chunk_list:
            complete_answer = ""
            llm_input = self._process_context(context=chunk)
            for i, decoded_token in enumerate(self._generate(llm_input)):
                complete_answer += decoded_token
                if ":" in decoded_token:
                    text, stop = self.stop_at_n_qa(complete_answer, self.hirag_config.local_QA_pair_limit)
                    if stop:
                        complete_answer = text
                        self._stop()
                        break
            complete_answer = complete_answer.replace("\n\n", "\n").strip()
            # Extract question-answer pairs
            pairs = self.pattern.findall(complete_answer)
            qa_list.extend(pairs)

        return qa_list

    def global_qa_convertion(self, context: str) -> list:
        self.generic_prompt = self.hirag_config.global_understanding_prompt
        qa_list = []
        chunk_list = self.global_text_splitter.split_text(context)
        for chunk in chunk_list:
            complete_answer = ""
            llm_input = self._process_context(context=chunk)
            for i, decoded_token in enumerate(self._generate(llm_input)):
                complete_answer += decoded_token
                if ":" in decoded_token:
                    text, stop = self.stop_at_n_qa(complete_answer, self.hirag_config.global_understanding_QA_pair_limit)
                    if stop:
                        complete_answer = text
                        self._stop()
                        break
            complete_answer = complete_answer.replace("\n\n", "\n").strip()
            # Extract question-answer pairs
            pairs = self.pattern.findall(complete_answer)
            qa_list.extend(pairs)

        return qa_list

    def qa_data_augmentation(self, qa_pair: tuple) -> list:
        (q, a) = qa_pair
        context = f"Q: {q}, A: {a}"
        self.generic_prompt = self.hirag_config.data_augmentation_prompt
        qa_list = []
        complete_answer = ""
        llm_input = self._process_context(context=context)
        for i, decoded_token in enumerate(self._generate(llm_input)):
            complete_answer += decoded_token
            if ":" in decoded_token:
                text, stop = self.stop_at_n_qa(complete_answer, self.hirag_config.data_augmentation_QA_pair_limit)
                if stop:
                    complete_answer = text
                    self._stop()
                    break
        complete_answer = complete_answer.replace("\n\n", "\n").strip()
        # Extract question-answer pairs
        pairs = self.pattern.findall(complete_answer)
        qa_list.extend(pairs)

        return qa_list

    def generate_hirag_chunks(self, data: dict | str) -> dict:
        chunks = {}
        id = 0

        if isinstance(data, dict):
            logger.warning("The HiRAG global understanding feature is not yet available for documents parsed into JSON "
                            "file")
            failed_qa_conversion = {}
            failed_data_augmentation = {}
            for key, value in tqdm(data.items(), desc=f"Chunking text with HiRAG"):
                text = value.pop('text')
                # Error check
                if text == "":
                    logger.warning(f"You have an empty string for chunk n° {key} in {value.get('origin_file', 'unknown')} file.")
                    continue
                # Add metadata
                metadata = {}
                for k, v in value.items():
                    metadata[k] = v

                qa_chunk_list = self.local_qa_convertion(text)
                # Error check
                if len(qa_chunk_list) == 0:
                    failed_qa_conversion[key] = {"text": text}.update(metadata)
                    logger.warning(f"Could not generate QA chunks for {failed_qa_conversion[key]}")

                # Perform data augmentation
                for original_qa in qa_chunk_list:
                    qa_augmented_list = [original_qa]
                    qa_augmented_list.extend(self.qa_data_augmentation(original_qa))
                    # Error check
                    if len(qa_augmented_list) == 1:
                        if key not in failed_data_augmentation:
                            failed_data_augmentation[key] = []
                        failed_data_augmentation[key].extend(original_qa)
                        logger.warning(f"Could not generate QA data augmentation chunks for {original_qa}")
                    for (q, a) in qa_augmented_list:
                        # Emb_QA -> Ret_A
                        chunks_entry1 = {
                            "chunks": [a],
                            "complete_chunks": q.replace(';', ',') + ";" + a.replace(';', ','),
                            "question": q,
                            "answer": a
                        }
                        # Add metadata
                        chunks_entry1.update(metadata)
                        chunks[id] = chunks_entry1
                        id += 1

                        # Emb_Q -> Ret_A
                        chunks_entry2 = {
                            "chunks": [a],
                            "complete_chunks": q.replace(';', ','),
                            "question": q,
                            "answer": a}
                        # Add metadata
                        chunks_entry2.update(metadata)
                        chunks[id] = chunks_entry2
                        id += 1

        elif isinstance(data, str):
            failed_data_augmentation = []
            qa_chunk_list = self.local_qa_convertion(data)
            global_qa_chunk_list = self.global_qa_convertion(data)
            combined_qa_chunk_list = qa_chunk_list + global_qa_chunk_list
            # Error check
            if len(qa_chunk_list) == 0:
                logger.warning("Could not generate local QA chunks.")
            if len(global_qa_chunk_list) == 0:
                logger.warning("Could not generate global QA chunks.")

            # Perform data augmentation
            for original_qa in tqdm(combined_qa_chunk_list, desc=f"Chunking text with HiRAG"):
                qa_augmented_list = [original_qa]
                qa_augmented_list.extend(self.qa_data_augmentation(original_qa))
                # Error check
                if len(qa_augmented_list) == 1:
                    failed_data_augmentation.extend(original_qa)
                    logger.warning(f"Could not generate QA data augmentation chunks for {original_qa}")
                for (q, a) in qa_augmented_list:
                    # Emb_QA -> Ret_A
                    chunks_entry1 = {
                        "chunks": [a],
                        "complete_chunks": q.replace(';', ',') + ";" + a.replace(';', ','),
                        "question": q,
                        "answer": a
                    }
                    chunks[id] = chunks_entry1
                    id += 1

                    # Emb_Q -> Ret_A
                    chunks_entry2 = {
                        "chunks": [a],
                        "complete_chunks": q.replace(';', ','),
                        "question": q,
                        "answer": a}
                    chunks[id] = chunks_entry2
                    id += 1

        return chunks
