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
from colorama import Fore
from transformers import AutoTokenizer, LogitsProcessorList, TopKLogitsWarper, TopPLogitsWarper, LogitNormalization, \
    TemperatureLogitsWarper, SequenceBiasLogitsProcessor, AutoModelForCausalLM, RepetitionPenaltyLogitsProcessor, \
    DynamicCache
from rag.models.llms.huggingface_llm import AvailableLLMs
from rag.preprocessing.generate_chunks import init_text_splitter

warnings.filterwarnings("ignore")


class HiRAGTextSplitter:
    def __init__(self):
        self.model_config = AvailableLLMs("llama3.1-8B").config()
        self.generic_prompt = ("Give me as much question-answer pairs as needed to cover the whole following text. "
                   "Question (Q) and answer (A) must be short and precise. The pairs must be formatted as: Q:, A:."
                   )
        self.max_QA_pair_limit = 15
        self.pattern = re.compile(r"Q:\s*(.*?),?\s*A:\s*(.*?)(?=\s*Q:|\s*Note: I|$)", re.DOTALL)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.model_id,
                                                       torch_dtype=self.model_config.torch_dtype)
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

        self.local_text_splitter, _ = init_text_splitter("SpaCy", 1000, 200)
        self.global_text_splitter, _ = init_text_splitter("SpaCy", 7000, 500)
        self.global_understanding_QA_pair_limit = 25
        self.local_QA_pair_limit = 25
        self.data_augmentation_QA_pair_limit = 5

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
            raise NotImplementedError(
                f"You need to specify how to decode the generated tokens for the {self.model_config.name} model?")
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
                    if count == self.max_QA_pair_limit + 1:
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
        self.generic_prompt = """
        Extract a list of concise question-answer pairs from the text, covering all main topics and entities 
        mentioned. Ensure each pair is accurate, relevant, and well-formatted as Q: ..., A: ...
    
        When creating the pairs, consider the following guidelines:
    
        Entity-based questions: Extract questions about the main entities mentioned in the text, such as people, 
        places, organizations, and objects. For example, \"Who is...\", \"What is...\", \"Where is...\".
    
        Action-based questions: Identify actions, events, or processes mentioned in the text and ask questions about
        them, like \"What is happening...\", \"What is being done...\", or \"What action is...\".
    
        Attribute-based questions: Extract questions about attributes or characteristics of entities mentioned in 
        the text, such as \"What color is...\", \"How old is...\", or \"What is the name of...\".
    
        Contextual questions: Formulate questions that provide context or additional information about the main 
        topic, like \"Why is this happening...\", \"What is the purpose of...\", or \"How does this relate to...\".
    
        When answering the questions, prioritize conciseness while ensuring each response is at least one sentence 
        long. Replace acronyms and technical words by their definition if they are clearly defined in the text. Avoid 
        inventing information not explicitly stated in the text. 
    
        Example of the output for the given text \"The white dog is running.\":
        Q: Who is running?, A: The white dog is running.
        Q: What color is the dog?, A: The dog is white.
        Q: What is the dog doing?, A: The dog is running.
    """
        qa_list = []
        chunk_list = self.local_text_splitter.split_text(context)

        for chunk in chunk_list:
            complete_answer = ""
            llm_input = self._process_context(context=chunk)
            for i, decoded_token in enumerate(self._generate(llm_input)):
                complete_answer += decoded_token
                if ":" in decoded_token:
                    text, stop = self.stop_at_n_qa(complete_answer, self.local_QA_pair_limit)
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
        self.generic_prompt = """
            Given the following text, generate a comprehensive serie of question-answer pairs that demonstrate a thorough understanding of the global content. 
            The questions should be clear and concise, and the answers should provide a brief summary or explanation. 
            Avoid inventing information that is not stated in the text. 
            Use the text to support the answers, and avoid making assumptions. 
            When answering the questions, prioritize conciseness while ensuring each response is at least one sentence long. 
            If a question has multiple possible answers, provide multiple question-answer pairs.
            
            
            Example:
            Useful questions: \"What is this document about?\", \"Who is speaking?\", \"What is the title of this document?\", 
            
            Or if the LLM were given:
            \"
            Menu:
            - Hamburger (bread, steak, salad, tomato, onion) 16$
            - Cheeseburger (bread, steak, cheese, salad, tomato, onion) 17$
            - Rice bowl eggs (rice, egg, beans) 12$
            \"
            The expected output would be something like:
            Q: What is the most expensive meal on the menu?, A: The cheeseburger (17$) is the most expensive meal.
            Q: What is the cheapest meal?, A: The rice bowl egg (12$) is the cheapest meal.
            Q: What are the vegetarian meals?, A: The rice bowl egg is the only vegetarian meal.
        """
        qa_list = []
        chunk_list = self.global_text_splitter.split_text(context)
        for chunk in chunk_list:
            complete_answer = ""
            llm_input = self._process_context(context=chunk)
            for i, decoded_token in enumerate(self._generate(llm_input)):
                complete_answer += decoded_token
                if ":" in decoded_token:
                    text, stop = self.stop_at_n_qa(complete_answer, self.global_understanding_QA_pair_limit)
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
        self.generic_prompt = """
        Given a Question-Answer (Q-A) pair, perform data augmentation by generating new Q-A pairs with the following modifications:
        - Synonym replacement: Replace key words or phrases with their synonyms or their definition, while preserving the original meaning.
        - Passive voice conversion: Convert the answer to the passive voice, where possible, to create alternative expressions.
        - Sentence rephrasing: Rephrase the answer to provide different ways of expressing the same idea, without altering the core meaning.
        
        Input format:
        Q: <original question>, A: <original answer>
        Output format:
        Q: <new question1>, A: <new answer1>
        Q: <new question2>, A: <new answer2>
        
        Example of the output for the given text \"Q: What causes global warming?, A: Greenhouse gas emissions trap heat in the atmosphere.\":
        Q: What leads to climate change?, A: Heat is retained in the atmosphere due to greenhouse gases.
        Q: What is global warming caused by?, A: Heat is trapped in the atmosphere by greenhouse gas emissions.
        Q: What factors contribute to rising global temperatures?, A: The atmosphere is warmed due to the accumulation of greenhouse gases.
        Q: Why is the Earth getting hotter?, A: Gases like CO2 keep heat from escaping into space.
    """
        qa_list = []
        complete_answer = ""
        llm_input = self._process_context(context=context)
        for i, decoded_token in enumerate(self._generate(llm_input)):
            complete_answer += decoded_token
            if ":" in decoded_token:
                text, stop = self.stop_at_n_qa(complete_answer, self.data_augmentation_QA_pair_limit)
                if stop:
                    complete_answer = text
                    self._stop()
                    break
        complete_answer = complete_answer.replace("\n\n", "\n").strip()
        # Extract question-answer pairs
        pairs = self.pattern.findall(complete_answer)
        qa_list.extend(pairs)

        return qa_list

    def generate_hirag_chunks(self, data: str) -> dict:
        chunks = {}
        id = 0

        failed_data_augmentation = []
        qa_chunk_list = self.local_qa_convertion(data)
        global_qa_chunk_list = self.global_qa_convertion(data)
        combined_qa_chunk_list = qa_chunk_list + global_qa_chunk_list
        # Error check
        if len(qa_chunk_list) == 0:
            print(Fore.RED, "Warning: Could not generate local QA chunks.", Fore.RESET)
        if len(global_qa_chunk_list) == 0:
            print(Fore.RED, "Warning: Could not generate global QA chunks.", Fore.RESET)

        # Perform data augmentation
        for original_qa in combined_qa_chunk_list:
            qa_augmented_list = [original_qa]
            qa_augmented_list.extend(self.qa_data_augmentation(original_qa))
            # Error check
            if len(qa_augmented_list) == 1:
                failed_data_augmentation.extend(original_qa)
                print(Fore.RED, f"Warning: Could not generate QA data augmentation chunks for {original_qa}",
                      Fore.RESET)
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
