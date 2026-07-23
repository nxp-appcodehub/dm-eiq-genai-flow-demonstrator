# Copyright 2025-2026 NXP
# NXP Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.

import time
import logging
from typing import Generator
from eiq_genai_flow.llm_backends.generic_client import LLMClientBase
from eiq_genai_flow.llm_backends.utils.msg_q import msg_q

logger = logging.getLogger(__name__)

# This class is used to communicate with the GUI DNPU backend, especially the Package 4 LLaVA Demo GUI


class GuiDNPUClient(LLMClientBase):
    def __init__(self, device_name=None, prompt=None):
        super().__init__(device_name, prompt)
        self.name = "gui_dnpu_client"
        self.asr_end_token = "<end>"
        self.tts_end_token = "TTS_END"
        self.init_signal = "INITED"
        self.msg_q = msg_q()
        self._running = True

    def shutdown(self):
        # TODO: Warn the Server
        pass

    def send(self, data):
        if self.msg_q:
            self.msg_q.send_data(data)
        else:
            logger.error("msg_q does not exist")
            exit(1)

    def get(self, timeout):
        if self.msg_q:
            return self.msg_q.get_data(timeout)
        else:
            logger.error("msg_q does not exist")
            exit(1)

    def send_inited(self):
        self.send(self.init_signal)

    def send_wakeword_detected(self):
        self.send("WWD")

    def send_asr_text(self, asr_text):
        self.send(asr_text)

    def send_asr_end(self):
        self.send(self.asr_end_token)

    def send_tts_end(self):
        self.send(self.tts_end_token)

    def wait_for_server_init(self):
        logger.info("Wait for ext NPU Server...")
        # FIXME: if the server is already ready, we wait for ever
        # TODO: Implement a connection/disconnection mechanism between server and client
        while True:
            response_data = self.get()
            if response_data == self.init_signal:
                break
        return True

    def format_input(self, query: str, prompt=None):
        """
        Format the input query with optional prompt.

        Args:
            query: The user's question
            prompt: Optional prompt (can be str or tuple[str, list] for RAG context)

        Returns:
            Formatted question string
        """
        if prompt is None:
            # No prompt provided, use just the query
            formatted_question = query
        elif isinstance(prompt, str):
            # Only a prompt is provided
            formatted_question = f"{prompt}\n{query}"
        elif (
            isinstance(prompt, tuple)
            and len(prompt) == 2
            and isinstance(prompt[0], str)
            and isinstance(prompt[1], list)
        ):
            # Prompt with RAG context chunks
            context = " ".join(prompt[1])
            formatted_question = f"{prompt[0]}\n{context}\n{query}"
        else:
            # Unsupported prompt type
            logger.warning(f"Unsupported prompt type, must be str or (str, list), have: {type(prompt)}")
            formatted_question = query

        return formatted_question

    def generate(self, text: str) -> Generator:
        """
        Generate response tokens from the formatted input text.

        Args:
            text: The formatted question/prompt text

        Yields:
            Individual response tokens
        """
        start = time.time()
        logger.debug(f"GuiDNPUClient input:\n{text}")

        self.msg_q.clear_mq_from_llmc()

        self.send_asr_text(text)
        self.send_asr_end()

        self._running = True

        while self._running:
            response_data = self.get(10)

            if response_data == "TTS_BEGIN":
                continue

            if response_data == "TIMEOUT":
                break

            end_token_index = response_data.rfind(self.llm_end_token)
            if end_token_index != -1:
                break

            yield response_data

        logger.info(f"\nRequest time: {round(time.time() - start, 3)}s")

    def stop(self):
        """Stop the generation process."""
        self._running = False

    def __call__(self, query: str, prompt: str | tuple[str, list] = None, forced_answer_prefix: str = "") -> Generator:
        """
        Main interface to generate responses from the LLM client.

        This method follows the same API pattern as the local LLM models,
        making it a drop-in replacement.

        Args:
            query: The user's question
            prompt: Optional system prompt (str) or RAG context (tuple[str, list])
            forced_answer_prefix: Optional prefix to force in the answer (not used in this client)

        Yields:
            Individual response tokens
        """
        formatted_question = self.format_input(query, prompt)
        logger.debug(f"GuiDNPUClient formatted input: {formatted_question}")

        if forced_answer_prefix:
            logger.debug(f"GuiDNPUClient answer prefix (note: not supported on server side): {forced_answer_prefix}")
            # Note: forced_answer_prefix is not currently supported by the discrete NPU server
            # but we maintain the parameter for API compatibility

        yield from self.generate(formatted_question)

    def process_question(self, prompt, question):
        """
        Legacy method for backward compatibility.

        Deprecated: Use __call__ instead.
        """
        logger.warning("process_question() is deprecated, use __call__() instead")
        return question, ""
