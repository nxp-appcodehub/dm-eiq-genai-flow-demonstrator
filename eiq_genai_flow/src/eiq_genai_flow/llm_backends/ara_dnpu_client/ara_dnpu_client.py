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
import json
import logging
import requests
from typing import Generator
from eiq_genai_flow.llm_backends.generic_client import LLMClientBase

logger = logging.getLogger(__name__)


class AraDNPUClient(LLMClientBase):
    """Client for ARA2 Discrete NPU via aaf-connector REST API (OpenAI-compatible)."""

    def __init__(self, model_name: str, device_name=None, prompt=None):
        super().__init__(device_name, prompt)
        self.model_name = model_name.replace("-ara", "")  # Remove -ara suffix
        self.name = f"{self.model_name}-ara"
        self.base_url = "http://0.0.0.0:8000"  # AAF connector default
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        self._running = True
        self.system_prompt = prompt or "You are a helpful assistant."

        logger.info(f"Initializing ARA DNPU client for model: {self.model_name}")

        # Import and create detector
        from eiq_genai_flow.config import Config
        from eiq_genai_flow.llm_backends.ara_dnpu_client.ara_detector import AraDetector

        self.config = Config()
        detector = AraDetector(self.config)

        # Ensure connector is ready and model is loaded
        if not detector.ensure_connector_ready(timeout=600, model_name=self.model_name):
            raise ConnectionError(
                f"Cannot connect to ARA connector or load model '{self.model_name}'\n"
                f"The connector did not become ready within the timeout period.\n"
                f"\n"
                f"Troubleshooting:\n"
                f"1. Check if connector process is running:\n"
                f"   ps -aux | grep connector\n"
                f"\n"
                f"2. Check connector logs:\n"
                f"   journalctl -u {self.config.discrete_npu_service} -f\n"
                f"\n"
                f"3. Manually start connector:\n"
                f"   source /usr/share/eiq/aaf-connector/venv/bin/activate\n"
                f"   connector\n"
                f"\n"
                f"4. Check if models are present:\n"
                f"   ls -la /usr/share/llm/\n"
            )

        # Log active LLM parameters from server_config.json
        self._log_server_llm_params()

        logger.info("✓ Successfully connected to ARA connector")
        logger.info(f"  Model: {self.model_name}")
        logger.info(f"  Endpoint: {self.base_url}")

    def _log_server_llm_params(self):
        """
        Read and log the active LLM parameters from server_config.json.
        These are the parameters the connector will use for inference.
        To change them, edit /usr/share/eiq/aaf-connector/server_config.json
        and restart the connector.
        """
        server_config_path = "/usr/share/eiq/aaf-connector/server_config.json"
        try:
            with open(server_config_path, "r") as f:
                server_config = json.load(f)

            llm_params = server_config.get("llm_params", {})

            logger.info("=== ARA Connector LLM Parameters (from server_config.json) ===")
            logger.info(f"  generate_max_tokens : {llm_params.get('generate_max_tokens', 'not set')}")
            logger.info(f"  temperature         : {llm_params.get('temperature', 'not set')}")
            logger.info(f"  top_k               : {llm_params.get('top_k', 'not set')}")
            logger.info(f"  top_p               : {llm_params.get('top_p', 'not set')}")
            logger.info(f"  repeat_penalty      : {llm_params.get('repeat_penalty', 'not set')}")
            logger.info(f"  repeat_last_n       : {llm_params.get('repeat_last_n', 'not set')}")
            logger.info(f"  ngram_penalty       : {llm_params.get('ngram_penalty', 'not set')}")
            logger.info(f"  frequency_penalty   : {llm_params.get('frequency_penalty', 'not set')}")
            logger.info(f"  suppress_penalty    : {llm_params.get('suppress_penalty', 'not set')}")
            logger.info(f"  seed                : {llm_params.get('seed', 'not set')}")
            logger.info(f"  To override: edit {server_config_path} and restart the connector")
            logger.info("=" * 60)

        except FileNotFoundError:
            logger.warning(f"server_config.json not found at {server_config_path}")
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse server_config.json: {e}")
        except Exception as e:
            logger.warning(f"Could not read server LLM parameters: {e}")

    def _verify_connection(self) -> bool:
        """Verify connection to aaf-connector."""
        try:
            response = self.session.get(f"{self.base_url}/", timeout=5)
            return True
        except requests.RequestException:
            try:
                response = self.session.get(f"{self.base_url}/health", timeout=5)
                return response.status_code == 200
            except requests.RequestException as e:
                logger.error(f"Failed to connect to ARA connector: {e}")
                return False

    def shutdown(self):
        """Close the session."""
        self._running = False
        self.session.close()

    def format_messages(self, query: str, prompt=None) -> list[dict]:
        """
        Format the input as OpenAI-compatible messages array.

        Args:
            query: The user's question
            prompt: Optional system prompt (str or tuple[str, list] for RAG)

        Returns:
            List of message dictionaries
        """
        messages = []

        # System message
        if prompt is None:
            system_content = self.system_prompt
        elif isinstance(prompt, str):
            system_content = prompt
        elif isinstance(prompt, tuple) and len(prompt) == 2:
            # RAG context: (system_prompt, chunk_list)
            context = " ".join(prompt[1])
            system_content = f"{prompt[0]}\n\nContext:\n{context}"
        else:
            logger.warning(f"Unsupported prompt type: {type(prompt)}")
            system_content = self.system_prompt

        messages.append({"role": "system", "content": system_content})
        messages.append({"role": "user", "content": query})

        return messages

    def generate(self, messages: list[dict]) -> Generator[str, None, None]:
        """
        Generate response tokens from the ARA NPU via streaming.
        LLM parameters are controlled by server_config.json on the connector side.

        Args:
            messages: List of message dictionaries

        Yields:
            Individual response tokens
        """
        start_time = time.time()
        logger.debug(f"AraNPUClient request messages: {messages}")

        # No llm_params in payload — connector uses server_config.json defaults
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
        }

        logger.debug(f"AraNPUClient payload: {json.dumps(payload, indent=2)}")

        response = None
        try:
            # (connect_timeout, read_timeout):
            #   - 10s to establish the TCP connection
            #   - 120s between successive SSE chunks (covers slow TTFT on
            #     complex prompts while still detecting a truly stuck server)
            response = self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                stream=True,
                timeout=(10, 120),
            )
            response.raise_for_status()

            token_count = 0
            first_token_time = None

            # Process Server-Sent Events (SSE) stream
            for line in response.iter_lines():
                if not self._running:
                    logger.info("Generation stopped by user")
                    break

                if not line:
                    continue

                line = line.decode("utf-8").strip()

                # SSE format: "data: {json}"
                if line.startswith("data: "):
                    data_str = line[6:]  # Remove "data: " prefix

                    if data_str == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data_str)

                        choices = chunk.get("choices", [])
                        if choices and len(choices) > 0:
                            delta = choices[0].get("delta", {})

                            # First chunk often contains role, skip it
                            if "role" in delta and not delta.get("content"):
                                continue

                            token = delta.get("content", "")

                            if token:
                                if token_count == 0:
                                    first_token_time = time.time() - start_time
                                    logger.info(f"Time to first token: {first_token_time:.2f}s")

                                token_count += 1
                                yield token

                    except json.JSONDecodeError as e:
                        logger.debug(f"Failed to parse JSON: {data_str[:100]}... Error: {e}")
                        continue

            total_time = time.time() - start_time
            if token_count > 0:
                tps = token_count / (total_time - (first_token_time or 0))
                logger.info(
                    f"Generated {token_count} tokens in {total_time:.2f}s "
                    f"({tps:.2f} tok/s, TTFT: {first_token_time:.2f}s)"
                )
            else:
                logger.warning("No tokens generated")

        except GeneratorExit:
            logger.debug("Generator closed by consumer")
        except requests.Timeout:
            logger.error(
                "ARA NPU read timed out — the server did not send data within "
                "the timeout window. The connector may be overloaded or stalled."
            )
        except requests.RequestException as e:
            logger.error(f"ARA NPU request failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during generation: {e}", exc_info=True)
        finally:
            if response is not None:
                try:
                    response.close()
                    logger.debug("Streaming response closed")
                except Exception as e:
                    logger.debug(f"Error closing response: {e}")

    def stop(self):
        """Stop the generation process."""
        logger.info("Stopping ARA NPU generation")
        self._running = False

    def __call__(
        self, query: str, prompt: str | tuple[str, list] = None, forced_answer_prefix: str = ""
    ) -> Generator[str, None, None]:
        """
        Main interface to generate responses.

        Args:
            query: The user's question
            prompt: Optional system prompt (str) or RAG context (tuple[str, list])
            forced_answer_prefix: Optional prefix to prepend to answer (yielded first)

        Yields:
            Individual response tokens
        """
        self._running = True

        messages = self.format_messages(query, prompt)

        logger.debug(f"AraNPUClient formatted messages: {json.dumps(messages, indent=2)}")

        if forced_answer_prefix:
            logger.debug(f"AraNPUClient answer prefix: {forced_answer_prefix}")
            yield forced_answer_prefix

        yield from self.generate(messages)

    def close(self):
        """Compatibility method for cleanup."""
        self.shutdown()
