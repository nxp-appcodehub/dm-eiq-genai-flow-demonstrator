#!/usr/bin/env python3
# Copyright 2026 NXP
# NXP Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.

import logging
import threading
from typing import Any, Optional, Dict
from queue import Queue, Empty

logger = logging.getLogger(__name__)


class BaseAdapter:
    """
    Base class for all GenAI Flow module adapters with thread management.

    Provides common functionality for:
    - Thread lifecycle management
    - Result queue handling
    - Enable/disable pattern
    - Audio reader management
    - Thread-safe state management

    Subclasses should override:
    - _worker_loop(): Main processing logic
    - shutdown(): Custom cleanup (optional, has default implementation)
    - process(): For adapters accepting external input (optional)
    """

    def __init__(self, config, audio_manager=None, verbose=False):
        """
        Initialize base adapter.

        Args:
            config: Module-specific configuration object
            audio_manager: AudioManager instance for audio I/O
            verbose: Enable verbose logging
        """
        self.config = config
        self.audio_manager = audio_manager
        self.verbose = verbose

        # State management
        self.is_running = False
        self._state_lock = threading.Lock()

        # Thread management
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._thread_name = f"{self.__class__.__name__}_Worker"

        # Result management
        self._result_queue = Queue()
        self._latest_result: Optional[Any] = None
        self._latest_info: Optional[Dict] = None
        self._result_lock = threading.Lock()
        self._result_available_cv = threading.Condition(self._result_lock)

        # Audio reader (if needed by subclass)
        self.audio_reader = None

        if self.verbose:
            logger.setLevel(logging.DEBUG)

    # ==================== Methods to Override in Subclasses ====================

    def shutdown(self):
        """
        Cleanup and shutdown adapter.

        Default implementation:
        - Calls disable() to stop worker thread
        - Unregisters audio reader if present

        Override this if your adapter needs additional cleanup
        (e.g., closing model resources, network connections, etc.)

        Example override:
            def shutdown(self):
                # Custom cleanup
                if self.model:
                    self.model.close()

                # Call base implementation
                super().shutdown()
        """
        logger.info(f"Shutting down {self.__class__.__name__}")

        # Disable to stop worker thread
        if self.is_running:
            self.disable()

        # Unregister audio reader
        if self.audio_reader:
            self.audio_reader.unregister()

    def _worker_loop(self):
        """
        Main processing loop running in background thread.

        **Subclasses SHOULD override this method** to implement their processing logic.

        Guidelines:
        - Check self._stop_event.is_set() to know when to stop
        - Read input data (from audio stream, queue, etc.)
        - Process the data according to your module's needs
        - Store results using self._store_result(result, info)
        - Either loop continuously or break after result (adapter-specific)
        - Handle exceptions gracefully

        Example patterns:

        Continuous processing (VIT, STT):
            while not self._stop_event.is_set():
                data = self.audio_reader.read(num_samples)
                if data is not None:
                    result = self._process_internal(data)
                    if result:
                        self._store_result(result)
                        break  # Stop after detection
                else:
                    time.sleep(0.01)

        Queue-driven processing (TTS):
            while not self._stop_event.is_set():
                try:
                    item = self.queue.get(timeout=0.1)
                    self._process_internal(item)
                except queue.Empty:
                    continue
        """
        logger.warning(
            f"{self.__class__.__name__}._worker_loop() not implemented. "
            "This adapter will not process anything until _worker_loop() is overridden."
        )
        # Default: just wait for stop signal
        self._stop_event.wait()

    # ==================== Optional Public API ====================

    def process(self, *args: Any, **kwargs: Any) -> Any:
        """
        Optional: Public API for external processing triggers.

        Override this method ONLY if your adapter needs to accept external input
        after being enabled (e.g., TTS.process(text, eos=True) to queue text for synthesis).

        For continuous processing adapters (VIT, STT), leave this as default.
        Those adapters process audio streams automatically via _worker_loop().

        Args:
            *args: Positional arguments (adapter-specific)
            **kwargs: Keyword arguments (adapter-specific)

        Returns:
            Processed result (adapter-specific)

        Raises:
            NotImplementedError: If adapter doesn't support direct process() calls

        Example implementations:
            # TTS adapter
            def process(self, text: Optional[str] = None, eos: bool = False):
                ...

            # Hypothetical image processor
            def process(self, image: np.ndarray, config: dict):
                ...
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} uses continuous processing via enable(). "
            "Direct process() calls are not supported."
        )

    # ==================== Thread Management ====================

    def enable(self, sync_to_current: bool = True):
        """
        Enable the adapter (start processing).

        Args:
            sync_to_current: If True, audio reader starts from current position
        """
        with self._state_lock:
            if self.is_running:
                logger.warning(f"{self.__class__.__name__} already enabled")
                return

            logger.debug(f"Enabling {self.__class__.__name__}")

            # Clear previous state
            self._stop_event.clear()
            self._clear_results()  # Always clear results

            # Enable audio reader if present
            if self.audio_reader:
                self.audio_reader.enable(sync_to_current=sync_to_current)
                logger.debug(f"Audio reader enabled (sync_to_current={sync_to_current})")

            # Start worker thread
            self._worker_thread = threading.Thread(target=self._worker_wrapper, name=self._thread_name, daemon=True)
            self._worker_thread.start()

            self.is_running = True
            logger.debug(f"{self.__class__.__name__} enabled")

    def disable(self, timeout: float = 5.0):
        """
        Disable the adapter (stop processing).

        Args:
            timeout: Maximum time to wait for thread to stop (seconds)
        """
        with self._state_lock:
            if not self.is_running:
                logger.debug(f"{self.__class__.__name__} already disabled")
                return

            logger.debug(f"Disabling {self.__class__.__name__}")
            # Signal thread to stop
            self._stop_event.set()

            # Save thread reference
            worker_thread = self._worker_thread

        # Wait for thread to finish
        if worker_thread and worker_thread.is_alive():
            logger.debug(f"Waiting for worker thread to stop (timeout={timeout}s)...")
            worker_thread.join(timeout=timeout)

            if worker_thread.is_alive():
                logger.warning(f"Worker thread did not stop within {timeout}s")

        # Disable audio reader if present
        if self.audio_reader:
            self.audio_reader.disable()
            logger.debug("Audio reader disabled")

        # Update state
        with self._state_lock:
            self.is_running = False

        logger.debug(f"{self.__class__.__name__} disabled")

    def _worker_wrapper(self):
        """Wrapper around worker loop to handle exceptions."""
        try:
            logger.debug(f"{self._thread_name} started")
            self._worker_loop()
        except Exception as e:
            logger.error(f"Error in {self._thread_name}: {e}", exc_info=True)
        finally:
            # Update is_running flag when worker exits
            with self._state_lock:
                self.is_running = False
            logger.debug(f"{self._thread_name} stopped (is_running={self.is_running})")

    # ==================== Result Management ====================

    def _store_result(self, result: Any, info: Optional[Dict] = None):
        """
        Store processing result in thread-safe manner.

        Args:
            result: Main result to store
            info: Optional additional information dictionary
        """
        with self._result_lock:
            self._latest_result = result
            self._latest_info = info
            self._result_queue.put((result, info))
            self._result_available_cv.notify_all()
            logger.debug(f"Result stored: {result}")

    def _clear_results(self):
        """Clear all stored results."""
        with self._result_lock:
            self._latest_result = None
            self._latest_info = None
            # Clear queue
            while not self._result_queue.empty():
                try:
                    self._result_queue.get_nowait()
                except Empty:
                    break

    def wait_for_result(self, stop_requested: bool = False, timeout: float = 0.01) -> Any:
        """
        Wait for and retrieve processing result (blocking).

        This method blocks until:
        - A result is available, OR
        - The worker thread stops, OR
        - stop_requested becomes True

        Used by adapters that need to wait for detection/processing to complete
        (e.g., VIT waiting for wake word, STT waiting for transcription).

        Args:
            stop_requested: External signal to stop waiting
            timeout: Timeout for each condition variable wait (seconds)
                    Default 0.01s means it checks every 10ms

        Returns:
            Result object (adapter-specific) or empty string if no result

        Example:
            >>> vit.enable()
            >>> result = vit.wait_for_result()  # Blocks until wake word detected
            >>> print(f"Wake word detected: {result}")
        """
        with self._result_available_cv:
            while not stop_requested:
                # Check if result is available
                if self._latest_result is not None:
                    result = self._latest_result
                    self._latest_result = None  # Clear after reading
                    return result

                # Check if thread is still running
                if not self.is_running or not self._worker_thread or not self._worker_thread.is_alive():
                    break

                # Wait efficiently for result or timeout
                self._result_available_cv.wait(timeout=timeout)

        return ""

    def get_detailed_info(self) -> Optional[Dict]:
        """
        Get detailed information about last result (non-blocking).

        Returns:
            Dictionary with detailed info or None
        """
        with self._result_lock:
            info = self._latest_info
            self._latest_info = None  # Clear after reading
            return info

    # ==================== Status Methods ====================

    def get_status(self) -> Dict[str, Any]:
        """
        Get adapter status.

        Returns:
            Dictionary with status information
        """
        return {
            "name": self.__class__.__name__,
            "is_running": self.is_running,
            "thread_alive": self._worker_thread.is_alive() if self._worker_thread else False,
            "audio_reader_enabled": self.audio_reader.enabled if self.audio_reader else None,
        }

    def is_thread_alive(self) -> bool:
        """Check if worker thread is alive (used by STT.mic_to_text)."""
        return self._worker_thread is not None and self._worker_thread.is_alive()
