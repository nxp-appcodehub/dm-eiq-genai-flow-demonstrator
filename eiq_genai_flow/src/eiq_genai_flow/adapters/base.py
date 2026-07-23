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
import queue
import threading
from typing import Any, Optional, Callable
from eiq_genai_flow.adapters.event_manager import EventManager, EventType, Event

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

    def __init__(self, audio_manager=None, event_manager=EventManager | None):
        """
        Initialize base adapter.

        Args:
            config: Module-specific configuration object
            audio_manager: AudioManager instance for audio I/O
            verbose: Enable verbose logging
        """
        self.audio_manager = audio_manager

        # State management
        self._state_lock = threading.Lock()
        self._enabled = threading.Event()

        # Thread management
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._thread_name = self.__class__.__name__

        # Audio reader (if needed by subclass)
        self.audio_reader = None
        self.event_manager = event_manager
        self._subscriptions_lock = threading.Lock()
        self._active_subscriptions: set[tuple] = set()

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
        self.disable()

        # Unregister audio reader
        if self.audio_reader:
            self.audio_reader.unregister()

    def publish(self, event_type: EventType | list[EventType], data: Any = None) -> None:
        if not self.event_manager:
            logger.warning(f"{self._thread_name}: cannot publish - event_manager not set")
            return

        if not isinstance(event_type, list):
            event_type = [event_type]

        self.event_manager.publish(
            [
                Event(
                    event_type=evt,
                    source=self._thread_name,
                    data=data,
                )
                for evt in event_type
            ]
        )

    def subscribe(self, event_type: EventType | list[EventType], callback: Callable[[Event], None]):
        if not isinstance(event_type, list):
            event_type = [event_type]

        for evt in event_type:
            with self._subscriptions_lock:
                self._active_subscriptions.add((evt, callback))

            self.event_manager.subscribe(evt, callback)
            logger.debug(f"{self._thread_name}: subscribed to {evt} with {callback.__name__}")

    def unsubscribe(self, event_type: EventType | list[EventType], callback: Callable[[Event], None]):
        if not self.event_manager:
            logger.warning(f"{self._thread_name}: cannot unsubscribe - event_manager not set")
            return

        if not isinstance(event_type, list):
            event_type = [event_type]

        for evt in event_type:
            with self._subscriptions_lock:
                self._active_subscriptions.discard((evt, callback))

            self.event_manager.unsubscribe(evt, callback)
            logger.debug(f"{self._thread_name}: unsubscribed to {evt} with {callback.__name__}")

    def unsubscribe_all(self):
        if not self.event_manager:
            logger.warning(f"{self._thread_name}: Cannot unsubscribe_all - event_manager not set")
            return

        # copy and clear under lock:
        with self._subscriptions_lock:
            subscriptions = list(self._active_subscriptions)
            self._active_subscriptions.clear()

        for evt, callback in subscriptions:
            self.event_manager.unsubscribe(evt, callback)
            logger.debug(f"{self._thread_name}: unsubscribed from {evt} with {callback.__name__}")

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
            f"{self.__class__.__name__}._worker_loop() not implemented."
            "This adapter will not process anything until _worker_loop() is overridden."
        )
        # Default: just wait for stop signal
        self._stop_event.wait()

    # ==================== Thread Management ====================

    def enable(self, sync_to_current: bool = True, start_worker_loop: bool = True):
        """
        Enable the adapter (start processing).

        Args:
            sync_to_current: If True, audio reader starts from current position
        """

        # Clear previous state
        self._stop_event.clear()  # does not work properly if inside 'with self._state_lock:'

        with self._state_lock:
            logger.debug(f"Enabling {self.__class__.__name__}")

            # Enable audio reader if present
            if self.audio_reader:
                self.audio_reader.enable(sync_to_current=sync_to_current)
                logger.debug(f"{self._thread_name}: Audio reader enabled (sync_to_current={sync_to_current})")

            # Start worker thread
            if start_worker_loop:
                self._worker_thread = threading.Thread(target=self._worker_wrapper, name=self._thread_name, daemon=True)
                self._worker_thread.start()
            self._enabled.set()

            logger.info(f"{self.__class__.__name__} enabled")

    def disable(self, timeout: float = 5.0):
        """
        Disable the adapter (stop processing).

        Args:
            timeout: Maximum time to wait for thread to stop (seconds)
        """
        with self._state_lock:
            logger.debug(f"Disabling {self.__class__.__name__}")

            # Signal thread to stop
            self._stop_event.set()

            # Wait for thread to finish
            if self._worker_thread and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=timeout)
                if self._worker_thread.is_alive():
                    logger.warning(f"Worker thread did not stop within {timeout}s")

            # Disable audio reader if present
            if self.audio_reader:
                self.audio_reader.disable()

            if self.event_manager and self._active_subscriptions:
                self.unsubscribe_all()

            self._enabled.clear()

            logger.info(f"{self.__class__.__name__} disabled")

    def _worker_wrapper(self):
        """Wrapper around worker loop to handle exceptions."""
        try:
            logger.debug(f"{self._thread_name} started")
            self._worker_loop()
        except Exception as e:
            logger.error(f"Error in {self._thread_name}: {e}", exc_info=True)
        finally:
            # worker exits
            logger.debug(f"{self._thread_name} _worker_loop ended")

    def is_thread_alive(self) -> bool:
        """Check if worker thread is alive (used by STT.mic_to_text)."""
        return self._worker_thread is not None and self._worker_thread.is_alive()

    def __enter__(self):
        self.enable()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disable()
        return False

    @property
    def is_running(self) -> bool:
        """
        Check if adapter is currently running.

        Returns:
            bool: True if adapter is enabled and worker thread is active
        """
        return self._enabled.is_set()


class EventAdapter(BaseAdapter):
    """
    Base class for asynchronous event handling with thread management.
    Subclasses should define the `events` class attribute.
    """

    # Subclasses should override this with their list of events
    subscribed_events: list = []

    def __init__(self, audio_manager=None, event_manager=EventManager | None):
        super().__init__(audio_manager=audio_manager,
                         event_manager=event_manager)
        self.queue = queue.Queue()
        self.disable_wait = threading.Event()

    def enable(self, sync_to_current=True):
        """Subscribe to events and start worker. Called by __enter__ in 'with' statement."""
        for e in self.subscribed_events:
            logger.debug(f"Subscribing to event: {e}")
            self.subscribe(e, self._on_event)
        super().enable(sync_to_current=sync_to_current)

    def disable(self, timeout: float = 5.0):
        """
        Called by __exit__ in 'with' statement.
            - unsubscribe from events.
            - Wait for all events already in queue to be processed.
            - disable the worker thread.
        """
        for e in self.subscribed_events:
            logger.debug(f"Unsuscribing to event: {e}")
            self.unsubscribe(e, self._on_event)

        if not self.queue.empty():
            self.disable_wait.clear()
            self.disable_wait.wait(timeout=timeout)

        logger.debug("All queued events processed, proceeding with disable")
        super().disable(timeout=timeout)

    def _on_event(self, event: Event):
        """Callback when subscribed event is triggered."""
        logger.debug(f"received event {event.event_type}")
        self.queue.put(event)

    def _worker_loop(self):
        """Main worker loop for processing queued events."""
        while not self._stop_event.is_set():
            try:
                event = self.queue.get(timeout=0.5)
                self._process_event(event)

                # Notify disable operation that queue is empty
                if self.queue.empty():
                    self.disable_wait.set()

            except queue.Empty:
                continue

    def _process_event(self, event: Event):
        """Process the event. Override in subclass for custom behavior."""
        logger.warning(f"Processing event function undefined for {self.__class__.__name__}")
