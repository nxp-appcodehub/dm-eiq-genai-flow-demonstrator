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
import threading
import queue
import logging
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Event types for adapter communication.

    NOTE: Keep entries in alphabetical order.
    """

    AUDIO_FILLER_PLAY = "AUDIO_FILLER_PLAY"
    AUDIO_FILLER_REGISTER = "AUDIO_FILLER_REGISTER"
    CONTINUOUS_WAKE = "CONTINUOUS_WAKE"
    END_OF_INPUT = "END_OF_INPUT"
    INIT = "INIT"
    INPUT_TEXT = "INPUT_TEXT"
    INTENT_DETECTED = "INTENT_DETECTED"
    KEYBOARD_KEYPRESS = "KEYBOARD_KEYPRESS"
    KEYBOARD_WAKE = "KEYBOARD_WAKE"
    LISTENING = "LISTENING"
    STT_END = "STT_END"
    TIMEOUT = "TIMEOUT"
    TTS_COMPLETE = "TTS_COMPLETE"
    TTS_PROCESS = "TTS_PROCESS"
    TTS_START_SEGMENT = "TTS_START_SEGMENT"
    UNVERIFIED_SPEAKER = "UNVERIFIED_SPEAKER"
    VAD_SPEECH_END = "VAD_SPEECH_END"
    VAD_SPEECH_START = "VAD_SPEECH_START"
    VERIFIED_SPEAKER = "VERIFIED_SPEAKER"
    VIT_WAKE = "VIT_WAKE"
    VOICE_ID_USED = "VOICE_ID_USED"
    VOICE_ID_WAKE = "VOICE_ID_WAKE"
    VOICE_ID_NO_WAKE = "VOICE_ID_NO_WAKE"
    VOICE_ID_STOP_COMMAND = "VOICE_ID_STOP_COMMAND"


@dataclass
class Event:
    """Event data structure for adapter communication."""

    event_type: EventType
    source: str = None  # Adapter name
    timestamp: float = field(default_factory=time.monotonic)
    data: Optional[Dict[str, Any]] = None


class EventManager:
    """
    Thread-safe event bus for inter-adapter communication.

    Allows adapters to publish events and subscribe to events from other adapters.
    Works across threads using queue-based message passing.
    """

    def __init__(self):
        # using set for Callable to avoid duplicates
        self._subscribers: dict[EventType, set[Callable[[Event], None]]] = {}
        self._event_queue = queue.Queue()
        self._wait_list = []
        self._running = threading.Event()
        self._dispatch_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        logger.info("EventManager initialized successfully")

    def start(self):
        """Start the event dispatch thread."""
        if self._running.is_set():
            logger.warning("EventManager already running")
            return

        self._running.set()
        self._dispatch_thread = threading.Thread(
            target=self._dispatch_loop, name="EventManager_Dispatcher", daemon=True
        )
        self._dispatch_thread.start()
        logger.debug("EventManager dispatch thread started")

    def stop(self):
        """Stop the event dispatch thread."""
        if not self._running.is_set():
            return

        self._running.clear()

        # Add sentinel to unblock queue
        self._event_queue.put(None)

        if self._dispatch_thread:
            self._dispatch_thread.join(timeout=2.0)

        logger.debug("EventManager stopped")

    def subscribe(self, event_type: EventType | list[EventType], callback: Callable[[Event], None]):
        """
        Subscribe to an event type.

        Args:
            event_type: Type of event to subscribe to
            callback: Function to call when event occurs (must be thread-safe)
        """
        if not isinstance(event_type, list):
            event_type = [event_type]

        frame = inspect.stack()[1]
        caller_self = frame[0].f_locals.get("self")
        caller_name = caller_self.__class__.__name__ if caller_self else frame.function

        with self._lock:
            for evt in event_type:
                self._subscribers.setdefault(evt, set()).add(callback)
                logger.debug(f"Subscribed to: {evt.value} from {caller_name}")

    def unsubscribe(self, event_type: EventType | list[EventType], callback: Callable):
        """
        Unsubscribe from an event type.

        Args:
            event_type: Type of event to unsubscribe from
            callback: Callback function to remove
        """
        if not isinstance(event_type, list):
            event_type = [event_type]

        frame = inspect.stack()[1]
        caller_self = frame[0].f_locals.get("self")
        caller_name = caller_self.__class__.__name__ if caller_self else frame.function

        with self._lock:
            for evt in event_type:
                if evt in self._subscribers:
                    if callback in self._subscribers[evt]:
                        self._subscribers[evt].discard(callback)
                        logger.debug(f"Unsubscribed from: {evt.value} from {caller_name}")
                    else:
                        logger.debug(f"Callback not found for {evt.value} from {caller_name}")

    def publish(self, event: Event | list[Event]):
        """
        Publish one or more events to all subscribers.

        Args:
            event: Event or list of Events to publish
        """
        if not self._running.is_set():
            if isinstance(event, list):
                for e in event:
                    logger.warning(f"EventManager not running, dropping event: {e.event_type.value}")
            else:
                logger.warning(f"EventManager not running, dropping event: {event.event_type.value}")
            return

        if not isinstance(event, list):
            event = [event]

        for e in event:
            self._event_queue.put(e)
            logger.debug(f"Event published: {e.event_type.value} from {e.source}")

    def wait(self, event_type: EventType, timeout: Optional[float] = None) -> Optional[Event]:
        """
        Wait for an event of the given type to be published.

        Args:
            event_type: Type of event to wait for
            timeout: Maximum time to wait in seconds (None = wait forever)

        Returns:
            The Event that was published, or None if timeout occurred
        """
        waiter = {
            "event_type": event_type,
            "signal": threading.Event(),
            "result": None,
        }

        with self._lock:
            self._wait_list.append(waiter)

        # Block until event is received or timeout
        triggered = waiter["signal"].wait(timeout=timeout)

        with self._lock:
            if waiter in self._wait_list:
                self._wait_list.remove(waiter)

        if triggered:
            return waiter["result"]

        return None

    def _dispatch_loop(self):
        """Event dispatch loop running in background thread."""
        logger.debug("Event dispatch loop started")

        while self._running.is_set():
            try:
                # Block with timeout to check running flag periodically
                event = self._event_queue.get(timeout=0.5)

                # Sentinel value to exit
                if event is None:
                    break

                self._dispatch_event(event)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in event dispatch loop: {e}", exc_info=True)

        logger.debug("Event dispatch loop stopped")

    def _dispatch_event(self, event: Event):
        """
        Dispatch event to subscribers.

        Args:
            event: Event to dispatch
        """
        with self._lock:
            # Notify waiting threads
            for waiter in self._wait_list:
                if waiter["event_type"] == event.event_type:
                    waiter["result"] = event
                    waiter["signal"].set()

            subscribers = self._subscribers.get(event.event_type, []).copy()

        if not subscribers:
            logger.debug(f"No subscribers for {event.event_type.value}")
            return

        logger.debug(f"Dispatching {event.event_type.value} to {len(subscribers)} subscriber(s)")

        for callback in subscribers:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in event callback for {event.event_type.value}: {e}", exc_info=True)

    def get_status(self) -> dict:
        """Get event bus status."""
        with self._lock:
            subscriber_counts = {
                event_type.value: len(callbacks) for event_type, callbacks in self._subscribers.items()
            }

        return {
            "running": self._running.is_set(),
            "queue_size": self._event_queue.qsize(),
            "subscribers": subscriber_counts,
        }
