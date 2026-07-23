# Copyright 2026 NXP
# NXP Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.

import os
import re
import sys
import tty
import select
import termios
from eiq_genai_flow.adapters.base import BaseAdapter
from eiq_genai_flow.adapters.event_manager import EventManager, EventType


class KeyboardAdapter(BaseAdapter):
    def __init__(self, event_manager: EventManager, wake_only: bool = False):
        super().__init__(event_manager=event_manager)
        self.wake_only = wake_only
        self.input_buffer = []
        self.fd = sys.stdin.fileno()

    def _worker_loop(self):
        old_settings = termios.tcgetattr(self.fd)
        try:
            tty.setcbreak(self.fd)

            while not self._stop_event.is_set():
                try:
                    ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                    if not ready:
                        continue

                    # Read up to 1024 bytes (larger pastes handled by next iteration)
                    ch = os.read(self.fd, 1024).decode('utf-8')

                    # Strip irrelevant sequences (spaces, arrows, etc.)
                    ch = re.sub(r'\x1b\[[0-9;]*[A-Za-z~]', '', ch)

                    for c in ch:
                        # Handle Ctrl+C
                        if ord(c) == 3:  # Ctrl+C
                            raise KeyboardInterrupt
                        # Handle Enter key
                        if c == "\r" or c == "\n":
                            print()  # Add newline after Enter
                            user_input = "".join(self.input_buffer).strip()
                            if not user_input:
                                # Just ENTER pressed without text
                                self.publish(EventType.KEYBOARD_WAKE)
                            else:
                                # Text was typed, send it
                                self.publish(EventType.KEYBOARD_WAKE)
                                self.publish(EventType.INPUT_TEXT, data=user_input)
                                self.publish(EventType.END_OF_INPUT)
                            self.input_buffer.clear()
                            continue

                        if self.wake_only:
                            continue

                        # Handle Backspace
                        if c == "\x7f":  # Backspace
                            if self.input_buffer:
                                self.input_buffer.pop()
                                self.publish(
                                    EventType.KEYBOARD_KEYPRESS,
                                    data={"key": "backspace", "buffer": "".join(self.input_buffer)},
                                )
                                sys.stdout.write("\b \b")
                                sys.stdout.flush()
                        # Handle regular characters
                        else:
                            self.input_buffer.append(c)
                            self.publish(
                                EventType.KEYBOARD_KEYPRESS,
                                data={"key": c, "buffer": "".join(self.input_buffer)},
                            )
                            sys.stdout.write(c)
                            sys.stdout.flush()

                except KeyboardInterrupt:
                    exit()

        finally:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, old_settings)
