# Copyright 2025-2026 NXP
# NXP Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.

from dataclasses import dataclass
from gui.config import GuiConfig


@dataclass
class ChatInterfaceConfig(GuiConfig):
    gui_app_name: str = "Chat_Interface"
    llmp_to_gui_queue_path: str = "/egf_to_gui"  # Must be aligned with launch_gui.sh
    max_message_size: int = 1024  # Max message size in the queue
    max_messages: int = 10  # Max message count in the queue
