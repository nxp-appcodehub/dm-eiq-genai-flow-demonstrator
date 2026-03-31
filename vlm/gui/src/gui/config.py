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

end_token = "<end>"
stop_token = "<stop>"
vit_token = "<vit>"


@dataclass
class GuiConfig:
    gui_app_name: str = 'GuiApp'
    verbose: bool = False
    llmp_to_gui_queue_path: str = '/llmp_to_gui_queue'
    gui_to_llmp_queue_path: str = '/gui_to_llmp_queue'
    max_message_size: int = 1024  # Max message size in the queue
    max_messages: int = 10  # Max message count in the queue
    connect_sig: str = '<con>'  # Connection message from App
    disconnect_sig: str = '<dis>'  # Disconnection message from App
