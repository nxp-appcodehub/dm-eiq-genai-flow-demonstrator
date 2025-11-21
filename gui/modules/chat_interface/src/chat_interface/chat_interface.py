# -*- coding: utf-8 -*-

# Copyright 2025 NXP
# NXP Proprietary.
# This software is owned or controlled by NXP and may only be used strictly in
# accordance with the applicable license terms. By expressly accepting such
# terms or by downloading, installing, activating and/or otherwise using the
# software, you are agreeing that you have read, and that you agree to comply
# with and are bound by, such license terms. If you do not agree to be bound
# by the applicable license terms, then you may not retain, install, activate
# or otherwise use the software.

import os
import sys
import logging
import threading
import time

import posix_ipc
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QScrollArea, QSizePolicy
)
from PyQt6.QtGui import QFont, QPixmap
from PyQt6.QtCore import QTimer, Qt, pyqtSignal, QPropertyAnimation

logger = logging.getLogger(__name__)


def resource_path(relative_path):
    """Get absolute path to resource, works for PyInstaller."""
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)


# ---------------- Chat Bubble ----------------
class ChatBubble(QWidget):
    def __init__(self, message="", is_user=True):
        super().__init__()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        img_path = resource_path("assets/user.png" if is_user else "assets/cpu.png")
        profile = QLabel()
        pixmap = QPixmap(img_path).scaled(25, 28, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        profile.setPixmap(pixmap)
        profile.setFixedSize(40, 40)
        profile.setAlignment(Qt.AlignmentFlag.AlignCenter)
        profile.setStyleSheet(f"""
            border: 2px solid {'#0EAFE0' if is_user else '#69CA00'};
            border-radius: 20px;
        """)

        bg_color = '#0EAFE0' if is_user else '#69CA00'
        self.label = QLabel(message)
        self.label.setWordWrap(True)
        self.label.setFont(QFont("Arial", 12))
        self.label.setStyleSheet(
            f"background-color: {bg_color}; color:white; border-radius:15px; padding:8px;"
        )
        self.label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        if is_user:
            layout.addStretch()
            layout.addWidget(self.label)
            layout.addWidget(profile)
        else:
            layout.addWidget(profile)
            layout.addWidget(self.label)
            layout.addStretch()

        self.setLayout(layout)

    def update_message(self, new_message: str):
        self.label.setText(new_message)


# ---------------- Main UI ----------------
class MainUI(QWidget):
    bubble_signal = pyqtSignal(str, str, str)
    system_signal = pyqtSignal(str, str)

    def __init__(self, egf_to_gui_queue):
        super().__init__()
        self.egf_to_gui_queue = egf_to_gui_queue

        # --- Layout setup ---
        main_layout = QVBoxLayout()
        self.chat_container = QWidget()
        self.chat_layout = QVBoxLayout()
        self.chat_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.chat_container.setLayout(self.chat_layout)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.chat_container)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollBar:vertical {
                background: transparent;
                width: 12px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #CCCCCC;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
        """)

        main_layout.addWidget(self.scroll_area, 2)
        self.setLayout(main_layout)

        # --- Logo ---
        bottom_image = QLabel()
        pixmap = QPixmap(resource_path("assets/NXP_Logo.png"))
        pixmap = pixmap.scaledToHeight(100, Qt.TransformationMode.SmoothTransformation)
        bottom_image.setPixmap(pixmap)
        bottom_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(bottom_image)

        # --- Signals ---
        self.bubble_signal.connect(self._manage_bubble)
        self.system_signal.connect(self._manage_system_message)

        # --- State ---
        self.current_qst_bubble = None
        self.current_rsp_bubble = None
        self.running = True
        self.connected = False

        # --- Initial system message ---
        self._manage_system_message("create", "○ Connecting to eIQ GenAI Flow...")

        # --- Start listener thread ---
        self.listener_thread = threading.Thread(target=self.queue_listener, daemon=True)
        self.listener_thread.start()

    def _manage_system_message(self, action_type: str, message: str = ""):
        label = getattr(self, "last_sys_message", None)
        if label is None:
            action_type = "create"

        if action_type == "create":
            label = QLabel(message)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setWordWrap(True)
            label.setStyleSheet("""
                QLabel {
                    color: #666666;
                    background-color: transparent;
                    font-style: italic;
                    padding: 6px 0px;
                    margin-top: 8px;
                    margin-bottom: 8px;
                    border: none;
                }
            """)
            self.chat_layout.addWidget(label)
            self.last_sys_message = label

        elif action_type == "update":
            if label is not None:
                label.setText(message)

        elif action_type == "next":
            self.last_sys_message = None

        else:  # action_type == "close":
            if label is not None:
                self.chat_layout.removeWidget(label)
                label.hide()
                label.deleteLater()
                self.last_sys_message = None

        QTimer.singleShot(0, self.smooth_scroll_to_bottom)

    def _manage_bubble(self, type, speaker, message):
        action = type
        if ((self.current_qst_bubble is not None and speaker == "User" and type == 'create')
                or (self.current_rsp_bubble is not None and speaker == "AI" and type == 'create')):
            # Prevent multiple bubbles from being created simultaneously
            action = 'update'

        if action == 'create':
            bubble = ChatBubble(message, is_user=(speaker == "User"))
            self.chat_layout.addWidget(bubble)
            if speaker == "User":
                self.current_qst_bubble = bubble
                self.current_rsp_bubble = None
            else:
                self.current_qst_bubble = None
                self.current_rsp_bubble = bubble
        elif action == 'update':
            if speaker == "User":
                if self.current_qst_bubble:
                    self.current_qst_bubble.update_message(message)
            else:
                if self.current_rsp_bubble:
                    self.current_rsp_bubble.update_message(message)
        elif type == 'next':
            if speaker == "User":
                self.current_qst_bubble = None
            else:
                self.current_rsp_bubble = None
        else:  # action_type == "close":
            if speaker == "User" and self.current_qst_bubble:
                self.chat_layout.removeWidget(self.current_qst_bubble)
                self.current_qst_bubble.hide()
                self.current_qst_bubble.deleteLater()
                self.current_qst_bubble = None
            if speaker == "AI" and self.current_rsp_bubble:
                self.chat_layout.removeWidget(self.current_rsp_bubble)
                self.current_rsp_bubble.hide()
                self.current_rsp_bubble.deleteLater()
                self.current_rsp_bubble = None
        QTimer.singleShot(0, self.smooth_scroll_to_bottom)

    def smooth_scroll_to_bottom(self):
        QTimer.singleShot(0, self._do_smooth_scroll)

    def _do_smooth_scroll(self):
        bar = self.scroll_area.verticalScrollBar()
        if hasattr(self, "_scroll_animation"):
            self._scroll_animation.stop()

        animation = QPropertyAnimation(bar, b"value", self)
        animation.setDuration(250)
        animation.setStartValue(bar.value())
        animation.setEndValue(bar.maximum())
        animation.start()
        self._scroll_animation = animation

    def queue_listener(self):
        logger.debug("Listener thread started")
        buffer_rsp = []

        while self.running:
            try:
                message, _ = self.egf_to_gui_queue.receive(timeout=1)
                message = message.decode().strip()

                if message.startswith("CON:"):
                    self.connected = True
                    self.system_signal.emit("update", "● Connected to eIQ GenAI Flow!")
                    self.system_signal.emit("next", "")
                    logger.info("Connection established with eIQ GenAI Flow.")

                if not self.connected:
                    logger.debug(f"Ignoring message before connection: {message}")
                    continue

                if message.startswith("QST:"):
                    content = message[4:]
                    if content == "<end>":
                        self.bubble_signal.emit("next", "User", "")
                    elif content == "<stop>":
                        self.system_signal.emit("close", "")
                        self.bubble_signal.emit("close", "User", "")
                    else:
                        if self.current_qst_bubble is None:
                            self.system_signal.emit("close", "")
                            self.bubble_signal.emit("create", "User", content)
                            time.sleep(0.1)  # Idle sleep to prevent UI bad rendering
                        else:
                            self.bubble_signal.emit("update", "User", content)

                elif message.startswith("RSP:"):
                    content = message[4:]
                    if content == "<end>":
                        self.bubble_signal.emit("next", "AI", "")
                        buffer_rsp.clear()
                    else:
                        buffer_rsp.append(content)
                        full = "".join(buffer_rsp)
                        if self.current_rsp_bubble is None:
                            self.bubble_signal.emit("create", "AI", full)
                        else:
                            self.bubble_signal.emit("update", "AI", full)

                elif message.startswith("CMD:"):
                    command = message[4:]
                    self.system_signal.emit("create", f"▶ Intent detected: {command}")
                    self.system_signal.emit("next", "")

                elif message.startswith("DIS:"):
                    self.system_signal.emit("create", "○ Disconnected from eIQ GenAI Flow.")
                    self.stop()

                # WakeWord detection to Gui
                elif message.startswith("WWD:"):
                    self.system_signal.emit("update", "◉ Listening...")

                # VIT is started info to Gui
                elif message.startswith("VIS:"):
                    self.system_signal.emit("create", "◎ Say the wake-word...")
                # TTS has finished info to Gui
                elif message.startswith("THF:"):
                    pass

                else:
                    pass

            except posix_ipc.BusyError:
                continue
            except Exception as e:
                logger.error(f"[ERROR] Queue listener error: {e}")
                break

    def stop(self):
        self.running = False
        if self.listener_thread.is_alive():
            self.listener_thread.join()


# ---------------- Main ----------------
if __name__ == "__main__":
    # Open queue
    egf_to_gui_queue = posix_ipc.MessageQueue('/egf_to_gui', flags=posix_ipc.O_CREAT)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    ui = MainUI(egf_to_gui_queue)
    ui.setWindowTitle("NXP® eIQ® GenAI Flow Demonstrator - GUI Example")
    ui.resize(600, 900)
    ui.setStyleSheet("background-color: white; color: black;")
    ui.show()

    try:
        sys.exit(app.exec())
    finally:
        ui.stop()
