# -*- coding: utf-8 -*-

# Copyright 2025-2026 NXP
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
from PyQt6.QtGui import QFont, QPixmap, QPainter, QPainterPath, QFontMetrics
from PyQt6.QtCore import QTimer, Qt, pyqtSignal, QPropertyAnimation, QRectF, QRect

logger = logging.getLogger(__name__)

COLOR_MAP = {
    "RED": "#e15554",
    "GREEN": "#3bb273",
}


def resource_path(relative_path):
    """
    Get absolute path to resource
    """
    base_path = os.path.dirname(os.path.abspath(__file__))
    resolved = os.path.join(base_path, relative_path)
    logger.debug(f"[RESOURCE] Resolved path: {resolved}")
    return resolved


def is_image_path(payload: str) -> bool:
    """
    Check whether the payload is a valid image file path.
    """
    if not payload:
        return False
    if not os.path.isfile(payload):
        return False

    return payload.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))


def rounded_image(src: QPixmap, radius: int) -> QPixmap:
    """
    Apply rounded corners to a QPixmap.
    """
    logger.debug("[IMAGE] Applying rounded corners")

    result = QPixmap(src.size())
    result.fill(Qt.GlobalColor.transparent)

    painter = QPainter(result)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

    path = QPainterPath()
    path.addRoundedRect(QRectF(0, 0, src.width(), src.height()), radius, radius)

    painter.setClipPath(path)
    painter.drawPixmap(0, 0, src)
    painter.end()

    return result


class ChatBubble(QWidget):
    def __init__(self, content="", is_user=True):
        super().__init__()

        role = "USER" if is_user else "ASSISTANT"
        logger.info(f"[BUBBLE] Creating {role} bubble")

        layout = QHBoxLayout()
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(6)

        avatar_path = resource_path("assets/user.png" if is_user else "assets/cpu.png")

        avatar = QLabel()
        avatar_pix = QPixmap(avatar_path).scaled(
            25, 28, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )

        avatar.setPixmap(avatar_pix)
        avatar.setFixedSize(40, 40)
        avatar.setAlignment(Qt.AlignmentFlag.AlignCenter)

        avatar.setStyleSheet(
            f"""
            border: 2px solid {'#0EAFE0' if is_user else '#69CA00'};
            border-radius: 20px;
            """
        )

        self.label = QLabel()
        self.label.setWordWrap(True)

        bg_color = "#0EAFE0" if is_user else "#69CA00"

        # Image Bubble
        if is_image_path(content):
            logger.debug("[BUBBLE] Rendering image bubble")

            pix = QPixmap(content)
            MAX_WIDTH = 320
            MAX_HEIGHT = 420

            if pix.width() >= pix.height():
                pix = pix.scaledToWidth(
                    MAX_WIDTH,
                    Qt.TransformationMode.SmoothTransformation
                )
            else:
                pix = pix.scaledToHeight(
                    MAX_HEIGHT,
                    Qt.TransformationMode.SmoothTransformation
                )

            pix = rounded_image(pix, radius=14)

            self.label.setPixmap(pix)
            self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.label.setWordWrap(False)

            self.label.setSizePolicy(
                QSizePolicy.Policy.Fixed,
                QSizePolicy.Policy.Fixed
            )

            self.label.setStyleSheet(
                f"""
                background-color: {bg_color};
                border-radius: 18px;
                padding: 4px;
                """
            )

        # Text Bubble
        else:
            logger.debug("[BUBBLE] Rendering text bubble")

            self.label.setFont(QFont("Arial", 12))
            self.label.setAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            )
            self.label.setText(content)

            self.label.setStyleSheet(
                f"""
                background-color: {bg_color};
                color: white;
                border-radius: 18px;
                padding: 8px;
                """
            )

        self.label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred
        )

        # Layout positioning
        if is_user:
            layout.addStretch()
            layout.addWidget(self.label)
            layout.addWidget(avatar)
        else:
            layout.addWidget(avatar)
            layout.addWidget(self.label)
            layout.addStretch()

        self.setLayout(layout)

    def update_message(self, new_message: str):
        """
        Update text content of an existing bubble.
        """
        logger.debug("[BUBBLE] Updating bubble content")

        if not is_image_path(new_message):
            self.label.setText(new_message)


class MainUI(QWidget):
    bubble_signal = pyqtSignal(str, str, str)
    system_signal = pyqtSignal(str, str, str)

    def __init__(self, egf_to_gui_queue):
        super().__init__()

        logger.info("[INIT] Initializing MainUI")

        self.egf_to_gui_queue = egf_to_gui_queue
        self.running = True
        self.connected = False

        # Track active bubbles
        self.current_qst_bubble = None
        self.current_rsp_bubble = None

        # Layout
        main_layout = QVBoxLayout()

        self.chat_container = QWidget()
        self.chat_layout = QVBoxLayout()
        self.chat_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.chat_container.setLayout(self.chat_layout)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.chat_container)
        self.scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOn
        )
        self.scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
        self.scroll_area.setStyleSheet(
            """
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
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical {
                background: none;
            }
            QScrollBar:horizontal {
                background: transparent;
                height: 12px;
                margin: 0px;
            }
            QScrollBar::handle:horizontal {
                background: #CCCCCC;
                border-radius: 6px;
                min-width: 20px;
            }
            QScrollBar::add-line:horizontal,
            QScrollBar::sub-line:horizontal {
                width: 0px;
            }
            QScrollBar::add-page:horizontal,
            QScrollBar::sub-page:horizontal {
                background: none;
            }
            """
        )

        main_layout.addWidget(self.scroll_area, 2)
        self.setLayout(main_layout)

        # Logo
        bottom_image = QLabel()
        pixmap = QPixmap(resource_path("assets/NXP_Logo.png"))
        pixmap = pixmap.scaledToHeight(
            100, Qt.TransformationMode.SmoothTransformation
        )
        bottom_image.setPixmap(pixmap)
        bottom_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(bottom_image)

        # Signal Wiring
        self.bubble_signal.connect(self._manage_bubble)
        self.system_signal.connect(self._manage_system_message)

        # Initial state message
        self._manage_system_message(
            "create",
            "○ Connecting to eIQ GenAI Flow...",
            ""
        )

        # Start listener thread
        self.listener_thread = threading.Thread(
            target=self.queue_listener,
            daemon=True
        )
        self.listener_thread.start()

        logger.info("[INIT] MainUI ready")

    def resizeEvent(self, event):
        logger.debug("[UI] Resize event triggered")
        super().resizeEvent(event)
        self._update_bubble_widths()

    def _update_bubble_widths(self):
        """
        Dynamically resize bubbles to max 65% of viewport width.
        """
        viewport_width = self.scroll_area.viewport().width()
        if viewport_width <= 0:
            return

        max_width = int(viewport_width * 0.65)

        for i in range(self.chat_layout.count()):
            widget = self.chat_layout.itemAt(i).widget()

            if isinstance(widget, ChatBubble):
                label = widget.label
                text = label.text()

                if not text:
                    continue

                fm = QFontMetrics(label.font())
                rect = fm.boundingRect(
                    QRect(0, 0, max_width, 10000),
                    Qt.TextFlag.TextWordWrap,
                    text
                )

                ideal_width = rect.width() + 24
                label.setMaximumWidth(min(ideal_width, max_width))
                label.setWordWrap(True)

    def _do_smooth_scroll(self):
        """
        Animate scroll to bottom.
        """
        logger.debug("[UI] Smooth scroll")

        # Force layout update before scrolling
        self.chat_container.updateGeometry()
        QApplication.processEvents()

        bar = self.scroll_area.verticalScrollBar()

        if hasattr(self, "_scroll_animation"):
            self._scroll_animation.stop()

        animation = QPropertyAnimation(bar, b"value", self)
        animation.setDuration(250)
        animation.setStartValue(bar.value())
        animation.setEndValue(bar.maximum())
        animation.start()

        self._scroll_animation = animation

    def _manage_system_message(self, action_type: str, message: str = "", color: str = ""):
        logger.info(f"System message received: action_type={action_type}, message={message}, color={color}")
        label = getattr(self, "last_sys_message", None)
        if label is None:
            action_type = "create"

        color_value = COLOR_MAP.get(color, "#666666")

        if action_type == "create":
            label = QLabel()
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)

            if is_image_path(message):
                logger.debug("[SYSTEM] Rendering image system message")
                pix = QPixmap(message)
                viewport_width = self.scroll_area.viewport().width()
                max_width = int(viewport_width * 0.9)
                max_height = int(viewport_width * 0.6)
                if pix.width() >= pix.height():
                    pix = pix.scaledToWidth(
                        max_width,
                        Qt.TransformationMode.SmoothTransformation
                    )
                else:
                    pix = pix.scaledToHeight(
                        max_height,
                        Qt.TransformationMode.SmoothTransformation
                    )
                label.setPixmap(pix)
                self.last_sys_message = label

            else:
                logger.debug("[SYSTEM] Rendering text system message")
                label = QLabel(message)
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                label.setWordWrap(True)
                label.setStyleSheet(
                    f"""
                    color: {color_value};
                    background-color: transparent;
                    font-style: italic;
                    padding: 6px 0px;
                    margin-top: 8px;
                    margin-bottom: 8px;
                    border: none;
                    """
                )
                self.last_sys_message = label
            self.chat_layout.addWidget(label)

        elif action_type == "update":
            if label is not None:
                label.setText(message)
                label.setStyleSheet(f"color: {color_value};")

        elif action_type == "next":
            self.last_sys_message = None

        else:  # action_type == "close":
            if label is not None:
                self.chat_layout.removeWidget(label)
                label.hide()
                label.deleteLater()
                self.last_sys_message = None

        QTimer.singleShot(30, self._do_smooth_scroll)

    def _manage_bubble(self, type, speaker, message):
        logger.info(f"Bubble message received: action_type={type}, speaker={speaker}, message={message}")
        action = type
        if (self.current_qst_bubble is not None and speaker == "User" and type == "create") or (
            self.current_rsp_bubble is not None and speaker == "AI" and type == "create"
        ):
            # Prevent multiple bubbles from being created simultaneously
            action = "update"

        is_user = speaker == "User"

        if action == "create":
            bubble = ChatBubble(content=message, is_user=is_user)
            self.chat_layout.addWidget(bubble)

            if is_user:
                self.current_qst_bubble = bubble
                self.current_rsp_bubble = None
            else:
                self.current_qst_bubble = None
                self.current_rsp_bubble = bubble

        elif action == "update":
            if is_user:
                if self.current_qst_bubble:
                    self.current_qst_bubble.update_message(message)
            else:
                if self.current_rsp_bubble:
                    self.current_rsp_bubble.update_message(message)

        elif type == "next":
            if is_user:
                self.current_qst_bubble = None
            else:
                self.current_rsp_bubble = None

        else:  # action_type == "close":
            if is_user and self.current_qst_bubble:
                self.chat_layout.removeWidget(self.current_qst_bubble)
                self.current_qst_bubble.hide()
                self.current_qst_bubble.deleteLater()
                self.current_qst_bubble = None

            if not is_user and self.current_rsp_bubble:
                self.chat_layout.removeWidget(self.current_rsp_bubble)
                self.current_rsp_bubble.hide()
                self.current_rsp_bubble.deleteLater()
                self.current_rsp_bubble = None

        QTimer.singleShot(0, self._update_bubble_widths)
        QTimer.singleShot(30, self._do_smooth_scroll)

    def queue_listener(self):
        """
        Background thread:
        - Reads POSIX message queue
        - Emits Qt signals
        """
        logger.info("[THREAD] Queue listener started")

        buffer_rsp = []

        while self.running:
            try:
                message, _ = self.egf_to_gui_queue.receive(timeout=1)
                message = message.decode()
                logger.debug(f"[EIQ_TO_GUI_QUEUE] {message}")

                # Connection
                if message.startswith("CON:"):
                    logger.info("[STATE] Connected to backend")
                    self.connected = True
                    self.system_signal.emit(
                        "update",
                        "● Connected to eIQ GenAI Flow!",
                        "GREEN"
                    )
                    self.system_signal.emit("next", "", "")
                    continue

                if not self.connected:
                    continue

                # User Message
                if message.startswith("QST:"):
                    content = message[4:]

                    if content == "<end>":
                        self.bubble_signal.emit("next", "User", "")

                    elif content == "<stop>":
                        self.system_signal.emit("close", "", "")
                        self.bubble_signal.emit("close", "User", "")

                    else:
                        if self.current_qst_bubble is None:
                            self.system_signal.emit("close", "", "")
                            self.bubble_signal.emit(
                                "create", "User", content
                            )
                            time.sleep(0.1)
                        else:
                            self.bubble_signal.emit(
                                "update", "User", content
                            )

                # Assistant Message
                elif message.startswith("RSP:"):
                    content = message[4:]

                    if content == "<end>":
                        self.bubble_signal.emit("next", "AI", "")
                        buffer_rsp.clear()

                    else:
                        buffer_rsp.append(content)
                        full = "".join(buffer_rsp)

                        if self.current_rsp_bubble is None:
                            self.bubble_signal.emit(
                                "create", "AI", full
                            )
                        else:
                            self.bubble_signal.emit(
                                "update", "AI", full
                            )

                elif message.startswith("CMD:"):
                    logger.info("Command message received")
                    command = message[4:]
                    self.system_signal.emit("create", command, "")
                    self.system_signal.emit("next", "", "")

                # WakeWord detection to Gui
                elif message.startswith("WWD:"):
                    logger.info("Wake message received")
                    if len(message) > 4:
                        color = message[4:]
                        self.system_signal.emit("update", "◉ Listening...", color)
                    else:
                        self.system_signal.emit("update", "◉ Listening...", "")

                # VIT is started info to Gui
                elif message.startswith("VIS:"):
                    logger.info("VIT message received")
                    self.system_signal.emit("create", "◎ Say the wake-word...", "")

                # TTS has finished info to Gui
                elif message.startswith("THF:"):
                    logger.info("TTS has finished message received")
                    pass

                # Disconnect
                elif message.startswith("DIS:"):
                    logger.warning("[STATE] Disconnected from backend")
                    self.system_signal.emit(
                        "create",
                        "○ Disconnected from eIQ GenAI Flow.",
                        "RED"
                    )
                    self.stop()

                else:
                    pass

            except posix_ipc.BusyError:
                continue
            except Exception as e:
                logger.exception(f"[ERROR] Queue listener failure: {e}")
                break

    # Shutdown
    def stop(self):
        logger.info("[SHUTDOWN] Stopping UI")
        self.running = False
        if (self.listener_thread.is_alive() and threading.current_thread() != self.listener_thread):
            self.listener_thread.join()

        logger.info("[SHUTDOWN] UI stopped")


if __name__ == "__main__":

    egf_to_gui_queue = posix_ipc.MessageQueue(
        '/egf_to_gui',
        flags=posix_ipc.O_CREAT
    )

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
