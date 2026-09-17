"""Chat tab for Prothynesis GUI."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Optional, List, Dict, Any

from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QTextCursor, QTextCharFormat, QColor, QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit,
    QPushButton, QLabel, QGroupBox, QFormLayout, QSpinBox,
    QDoubleSpinBox, QComboBox, QCheckBox, QProgressBar,
    QSplitter, QFrame
)


class ChatWorker(QThread):
    """Worker thread for chat generation."""
    text_chunk = Signal(str)
    finished = Signal(str)
    error = Signal(str)
    telemetry = Signal(dict)
    
    def __init__(self, prompt: str, config: dict, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.prompt = prompt
        self.config = config
        self._is_running = True
    
    def run(self):
        try:
            start_time = time.time()
            
            # Simulate orchestration telemetry
            telemetry = {
                "models_recruited": 3,
                "orchestral_levels": ["Orchestral", "Chief"],
                "deliberation_rounds": 2,
                "verification_state": "passed",
                "confidence": 0.87,
                "latency_ms": 0,
                "compute_usage": "low",
                "timestamp": datetime.now().isoformat(),
            }
            
            # Simulate streaming response
            response_parts = [
                "Based on ", "the ", "orchestrated ", "analysis ", "from ", 
                "multiple ", "solver ", "models, ", "the ", "answer ", "is:",
                "\n\n", "This ", "is ", "a ", "simulated ", "response ", "from ",
                "the ", "Prothynesis ", "system. ", "In ", "production, ", "this ",
                "would ", "invoke ", "real ", "75M ", "parameter ", "solver ", 
                "models ", "with ", "hierarchical ", "orchestration."
            ]
            
            full_response = ""
            for part in response_parts:
                if not self._is_running:
                    break
                time.sleep(0.05)
                full_response += part
                self.text_chunk.emit(part)
            
            telemetry["latency_ms"] = int((time.time() - start_time) * 1000)
            self.telemetry.emit(telemetry)
            self.finished.emit(full_response)
            
        except Exception as e:
            self.error.emit(str(e))
    
    def stop(self):
        self._is_running = False


class ChatTab(QWidget):
    """Chat interface tab."""
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.main_window = parent
        self.worker: Optional[ChatWorker] = None
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # Main splitter: chat area + telemetry
        splitter = QSplitter(Qt.Horizontal)
        
        # Left: Chat area
        chat_widget = QWidget()
        chat_layout = QVBoxLayout(chat_widget)
        chat_layout.setContentsMargins(0, 0, 0, 0)
        chat_layout.setSpacing(8)
        
        # Chat history
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setFont(QFont("Consolas", 10))
        self.chat_history.setStyleSheet("""
            QTextEdit { background: #1e1e1e; color: #d4d4d4; border: 1px solid #555; }
        """)
        chat_layout.addWidget(self.chat_history, stretch=1)
        
        # Input area
        input_layout = QHBoxLayout()
        
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Enter your prompt...")
        self.input_field.setFont(QFont("Consolas", 10))
        self.input_field.returnPressed.connect(self._send_message)
        input_layout.addWidget(self.input_field, stretch=1)
        
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self._send_message)
        input_layout.addWidget(self.send_button)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self._stop_generation)
        self.stop_button.setEnabled(False)
        input_layout.addWidget(self.stop_button)
        
        chat_layout.addLayout(input_layout)
        
        # Generation config
        config_layout = QHBoxLayout()
        
        config_layout.addWidget(QLabel("Temperature:"))
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.0, 2.0)
        self.temp_spin.setSingleStep(0.1)
        self.temp_spin.setValue(0.7)
        config_layout.addWidget(self.temp_spin)
        
        config_layout.addWidget(QLabel("Top-K:"))
        self.top_k_spin = QSpinBox()
        self.top_k_spin.setRange(1, 100)
        self.top_k_spin.setValue(2)
        config_layout.addWidget(self.top_k_spin)
        
        config_layout.addWidget(QLabel("Max Tokens:"))
        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(10, 2000)
        self.max_tokens_spin.setSingleStep(10)
        self.max_tokens_spin.setValue(100)
        config_layout.addWidget(self.max_tokens_spin)
        
        config_layout.addStretch()
        chat_layout.addLayout(config_layout)
        
        splitter.addWidget(chat_widget)
        
        # Right: Telemetry panel
        telemetry_widget = QGroupBox("Telemetry")
        telemetry_layout = QFormLayout(telemetry_widget)
        telemetry_widget.setMaximumWidth(280)
        
        self.telemetry_labels = {}
        telemetry_fields = [
            ("models_recruited", "Models Recruited:"),
            ("orchestral_levels", "Orchestral Levels:"),
            ("deliberation_rounds", "Deliberation Rounds:"),
            ("verification_state", "Verification:"),
            ("confidence", "Confidence:"),
            ("latency_ms", "Latency (ms):"),
            ("compute_usage", "Compute Usage:"),
            ("timestamp", "Timestamp:"),
        ]
        
        for key, label in telemetry_fields:
            lbl = QLabel("--")
            lbl.setWordWrap(True)
            lbl.setFont(QFont("Consolas", 9))
            telemetry_layout.addRow(label, lbl)
            self.telemetry_labels[key] = lbl
        
        self.telemetry_labels["confidence"].setToolTip("Confidence score 0.0-1.0")
        self.telemetry_labels["latency_ms"].setToolTip("End-to-end latency in milliseconds")
        
        splitter.addWidget(telemetry_widget)
        splitter.setSizes([900, 280])
        
        layout.addWidget(splitter, stretch=1)
    
    def _send_message(self):
        text = self.input_field.text().strip()
        if not text or self.worker is not None:
            return
        
        self.input_field.clear()
        self.send_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        
        # Add user message to chat
        self._append_message("User", text, "#0078d7")
        
        # Add assistant placeholder
        self._assistant_started = False
        self._assistant_cursor = self._append_message("Assistant", "", "#4ec9b0")
        
        config = {
            "temperature": self.temp_spin.value(),
            "top_k": self.top_k_spin.value(),
            "max_new_tokens": self.max_tokens_spin.value(),
        }
        
        self.worker = ChatWorker(text, config, self)
        self.worker.text_chunk.connect(self._on_text_chunk)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.telemetry.connect(self._on_telemetry)
        self.worker.start()
    
    def _stop_generation(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait(2000)
            self.worker = None
        self._reset_input_state()
    
    def _on_text_chunk(self, chunk: str):
        if not hasattr(self, '_assistant_started') or not self._assistant_started:
            self._assistant_cursor.insertText(chunk)
            self._assistant_started = True
        else:
            self._assistant_cursor.insertText(chunk)
        self.chat_history.moveCursor(QTextCursor.End)
    
    def _on_finished(self, response: str):
        self.worker = None
        self._reset_input_state()
        if self.main_window:
            self.main_window.status_label.setText("Ready")
    
    def _on_error(self, error: str):
        self.worker = None
        self._reset_input_state()
        self._append_message("System", f"Error: {error}", "#f48771")
    
    def _on_telemetry(self, telemetry: dict):
        for key, value in telemetry.items():
            if key in self.telemetry_labels:
                lbl = self.telemetry_labels[key]
                if isinstance(value, float):
                    if key == "confidence":
                        lbl.setText(f"{value:.2f}")
                    else:
                        lbl.setText(f"{value:.1f}")
                elif isinstance(value, list):
                    lbl.setText(", ".join(str(v) for v in value))
                else:
                    lbl.setText(str(value))
    
    def _append_message(self, role: str, text: str, color: str) -> QTextCursor:
        cursor = self.chat_history.textCursor()
        cursor.moveCursor(QTextCursor.End)
        
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        
        cursor.insertText(f"\n[{role}]\n", fmt)
        cursor.insertText(f"{text}\n")
        
        self.chat_history.setTextCursor(cursor)
        return cursor
    
    def _reset_input_state(self):
        self.send_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if self.main_window:
            self.main_window.status_label.setText("Ready")
    
    def clear_chat(self):
        self.chat_history.clear()
        for lbl in self.telemetry_labels.values():
            lbl.setText("--")
    
    def save_session(self):
        pass
