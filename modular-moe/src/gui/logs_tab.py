"""Logs tab for Prothynesis GUI."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QColor, QFont, QTextCursor, QTextCharFormat
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTextEdit, QCheckBox, QComboBox, QSplitter, QFrame
)


class LogsTab(QWidget):
    """Log viewing tab."""
    
    # Signal to match main.py's log handler
    log_signal = None  # Will be set by main window if needed
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.main_window = parent
        self._level_colors = {
            "DEBUG": "#6a9955",
            "INFO": "#d4d4d4",
            "WARNING": "#dcdcaa",
            "ERROR": "#f48771",
            "CRITICAL": "#f44747",
        }
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # Toolbar
        toolbar = QHBoxLayout()
        
        self.level_filter = QComboBox()
        self.level_filter.addItems(["ALL", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        self.level_filter.currentTextChanged.connect(self._filter_logs)
        toolbar.addWidget(QLabel("Level:"))
        toolbar.addWidget(self.level_filter)
        
        self.auto_scroll_cb = QCheckBox("Auto-scroll")
        self.auto_scroll_cb.setChecked(True)
        toolbar.addWidget(self.auto_scroll_cb)
        
        toolbar.addStretch()
        
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_logs)
        toolbar.addWidget(self.clear_btn)
        
        self.export_btn = QPushButton("Export")
        self.export_btn.clicked.connect(self._export_logs)
        toolbar.addWidget(self.export_btn)
        
        layout.addLayout(toolbar)
        
        # Log viewer
        self.log_viewer = QTextEdit()
        self.log_viewer.setReadOnly(True)
        self.log_viewer.setFont(QFont("Consolas", 9))
        self.log_viewer.setStyleSheet("QTextEdit { background: #1e1e1e; color: #d4d4d4; }")
        layout.addWidget(self.log_viewer, stretch=1)
        
        # Status
        self.status_label = QLabel("Logs ready")
        layout.addWidget(self.status_label)
    
    def append_log(self, message: str, level: str = "INFO"):
        """Append a log message (called by main window log handler)."""
        cursor = self.log_viewer.textCursor()
        cursor.movePosition(QTextCursor.End)
        
        fmt = QTextCharFormat()
        color = self._level_colors.get(level.upper(), "#d4d4d4")
        fmt.setForeground(QColor(color))
        
        timestamp = self._get_timestamp()
        cursor.insertText(f"[{timestamp}] [{level}] {message}\n", fmt)
        
        self.log_viewer.setTextCursor(cursor)
        
        if self.auto_scroll_cb.isChecked():
            self.log_viewer.moveCursor(QTextCursor.End)
        
        self.status_label.setText(f"Logs: {self.log_viewer.document().blockCount()} entries")
    
    def _get_timestamp(self) -> str:
        from datetime import datetime
        return datetime.now().strftime("%H:%M:%S")
    
    def _filter_logs(self):
        # Simple filter - in a real implementation we'd store structured logs
        level = self.level_filter.currentText()
        if level == "ALL":
            self.status_label.setText("Showing all logs")
        else:
            self.status_label.setText(f"Filtering: {level}")
    
    def clear_logs(self):
        self.log_viewer.clear()
        self.status_label.setText("Logs cleared")
    
    def _export_logs(self):
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Logs", "", "Log Files (*.log);;Text Files (*.txt)"
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.log_viewer.toPlainText())
            self.status_label.setText(f"Exported to {path}")
