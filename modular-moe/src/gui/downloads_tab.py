"""Downloads tab for Prothynesis GUI."""

from __future__ import annotations

import time
from typing import Optional, Dict, Any, List
from datetime import datetime

from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QFormLayout, QLineEdit, QComboBox, QProgressBar,
    QTextEdit, QTableWidget, QTableWidgetItem, QHeaderView,
    QSplitter, QFrame, QMessageBox, QCheckBox, QSpinBox
)


class DownloadWorker(QThread):
    """Worker for download tasks."""
    progress = Signal(int, str)
    finished = Signal(str)
    log = Signal(str)
    
    def __init__(self, url: str, dest: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.url = url
        self.dest = dest
        self._is_running = True
    
    def run(self):
        try:
            self.log.emit(f"Starting download: {self.url}")
            steps = 20
            for i in range(steps):
                if not self._is_running:
                    return
                time.sleep(0.1)
                progress = int((i + 1) / steps * 100)
                self.progress.emit(progress, f"Downloading... {progress}%")
            
            self.log.emit(f"Download complete: {self.dest}")
            self.finished.emit(self.dest)
        except Exception as e:
            self.log.emit(f"Download error: {e}")
    
    def stop(self):
        self._is_running = False


class DownloadsTab(QWidget):
    """Downloads management tab."""
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.main_window = parent
        self.downloads: List[Dict[str, Any]] = []
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # Add download
        add_group = QGroupBox("Add Download")
        add_form = QFormLayout(add_group)
        
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("https://huggingface.co/...")
        add_form.addRow("URL:", self.url_input)
        
        self.dest_input = QLineEdit()
        self.dest_input.setPlaceholderText("models/ or datasets/")
        add_form.addRow("Destination:", self.dest_input)
        
        type_layout = QHBoxLayout()
        self.type_combo = QComboBox()
        self.type_combo.addItems(["model", "dataset", "tokenizer"])
        type_layout.addWidget(QLabel("Type:"))
        type_layout.addWidget(self.type_combo)
        type_layout.addStretch()
        add_form.addRow("", type_layout)
        
        self.add_btn = QPushButton("Add to Queue")
        self.add_btn.clicked.connect(self._add_download)
        add_form.addRow("", self.add_btn)
        
        layout.addWidget(add_group)
        
        # Queue table
        self.queue_table = QTableWidget()
        self.queue_table.setColumnCount(5)
        self.queue_table.setHorizontalHeaderLabels(["URL/Name", "Type", "Progress", "Status", "Added"])
        self.queue_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.queue_table, stretch=1)
        
        # Log
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setFont(QFont("Consolas", 9))
        self.log_area.setStyleSheet("QTextEdit { background: #1e1e1e; color: #d4d4d4; }")
        self.log_area.setMaximumHeight(120)
        layout.addWidget(self.log_area)
    
    def _add_download(self):
        url = self.url_input.text().strip()
        dest = self.dest_input.text().strip()
        dtype = self.type_combo.currentText()
        
        if not url:
            QMessageBox.warning(self, "Add Download", "Please enter a URL.")
            return
        
        row = self.queue_table.rowCount()
        self.queue_table.insertRow(row)
        self.queue_table.setItem(row, 0, QTableWidgetItem(url))
        self.queue_table.setItem(row, 1, QTableWidgetItem(dtype))
        
        progress_bar = QProgressBar()
        progress_bar.setValue(0)
        self.queue_table.setCellWidget(row, 2, progress_bar)
        
        self.queue_table.setItem(row, 3, QTableWidgetItem("Queued"))
        self.queue_table.setItem(row, 4, QTableWidgetItem(datetime.now().strftime("%H:%M:%S")))
        
        self.url_input.clear()
        self.dest_input.clear()
        
        if self.main_window:
            self.main_window.status_label.setText(f"Added download: {url}")
        self.log_area.append(f"[{datetime.now().strftime('%H:%M:%S')}] Added: {url}")
    
    def refresh(self):
        pass
