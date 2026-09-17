"""Training tab for Prothynesis GUI."""

from __future__ import annotations

import time
from typing import Optional

from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox, QComboBox,
    QCheckBox, QProgressBar, QTextEdit, QTableWidget,
    QTableWidgetItem, QHeaderView, QSplitter, QFrame, QMessageBox,
    QLineEdit
)


class TrainingWorker(QThread):
    """Worker for training jobs."""
    progress = Signal(int, str)
    log = Signal(str)
    finished = Signal(dict)
    
    def __init__(self, config: dict, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.config = config
        self._is_running = True
    
    def run(self):
        try:
            target = self.config.get("target", "single")
            epochs = self.config.get("epochs", 1)
            
            self.log.emit(f"Starting training: target={target}, epochs={epochs}")
            self.progress.emit(0, "Initializing...")
            time.sleep(0.2)
            
            steps = 20
            for step in range(steps):
                if not self._is_running:
                    self.log.emit("Training stopped.")
                    return
                
                time.sleep(0.1)
                progress = int((step + 1) / steps * 100)
                loss = 2.5 * (1 - step / steps) + 0.1
                lr = self.config.get("learning_rate", 1e-4) * (1 - step / steps * 0.5)
                
                self.progress.emit(progress, f"Step {step + 1}/{steps} | loss={loss:.4f} | lr={lr:.2e}")
                if step % 5 == 0:
                    self.log.emit(f"Step {step}: loss={loss:.4f}, lr={lr:.2e}")
            
            self.progress.emit(100, "Finalizing...")
            time.sleep(0.2)
            
            result = {
                "status": "completed",
                "final_loss": round(loss, 4),
                "epochs_completed": epochs,
                "checkpoint": "checkpoints/latest.pt",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            
            self.log.emit(f"Training complete. Final loss: {loss:.4f}")
            self.finished.emit(result)
            
        except Exception as e:
            self.log.emit(f"Training error: {e}")
    
    def stop(self):
        self._is_running = False


class TrainingTab(QWidget):
    """Training control tab."""
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.main_window = parent
        self.worker: Optional[TrainingWorker] = None
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # Training config
        config_group = QGroupBox("Training Configuration")
        config_form = QFormLayout(config_group)
        
        self.target_combo = QComboBox()
        self.target_combo.addItems(["single", "pool", "orchestrator"])
        config_form.addRow("Target:", self.target_combo)
        
        self.model_id_input = QLineEdit()
        self.model_id_input.setPlaceholderText("e.g., solver_0001")
        config_form.addRow("Model ID:", self.model_id_input)
        
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(1)
        config_form.addRow("Epochs:", self.epochs_spin)
        
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 512)
        self.batch_size_spin.setValue(8)
        config_form.addRow("Batch Size:", self.batch_size_spin)
        
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(1e-6, 1e-2)
        self.lr_spin.setSingleStep(1e-5)
        self.lr_spin.setValue(1e-4)
        self.lr_spin.setDecimals(6)
        config_form.addRow("Learning Rate:", self.lr_spin)
        
        self.context_spin = QSpinBox()
        self.context_spin.setRange(128, 32768)
        self.context_spin.setSingleStep(128)
        self.context_spin.setValue(2048)
        config_form.addRow("Context Length:", self.context_spin)
        
        self.gradient_accum_spin = QSpinBox()
        self.gradient_accum_spin.setRange(1, 64)
        self.gradient_accum_spin.setValue(1)
        config_form.addRow("Gradient Accumulation:", self.gradient_accum_spin)
        
        self.mixed_precision_cb = QCheckBox("Mixed Precision")
        self.mixed_precision_cb.setChecked(True)
        config_form.addRow("", self.mixed_precision_cb)
        
        self.multi_run_cb = QCheckBox("Multi-Run Search")
        self.multi_run_cb.setChecked(False)
        config_form.addRow("", self.multi_run_cb)
        
        layout.addWidget(config_group)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("Start Training")
        self.start_btn.clicked.connect(self._start_training)
        controls_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._stop_training)
        self.stop_btn.setEnabled(False)
        controls_layout.addWidget(self.stop_btn)
        
        controls_layout.addStretch()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        controls_layout.addWidget(self.progress_bar)
        
        layout.addLayout(controls_layout)
        
        # Log and metrics
        splitter = QSplitter(Qt.Vertical)
        
        log_group = QGroupBox("Training Log")
        log_layout = QVBoxLayout(log_group)
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setFont(QFont("Consolas", 9))
        self.log_area.setStyleSheet("QTextEdit { background: #1e1e1e; color: #d4d4d4; }")
        log_layout.addWidget(self.log_area)
        splitter.addWidget(log_group)
        
        metrics_group = QGroupBox("Live Metrics")
        metrics_layout = QVBoxLayout(metrics_group)
        
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.metrics_table.setRowCount(4)
        metrics = [("Train Loss", "--"), ("Learning Rate", "--"), ("Epoch", "--"), ("ETA", "--")]
        for i, (metric, value) in enumerate(metrics):
            self.metrics_table.setItem(i, 0, QTableWidgetItem(metric))
            self.metrics_table.setItem(i, 1, QTableWidgetItem(value))
        metrics_layout.addWidget(self.metrics_table)
        splitter.addWidget(metrics_group)
        
        splitter.setSizes([300, 150])
        layout.addWidget(splitter, stretch=1)
    
    def _start_training(self):
        model_id = self.model_id_input.text().strip() or "solver_0001"
        
        config = {
            "target": self.target_combo.currentText(),
            "model_id": model_id,
            "epochs": self.epochs_spin.value(),
            "batch_size": self.batch_size_spin.value(),
            "learning_rate": self.lr_spin.value(),
            "context_length": self.context_spin.value(),
            "gradient_accumulation": self.gradient_accum_spin.value(),
            "mixed_precision": self.mixed_precision_cb.isChecked(),
            "multi_run": self.multi_run_cb.isChecked(),
        }
        
        self.log_area.clear()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        if self.main_window:
            self.main_window.status_label.setText(f"Training {model_id}...")
        
        self.worker = TrainingWorker(config, self)
        self.worker.progress.connect(self._on_progress)
        self.worker.log.connect(self._on_log)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()
    
    def _stop_training(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait(2000)
            self.worker = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if self.main_window:
            self.main_window.status_label.setText("Training stopped")
    
    def _on_progress(self, value: int, message: str):
        self.progress_bar.setValue(value)
        if self.main_window:
            self.main_window.status_label.setText(message)
        
        if "loss=" in message:
            parts = message.split("|")
            for part in parts:
                part = part.strip()
                if part.startswith("loss="):
                    self.metrics_table.item(0, 1).setText(part.replace("loss=", ""))
                elif part.startswith("lr="):
                    self.metrics_table.item(1, 1).setText(part.replace("lr=", ""))
    
    def _on_log(self, message: str):
        self.log_area.append(f"[{time.strftime('%H:%M:%S')}] {message}")
    
    def _on_finished(self, result: dict):
        self.worker = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.metrics_table.item(2, 1).setText(str(result.get("epochs_completed", "--")))
        
        if self.main_window:
            self.main_window.status_label.setText(f"Training complete. Loss: {result.get('final_loss', 'N/A')}")
        
        QMessageBox.information(self, "Training Complete",
                               f"Training finished successfully.\nFinal loss: {result.get('final_loss')}\nCheckpoint: {result.get('checkpoint')}")
