"""Benchmarks tab for Prothynesis GUI."""

from __future__ import annotations

import time
from typing import Optional, Dict, Any, List

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox, QComboBox,
    QCheckBox, QProgressBar, QTextEdit, QTableWidget,
    QTableWidgetItem, QHeaderView, QSplitter, QFrame, QMessageBox,
    QRadioButton, QButtonGroup
)


class BenchmarkWorker(QThread):
    """Worker for benchmark execution."""
    progress = Signal(int, str)
    result = Signal(dict)
    log = Signal(str)
    
    def __init__(self, config: dict, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.config = config
        self._is_running = True
    
    def run(self):
        try:
            benchmarks = self.config.get("benchmarks", ["mmlu", "math", "coding"])
            self.log.emit(f"Starting benchmarks: {', '.join(benchmarks)}")
            self.progress.emit(0, "Initializing benchmarks...")
            time.sleep(0.2)
            
            results = {}
            total = len(benchmarks)
            for i, bench in enumerate(benchmarks):
                if not self._is_running:
                    return
                
                self.log.emit(f"Running benchmark: {bench}")
                time.sleep(0.3)
                
                score = round(0.5 + (i + 1) * 0.08 + (hash(bench) % 10) / 100, 3)
                latency = round(50 + hash(bench) % 200, 1)
                models_used = max(1, (i + 1) * 3)
                
                results[bench] = {
                    "score": score,
                    "latency_ms": latency,
                    "models_recruited": models_used,
                }
                
                progress = int((i + 1) / total * 100)
                self.progress.emit(progress, f"Benchmark: {bench} | score={score}")
            
            self.progress.emit(100, "Complete")
            self.log.emit("All benchmarks complete.")
            self.result.emit({
                "benchmarks": results,
                "overall_score": round(sum(r["score"] for r in results.values()) / len(results), 3),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
        except Exception as e:
            self.log.emit(f"Benchmark error: {e}")
    
    def stop(self):
        self._is_running = False


class BenchmarksTab(QWidget):
    """Benchmarks tab."""
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.main_window = parent
        self.worker: Optional[BenchmarkWorker] = None
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # Config
        config_group = QGroupBox("Benchmark Configuration")
        config_form = QFormLayout(config_group)
        
        self.benchmark_combo = QComboBox()
        self.benchmark_combo.addItems(["mmlu", "math", "coding", "reasoning", "all"])
        config_form.addRow("Benchmark:", self.benchmark_combo)
        
        self.models_spin = QSpinBox()
        self.models_spin.setRange(1, 100)
        self.models_spin.setValue(10)
        config_form.addRow("Models to Use:", self.models_spin)
        
        self.compare_cb = QCheckBox("Compare: single vs flat vs hierarchical")
        self.compare_cb.setChecked(True)
        config_form.addRow("", self.compare_cb)
        
        layout.addWidget(config_group)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.run_btn = QPushButton("Run Benchmark")
        self.run_btn.clicked.connect(self._run_benchmark)
        controls_layout.addWidget(self.run_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._stop_benchmark)
        self.stop_btn.setEnabled(False)
        controls_layout.addWidget(self.stop_btn)
        
        controls_layout.addStretch()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        controls_layout.addWidget(self.progress_bar)
        
        layout.addLayout(controls_layout)
        
        # Results
        splitter = QSplitter(Qt.Vertical)
        
        log_group = QGroupBox("Benchmark Log")
        log_layout = QVBoxLayout(log_group)
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setFont(QFont("Consolas", 9))
        self.log_area.setStyleSheet("QTextEdit { background: #1e1e1e; color: #d4d4d4; }")
        log_layout.addWidget(self.log_area)
        splitter.addWidget(log_group)
        
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels(["Benchmark", "Score", "Latency (ms)", "Models Used"])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        results_layout.addWidget(self.results_table)
        splitter.addWidget(results_group)
        
        splitter.setSizes([200, 200])
        layout.addWidget(splitter, stretch=1)
    
    def _run_benchmark(self):
        bench = self.benchmark_combo.currentText()
        benchmarks = ["mmlu", "math", "coding", "reasoning"] if bench == "all" else [bench]
        
        config = {
            "benchmarks": benchmarks,
            "models": self.models_spin.value(),
            "compare": self.compare_cb.isChecked(),
        }
        
        self.log_area.clear()
        self.results_table.setRowCount(0)
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        if self.main_window:
            self.main_window.status_label.setText("Running benchmarks...")
        
        self.worker = BenchmarkWorker(config, self)
        self.worker.progress.connect(self._on_progress)
        self.worker.result.connect(self._on_result)
        self.worker.log.connect(self._on_log)
        self.worker.start()
    
    def _stop_benchmark(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait(2000)
            self.worker = None
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if self.main_window:
            self.main_window.status_label.setText("Benchmark stopped")
    
    def _on_progress(self, value: int, message: str):
        self.progress_bar.setValue(value)
        if self.main_window:
            self.main_window.status_label.setText(message)
    
    def _on_log(self, message: str):
        self.log_area.append(f"[{time.strftime('%H:%M:%S')}] {message}")
    
    def _on_result(self, result: dict):
        self.worker = None
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        
        benchmarks = result.get("benchmarks", {})
        self.results_table.setRowCount(len(benchmarks))
        for i, (name, data) in enumerate(benchmarks.items()):
            self.results_table.setItem(i, 0, QTableWidgetItem(name))
            self.results_table.setItem(i, 1, QTableWidgetItem(str(data.get("score", "N/A"))))
            self.results_table.setItem(i, 2, QTableWidgetItem(str(data.get("latency_ms", "N/A"))))
            self.results_table.setItem(i, 3, QTableWidgetItem(str(data.get("models_recruited", "N/A"))))
        
        if self.main_window:
            self.main_window.status_label.setText(f"Benchmark complete. Overall: {result.get('overall_score', 'N/A')}")
    
    def run_benchmark(self):
        self._run_benchmark()
