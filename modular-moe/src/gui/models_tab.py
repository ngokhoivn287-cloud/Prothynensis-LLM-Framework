"""Models tab for Prothynesis GUI."""

from __future__ import annotations

from typing import Optional, List, Dict, Any

from PySide6.QtCore import Qt, QThread, Signal, QModelIndex, QSortFilterProxyModel
from PySide6.QtGui import QStandardItemModel, QStandardItem, QColor, QIcon
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTableView, QHeaderView, QAbstractItemView, QGroupBox,
    QFormLayout, QLineEdit, QComboBox, QSpinBox, QCheckBox,
    QMessageBox, QProgressBar, QTextEdit, QSplitter, QFrame,
    QMenu, QInputDialog, QFileDialog
)


class ModelsTab(QWidget):
    """Models management tab."""
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.main_window = parent
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # Toolbar
        toolbar = QHBoxLayout()
        
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh)
        toolbar.addWidget(self.refresh_btn)
        
        self.generate_btn = QPushButton("Generate Pool")
        self.generate_btn.clicked.connect(self._generate_pool)
        toolbar.addWidget(self.generate_btn)
        
        self.load_btn = QPushButton("Load Selected")
        self.load_btn.clicked.connect(self._load_selected)
        toolbar.addWidget(self.load_btn)
        
        self.unload_btn = QPushButton("Unload Selected")
        self.unload_btn.clicked.connect(self._unload_selected)
        toolbar.addWidget(self.unload_btn)
        
        toolbar.addStretch()
        
        # Filter
        toolbar.addWidget(QLabel("Filter:"))
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Filter by ID, specialization...")
        self.filter_input.textChanged.connect(self._apply_filter)
        toolbar.addWidget(self.filter_input)
        
        layout.addLayout(toolbar)
        
        # Models table
        self.model = QStandardItemModel()
        self.model.setHorizontalHeaderLabels([
            "Model ID", "Parameters", "Specialization", "Status",
            "Version", "Loaded", "Memory"
        ])
        
        self.proxy = QSortFilterProxyModel()
        self.proxy.setSourceModel(self.model)
        self.proxy.setFilterCaseSensitivity(Qt.CaseInsensitive)
        
        self.table = QTableView()
        self.table.setModel(self.proxy)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.MultiSelection)
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._show_context_menu)
        layout.addWidget(self.table, stretch=1)
        
        # Stats bar
        self.stats_label = QLabel("Models: 0 | Loaded: 0 | GPU: 0 | CPU: 0")
        layout.addWidget(self.stats_label)
    
    def refresh(self):
        self.model.removeRows(0, self.model.rowCount())
        self._populate_demo_data()
        self._update_stats()
    
    def _populate_demo_data(self):
        demo_models = [
            ("solver_0001", "75.2M", "general", "trained", "1.0", "No", "N/A"),
            ("solver_0002", "75.1M", "mathematics", "trained", "1.0", "No", "N/A"),
            ("solver_0003", "75.3M", "programming", "training", "1.0", "No", "N/A"),
            ("solver_0004", "75.0M", "science", "trained", "1.0", "GPU", "0.8 GB"),
            ("solver_0005", "75.2M", "reasoning", "trained", "1.0", "GPU", "0.8 GB"),
            ("solver_0006", "75.1M", "verification", "untrained", "1.0", "No", "N/A"),
            ("orch_0001", "75.5M", "orchestral", "trained", "1.0", "No", "N/A"),
            ("chief_0001", "76.0M", "chief_orchestral", "trained", "1.0", "No", "N/A"),
        ]
        
        for row_data in demo_models:
            items = []
            for col, text in enumerate(row_data):
                item = QStandardItem(text)
                if col == 4 and text == "trained":
                    item.setForeground(QColor("#4ec9b0"))
                elif col == 3 and text == "training":
                    item.setForeground(QColor("#dcdcaa"))
                elif col == 3 and text == "untrained":
                    item.setForeground(QColor("#f48771"))
                items.append(item)
            self.model.appendRow(items)
    
    def _apply_filter(self):
        text = self.filter_input.text()
        self.proxy.setFilterFixedString(text)
    
    def _update_stats(self):
        total = self.model.rowCount()
        loaded_gpu = sum(1 for r in range(total) if self.model.item(r, 5).text() == "GPU")
        loaded_cpu = sum(1 for r in range(total) if self.model.item(r, 5).text() == "CPU")
        self.stats_label.setText(f"Models: {total} | Loaded: {loaded_gpu + loaded_cpu} | GPU: {loaded_gpu} | CPU: {loaded_cpu}")
    
    def _generate_pool(self):
        if self.main_window:
            self.main_window.status_label.setText("Generating model pool...")
        self.model.removeRows(0, self.model.rowCount())
        
        specializations = ["general", "mathematics", "programming", "science", "reasoning", "verification", "systems", "languages"]
        statuses = ["trained", "training", "untrained"]
        
        for i in range(1, 33):
            spec = specializations[i % len(specializations)]
            status = statuses[i % len(statuses)]
            params = f"{75.0 + (i % 5) * 0.1:.1f}M"
            loaded = "No"
            memory = "N/A"
            if status == "trained" and i < 5:
                loaded = "GPU" if i % 2 == 0 else "CPU"
                memory = f"{(0.5 + i * 0.1):.1f} GB"
            
            row = [f"solver_{i:04d}", params, spec, status, "1.0", loaded, memory]
            items = []
            for col, text in enumerate(row):
                item = QStandardItem(text)
                if col == 3:
                    if text == "trained":
                        item.setForeground(QColor("#4ec9b0"))
                    elif text == "training":
                        item.setForeground(QColor("#dcdcaa"))
                    else:
                        item.setForeground(QColor("#f48771"))
                items.append(item)
            self.model.appendRow(items)
        
        self._update_stats()
        if self.main_window:
            self.main_window.status_label.setText("Generated 32 demo models")
    
    def _load_selected(self):
        indexes = self.table.selectionModel().selectedRows()
        if not indexes:
            QMessageBox.information(self, "Load Models", "No models selected.")
            return
        
        for idx in indexes:
            row = self.proxy.mapToSource(idx).row()
            model_id = self.model.item(row, 0).text()
            self.model.item(row, 5).setText("GPU")
            self.model.item(row, 6).setText("0.8 GB")
        self._update_stats()
        QMessageBox.information(self, "Load Models", f"Loading {len(indexes)} model(s)...")
    
    def _unload_selected(self):
        indexes = self.table.selectionModel().selectedRows()
        if not indexes:
            return
        for idx in indexes:
            row = self.proxy.mapToSource(idx).row()
            self.model.item(row, 5).setText("No")
            self.model.item(row, 6).setText("N/A")
        self._update_stats()
    
    def _show_context_menu(self, pos):
        menu = QMenu()
        inspect_action = menu.addAction("Inspect")
        load_action = menu.addAction("Load to GPU")
        unload_action = menu.addAction("Unload")
        menu.addSeparator()
        delete_action = menu.addAction("Delete")
        
        action = menu.exec_(self.table.viewport().mapToGlobal(pos))
        if action == inspect_action:
            self._inspect_selected()
        elif action == load_action:
            self._load_selected()
        elif action == unload_action:
            self._unload_selected()
    
    def _inspect_selected(self):
        indexes = self.table.selectionModel().selectedRows()
        if not indexes:
            return
        row = self.proxy.mapToSource(indexes[0]).row()
        model_id = self.model.item(row, 0).text()
        params = self.model.item(row, 1).text()
        spec = self.model.item(row, 2).text()
        status = self.model.item(row, 3).text()
        
        info = (
            f"Model ID: {model_id}\n"
            f"Parameters: {params}\n"
            f"Specialization: {spec}\n"
            f"Status: {status}\n"
            f"Architecture: Transformer (12 layers, 768 hidden, 12 heads)\n"
            f"Tokenizer: GPT-NeoX\n"
            f"Precision: FP16\n"
            f"Estimated Memory: 300 MB (FP16)"
        )
        QMessageBox.information(self, f"Inspect {model_id}", info)
    
    def set_registry_path(self, path: str):
        pass
