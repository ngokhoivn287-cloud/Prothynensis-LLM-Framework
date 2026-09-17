"""Settings tab for Prothynesis GUI."""

from __future__ import annotations

import os
from typing import Optional

from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QFormLayout, QLineEdit, QSpinBox, QDoubleSpinBox,
    QComboBox, QCheckBox, QTextEdit, QMessageBox, QFileDialog,
    QTabWidget, QFrame
)


class SettingsTab(QWidget):
    """Settings and configuration tab."""
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.main_window = parent
        self.settings = QSettings("Prothynesis", "GUI")
        self._setup_ui()
        self._load_settings()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        tabs = QTabWidget()
        
        # General settings
        general_tab = QWidget()
        general_layout = QFormLayout(general_tab)
        
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["dark", "light", "system"])
        general_layout.addRow("Theme:", self.theme_combo)
        
        self.font_combo = QComboBox()
        self.font_combo.addItems(["Consolas", "Courier New", "Fira Code", "Monaco"])
        general_layout.addRow("Font Family:", self.font_combo)
        
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(8, 20)
        self.font_size_spin.setValue(10)
        general_layout.addRow("Font Size:", self.font_size_spin)
        
        self.registry_input = QLineEdit()
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_registry)
        reg_layout = QHBoxLayout()
        reg_layout.addWidget(self.registry_input, stretch=1)
        reg_layout.addWidget(browse_btn)
        general_layout.addRow("Registry Path:", reg_layout)
        
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        general_layout.addRow("Log Level:", self.log_level_combo)
        
        self.auto_save_cb = QCheckBox("Auto-save session")
        self.auto_save_cb.setChecked(True)
        general_layout.addRow("", self.auto_save_cb)
        
        tabs.addTab(general_tab, "General")
        
        # Model settings
        model_tab = QWidget()
        model_layout = QFormLayout(model_tab)
        
        self.max_gpu_spin = QSpinBox()
        self.max_gpu_spin.setRange(1, 16)
        self.max_gpu_spin.setValue(2)
        model_layout.addRow("Max GPU Models:", self.max_gpu_spin)
        
        self.max_cpu_spin = QSpinBox()
        self.max_cpu_spin.setRange(1, 64)
        self.max_cpu_spin.setValue(8)
        model_layout.addRow("Max CPU Models:", self.max_cpu_spin)
        
        self.lazy_loading_cb = QCheckBox("Enable Lazy Loading")
        self.lazy_loading_cb.setChecked(True)
        model_layout.addRow("", self.lazy_loading_cb)
        
        self.cpu_offload_cb = QCheckBox("Enable CPU Offload")
        self.cpu_offload_cb.setChecked(True)
        model_layout.addRow("", self.cpu_offload_cb)
        
        tabs.addTab(model_tab, "Models")
        
        # Training settings
        training_tab = QWidget()
        training_layout = QFormLayout(training_tab)
        
        self.default_batch_spin = QSpinBox()
        self.default_batch_spin.setRange(1, 512)
        self.default_batch_spin.setValue(8)
        training_layout.addRow("Default Batch Size:", self.default_batch_spin)
        
        self.default_lr_spin = QDoubleSpinBox()
        self.default_lr_spin.setRange(1e-6, 1e-2)
        self.default_lr_spin.setSingleStep(1e-5)
        self.default_lr_spin.setValue(1e-4)
        self.default_lr_spin.setDecimals(6)
        training_layout.addRow("Default Learning Rate:", self.default_lr_spin)
        
        self.default_context_spin = QSpinBox()
        self.default_context_spin.setRange(128, 32768)
        self.default_context_spin.setSingleStep(128)
        self.default_context_spin.setValue(2048)
        training_layout.addRow("Default Context Length:", self.default_context_spin)
        
        self.mixed_precision_cb = QCheckBox("Default Mixed Precision")
        self.mixed_precision_cb.setChecked(True)
        training_layout.addRow("", self.mixed_precision_cb)
        
        tabs.addTab(training_tab, "Training")
        
        # About
        about_tab = QWidget()
        about_layout = QVBoxLayout(about_tab)
        
        about_text = QTextEdit()
        about_text.setReadOnly(True)
        about_text.setHtml(
            "<h2>Prothynesis</h2>"
            "<p>Pure Hierarchical Orchestral Mix of Many Models (MoMMs)</p>"
            "<p>Version: 0.1.0</p>"
            "<p>Architecture: 75M parameter independent solver models with hierarchical learned orchestration.</p>"
            "<p>Built with PySide6 and PyTorch.</p>"
            "<br>"
            "<p>Components:</p>"
            "<ul>"
            "<li>Solver Models: Independently trainable 75M parameter models</li>"
            "<li>Orchestral Models: ~75M learned reasoning coordinators</li>"
            "<li>Chief Orchestral: Manages multiple Orchestral units</li>"
            "<li>Master Orchestral: Coordinates Chiefs</li>"
            "<li>Ultimate Orchestral: Global reasoning coordinator</li>"
            "</ul>"
        )
        about_layout.addWidget(about_text)
        tabs.addTab(about_tab, "About")
        
        layout.addWidget(tabs, stretch=1)
        
        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        self.save_btn = QPushButton("Save Settings")
        self.save_btn.clicked.connect(self.save_settings)
        btn_layout.addWidget(self.save_btn)
        
        self.reset_btn = QPushButton("Reset to Defaults")
        self.reset_btn.clicked.connect(self._reset_settings)
        btn_layout.addWidget(self.reset_btn)
        
        layout.addLayout(btn_layout)
    
    def _load_settings(self):
        self.theme_combo.setCurrentText(self.settings.value("theme", "dark"))
        self.font_combo.setCurrentText(self.settings.value("font_family", "Consolas"))
        self.font_size_spin.setValue(self.settings.value("font_size", 10, type=int))
        self.registry_input.setText(self.settings.value("registry_path", "models"))
        self.log_level_combo.setCurrentText(self.settings.value("log_level", "INFO"))
        self.auto_save_cb.setChecked(self.settings.value("auto_save", True, type=bool))
    
    def _browse_registry(self):
        path = QFileDialog.getExistingDirectory(self, "Select Registry Directory")
        if path:
            self.registry_input.setText(path)
    
    def save_settings(self):
        self.settings.setValue("theme", self.theme_combo.currentText())
        self.settings.setValue("font_family", self.font_combo.currentText())
        self.settings.setValue("font_size", self.font_size_spin.value())
        self.settings.setValue("registry_path", self.registry_input.text())
        self.settings.setValue("log_level", self.log_level_combo.currentText())
        self.settings.setValue("auto_save", self.auto_save_cb.isChecked())
        
        if self.main_window:
            self.main_window.config.theme = self.theme_combo.currentText()
            self.main_window.config.font_family = self.font_combo.currentText()
            self.main_window.config.font_size = self.font_size_spin.value()
            self.main_window.config.registry_path = self.registry_input.text()
            self.main_window.config.log_level = self.log_level_combo.currentText()
            self.main_window.config.auto_save = self.auto_save_cb.isChecked()
            self.main_window._save_config()
            self.main_window._apply_theme()
        
        QMessageBox.information(self, "Settings", "Settings saved successfully.")
    
    def _reset_settings(self):
        reply = QMessageBox.question(self, "Reset Settings",
                                     "Reset all settings to defaults?",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.settings.clear()
            self._load_settings()
            QMessageBox.information(self, "Settings", "Settings reset to defaults.")
    
    def validate_config(self):
        issues = []
        if not self.registry_input.text().strip():
            issues.append("Registry path is empty.")
        if self.font_size_spin.value() < 8:
            issues.append("Font size too small.")
        
        if issues:
            QMessageBox.warning(self, "Configuration Issues", "\n".join(issues))
        else:
            QMessageBox.information(self, "Configuration", "Configuration is valid.")
