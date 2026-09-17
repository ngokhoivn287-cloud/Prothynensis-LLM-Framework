"""Main GUI application for Prothynesis."""

from __future__ import annotations

import sys
import json
import zipfile
import logging
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from datetime import datetime

from PySide6.QtCore import (
    Qt, QThread, Signal, QTimer, QSize, QSettings,
    QAbstractTableModel, QModelIndex, QSortFilterProxyModel
)
from PySide6.QtGui import (
    QAction, QIcon, QFont, QColor, QPalette,
    QTextCursor, QTextCharFormat, QSyntaxHighlighter
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QSplitter, QTextEdit, QLineEdit, QPushButton,
    QLabel, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QTableView, QHeaderView, QAbstractItemView, QFileDialog,
    QMessageBox, QProgressBar, QGroupBox, QFormLayout,
    QScrollArea, QTreeWidget, QTreeWidgetItem, QListWidget,
    QListWidgetItem, QDialog, QDialogButtonBox, QTextBrowser,
    QPlainTextEdit, QToolBar, QStatusBar, QDockWidget,
    QFrame, QSizePolicy, QMenu, QInputDialog
)

from .chat_tab import ChatTab
from .models_tab import ModelsTab
from .orchestration_tab import OrchestrationTab
from .training_tab import TrainingTab
from .datasets_tab import DatasetsTab
from .downloads_tab import DownloadsTab
from .benchmarks_tab import BenchmarksTab
from .settings_tab import SettingsTab
from .logs_tab import LogsTab
from pmo.package import PMOPackage, PMOMetadata


@dataclass
class AppConfig:
    """Application configuration."""
    window_width: int = 1400
    window_height: int = 900
    theme: str = "dark"
    font_family: str = "Consolas"
    font_size: int = 10
    registry_path: str = "models"
    auto_save: bool = True
    log_level: str = "INFO"


class LogHandler(logging.Handler):
    """Logging handler that emits signals for GUI."""
    
    def __init__(self, signal):
        super().__init__()
        self.signal = signal
    
    def emit(self, record):
        msg = self.format(record)
        self.signal.emit(msg, record.levelname)


class MainWindow(QMainWindow):
    """Main application window."""
    
    log_signal = Signal(str, str)  # message, level
    
    def __init__(self, runtime: Any = None):
        super().__init__()
        self.runtime = runtime
        
        self.config = self._load_config()
        self._setup_logging()
        self._setup_ui()
        self._setup_menu()
        self._setup_toolbar()
        self._setup_statusbar()
        self._apply_theme()
        
        # Connect log signal
        self.log_signal.connect(self._append_log)
        
        # Load tabs
        self._init_tabs()
        
        # Restore window state
        self._restore_state()
        
        # Update system info
        self._update_system_info()
    
    def _load_config(self) -> AppConfig:
        """Load application configuration."""
        settings = QSettings("Prothynesis", "GUI")
        return AppConfig(
            window_width=settings.value("window_width", 1400, type=int),
            window_height=settings.value("window_height", 900, type=int),
            theme=settings.value("theme", "dark"),
            font_family=settings.value("font_family", "Consolas"),
            font_size=settings.value("font_size", 10, type=int),
            registry_path=settings.value("registry_path", "models"),
            auto_save=settings.value("auto_save", True, type=bool),
            log_level=settings.value("log_level", "INFO"),
        )
    
    def _save_config(self):
        """Save application configuration."""
        settings = QSettings("Prothynesis", "GUI")
        settings.setValue("window_width", self.width())
        settings.setValue("window_height", self.height())
        settings.setValue("theme", self.config.theme)
        settings.setValue("font_family", self.config.font_family)
        settings.setValue("font_size", self.config.font_size)
        settings.setValue("registry_path", self.config.registry_path)
        settings.setValue("auto_save", self.config.auto_save)
        settings.setValue("log_level", self.config.log_level)
    
    def _setup_logging(self):
        """Setup logging."""
        self.logger = logging.getLogger("prothynesis.gui")
        self.logger.setLevel(getattr(logging, self.config.log_level))
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(console_handler)
        
        # GUI handler
        gui_handler = LogHandler(self.log_signal)
        gui_handler.setFormatter(
            logging.Formatter('[%(levelname)s] %(name)s: %(message)s')
        )
        self.logger.addHandler(gui_handler)
    
    def _setup_ui(self):
        """Setup main UI."""
        self.setWindowTitle("Prothynesis - Pure Hierarchical Orchestral MoMMs")
        self.resize(self.config.window_width, self.config.window_height)
        
        # Central widget with tabs
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.North)
        self.tab_widget.setMovable(True)
        self.tab_widget.setTabsClosable(False)
        self.setCentralWidget(self.tab_widget)
    
    def _setup_menu(self):
        """Setup menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        new_action = QAction("&New Session", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self._new_session)
        file_menu.addAction(new_action)
        
        open_action = QAction("&Open Project...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_project)
        file_menu.addAction(open_action)
        
        save_action = QAction("&Save Session", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._save_session)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        export_action = QAction("&Export PMO...", self)
        export_action.triggered.connect(self._export_pmo)
        file_menu.addAction(export_action)
        
        import_action = QAction("&Import PMO...", self)
        import_action.triggered.connect(self._import_pmo)
        file_menu.addAction(import_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = menubar.addMenu("&Edit")
        
        prefs_action = QAction("&Preferences...", self)
        prefs_action.setShortcut("Ctrl+,")
        prefs_action.triggered.connect(self._show_preferences)
        edit_menu.addAction(prefs_action)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        
        toggle_logs_action = QAction("&Show Logs Dock", self)
        toggle_logs_action.setCheckable(True)
        toggle_logs_action.setChecked(True)
        toggle_logs_action.triggered.connect(self._toggle_logs_dock)
        view_menu.addAction(toggle_logs_action)
        
        toggle_models_action = QAction("&Show Models Dock", self)
        toggle_models_action.setCheckable(True)
        toggle_models_action.setChecked(False)
        toggle_models_action.triggered.connect(self._toggle_models_dock)
        view_menu.addAction(toggle_models_action)
        
        view_menu.addSeparator()
        
        theme_menu = view_menu.addMenu("&Theme")
        for theme_name in ["dark", "light", "system"]:
            theme_action = QAction(theme_name.capitalize(), self)
            theme_action.setCheckable(True)
            theme_action.setChecked(theme_name == self.config.theme)
            theme_action.triggered.connect(lambda checked, t=theme_name: self._set_theme(t))
            theme_menu.addAction(theme_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("&Tools")
        
        benchmark_action = QAction("Run &Benchmark...", self)
        benchmark_action.triggered.connect(self._run_benchmark)
        tools_menu.addAction(benchmark_action)
        
        simulate_action = QAction("Run &Simulation...", self)
        simulate_action.triggered.connect(self._run_simulation)
        tools_menu.addAction(simulate_action)
        
        validate_action = QAction("&Validate Configuration", self)
        validate_action.triggered.connect(self._validate_config)
        tools_menu.addAction(validate_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        docs_action = QAction("&Documentation", self)
        docs_action.triggered.connect(self._show_docs)
        help_menu.addAction(docs_action)
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _setup_toolbar(self):
        """Setup toolbar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(24, 24))
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        
        # Quick actions
        self.chat_action = QAction("💬 Chat", self)
        self.chat_action.triggered.connect(lambda: self.tab_widget.setCurrentIndex(0))
        toolbar.addAction(self.chat_action)
        
        self.models_action = QAction("🤖 Models", self)
        self.models_action.triggered.connect(lambda: self.tab_widget.setCurrentIndex(1))
        toolbar.addAction(self.models_action)
        
        self.orch_action = QAction("🎭 Orchestration", self)
        self.orch_action.triggered.connect(lambda: self.tab_widget.setCurrentIndex(2))
        toolbar.addAction(self.orch_action)
        
        toolbar.addSeparator()
        
        self.train_action = QAction("🏋️ Train", self)
        self.train_action.triggered.connect(lambda: self.tab_widget.setCurrentIndex(3))
        toolbar.addAction(self.train_action)
        
        self.data_action = QAction("📊 Datasets", self)
        self.data_action.triggered.connect(lambda: self.tab_widget.setCurrentIndex(4))
        toolbar.addAction(self.data_action)
        
        toolbar.addSeparator()
        
        self.refresh_action = QAction("🔄 Refresh", self)
        self.refresh_action.triggered.connect(self._refresh_all)
        toolbar.addAction(self.refresh_action)
        
        toolbar.addSeparator()
        
        # System info
        self.sys_info_label = QLabel("System: Initializing...")
        self.sys_info_label.setMinimumWidth(300)
        toolbar.addWidget(self.sys_info_label)
        
        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_system_info)
        self.update_timer.start(5000)
    
    def _setup_statusbar(self):
        """Setup status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_label)
        
        self.memory_label = QLabel("Memory: --")
        self.status_bar.addPermanentWidget(self.memory_label)
        
        self.model_count_label = QLabel("Models: 0")
        self.status_bar.addPermanentWidget(self.model_count_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
    
    def _init_tabs(self):
        """Initialize all tabs."""
        # Chat tab
        self.chat_tab = ChatTab(self)
        self.tab_widget.addTab(self.chat_tab, "💬 Chat")
        
        # Models tab
        self.models_tab = ModelsTab(self)
        self.tab_widget.addTab(self.models_tab, "🤖 Models")
        
        # Orchestration tab
        self.orchestration_tab = OrchestrationTab(self)
        self.tab_widget.addTab(self.orchestration_tab, "🎭 Orchestration")
        
        # Training tab
        self.training_tab = TrainingTab(self)
        self.tab_widget.addTab(self.training_tab, "🏋️ Training")
        
        # Datasets tab
        self.datasets_tab = DatasetsTab(self)
        self.tab_widget.addTab(self.datasets_tab, "📊 Datasets")
        
        # Downloads tab
        self.downloads_tab = DownloadsTab(self)
        self.tab_widget.addTab(self.downloads_tab, "⬇️ Downloads")
        
        # Benchmarks tab
        self.benchmarks_tab = BenchmarksTab(self)
        self.tab_widget.addTab(self.benchmarks_tab, "📈 Benchmarks")
        
        # Settings tab
        self.settings_tab = SettingsTab(self)
        self.tab_widget.addTab(self.settings_tab, "⚙️ Settings")
        
        # Logs tab
        self.logs_tab = LogsTab(self)
        self.tab_widget.addTab(self.logs_tab, "📝 Logs")
        
        # Connect tab change signal
        self.tab_widget.currentChanged.connect(self._on_tab_changed)
    
    def _apply_theme(self):
        """Apply theme."""
        if self.config.theme == "dark":
            self._apply_dark_theme()
        elif self.config.theme == "light":
            self._apply_light_theme()
        else:
            self._apply_system_theme()
    
    def _apply_dark_theme(self):
        """Apply dark theme."""
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(45, 45, 48))
        palette.setColor(QPalette.WindowText, QColor(212, 212, 212))
        palette.setColor(QPalette.Base, QColor(30, 30, 30))
        palette.setColor(QPalette.AlternateBase, QColor(45, 45, 48))
        palette.setColor(QPalette.ToolTipBase, QColor(212, 212, 212))
        palette.setColor(QPalette.ToolTipText, QColor(212, 212, 212))
        palette.setColor(QPalette.Text, QColor(212, 212, 212))
        palette.setColor(QPalette.Button, QColor(45, 45, 48))
        palette.setColor(QPalette.ButtonText, QColor(212, 212, 212))
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, QColor(0, 120, 215))
        palette.setColor(QPalette.Highlight, QColor(0, 120, 215))
        palette.setColor(QPalette.HighlightedText, Qt.black)
        self.setPalette(palette)
        
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #2d2d30; color: #d4d4d4; }
            QTabWidget::pane { border: 1px solid #3e3e42; background: #2d2d30; }
            QTabBar::tab { background: #2d2d30; color: #d4d4d4; padding: 8px 16px; border: 1px solid #3e3e42; }
            QTabBar::tab:selected { background: #1e1e1e; border-bottom: 2px solid #0078d7; }
            QTabBar::tab:hover { background: #3e3e42; }
            QPushButton { background: #3e3e42; color: #d4d4d4; border: 1px solid #555; padding: 6px 12px; border-radius: 3px; }
            QPushButton:hover { background: #4e4e52; }
            QPushButton:pressed { background: #2e2e32; }
            QPushButton:disabled { background: #2d2d30; color: #666; }
            QLineEdit, QTextEdit, QPlainTextEdit { background: #1e1e1e; color: #d4d4d4; border: 1px solid #555; border-radius: 3px; padding: 4px; }
            QComboBox { background: #1e1e1e; color: #d4d4d4; border: 1px solid #555; padding: 4px; }
            QComboBox::drop-down { border: none; }
            QSpinBox, QDoubleSpinBox { background: #1e1e1e; color: #d4d4d4; border: 1px solid #555; }
            QTableView { background: #1e1e1e; color: #d4d4d4; gridline-color: #3e3e42; }
            QHeaderView::section { background: #3e3e42; color: #d4d4d4; padding: 4px; border: 1px solid #555; }
            QGroupBox { border: 1px solid #555; border-radius: 3px; margin-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QScrollBar:vertical { background: #2d2d30; width: 12px; }
            QScrollBar::handle:vertical { background: #555; border-radius: 6px; min-height: 20px; }
            QScrollBar::handle:vertical:hover { background: #666; }
            QProgressBar { border: 1px solid #555; border-radius: 3px; text-align: center; background: #1e1e1e; }
            QProgressBar::chunk { background: #0078d7; border-radius: 2px; }
            QStatusBar { background: #1e1e1e; color: #d4d4d4; }
            QToolBar { background: #2d2d30; border: none; spacing: 4px; }
            QMenu { background: #2d2d30; color: #d4d4d4; border: 1px solid #555; }
            QMenu::item:selected { background: #0078d7; }
        """)
    
    def _apply_light_theme(self):
        """Apply light theme."""
        self.setPalette(QApplication.style().standardPalette())
        self.setStyleSheet("")
    
    def _apply_system_theme(self):
        """Apply system theme."""
        self.setPalette(QApplication.style().standardPalette())
        self.setStyleSheet("")
    
    def _set_theme(self, theme: str):
        """Set theme."""
        self.config.theme = theme
        self._apply_theme()
        self._save_config()
    
    def _on_tab_changed(self, index: int):
        """Handle tab change."""
        tab_names = [
            "Chat", "Models", "Orchestration", "Training",
            "Datasets", "Downloads", "Benchmarks", "Settings", "Logs"
        ]
        if 0 <= index < len(tab_names):
            self.status_label.setText(f"Current: {tab_names[index]}")
    
    def _append_log(self, message: str, level: str):
        """Append log message to logs tab."""
        if hasattr(self, 'logs_tab'):
            self.logs_tab.append_log(message, level)
    
    def _update_system_info(self):
        """Update system info in toolbar."""
        try:
            import torch
            import psutil
            
            info_parts = []
            
            # CPU
            cpu_percent = psutil.cpu_percent(interval=0.1)
            info_parts.append(f"CPU: {cpu_percent:.0f}%")
            
            # RAM
            ram = psutil.virtual_memory()
            info_parts.append(f"RAM: {ram.percent:.0f}% ({ram.used/1024**3:.1f}/{ram.total/1024**3:.1f} GB)")
            
            # GPU
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.memory_allocated() / 1024**3
                gpu_reserved = torch.cuda.memory_reserved() / 1024**3
                info_parts.append(f"GPU: {gpu_mem:.1f}/{gpu_reserved:.1f} GB")
            else:
                info_parts.append("GPU: N/A")
            
            self.sys_info_label.setText(" | ".join(info_parts))
            
            # Update memory label in status bar
            self.memory_label.setText(f"RAM: {ram.percent:.0f}% | GPU: {gpu_mem:.1f}GB" if torch.cuda.is_available() else f"RAM: {ram.percent:.0f}%")
            
        except Exception as e:
            self.logger.error(f"Failed to update system info: {e}")
    
    def _refresh_all(self):
        """Refresh all tabs."""
        self.logger.info("Refreshing all tabs...")
        self.models_tab.refresh()
        self.orchestration_tab.refresh()
        self.datasets_tab.refresh()
        self.downloads_tab.refresh()
        self.benchmarks_tab.refresh()
        self._update_system_info()
        self.status_label.setText("Refreshed")
    
    def _new_session(self):
        """Create new session."""
        reply = QMessageBox.question(
            self, "New Session",
            "Create a new session? This will clear current state.",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.chat_tab.clear_chat()
            self.orchestration_tab.reset()
            self.status_label.setText("New session created")
    
    def _open_project(self):
        """Open project directory."""
        path = QFileDialog.getExistingDirectory(self, "Open Project")
        if path:
            self.config.registry_path = path
            self.models_tab.set_registry_path(path)
            self.status_label.setText(f"Opened project: {path}")
    
    def _save_session(self):
        """Save session."""
        self._save_config()
        self.chat_tab.save_session()
        self.settings_tab.save_settings()
        self.status_label.setText("Session saved")
    
    def _export_pmo(self):
        """Export PMO package."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Export PMO", "", "PMO Files (*.pmo)"
        )
        if path:
            self.status_label.setText(f"Exporting PMO to {path}...")
            try:
                # Collect selected models or all models
                selected = self.models_tab.table.selectionModel().selectedRows()
                if selected:
                    model_ids = [
                        self.models_tab.model.item(r.row(), 0).text()
                        for r in selected
                    ]
                else:
                    model_ids = [
                        self.models_tab.model.item(r, 0).text()
                        for r in range(self.models_tab.model.rowCount())
                    ]

                if not model_ids:
                    QMessageBox.warning(self, "Export PMO", "No models to export")
                    self.status_label.setText("PMO export failed: no models")
                    return

                # Build manifest
                manifest = {
                    "version": "1.0",
                    "models": model_ids,
                    "exported_at": datetime.now().isoformat(),
                }

                # Collect files from registry
                files = {}
                registry_path = Path(self.config.registry_path)
                for model_id in model_ids:
                    model_dir = registry_path / model_id
                    if model_dir.exists():
                        for checkpoint_file in model_dir.glob("*.pt"):
                            arcname = f"models/{model_id}/{checkpoint_file.name}"
                            files[arcname] = checkpoint_file
                        for checkpoint_file in model_dir.glob("*.safetensors"):
                            arcname = f"models/{model_id}/{checkpoint_file.name}"
                            files[arcname] = checkpoint_file

                metadata = PMOMetadata(
                    version="1.0",
                    architecture="MoMMs",
                    solver_count=len(model_ids),
                    manifest_path="manifest.json",
                    model_index_path="model_index.json",
                )

                pmo = PMOPackage(path)
                pmo.build(metadata, manifest, files)
                self.status_label.setText(f"Exported PMO to {path}")
                QMessageBox.information(self, "Export PMO", f"Exported {len(model_ids)} models to {path}")
            except Exception as e:
                self.status_label.setText(f"PMO export failed: {e}")
                QMessageBox.critical(self, "Export PMO", f"Failed to export PMO: {e}")
    
    def _import_pmo(self):
        """Import PMO package."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Import PMO", "", "PMO Files (*.pmo)"
        )
        if path:
            self.status_label.setText(f"Importing PMO from {path}...")
            try:
                pmo = PMOPackage(path)
                if not pmo.verify():
                    raise ValueError("Invalid or corrupted PMO package")

                info = pmo.inspect()
                metadata = info.get("metadata", {})
                model_ids = []
                with zipfile.ZipFile(path, "r") as zf:
                    manifest_bytes = zf.read("manifest.json")
                    manifest = json.loads(manifest_bytes)
                    model_ids = manifest.get("models", [])

                if not model_ids:
                    QMessageBox.warning(self, "Import PMO", "No models found in package")
                    self.status_label.setText("PMO import failed: no models")
                    return

                # Extract to registry
                output_dir = Path(self.config.registry_path)
                output_dir.mkdir(parents=True, exist_ok=True)
                pmo.extract(str(output_dir))

                self.status_label.setText(f"Imported {len(model_ids)} models from {path}")
                QMessageBox.information(self, "Import PMO", f"Imported {len(model_ids)} models from {path}")
                self.models_tab.refresh()
            except Exception as e:
                self.status_label.setText(f"PMO import failed: {e}")
                QMessageBox.critical(self, "Import PMO", f"Failed to import PMO: {e}")
    
    def _show_preferences(self):
        """Show preferences dialog."""
        self.tab_widget.setCurrentWidget(self.settings_tab)
    
    def _toggle_logs_dock(self, checked: bool):
        """Toggle logs dock."""
        # For now, just switch to logs tab
        if checked:
            self.tab_widget.setCurrentWidget(self.logs_tab)
    
    def _toggle_models_dock(self, checked: bool):
        """Toggle models dock."""
        if checked:
            self.tab_widget.setCurrentWidget(self.models_tab)
    
    def _run_benchmark(self):
        """Run benchmark."""
        self.tab_widget.setCurrentWidget(self.benchmarks_tab)
        self.benchmarks_tab.run_benchmark()
    
    def _run_simulation(self):
        """Run simulation."""
        self.tab_widget.setCurrentWidget(self.orchestration_tab)
        self.orchestration_tab.run_simulation()
    
    def _validate_config(self):
        """Validate configuration."""
        self.tab_widget.setCurrentWidget(self.settings_tab)
        self.settings_tab.validate_config()
    
    def _show_docs(self):
        """Show documentation."""
        QMessageBox.information(
            self, "Documentation",
            "Prothynesis Documentation\n\n"
            "Repository: https://github.com/Prothynesis/Prothynesis\n"
            "See README.md for getting started guide."
        )
    
    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self, "About Prothynesis",
            "Prothynesis - Pure Hierarchical Orchestral MoMMs\n\n"
            "Version: 0.1.0\n"
            "Architecture: 75M parameter independent solver models\n"
            "with hierarchical learned orchestration.\n\n"
            "Built with PySide6 and PyTorch."
        )
    
    def _restore_state(self):
        """Restore window state."""
        settings = QSettings("Prothynesis", "GUI")
        geometry = settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        state = settings.value("windowState")
        if state:
            self.restoreState(state)
    
    def closeEvent(self, event):
        """Handle close event."""
        self._save_config()
        settings = QSettings("Prothynesis", "GUI")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())
        event.accept()
    
    def log(self, message: str, level: str = "INFO"):
        """Log a message."""
        self.log_signal.emit(message, level)


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    app.setApplicationName("Prothynesis")
    app.setApplicationVersion("0.1.0")
    app.setOrganizationName("Prothynesis")
    
    # Set application font
    font = QFont("Consolas", 10)
    app.setFont(font)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()