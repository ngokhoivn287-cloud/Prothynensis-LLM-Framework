"""GUI entry point for Prothynesis Runtime."""

from __future__ import annotations

import sys
import signal
from pathlib import Path

from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtCore import QTimer

from src.runtime.controller import ProthynesisRuntime
from src.gui.main import MainWindow


class RuntimeGUI:
    """GUI wrapper that integrates runtime controller with MainWindow."""
    
    def __init__(self, settings_path: Path | None = None):
        self.app = QApplication(sys.argv)
        self.app.setApplicationName("Prothynesis Runtime")
        self.app.setApplicationVersion("0.2.0")
        self.app.setOrganizationName("Prothynesis")
        
        self.runtime = ProthynesisRuntime(settings_path)
        self.window = MainWindow(runtime=self.runtime)
        
        # Handle SIGINT gracefully
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        quit_timer = QTimer()
        quit_timer.timeout.connect(lambda: None)
        quit_timer.start(100)
    
    def run(self) -> int:
        """Run the GUI application."""
        self.window.show()
        
        # Run first-run setup if needed
        if self.runtime.settings.get("app.first_run", True):
            self._show_first_run_dialog()
        
        return self.app.exec()
    
    def _show_first_run_dialog(self):
        """Show first-run hardware detection dialog."""
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton, QLabel
        
        dialog = QDialog(self.window)
        dialog.setWindowTitle("Prothynesis Runtime - First Run")
        dialog.resize(600, 400)
        
        layout = QVBoxLayout(dialog)
        
        label = QLabel("Hardware Detection Complete")
        label.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(label)
        
        text = QTextEdit()
        text.setReadOnly(True)
        from src.runtime.startup import FirstRunSetup
        setup = FirstRunSetup(self.runtime.settings)
        summary = setup.get_summary()
        text.setText(summary)
        layout.addWidget(text)
        
        btn = QPushButton("Continue")
        btn.clicked.connect(dialog.accept)
        layout.addWidget(btn)
        
        dialog.exec()


def main(settings_path: Path | None = None) -> int:
    """Main entry point for GUI."""
    gui = RuntimeGUI(settings_path)
    return gui.run()


if __name__ == "__main__":
    sys.exit(main())
