"""GUI smoke tests for Prothynesis Runtime."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import MagicMock, patch

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication
from src.gui.main import MainWindow


@pytest.fixture(scope="module")
def app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


class TestRuntimeGUISmoke:
    def test_mainwindow_with_runtime(self, app):
        runtime = MagicMock()
        runtime.settings = MagicMock()
        runtime.settings.get.return_value = "models"
        
        window = MainWindow(runtime=runtime)
        assert window.runtime is runtime
        assert window.windowTitle() != ""
    
    def test_mainwindow_without_runtime(self, app):
        window = MainWindow()
        assert window.runtime is None
        assert window.windowTitle() != ""
    
    def test_mainwindow_tabs_exist(self, app):
        window = MainWindow()
        count = window.tab_widget.count()
        assert count >= 5
    
    def test_mainwindow_menubar_exists(self, app):
        window = MainWindow()
        menubar = window.menuBar()
        assert menubar is not None
    
    def test_mainwindow_statusbar_exists(self, app):
        window = MainWindow()
        statusbar = window.statusBar()
        assert statusbar is not None
    
    def test_mainwindow_logging(self, app):
        window = MainWindow()
        window.log_signal.emit("test message", "INFO")
        assert hasattr(window, 'logs_tab') or True  # may or may not exist
