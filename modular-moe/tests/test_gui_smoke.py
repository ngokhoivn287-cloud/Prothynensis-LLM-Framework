"""Smoke test for Prothynesis GUI.

Validates:
- All GUI modules can be imported
- All tab widgets can be instantiated
- MainWindow can be created without crashing
- Basic tab functionality is wired correctly
"""

from __future__ import annotations

import sys
import os

# Use offscreen platform if available (no display needed)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Ensure modular-moe src is importable
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "modular-moe", "src"))


def test_imports():
    """Test that all GUI modules can be imported."""
    print("[1/6] Testing imports...")
    
    from gui.main import MainWindow, AppConfig, main, LogHandler
    from gui.chat_tab import ChatTab
    from gui.models_tab import ModelsTab
    from gui.orchestration_tab import OrchestrationTab
    from gui.training_tab import TrainingTab
    from gui.datasets_tab import DatasetsTab
    from gui.downloads_tab import DownloadsTab
    from gui.benchmarks_tab import BenchmarksTab
    from gui.settings_tab import SettingsTab
    from gui.logs_tab import LogsTab
    
    assert MainWindow is not None
    assert ChatTab is not None
    assert ModelsTab is not None
    print("  All imports OK")


def test_widget_creation():
    """Test that all widgets can be instantiated."""
    print("[2/6] Testing widget creation...")
    
    from PySide6.QtWidgets import QApplication
    from gui.chat_tab import ChatTab
    from gui.models_tab import ModelsTab
    from gui.orchestration_tab import OrchestrationTab
    from gui.training_tab import TrainingTab
    from gui.datasets_tab import DatasetsTab
    from gui.downloads_tab import DownloadsTab
    from gui.benchmarks_tab import BenchmarksTab
    from gui.settings_tab import SettingsTab
    from gui.logs_tab import LogsTab
    
    app = QApplication.instance() or QApplication(sys.argv)
    
    tabs = [
        ("ChatTab", ChatTab),
        ("ModelsTab", ModelsTab),
        ("OrchestrationTab", OrchestrationTab),
        ("TrainingTab", TrainingTab),
        ("DatasetsTab", DatasetsTab),
        ("DownloadsTab", DownloadsTab),
        ("BenchmarksTab", BenchmarksTab),
        ("SettingsTab", SettingsTab),
        ("LogsTab", LogsTab),
    ]
    
    for name, cls in tabs:
        widget = cls(parent=None)
        assert widget is not None, f"{name} creation returned None"
        assert widget.__class__.__name__ == name, f"Expected {name}, got {widget.__class__.__name__}"
        print(f"  {name}: OK")
    
    print("  All widgets created successfully")


def test_mainwindow_creation():
    """Test that MainWindow can be created."""
    print("[3/6] Testing MainWindow creation...")
    
    from PySide6.QtWidgets import QApplication
    from gui.main import MainWindow
    
    app = QApplication.instance() or QApplication(sys.argv)
    
    window = MainWindow()
    assert window is not None
    assert window.windowTitle() == "Prothynesis - Pure Hierarchical Orchestral MoMMs"
    
    # Verify tab count
    assert window.tab_widget.count() == 9, f"Expected 9 tabs, got {window.tab_widget.count()}"
    
    # Verify tab names
    expected_tabs = [
        "Chat", "Models", "Orchestration", "Training",
        "Datasets", "Downloads", "Benchmarks", "Settings", "Logs"
    ]
    for i, expected in enumerate(expected_tabs):
        actual = window.tab_widget.tabText(i)
        assert expected in actual, f"Tab {i}: expected '{expected}' in '{actual}'"
    
    print(f"  MainWindow created with {window.tab_widget.count()} tabs")
    print("  All tab names verified")
    
    # Test log signal
    window.log("Test message", "INFO")
    
    # Test refresh
    window._update_system_info()
    
    print("  MainWindow: OK")


def test_chat_tab_functionality():
    """Test basic ChatTab functionality."""
    print("[4/6] Testing ChatTab functionality...")
    
    from PySide6.QtWidgets import QApplication
    from gui.chat_tab import ChatTab
    
    app = QApplication.instance() or QApplication(sys.argv)
    tab = ChatTab(parent=None)
    
    # Test clear_chat
    tab.clear_chat()
    
    # Test telemetry labels exist
    assert "models_recruited" in tab.telemetry_labels
    assert "confidence" in tab.telemetry_labels
    assert "latency_ms" in tab.telemetry_labels
    
    print("  ChatTab: OK")


def test_models_tab_functionality():
    """Test basic ModelsTab functionality."""
    print("[5/6] Testing ModelsTab functionality...")
    
    from PySide6.QtWidgets import QApplication
    from gui.models_tab import ModelsTab
    
    app = QApplication.instance() or QApplication(sys.argv)
    tab = ModelsTab(parent=None)
    
    # Test populate demo data
    tab._populate_demo_data()
    assert tab.model.rowCount() > 0, "Demo data should populate table"
    
    # Test filter
    tab.filter_input.setText("solver")
    tab._apply_filter()
    
    # Test stats update
    tab._update_stats()
    
    print(f"  ModelsTab: {tab.model.rowCount()} demo models loaded")


def test_orchestration_tab_functionality():
    """Test basic OrchestrationTab functionality."""
    print("[6/6] Testing OrchestrationTab functionality...")
    
    from PySide6.QtWidgets import QApplication
    from gui.orchestration_tab import OrchestrationTab
    
    app = QApplication.instance() or QApplication(sys.argv)
    tab = OrchestrationTab(parent=None)
    
    # Test reset
    tab.reset()
    
    # Verify UI elements exist
    assert tab.pool_size_spin is not None
    assert tab.max_rounds_spin is not None
    assert tab.orchestral_levels_combo is not None
    
    print("  OrchestrationTab: OK")


def main():
    """Run all smoke tests."""
    print("=" * 60)
    print("PROTHYNESIS GUI SMOKE TEST")
    print("=" * 60)
    
    try:
        test_imports()
        test_widget_creation()
        test_mainwindow_creation()
        test_chat_tab_functionality()
        test_models_tab_functionality()
        test_orchestration_tab_functionality()
        
        print("=" * 60)
        print("ALL SMOKE TESTS PASSED")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print(f"\nSMOKE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
