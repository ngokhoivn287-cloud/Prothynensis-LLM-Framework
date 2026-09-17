"""Tests for Prothynesis Runtime."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.runtime.settings import SettingsManager, RuntimePaths, HardwareProfile
from src.runtime.hardware import detect_hardware, HardwareReport, CPUInfo
from src.runtime.model_manager import RuntimeModelManager, ModelInfo
from src.runtime.startup import FirstRunSetup, run_startup


class TestSettingsManager:
    def test_defaults(self, tmp_path):
        path = tmp_path / "settings.yaml"
        mgr = SettingsManager(path)
        assert mgr.get("app.version") == "0.2.0"
        assert mgr.get("runtime.backend") == "auto"
        assert mgr.get("models.storage_path") == str(mgr.paths.models)
    
    def test_set_and_get(self, tmp_path):
        path = tmp_path / "settings.yaml"
        mgr = SettingsManager(path)
        mgr.set("runtime.backend", "cuda")
        assert mgr.get("runtime.backend") == "cuda"
    
    def test_paths_exist(self, tmp_path):
        path = tmp_path / "settings.yaml"
        mgr = SettingsManager(path)
        for p in [mgr.paths.models, mgr.paths.downloads, mgr.paths.logs]:
            assert p.exists()


class TestHardwareDetection:
    def test_detect_hardware(self):
        report = detect_hardware()
        assert isinstance(report, HardwareReport)
        assert isinstance(report.platform, str)
        assert isinstance(report.cpu, CPUInfo)
        assert report.cpu.cores >= 1
        assert report.recommended_backend in {"cpu", "cuda"}
    
    def test_hardware_report_dict(self):
        report = detect_hardware()
        data = report.to_dict()
        assert "platform" in data
        assert "cpu" in data
        assert "gpus" in data
        assert "recommended_backend" in data


class TestModelManager:
    def test_model_manager_init(self, tmp_path):
        mgr = RuntimeModelManager(SettingsManager(tmp_path / "settings.yaml"))
        assert mgr.registry is not None
    
    def test_discover_models_empty(self, tmp_path):
        mgr = RuntimeModelManager(SettingsManager(tmp_path / "settings.yaml"))
        models = mgr.discover_models([tmp_path / "empty"])
        assert models == []


class TestFirstRunSetup:
    def test_first_run_setup(self, tmp_path):
        settings = SettingsManager(tmp_path / "settings.yaml")
        setup = FirstRunSetup(settings)
        result = setup.run()
        assert "hardware" in result
        assert "recommendations" in result
        assert result["ready"] is True
    
    def test_summary_nonempty(self, tmp_path):
        settings = SettingsManager(tmp_path / "settings.yaml")
        setup = FirstRunSetup(settings)
        setup.run()
        summary = setup.get_summary()
        assert "PROTHYNESIS RUNTIME" in summary
        assert "CPU:" in summary
