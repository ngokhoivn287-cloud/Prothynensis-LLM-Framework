"""Tests for execution environment detection."""

from __future__ import annotations

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.utils.environment import EnvironmentDetector, ExecutionEnvironment


def test_environment_detector_reports_current():
    info = EnvironmentDetector.get_environment_info()
    assert "environment" in info
    assert "os" in info
    assert "python_executable" in info
    assert "pytorch_version" in info
    assert "cuda_available" in info
    assert "device_selected" in info
    assert info["device_selected"] in ("cuda", "cpu")


def test_environment_detector_reports_gpu_name():
    info = EnvironmentDetector.get_environment_info()
    if info["cuda_available"]:
        assert info["gpu_name"] is not None
        assert "NVIDIA" in info["gpu_name"] or "AMD" in info["gpu_name"] or "Intel" in info["gpu_name"]


def test_environment_detector_get_device():
    device = EnvironmentDetector.get_device()
    assert str(device) in ("cuda", "cpu")


def test_environment_detector_report_environment():
    report = EnvironmentDetector.report_environment()
    assert "Environment:" in report
    assert "PyTorch:" in report
    assert "GPU:" in report or "Device selected:" in report


def test_detect_environment_returns_valid_string():
    env = EnvironmentDetector.detect_environment()
    assert env in (ExecutionEnvironment.WSL2_CUDA, ExecutionEnvironment.WINDOWS_CUDA, ExecutionEnvironment.CPU)
