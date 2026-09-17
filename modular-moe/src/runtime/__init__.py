"""Prothynesis Runtime package."""

from __future__ import annotations

try:
    from runtime.settings import SettingsManager, HardwareProfile, RuntimePaths
    from runtime.hardware import detect_hardware, HardwareReport
except ImportError:
    try:
        from src.runtime.settings import SettingsManager, HardwareProfile, RuntimePaths
        from src.runtime.hardware import detect_hardware, HardwareReport
    except ImportError:
        SettingsManager = None  # type: ignore[misc,assignment]
        HardwareProfile = None  # type: ignore[misc,assignment]
        RuntimePaths = None  # type: ignore[misc,assignment]
        detect_hardware = None  # type: ignore[assignment]
        HardwareReport = None  # type: ignore[assignment]

__all__ = [
    "SettingsManager",
    "HardwareProfile",
    "RuntimePaths",
    "detect_hardware",
    "HardwareReport",
]
