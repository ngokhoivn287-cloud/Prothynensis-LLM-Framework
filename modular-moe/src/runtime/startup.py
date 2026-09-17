"""First-run setup and startup for Prothynesis Runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional

from src.runtime.hardware import detect_hardware
from src.runtime.settings import SettingsManager, HardwareProfile


class FirstRunSetup:
    """First-run hardware detection and configuration."""
    
    def __init__(self, settings: SettingsManager):
        self.settings = settings
        self.report: Optional[Dict[str, Any]] = None
    
    def run(self) -> Dict[str, Any]:
        """Run first-run detection and return results."""
        hardware = detect_hardware()
        recommendations = self.settings.recommend_configuration()
        
        self.report = {
            "hardware": hardware.to_dict(),
            "recommendations": recommendations,
            "ready": True,
        }
        
        self.settings.set("app.first_run", False)
        self.settings.set("hardware", hardware.to_dict())
        self.settings.apply_recommendations()
        
        return self.report
    
    def get_summary(self) -> str:
        """Get human-readable summary of first-run results."""
        if self.report is None:
            self.run()
        
        hw = self.report["hardware"]
        recs = self.report["recommendations"]
        
        lines = [
            "=" * 60,
            "PROTHYNESIS RUNTIME - FIRST RUN",
            "=" * 60,
            f"Platform: {hw['platform']}",
            f"Python: {hw['python_version']}",
            f"PyTorch: {hw['pytorch_version']}",
            "",
            "CPU:",
            f"  Model: {hw['cpu']['model']}",
            f"  Cores: {hw['cpu']['cores']} physical / {hw['cpu']['threads']} logical",
            f"  RAM: {hw['cpu']['memory_gb']:.1f} GB total, {hw['cpu']['memory_available_gb']:.1f} GB available",
            "",
        ]
        
        if hw["cuda_available"]:
            lines.append("GPU:")
            for gpu in hw["gpus"]:
                lines.append(f"  [{gpu['index']}] {gpu['name']}")
                lines.append(f"      VRAM: {gpu['vram_total_gb']:.1f} GB total, {gpu['vram_free_gb']:.1f} GB free")
                lines.append(f"      Compute Capability: {gpu['compute_capability']}")
            lines.append(f"  CUDA: {hw['cuda_version']}")
            lines.append(f"  cuDNN: {hw['cudnn_version']}")
        else:
            lines.append("GPU: None detected (CPU-only mode)")
        
        lines.extend([
            "",
            "Recommendations:",
            f"  Backend: {recs['backend']}",
            f"  Max GPU models: {recs['max_gpu_models']}",
            f"  Max CPU models: {recs['max_cpu_models']}",
            f"  Default context: {recs['default_context_length']}",
            f"  Quantization: {recs['quantization']}",
        ])
        
        if recs.get("notes"):
            lines.append("")
            lines.append("Notes:")
            for note in recs["notes"]:
                lines.append(f"  - {note}")
        
        if hw["warnings"]:
            lines.append("")
            lines.append("Warnings:")
            for warning in hw["warnings"]:
                lines.append(f"  - {warning}")
        
        lines.append("=" * 60)
        return "\n".join(lines)


def run_startup(settings_path: Optional[Path] = None) -> Dict[str, Any]:
    """Run startup sequence."""
    settings = SettingsManager(settings_path)
    
    if settings.get("app.first_run", True):
        setup = FirstRunSetup(settings)
        result = setup.run()
    else:
        result = {
            "hardware": detect_hardware().to_dict(),
            "recommendations": settings.recommend_configuration(),
            "ready": True,
        }
    
    return result
