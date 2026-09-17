"""PMO package format for Prothynesis."""

from __future__ import annotations

import json
import zipfile
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime


@dataclass
class PMOMetadata:
    """Metadata for a PMO package."""
    version: str = "1.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    architecture: str = "custom"
    solver_count: int = 0
    orchestral_count: int = 0
    total_params: int = 0
    checksum: str = ""
    manifest_path: str = "manifest.json"
    model_index_path: str = "model_index.json"
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> "PMOMetadata":
        return cls(**d)


class PMOPackage:
    """
    Prothynesis-specific package format (.pmo).
    
    Contains:
    - manifest (JSON)
    - orchestral models (safetensors)
    - solver model blobs (safetensors)
    - model registry (JSON)
    - tokenizer
    - dataset profiles
    - training metadata
    - checksums
    - indexes
    """
    
    def __init__(self, path: str):
        self.path = Path(path)
        self.metadata: Optional[PMOMetadata] = None
        self._manifest: Optional[dict] = None
    
    def build(self, metadata: PMOMetadata, manifest: dict, files: Dict[str, Path]) -> None:
        """Build a PMO package."""
        self.metadata = metadata
        
        with zipfile.ZipFile(self.path, "w", zipfile.ZIP_DEFLATED) as zf:
            # Write metadata (placeholder - will update after computing checksum)
            zf.writestr("metadata.json", json.dumps(metadata.to_dict(), indent=2))
            
            # Write manifest
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))
            
            # Write files
            for arcname, filepath in files.items():
                if filepath.exists():
                    zf.write(filepath, arcname)
        
        # Update metadata with actual checksum
        actual_checksum = self._compute_checksum()
        self.metadata.checksum = actual_checksum
        
        # Rebuild with correct checksum
        with zipfile.ZipFile(self.path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("metadata.json", json.dumps(self.metadata.to_dict(), indent=2))
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))
            for arcname, filepath in files.items():
                if filepath.exists():
                    zf.write(filepath, arcname)
    
    def inspect(self) -> dict:
        """Inspect PMO package contents."""
        result = {
            "path": str(self.path),
            "exists": self.path.exists(),
            "size_mb": 0,
            "files": [],
        }
        
        if self.path.exists():
            result["size_mb"] = round(self.path.stat().st_size / 1024**2, 2)
            
            with zipfile.ZipFile(self.path, "r") as zf:
                for info in zf.infolist():
                    result["files"].append({
                        "name": info.filename,
                        "size": info.file_size,
                        "compressed": info.compress_size,
                    })
                
                # Try to read metadata
                try:
                    metadata_bytes = zf.read("metadata.json")
                    result["metadata"] = json.loads(metadata_bytes)
                except Exception:
                    pass
        
        return result
    
    def verify(self) -> bool:
        """Verify PMO package integrity."""
        if not self.path.exists():
            return False
        
        try:
            with zipfile.ZipFile(self.path, "r") as zf:
                # Check required files
                required = ["metadata.json", "manifest.json"]
                namelist = zf.namelist()
                
                for req in required:
                    if req not in namelist:
                        return False
                
                # Verify zipfile integrity
                bad_file = zf.testzip()
                if bad_file is not None:
                    return False
                
                # Verify metadata is valid JSON
                metadata_bytes = zf.read("metadata.json")
                metadata = json.loads(metadata_bytes)
                
                # Check required fields
                if "version" not in metadata or "architecture" not in metadata:
                    return False
                
                return True
        except Exception:
            return False
    
    def extract(self, output_dir: str, members: Optional[List[str]] = None) -> None:
        """Extract PMO package."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(self.path, "r") as zf:
            if members:
                for member in members:
                    zf.extract(member, output_path)
            else:
                zf.extractall(output_path)
    
    def _compute_checksum(self) -> str:
        """Compute SHA256 checksum of the package."""
        sha256 = hashlib.sha256()
        with open(self.path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    @classmethod
    def from_config(cls, path: str, config: dict) -> "PMOPackage":
        """Create PMO package from configuration."""
        package = cls(path)
        
        metadata = PMOMetadata(
            version="1.0",
            architecture=config.get("architecture", "custom"),
            solver_count=config.get("solver", {}).get("count", 0),
            orchestral_count=config.get("orchestration", {}).get("total_orchestral_models", 0),
            total_params=config.get("system", {}).get("total_params", 0),
        )
        
        manifest = {
            "architecture": config.get("architecture"),
            "solver": config.get("solver"),
            "orchestration": config.get("orchestration"),
            "system": config.get("system"),
            "datasets": config.get("datasets"),
            "training": config.get("training"),
            "recruitment": config.get("recruitment"),
            "cache": config.get("cache"),
            "inference": config.get("inference"),
        }
        
        package.metadata = metadata
        package._manifest = manifest
        
        return package


def create_pmo_package(path: str, config: dict) -> PMOPackage:
    """Factory function to create PMO package."""
    return PMOPackage.from_config(path, config)
