import hashlib
import json
import os
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from prothynesis_runtime.pmo.manifest import PMOManifest, validate_manifest


@dataclass
class PMOMetadata:
    version: str = "0.1.0"
    created_at: str = ""
    architecture: str = ""
    solver_count: int = 0
    orchestral_count: int = 0
    total_params: int = 0
    checksum: str = ""
    manifest_path: str = ""
    model_index_path: str = ""


class PMOPackage:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"PMO package not found: {path}")

    def _compute_checksum(self) -> str:
        sha256 = hashlib.sha256()
        with open(self.path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def inspect(self) -> dict:
        metadata: Dict[str, Any] = {}
        with zipfile.ZipFile(self.path, "r") as zf:
            names = zf.namelist()
            metadata["files"] = names
            if "metadata.json" in names:
                with zf.open("metadata.json") as f:
                    metadata["metadata"] = json.load(f)
            if "manifest.json" in names:
                with zf.open("manifest.json") as f:
                    metadata["manifest"] = json.load(f)
            if "model_index.json" in names:
                with zf.open("model_index.json") as f:
                    metadata["model_index"] = json.load(f)
        metadata["checksum"] = self._compute_checksum()
        return metadata

    def verify(self) -> bool:
        try:
            with zipfile.ZipFile(self.path, "r") as zf:
                names = set(zf.namelist())
                required = {"metadata.json", "manifest.json", "model_index.json"}
                missing = required - names
                if missing:
                    return False
                if "manifest.json" in names:
                    with zf.open("manifest.json") as f:
                        data = json.load(f)
                    ok, _ = validate_manifest(data)
                    if not ok:
                        return False
            return True
        except Exception:
            return False

    def extract(self, output_dir: str, members: Optional[List[str]] = None) -> None:
        with zipfile.ZipFile(self.path, "r") as zf:
            zf.extractall(output_dir, members=members)

    @classmethod
    def build(
        cls,
        metadata: PMOMetadata,
        manifest: Dict[str, Any],
        files: List[str],
        output_path: str,
    ) -> "PMOPackage":
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            metadata_bytes = json.dumps({
                "version": metadata.version,
                "created_at": metadata.created_at,
                "architecture": metadata.architecture,
                "solver_count": metadata.solver_count,
                "orchestral_count": metadata.orchestral_count,
                "total_params": metadata.total_params,
                "manifest_path": metadata.manifest_path,
                "model_index_path": metadata.model_index_path,
            }, indent=2).encode("utf-8")
            zf.writestr("metadata.json", metadata_bytes)
            zf.writestr("manifest.json", json.dumps(manifest, indent=2).encode("utf-8"))
            zf.writestr("model_index.json", json.dumps({"models": []}, indent=2).encode("utf-8"))
            for file_path in files:
                arcname = os.path.relpath(file_path, os.path.dirname(output_path))
                zf.write(file_path, arcname)
        return cls(output_path)

    @classmethod
    def from_config(cls, path: str, config: Dict[str, Any]) -> "PMOPackage":
        metadata = PMOMetadata(
            version=config.get("version", "0.1.0"),
            created_at=config.get("created_at", ""),
            architecture=config.get("architecture", ""),
            solver_count=config.get("solver_count", 0),
            orchestral_count=config.get("orchestral_count", 0),
            total_params=config.get("total_params", 0),
            manifest_path=config.get("manifest_path", "manifest.json"),
            model_index_path=config.get("model_index_path", "model_index.json"),
        )
        manifest = config.get("manifest", {})
        files = config.get("files", [])
        return cls.build(metadata, manifest, files, path)
