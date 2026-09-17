import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List


class ArtifactStore(ABC):
    @abstractmethod
    def upload(self, local_path: str, remote_path: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def download(self, remote_path: str, local_path: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def delete(self, remote_path: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def list(self, prefix: str) -> List[str]:
        raise NotImplementedError


class LocalArtifactStore(ArtifactStore):
    def __init__(self, base_dir: str) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def upload(self, local_path: str, remote_path: str) -> str:
        src = Path(local_path)
        dst = self.base_dir / remote_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
        return str(dst)

    def download(self, remote_path: str, local_path: str) -> bool:
        src = self.base_dir / remote_path
        if not src.exists():
            return False
        dst = Path(local_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
        return True

    def delete(self, remote_path: str) -> bool:
        target = self.base_dir / remote_path
        if target.exists():
            target.unlink()
            return True
        return False

    def list(self, prefix: str) -> List[str]:
        base = self.base_dir / prefix if prefix else self.base_dir
        if not base.exists():
            return []
        results = []
        for root, _, files in os.walk(base):
            for file in files:
                full = Path(root) / file
                results.append(str(full.relative_to(self.base_dir)))
        return results


class GitHubArtifactStore(ArtifactStore):
    def __init__(self, owner: str = "", repo: str = "", token: str = "") -> None:
        self.owner = owner
        self.repo = repo
        self.token = token

    def upload(self, local_path: str, remote_path: str) -> str:
        raise NotImplementedError("GitHubArtifactStore upload not implemented in MVP")

    def download(self, remote_path: str, local_path: str) -> bool:
        raise NotImplementedError("GitHubArtifactStore download not implemented in MVP")

    def delete(self, remote_path: str) -> bool:
        raise NotImplementedError("GitHubArtifactStore delete not implemented in MVP")

    def list(self, prefix: str) -> List[str]:
        raise NotImplementedError("GitHubArtifactStore list not implemented in MVP")
