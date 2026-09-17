import os
import tempfile
import zipfile
from dataclasses import asdict

import pytest

from prothynesis_runtime.pmo.manifest import PMOManifest
from prothynesis_runtime.pmo.package import PMOMetadata, PMOPackage


def test_pmo_metadata_serialization():
    metadata = PMOMetadata(
        version="0.1.0",
        created_at="2024-01-01T00:00:00",
        architecture="cpu",
        solver_count=5,
        orchestral_count=2,
        total_params=100_000_000,
        checksum="abc123",
        manifest_path="manifest.json",
        model_index_path="model_index.json",
    )
    data = asdict(metadata)
    restored = PMOMetadata(**data)
    assert restored.version == "0.1.0"
    assert restored.solver_count == 5
    assert restored.total_params == 100_000_000


def test_pmo_package_build_and_verify():
    with tempfile.TemporaryDirectory() as tmpdir:
        metadata = PMOMetadata(version="0.1.0", created_at="2024-01-01", architecture="cpu")
        manifest = PMOManifest(model_id="m1", version="0.1.0", architecture="cpu")
        data_file = os.path.join(tmpdir, "data.bin")
        with open(data_file, "wb") as f:
            f.write(b"dummy data")
        output_path = os.path.join(tmpdir, "package.pmo")
        pkg = PMOPackage.build(metadata, asdict(manifest), [data_file], output_path)
        assert os.path.exists(output_path)
        assert pkg.path.exists()
        assert pkg.verify() is True


def test_pmo_package_verify_invalid():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "corrupted.pmo")
        with open(output_path, "wb") as f:
            f.write(b"not a real zip")
        pkg = PMOPackage(output_path)
        assert pkg.verify() is False


def test_pmo_package_inspect():
    with tempfile.TemporaryDirectory() as tmpdir:
        metadata = PMOMetadata(version="0.1.0", created_at="2024-01-01", architecture="cpu")
        manifest = PMOManifest(model_id="m1", version="0.1.0", architecture="cpu")
        data_file = os.path.join(tmpdir, "data.bin")
        with open(data_file, "wb") as f:
            f.write(b"dummy data")
        output_path = os.path.join(tmpdir, "package.pmo")
        pkg = PMOPackage.build(metadata, asdict(manifest), [data_file], output_path)
        info = pkg.inspect()
        assert "files" in info
        assert "metadata.json" in info["files"]


def test_pmo_package_extract():
    with tempfile.TemporaryDirectory() as tmpdir:
        metadata = PMOMetadata(version="0.1.0", created_at="2024-01-01", architecture="cpu")
        manifest = PMOManifest(model_id="m1", version="0.1.0", architecture="cpu")
        data_file = os.path.join(tmpdir, "data.bin")
        with open(data_file, "wb") as f:
            f.write(b"dummy data")
        output_path = os.path.join(tmpdir, "package.pmo")
        pkg = PMOPackage.build(metadata, asdict(manifest), [data_file], output_path)
        extract_dir = os.path.join(tmpdir, "extracted")
        pkg.extract(extract_dir)
        assert os.path.exists(os.path.join(extract_dir, "metadata.json"))
        assert os.path.exists(os.path.join(extract_dir, "manifest.json"))
