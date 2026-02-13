from __future__ import annotations

from pathlib import Path

from bid_scoring.pipeline.infrastructure.minio_store import MinIOObjectStore


class _Storage:
    def upload_directory(self, local_dir, prefix, callback=None):  # pragma: no cover
        return [{"local_dir": str(local_dir), "prefix": prefix}]

    def upload_file(self, local_path, object_key):  # pragma: no cover
        return {"local_path": str(local_path), "object_key": object_key}


def test_minio_store_upload_directory_delegates(tmp_path: Path):
    store = MinIOObjectStore(storage=_Storage())
    result = store.upload_directory(local_dir=tmp_path, prefix="a/b")
    assert result[0]["prefix"] == "a/b"


def test_minio_store_upload_file_delegates(tmp_path: Path):
    file_path = tmp_path / "x.txt"
    file_path.write_text("x", encoding="utf-8")
    store = MinIOObjectStore(storage=_Storage())
    result = store.upload_file(local_path=file_path, object_key="k")
    assert result["object_key"] == "k"
