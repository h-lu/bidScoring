from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from mineru.minio_storage import MinIOStorage


class MinIOObjectStore:
    """Thin object-store adapter wrapping existing MinIO client."""

    def __init__(self, storage: MinIOStorage | None = None):
        self.storage = storage or MinIOStorage()

    def upload_directory(
        self,
        *,
        local_dir: Path,
        prefix: str,
        callback: Callable[[int, int], None] | None = None,
    ) -> list[dict[str, Any]]:
        return self.storage.upload_directory(local_dir=local_dir, prefix=prefix, callback=callback)

    def upload_file(self, *, local_path: Path, object_key: str) -> dict[str, Any]:
        return self.storage.upload_file(local_path=local_path, object_key=object_key)

