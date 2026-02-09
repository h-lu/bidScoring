"""MinIO Object Storage Module.

Provides file upload/download and presigned URL generation for MinIO/S3
compatible object storage.

Usage:
    storage = MinIOStorage(
        endpoint="localhost:9000",
        access_key="minioadmin",
        secret_key="minioadmin",
        bucket="bids",
        secure=False
    )

    # Upload file
    result = storage.upload_file(
        local_path=Path("file.pdf"),
        object_key="bids/project/version/files/original/file.pdf"
    )

    # Generate presigned URL
    url = storage.generate_presigned_url(
        object_key="bids/project/version/files/original/file.pdf",
        expires=timedelta(hours=1)
    )
"""

import logging
import os
from datetime import timedelta
from pathlib import Path
from typing import Any

try:
    from minio import Minio
    from minio.error import S3Error
except ImportError:
    Minio = None
    S3Error = Exception

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class MinIOStorage:
    """MinIO object storage client wrapper."""

    def __init__(
        self,
        endpoint: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        bucket: str | None = None,
        secure: bool = False,
    ):
        """Initialize MinIO storage client.

        Args:
            endpoint: MinIO server endpoint (host:port)
            access_key: Access key (username)
            secret_key: Secret key (password)
            bucket: Bucket name
            secure: Use HTTPS (default: False)
        """
        if Minio is None:
            raise ImportError(
                "minio package is required. Install with: pip install minio"
            )

        self.endpoint = endpoint or os.getenv("MINIO_ENDPOINT", "localhost:9000")
        self.access_key = access_key or os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        self.secret_key = secret_key or os.getenv("MINIO_SECRET_KEY", "minioadmin")
        self.bucket = bucket or os.getenv("MINIO_BUCKET", "bids")
        self.secure = secure

        # Initialize MinIO client
        self._client = Minio(
            endpoint=self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=self.secure,
        )

        # Ensure bucket exists
        self._ensure_bucket_exists()

    @property
    def client(self):
        """Get the underlying MinIO client."""
        return self._client

    def _ensure_bucket_exists(self) -> None:
        """Create bucket if it doesn't exist."""
        try:
            if not self._client.bucket_exists(self.bucket):
                logger.info(f"Creating bucket: {self.bucket}")
                self._client.make_bucket(self.bucket)
        except S3Error as e:
            logger.error(f"Failed to ensure bucket exists: {e}")
            raise

    def upload_file(
        self,
        local_path: Path,
        object_key: str,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Upload a file to MinIO.

        Args:
            local_path: Path to local file
            object_key: Object key in bucket (e.g., "bids/proj/version/file.pdf")
            content_type: MIME type (auto-detected from extension if None)
            metadata: Custom metadata to attach

        Returns:
            Dict with object_key, etag, size, metadata

        Raises:
            FileNotFoundError: If local file doesn't exist
            S3Error: If upload fails
        """
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"File not found: {local_path}")

        # Auto-detect content type from extension
        if content_type is None:
            content_type = self._guess_content_type(local_path)

        # Build MinIO metadata headers
        minio_metadata = {}
        if metadata:
            for key, value in metadata.items():
                minio_metadata[f"x-amz-meta-{key}"] = str(value)

        try:
            result = self._client.fput_object(
                bucket_name=self.bucket,
                object_name=object_key,
                file_path=str(local_path),
                content_type=content_type,
                metadata=minio_metadata,
            )

            # Handle different minio library versions
            size = getattr(result, "size", None)
            if size is None:
                # Try alternative attributes
                size = getattr(result, "object_size", getattr(result, "length", 0))

            etag = getattr(result, "etag", getattr(result, "etag", None))

            return {
                "object_key": object_key,
                "etag": etag,
                "size": size,
                "content_type": content_type,
                "metadata": metadata,
            }

        except S3Error as e:
            logger.error(f"Failed to upload {local_path} to {object_key}: {e}")
            raise

    def upload_directory(
        self,
        local_dir: Path,
        prefix: str,
        callback: callable | None = None,
    ) -> list[dict[str, Any]]:
        """Upload all files in a directory recursively.

        Args:
            local_dir: Local directory to upload
            prefix: Object key prefix (e.g., "bids/proj/version/files")
            callback: Optional callback(file_count, total_size) for progress

        Returns:
            List of upload results for each file
        """
        local_dir = Path(local_dir)
        if not local_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {local_dir}")

        results = []
        file_count = 0
        total_size = 0

        for file_path in local_dir.rglob("*"):
            if file_path.is_file():
                # Calculate relative path for object key
                rel_path = file_path.relative_to(local_dir)
                object_key = f"{prefix}/{rel_path}"

                try:
                    result = self.upload_file(file_path, object_key)
                    results.append(result)
                    file_count += 1
                    total_size += result["size"]

                    if callback:
                        callback(file_count, total_size)

                except S3Error as e:
                    logger.warning(f"Failed to upload {file_path}: {e}")
                    results.append(
                        {
                            "object_key": str(rel_path),
                            "error": str(e),
                        }
                    )

        return results

    def generate_presigned_url(
        self,
        object_key: str,
        expires: timedelta = timedelta(hours=1),
    ) -> str:
        """Generate a presigned URL for downloading.

        Args:
            object_key: Object key in bucket
            expires: URL expiration time (default: 1 hour)

        Returns:
            Presigned URL string
        """
        try:
            return self._client.presigned_get_object(
                bucket_name=self.bucket,
                object_name=object_key,
                expires=expires,
            )
        except S3Error as e:
            logger.error(f"Failed to generate presigned URL for {object_key}: {e}")
            raise

    def download_file(self, object_key: str, local_path: Path) -> None:
        """Download a file from MinIO.

        Args:
            object_key: Object key in bucket
            local_path: Where to save the file
        """
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self._client.fget_object(
                bucket_name=self.bucket,
                object_name=object_key,
                file_path=str(local_path),
            )
        except S3Error as e:
            logger.error(f"Failed to download {object_key}: {e}")
            raise

    def list_files(self, prefix: str, recursive: bool = True) -> list[dict[str, Any]]:
        """List objects with a given prefix.

        Args:
            prefix: Object key prefix
            recursive: List recursively (default: True)

        Returns:
            List of object info dicts
        """
        try:
            objects = self._client.list_objects(
                bucket_name=self.bucket,
                prefix=prefix,
                recursive=recursive,
            )

            return [
                {
                    "object_key": obj.object_name,
                    "size": obj.size,
                    "last_modified": obj.last_modified,
                    "etag": obj.etag,
                }
                for obj in objects
            ]

        except S3Error as e:
            logger.error(f"Failed to list objects with prefix {prefix}: {e}")
            raise

    def delete_file(self, object_key: str) -> None:
        """Delete an object from MinIO.

        Args:
            object_key: Object key to delete
        """
        try:
            self._client.remove_object(
                bucket_name=self.bucket,
                object_name=object_key,
            )
        except S3Error as e:
            logger.error(f"Failed to delete {object_key}: {e}")
            raise

    def file_exists(self, object_key: str) -> bool:
        """Check if an object exists in MinIO.

        Args:
            object_key: Object key to check

        Returns:
            True if object exists
        """
        try:
            self._client.stat_object(
                bucket_name=self.bucket,
                object_name=object_key,
            )
            return True
        except S3Error:
            return False

    @staticmethod
    def _guess_content_type(file_path: Path) -> str:
        """Guess content type from file extension.

        Args:
            file_path: File path

        Returns:
            MIME type string
        """
        content_types = {
            ".pdf": "application/pdf",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".svg": "image/svg+xml",
            ".webp": "image/webp",
            ".txt": "text/plain",
            ".md": "text/markdown",
            ".html": "text/html",
            ".css": "text/css",
            ".js": "application/javascript",
            ".json": "application/json",
            ".xml": "application/xml",
            ".zip": "application/zip",
        }

        ext = file_path.suffix.lower()
        return content_types.get(ext, "application/octet-stream")

    @staticmethod
    def build_object_key(
        project_id: str,
        version_id: str,
        file_type: str,
        file_name: str,
    ) -> str:
        """Build a standard MinIO object key.

        Args:
            project_id: Project UUID
            version_id: Version UUID
            file_type: File type (original, parsed, images, etc.)
            file_name: File name

        Returns:
            Object key string like "bids/{project_id}/{version_id}/files/{file_type}/{file_name}"
        """
        return f"bids/{project_id}/{version_id}/files/{file_type}/{file_name}"

    @staticmethod
    def get_file_type(object_key: str) -> str:
        """Extract file type from object key.

        Args:
            object_key: Object key string

        Returns:
            File type (original, parsed, images, etc.)
        """
        parts = object_key.split("/")
        # Format: bids/{project_id}/{version_id}/files/{file_type}/{file_name}
        if len(parts) >= 5 and parts[0] == "bids" and parts[3] == "files":
            return parts[4]
        return "unknown"


def create_storage_from_env() -> MinIOStorage:
    """Create MinIOStorage instance from environment variables.

    Environment variables:
        MINIO_ENDPOINT: Server endpoint (default: localhost:9000)
        MINIO_ACCESS_KEY: Access key (default: minioadmin)
        MINIO_SECRET_KEY: Secret key (default: minioadmin)
        MINIO_BUCKET: Bucket name (default: bids)
        MINIO_SECURE: Use HTTPS (default: false)

    Returns:
        MinIOStorage instance
    """
    secure = os.getenv("MINIO_SECURE", "false").lower() in ("true", "1", "yes")

    return MinIOStorage(
        endpoint=os.getenv("MINIO_ENDPOINT", "localhost:9000"),
        access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        bucket=os.getenv("MINIO_BUCKET", "bids"),
        secure=secure,
    )
