"""Processing Coordinator for MinerU + MinIO + Database pipeline.

This module orchestrates the complete PDF processing workflow:
1. Upload files to MinIO
2. Import content_list.json to database
3. Generate embeddings
4. Build HiChunk index

Usage:
    coordinator = ProcessingCoordinator()
    result = coordinator.process_existing_output(
        output_dir=Path("output/document_id"),
        project_id="proj-uuid",
        document_id="doc-uuid",
        version_id="version-uuid",
        conn=db_connection
    )
"""

import json
import logging
import uuid
from pathlib import Path
from typing import Any

from bid_scoring.ingest import ingest_content_list
from bid_scoring.files import FileRegistry
from bid_scoring.embeddings_batch import EmbeddingBatchService

from mineru.minio_storage import create_storage_from_env, MinIOStorage

logger = logging.getLogger(__name__)


class ProcessingCoordinator:
    """Coordinates the complete PDF processing pipeline."""

    def __init__(
        self,
        storage: MinIOStorage | None = None,
        embedder: EmbeddingBatchService | None = None,
    ):
        """Initialize the coordinator.

        Args:
            storage: MinIO storage instance (created from env if None)
            embedder: Embedding service instance (created from env if None)
        """
        self.storage = storage or create_storage_from_env()
        self.embedder = embedder or None  # Lazy init

    def process_pdf(
        self,
        pdf_path: Path,
        project_id: str | None = None,
        conn=None,
    ) -> dict[str, Any]:
        """Process a PDF file through the complete pipeline.

        This is a placeholder for future implementation that will:
        1. Call MinerU API to parse PDF
        2. Store results in MinIO
        3. Import to database
        4. Generate embeddings

        Args:
            pdf_path: Path to PDF file
            project_id: Existing project ID or None to create new
            conn: Database connection

        Returns:
            Processing result dict with IDs and status
        """
        logger.info(f"Processing PDF: {pdf_path}")

        # Generate IDs
        if project_id is None:
            project_id = self.generate_project_id(pdf_path.stem)

        document_id = self.generate_document_id(project_id, pdf_path.name)
        version_id = self.generate_version_id()

        logger.info(f"Generated IDs: project={project_id}, doc={document_id}, version={version_id}")

        # TODO: Call MinerU API
        # TODO: Process output through process_existing_output

        return {
            "project_id": project_id,
            "document_id": document_id,
            "version_id": version_id,
            "status": "pending_mineru_api",
        }

    def process_existing_output(
        self,
        output_dir: Path,
        project_id: str,
        document_id: str,
        version_id: str,
        conn,
        document_title: str = "untitled",
        skip_embeddings: bool = False,
        skip_hichunk: bool = False,
    ) -> dict[str, Any]:
        """Process an existing MinerU output directory.

        Args:
            output_dir: Path to MinerU output directory
            project_id: Project UUID
            document_id: Document UUID
            version_id: Version UUID
            conn: Database connection
            document_title: Document title
            skip_embeddings: Skip embedding generation
            skip_hichunk: Skip HiChunk building

        Returns:
            Processing result dict
        """
        logger.info(f"Processing existing output: {output_dir}")

        # Step 1: Upload files to MinIO
        logger.info("Step 1: Uploading files to MinIO...")
        prefix = f"{project_id}/{version_id}/files"
        uploaded = self.storage.upload_directory(
            local_dir=output_dir,
            prefix=prefix,
        )

        logger.info(f"Uploaded {len(uploaded)} files to MinIO")

        # Step 2: Register files in database
        logger.info("Step 2: Registering files in database...")
        registry = FileRegistry(conn)
        registered_count = 0

        for file_info in uploaded:
            if "error" not in file_info:
                object_key = file_info["object_key"]
                file_type = self._get_file_type_from_key(object_key)
                file_name = Path(object_key).name

                registry.register_file(
                    version_id=version_id,
                    file_type=file_type,
                    file_path=object_key,
                    file_name=file_name,
                    file_size=file_info.get("size"),
                    etag=file_info.get("etag"),
                )
                registered_count += 1

        logger.info(f"Registered {registered_count} files in database")

        # Step 3: Import content_list.json
        logger.info("Step 3: Importing content_list.json...")
        content_list = load_content_list(output_dir)

        if content_list:
            ingest_result = ingest_content_list(
                conn=conn,
                project_id=project_id,
                document_id=document_id,
                version_id=version_id,
                content_list=content_list,
                document_title=document_title,
                source_type="mineru",
                source_uri=f"minio://{prefix}",
                parser_version="1.0",
                status="ready",
            )
            logger.info(f"Imported {ingest_result['total_chunks']} chunks")
        else:
            logger.warning("No content_list.json found")
            ingest_result = {"total_chunks": 0}

        # Step 4: Generate embeddings
        embedding_result = {"total_processed": 0, "succeeded": 0, "failed": 0}
        if not skip_embeddings and ingest_result.get("total_chunks", 0) > 0:
            logger.info("Step 4: Generating embeddings...")

            if self.embedder is None:
                self.embedder = EmbeddingBatchService()

            embedding_result = self.embedder.process_version(
                version_id=version_id,
                conn=conn,
            )
            logger.info(f"Generated {embedding_result['succeeded']}/{embedding_result['total_processed']} embeddings")

        # Step 5: Build HiChunk (placeholder)
        hichunk_result = {}
        if not skip_hichunk and ingest_result.get("total_chunks", 0) > 0:
            logger.info("Step 5: Building HiChunk index...")
            # TODO: Implement HiChunk building
            hichunk_result = {"status": "skipped"}

        return {
            "project_id": project_id,
            "document_id": document_id,
            "version_id": version_id,
            "files_uploaded": len(uploaded),
            "files_registered": registered_count,
            "chunks_imported": ingest_result.get("total_chunks", 0),
            "embeddings_generated": embedding_result.get("succeeded", 0),
            "status": "completed",
        }

    def generate_project_id(self, name: str | None = None) -> str:
        """Generate or retrieve a project ID.

        Args:
            name: Project name (for new projects)

        Returns:
            Project UUID
        """
        # TODO: Check if project exists by name
        # For now, always generate new
        return str(uuid.uuid4())

    def generate_document_id(self, project_id: str, filename: str) -> str:
        """Generate a document ID.

        Args:
            project_id: Project UUID
            filename: Source filename

        Returns:
            Document UUID
        """
        # TODO: Check if document exists
        return str(uuid.uuid4())

    def generate_version_id(self) -> str:
        """Generate a new version ID.

        Returns:
            Version UUID
        """
        return str(uuid.uuid4())

    @staticmethod
    def _get_file_type_from_key(object_key: str) -> str:
        """Determine file type from MinIO object key.

        Args:
            object_key: MinIO object key

        Returns:
            File type string
        """
        parts = object_key.split("/")
        # Format: bids/{project_id}/{version_id}/files/{file_type}/{file_name}
        if len(parts) >= 5 and parts[3] == "files":
            return parts[4]
        return "unknown"


def load_content_list(output_dir: Path) -> list[dict]:
    """Load content_list.json from MinerU output directory.

    Args:
        output_dir: Path to MinerU output directory

    Returns:
        List of content items, empty list if file not found
    """
    content_list_path = output_dir / "content_list.json"

    if not content_list_path.exists():
        logger.warning(f"content_list.json not found in {output_dir}")
        return []

    try:
        with open(content_list_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse content_list.json: {e}")
        return []
    except Exception as e:
        logger.error(f"Failed to read content_list.json: {e}")
        return []
