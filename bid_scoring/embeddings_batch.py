"""Batch Embedding Service for vectorizing chunks.

This module provides efficient batch processing for generating embeddings
with retry logic and progress tracking.

Based on OpenAI Cookbook best practices:
- Batch size 100-1000 texts per API call
- Exponential backoff retry (1-20s)
- Up to 6 retry attempts

Usage:
    service = EmbeddingBatchService(api_key="sk-...")
    result = service.process_version(version_id, conn)
    print(result)  # {'total_processed': 150, 'succeeded': 148, 'failed': 2}
"""

import logging
import os
from typing import Any

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from bid_scoring.config import load_settings

logger = logging.getLogger(__name__)


class EmbeddingBatchService:
    """Batch embedding generation service with retry logic.

    Processes chunks in batches to generate embeddings using OpenAI's API.
    Supports progress tracking, error recovery, and resume capability.
    """

    DEFAULT_BATCH_SIZE = 100
    DEFAULT_MAX_RETRIES = 6
    DEFAULT_MODEL = "text-embedding-3-small"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        batch_size: int | None = None,
        max_retries: int | None = None,
    ):
        """Initialize the embedding service.

        Args:
            api_key: OpenAI API key (from env if None)
            base_url: OpenAI base URL (from env if None)
            model: Embedding model name
            batch_size: Texts per batch (default: 100)
            max_retries: Max retry attempts (default: 6)
        """
        if OpenAI is None:
            raise ImportError(
                "openai package is required. Install with: pip install openai"
            )

        settings = load_settings()

        self.api_key = api_key or settings.get("OPENAI_API_KEY")
        self.base_url = base_url or settings.get("OPENAI_BASE_URL")
        self.model = model or settings.get("OPENAI_EMBEDDING_MODEL", self.DEFAULT_MODEL)
        self.batch_size = batch_size or int(
            os.getenv("EMBEDDING_BATCH_SIZE", self.DEFAULT_BATCH_SIZE)
        )
        self.max_retries = max_retries or int(
            os.getenv("EMBEDDING_MAX_RETRIES", self.DEFAULT_MAX_RETRIES)
        )

        # Initialize OpenAI client
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    @retry(
        wait=wait_random_exponential(min=1, max=20),
        stop=stop_after_attempt(6),
    )
    def _get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for a batch of texts with retry logic.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            Exception: If all retries exhausted
        """
        try:
            response = self._client.embeddings.create(
                input=texts,
                model=self.model,
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.warning(f"Embedding API error: {e}, retrying...")
            raise

    def process_version(
        self,
        version_id: str,
        conn,
        batch_size: int | None = None,
    ) -> dict[str, Any]:
        """Process all pending chunks for a specific version.

        Args:
            version_id: Document version ID
            conn: Database connection
            batch_size: Override default batch size

        Returns:
            Dict with total_processed, succeeded, failed counts
        """
        batch_size = batch_size or self.batch_size

        # Get all pending chunks for this version
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT chunk_id, text_raw
                FROM chunks
                WHERE version_id = %s
                AND (embedding IS NULL OR embedding_status != 'completed')
                ORDER BY chunk_index
                """,
                (version_id,),
            )
            chunks = cur.fetchall()
            # Convert tuples to dicts for compatibility (handle both tuples and dicts)
            if chunks and isinstance(chunks[0], dict):
                chunks_list = chunks
            else:
                chunks_list = [{"chunk_id": row[0], "text_raw": row[1]} for row in chunks]
            chunks = chunks_list

        if not chunks:
            logger.info(f"No pending chunks for version {version_id}")
            return {"total_processed": 0, "succeeded": 0, "failed": 0}

        logger.info(f"Processing {len(chunks)} chunks for version {version_id}")

        succeeded = 0
        failed = 0

        # Process in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [chunk["text_raw"] for chunk in batch if chunk.get("text_raw")]
            chunk_ids = [chunk["chunk_id"] for chunk in batch]

            if not texts:
                continue

            try:
                # Mark as processing
                self._update_status(conn, chunk_ids, "processing")

                # Get embeddings
                embeddings = self._get_embeddings_batch(texts)

                # Update database
                self._save_embeddings(conn, chunk_ids, embeddings)
                succeeded += len(embeddings)

                logger.info(
                    f"Batch {i // batch_size + 1}: {len(embeddings)} embeddings saved"
                )

            except Exception as e:
                logger.error(f"Batch {i // batch_size + 1} failed: {e}")
                self._update_status(conn, chunk_ids, "failed")
                failed += len(batch)

        conn.commit()

        return {
            "total_processed": len(chunks),
            "succeeded": succeeded,
            "failed": failed,
        }

    def process_pending(
        self,
        conn,
        limit: int = 1000,
        version_id: str | None = None,
    ) -> dict[str, Any]:
        """Process pending chunks across all or specific versions.

        Args:
            conn: Database connection
            limit: Max chunks to process
            version_id: Optional specific version to process

        Returns:
            Dict with processing results
        """
        # Get pending chunks
        with conn.cursor() as cur:
            if version_id:
                cur.execute(
                    """
                    SELECT chunk_id, text_raw, version_id
                    FROM chunks
                    WHERE version_id = %s
                    AND embedding_status = 'pending'
                    ORDER BY version_id, chunk_index
                    LIMIT %s
                    """,
                    (version_id, limit),
                )
            else:
                cur.execute(
                    """
                    SELECT chunk_id, text_raw, version_id
                    FROM chunks
                    WHERE embedding_status = 'pending'
                    ORDER BY version_id, chunk_index
                    LIMIT %s
                    """,
                    (limit,),
                )
            chunks = cur.fetchall()

        if not chunks:
            return {"total_processed": 0, "succeeded": 0, "failed": 0}

        # Group by version_id for efficient processing
        from collections import defaultdict

        by_version: dict[str, list[dict]] = defaultdict(list)
        for chunk in chunks:
            by_version[chunk["version_id"]].append(chunk)

        total_succeeded = 0
        total_failed = 0

        for ver_id, ver_chunks in by_version.items():
            result = self._process_chunk_list(conn, ver_id, ver_chunks)
            total_succeeded += result["succeeded"]
            total_failed += result["failed"]

        return {
            "total_processed": len(chunks),
            "succeeded": total_succeeded,
            "failed": total_failed,
        }

    def _process_chunk_list(
        self,
        conn,
        version_id: str,
        chunks: list[dict],
    ) -> dict[str, int]:
        """Process a list of chunks for a version.

        Args:
            conn: Database connection
            version_id: Version ID
            chunks: List of chunk dicts

        Returns:
            Dict with succeeded/failed counts
        """
        succeeded = 0
        failed = 0

        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            texts = [c["text_raw"] for c in batch if c.get("text_raw")]
            chunk_ids = [c["chunk_id"] for c in batch]

            if not texts:
                continue

            try:
                self._update_status(conn, chunk_ids, "processing")
                embeddings = self._get_embeddings_batch(texts)
                self._save_embeddings(conn, chunk_ids, embeddings)
                succeeded += len(embeddings)

            except Exception as e:
                logger.error(f"Batch failed for version {version_id}: {e}")
                self._update_status(conn, chunk_ids, "failed")
                failed += len(batch)

        return {"succeeded": succeeded, "failed": failed}

    def _update_status(self, conn, chunk_ids: list[str], status: str) -> None:
        """Update embedding status for chunks.

        Args:
            conn: Database connection
            chunk_ids: List of chunk IDs
            status: New status ('pending', 'processing', 'completed', 'failed')
        """
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE chunks
                SET embedding_status = %s
                WHERE chunk_id = ANY(%s)
                """,
                (status, chunk_ids),
            )
        conn.commit()

    def _save_embeddings(
        self,
        conn,
        chunk_ids: list[str],
        embeddings: list[list[float]],
    ) -> None:
        """Save embeddings to database.

        Args:
            conn: Database connection
            chunk_ids: List of chunk IDs
            embeddings: List of embedding vectors
        """
        with conn.cursor() as cur:
            for chunk_id, embedding in zip(chunk_ids, embeddings):
                cur.execute(
                    """
                    UPDATE chunks
                    SET embedding = %s, embedding_status = 'completed'
                    WHERE chunk_id = %s
                    """,
                    (embedding, chunk_id),
                )
        conn.commit()


def create_service_from_env() -> EmbeddingBatchService:
    """Create EmbeddingBatchService from environment variables.

    Returns:
        EmbeddingBatchService instance
    """
    return EmbeddingBatchService()
