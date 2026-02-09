"""PDF Annotator Module for adding highlights to PDFs.

This module provides functionality to:
1. Retrieve chunk bbox coordinates from the database
2. Download original PDFs from MinIO
3. Add highlights and annotations using PyMuPDF
4. Upload annotated PDFs back to MinIO
5. Support cumulative layer additions for different analysis topics
"""

import logging
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

from psycopg.rows import dict_row

from mineru.minio_storage import MinIOStorage

from mcp_servers.annotation_insights import generate_annotation_content

logger = logging.getLogger(__name__)


# =============================================================================
# Color Coding Configuration
# =============================================================================

# Color coding by topic for visual organization
TOPIC_COLORS = {
    "risk": (1.0, 0.4, 0.4),        # Red
    "warranty": (0.4, 0.8, 0.4),    # Green
    "training": (0.9, 0.8, 0.3),    # Yellow
    "delivery": (0.9, 0.6, 0.2),    # Orange
    "financial": (0.4, 0.6, 0.9),   # Blue
    "technical": (0.6, 0.4, 0.8),   # Purple
    "default": (1.0, 0.9, 0.6),     # Light yellow
}


def parse_color(color: str) -> tuple[float, float, float] | None:
    """Parse color string to RGB tuple.

    Args:
        color: Color as hex (#RRGGBB) or topic name

    Returns:
        RGB tuple (0-1 range) or None
    """
    # Check if it's a topic name
    if color.lower() in TOPIC_COLORS:
        return TOPIC_COLORS[color.lower()]

    # Parse hex color
    if color.startswith("#"):
        hex_color = color[1:]
        if len(hex_color) == 6:
            try:
                r = int(hex_color[0:2], 16) / 255.0
                g = int(hex_color[2:4], 16) / 255.0
                b = int(hex_color[4:6], 16) / 255.0
                return (r, g, b)
            except ValueError:
                pass

    return None


@dataclass
class HighlightRequest:
    """Request for adding highlights to a PDF.

    Attributes:
        version_id: Document version UUID
        chunk_ids: List of chunk IDs to highlight
        topic: Topic name for color coding
        color: Optional custom RGB color (overrides topic color)
        increment: If True, add to existing annotated PDF
    """

    version_id: str
    chunk_ids: list[str]
    topic: str
    color: tuple[float, float, float] | None = None
    increment: bool = True


@dataclass
class HighlightResult:
    """Result of PDF highlighting operation.

    Attributes:
        success: Whether highlighting succeeded
        annotated_url: Presigned URL to annotated PDF
        highlights_added: Number of highlights added
        file_path: MinIO object key
        file_id: Database file record ID
        topics: List of topics in the annotated PDF
        error: Error message if failed
    """

    success: bool
    annotated_url: str | None = None
    highlights_added: int = 0
    file_path: str | None = None
    file_id: str | None = None
    topics: list[str] | None = None
    error: str | None = None


# =============================================================================
# PDF Annotator
# =============================================================================


class PDFAnnotator:
    """Annotates PDFs with highlights based on chunk bbox coordinates.

    Usage:
        annotator = PDFAnnotator(conn, minio_storage)

        result = annotator.highlight_chunks(
            version_id="xxx",
            chunk_ids=["chunk1", "chunk2"],
            topic="warranty"
        )

        url = result.annotated_url
    """

    def __init__(self, conn, storage: MinIOStorage | None = None):
        """Initialize PDFAnnotator.

        Args:
            conn: psycopg database connection
            storage: MinIO storage instance (created from env if None)
        """
        self.conn = conn
        self.storage = storage or MinIOStorage()

    def highlight_chunks(
        self,
        version_id: str,
        chunk_ids: list[str],
        topic: str,
        color: tuple[float, float, float] | None = None,
        increment: bool = True,
    ) -> HighlightResult:
        """Add highlights to PDF for specified chunks.

        Args:
            version_id: Document version ID
            chunk_ids: List of chunk IDs to highlight
            topic: Topic name for color coding
            color: Optional RGB color (0-1 range, overrides topic)
            increment: If True, add to existing annotated PDF

        Returns:
            HighlightResult with annotated URL and metadata
        """
        try:
            # Validate inputs
            if not chunk_ids:
                return HighlightResult(
                    success=False, error="chunk_ids cannot be empty"
                )

            # Get chunk bbox coordinates
            chunks = self._get_chunk_bboxes(version_id, chunk_ids)
            if not chunks:
                return HighlightResult(
                    success=False, error=f"No chunks found for IDs: {chunk_ids}"
                )

            # Determine source PDF (annotated or original)
            if increment:
                source_pdf = self._get_annotated_pdf(version_id)
            else:
                source_pdf = None

            if not source_pdf:
                # Download original PDF
                source_pdf = self._download_original_pdf(version_id)

            if not source_pdf:
                return HighlightResult(
                    success=False, error="No PDF found for version"
                )

            # Get color for topic
            if color is None:
                color = TOPIC_COLORS.get(topic.lower(), TOPIC_COLORS["default"])

            # Add highlights
            pdf_path = Path(source_pdf["local_path"])
            highlights_added = self._add_highlights(
                pdf_path, chunks, topic, color
            )

            # Get metadata for tracking
            metadata = source_pdf.get("metadata", {})
            topics = metadata.get("topics", [])

            if increment and topic not in topics:
                topics.append(topic)
            elif not increment:
                topics = [topic]

            metadata["topics"] = topics
            metadata["highlights_count"] = metadata.get("highlights_count", 0) + highlights_added
            metadata["last_updated"] = datetime.utcnow().isoformat()

            # Upload annotated PDF
            project_id = self._get_project_id(version_id)
            object_key, file_id = self._upload_annotated_pdf(
                version_id=version_id,
                project_id=project_id,
                pdf_path=pdf_path,
                metadata=metadata,
            )

            # Generate presigned URL
            presigned_url = self.storage.generate_presigned_url(object_key)

            return HighlightResult(
                success=True,
                annotated_url=presigned_url,
                highlights_added=highlights_added,
                file_path=object_key,
                file_id=file_id,
                topics=topics,
            )

        except Exception as e:
            logger.error(f"Failed to highlight PDF: {e}", exc_info=True)
            return HighlightResult(success=False, error=str(e))

    def _get_chunk_bboxes(
        self, version_id: str, chunk_ids: list[str]
    ) -> list[dict[str, Any]]:
        """Retrieve bbox coordinates for chunks.

        Args:
            version_id: Document version UUID
            chunk_ids: List of chunk IDs

        Returns:
            List of chunks with bbox and page info
        """
        with self.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT
                    c.chunk_id,
                    c.page_idx,
                    c.bbox,
                    c.text_raw,
                    c.element_type,
                    dp.coord_sys,
                    dp.page_w,
                    dp.page_h
                FROM chunks c
                LEFT JOIN document_pages dp
                    ON c.version_id = dp.version_id AND c.page_idx = dp.page_idx
                WHERE c.chunk_id = ANY(%s)
                """,
                (chunk_ids,),
            )
            return cur.fetchall()

    def _get_annotated_pdf(self, version_id: str) -> dict[str, Any] | None:
        """Get existing annotated PDF record.

        Args:
            version_id: Document version UUID

        Returns:
            File record with local_path, or None if not found
        """
        from bid_scoring.files import FileRegistry

        registry = FileRegistry(self.conn)
        record = registry.get_files_by_type(version_id, "annotated")

        if record:
            # Download to temp file
            temp_dir = Path(tempfile.gettempdir())
            temp_path = temp_dir / f"annotated_{version_id}.pdf"

            try:
                self.storage.download_file(record[0]["file_path"], temp_path)
                return {
                    "local_path": str(temp_path),
                    "metadata": record[0].get("metadata"),
                    "file_path": record[0]["file_path"],
                }
            except Exception as e:
                logger.warning(f"Failed to download annotated PDF: {e}")
                return None

        return None

    def _download_original_pdf(self, version_id: str) -> dict[str, Any] | None:
        """Download original PDF from MinIO.

        Args:
            version_id: Document version UUID

        Returns:
            Dict with local_path, or None if not found
        """
        from bid_scoring.files import FileRegistry

        registry = FileRegistry(self.conn)
        record = registry.get_original_pdf(version_id)

        if not record:
            return None

        # Download to temp file
        temp_dir = Path(tempfile.gettempdir())
        temp_path = temp_dir / f"original_{version_id}.pdf"

        try:
            self.storage.download_file(record["file_path"], temp_path)
            return {
                "local_path": str(temp_path),
                "file_name": record.get("file_name", "document.pdf"),
            }
        except Exception as e:
            logger.error(f"Failed to download PDF: {e}")
            return None

    def _add_highlights(
        self,
        pdf_path: Path,
        chunks: list[dict[str, Any]],
        topic: str,
        color: tuple[float, float, float],
    ) -> int:
        """Add highlights to PDF for chunks.

        Args:
            pdf_path: Path to PDF file
            chunks: List of chunks with bbox coordinates
            topic: Topic name for annotation
            color: RGB color tuple

        Returns:
            Number of highlights added
        """
        doc = fitz.open(pdf_path)
        highlights_added = 0

        for chunk in chunks:
            bbox = chunk.get("bbox")
            if not bbox or len(bbox) != 4:
                continue

            page_idx = chunk.get("page_idx", 0)
            if page_idx >= len(doc):
                logger.warning(f"Page {page_idx} out of range")
                continue

            page = doc[page_idx]
            page_width = page.rect.width
            page_height = page.rect.height

            # Check coordinate system and convert if needed
            # MinerU mineru_bbox_v1 uses normalized coordinates (0-1000)
            coord_sys = chunk.get("coord_sys", "mineru_bbox_v1")

            if coord_sys == "mineru_bbox_v1":
                # Convert from normalized (0-1000) to actual PDF coordinates
                x0 = bbox[0] / 1000.0 * page_width
                y0 = bbox[1] / 1000.0 * page_height
                x1 = bbox[2] / 1000.0 * page_width
                y1 = bbox[3] / 1000.0 * page_height
                rect = fitz.Rect(x0, y0, x1, y1)
            else:
                # Use bbox as-is (assuming already in PDF coordinates)
                rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])

            # Add highlight annotation
            annot = page.add_highlight_annot(rect)

            # Set color
            annot.set_colors(stroke=color)
            annot.set_opacity(0.3)

            # Generate intelligent annotation content instead of just repeating text
            text = chunk.get("text_raw", "")
            if text:
                annotation_content = generate_annotation_content(
                    text=text,
                    topic=topic,
                    max_length=300,
                )
                annot.set_info(annotation_content)

            annot.update()
            highlights_added += 1

        # Save changes
        doc.save(pdf_path, incremental=False)
        doc.close()

        return highlights_added

    def _get_project_id(self, version_id: str) -> str | None:
        """Get project_id for a version.

        Args:
            version_id: Document version UUID

        Returns:
            Project UUID or None
        """
        with self.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT d.project_id
                FROM document_versions v
                JOIN documents d ON v.doc_id = d.doc_id
                WHERE v.version_id = %s
                """,
                (version_id,),
            )
            result = cur.fetchone()
            return result["project_id"] if result else None

    def _upload_annotated_pdf(
        self,
        version_id: str,
        project_id: str,
        pdf_path: Path,
        metadata: dict[str, Any],
    ) -> tuple[str, str]:
        """Upload annotated PDF to MinIO.

        Args:
            version_id: Document version UUID
            project_id: Project UUID
            pdf_path: Path to annotated PDF
            metadata: Metadata dict

        Returns:
            Tuple of (object_key, file_id)
        """
        from bid_scoring.files import FileRegistry
        from psycopg.types.json import Jsonb

        registry = FileRegistry(self.conn)

        # Build object key
        file_name = f"{version_id}_annotated.pdf"
        object_key = f"bids/{project_id}/{version_id}/files/annotated/{file_name}"

        # Upload to MinIO
        result = self.storage.upload_file(
            local_path=pdf_path,
            object_key=object_key,
            content_type="application/pdf",
        )

        # Register in database
        with self.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO document_files (
                    file_id, version_id, file_type, file_path, file_name,
                    file_size, content_type, etag, metadata
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (version_id, file_path) DO UPDATE SET
                    file_type = EXCLUDED.file_type,
                    file_name = EXCLUDED.file_name,
                    file_size = EXCLUDED.file_size,
                    content_type = EXCLUDED.content_type,
                    etag = EXCLUDED.etag,
                    metadata = EXCLUDED.metadata
                RETURNING file_id
                """,
                (
                    str(uuid.uuid4()),
                    version_id,
                    "annotated",
                    object_key,
                    file_name,
                    result.get("size"),
                    "application/pdf",
                    result.get("etag"),
                    Jsonb(metadata),
                ),
            )
            file_record = cur.fetchone()
            self.conn.commit()

        return object_key, file_record["file_id"] if file_record else None
