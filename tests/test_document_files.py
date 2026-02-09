"""Tests for document_files table and MinIO integration.

This module tests:
1. document_files table schema
2. FileRegistry operations
3. embedding_status column on chunks table
"""

import uuid
import os
import pytest
import psycopg
from psycopg.rows import dict_row

from bid_scoring.config import load_settings

# Skip all tests if DATABASE_URL not set
needs_database = pytest.mark.skipif(
    not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set"
)


def get_test_conn():
    """Get a test database connection."""
    settings = load_settings()
    dsn = settings["DATABASE_URL"]
    return psycopg.connect(dsn)


@needs_database
class TestDocumentFilesSchema:
    """Test document_files table schema."""

    def test_document_files_table_exists(self):
        """document_files table should exist."""
        with get_test_conn() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_schema = 'public'
                        AND table_name = 'document_files'
                    )
                """)
                result = cur.fetchone()
                assert result["exists"] is True, "document_files table should exist"

    def test_document_files_columns(self):
        """document_files should have required columns."""
        with get_test_conn() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute("""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_name = 'document_files'
                    ORDER BY ordinal_position
                """)
                columns = {row["column_name"]: row for row in cur.fetchall()}

                required_columns = {
                    "file_id": ("uuid", "NO"),
                    "version_id": ("uuid", "YES"),
                    "file_type": ("text", "YES"),
                    "file_path": ("text", "YES"),
                    "file_name": ("text", "YES"),
                    "file_size": ("bigint", "YES"),
                    "content_type": ("text", "YES"),
                    "etag": ("text", "YES"),
                    "metadata": ("jsonb", "YES"),
                    "created_at": ("timestamp with time zone", "YES"),
                }

                for col_name, (expected_type, _) in required_columns.items():
                    assert col_name in columns, f"Column {col_name} should exist"
                    # Type may vary by PostgreSQL version, check contains
                    assert expected_type in columns[col_name]["data_type"], (
                        f"Column {col_name} should be {expected_type} type"
                    )

    def test_document_files_indexes(self):
        """document_files should have required indexes."""
        with get_test_conn() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute("""
                    SELECT indexname
                    FROM pg_indexes
                    WHERE tablename = 'document_files'
                """)
                indexes = {row["indexname"] for row in cur.fetchall()}

                assert "idx_document_files_version" in indexes
                assert "idx_document_files_type" in indexes

    def test_document_files_foreign_key(self):
        """document_files.version_id should reference document_versions."""
        with get_test_conn() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute("""
                    SELECT
                        tc.constraint_name,
                        tc.table_name,
                        kcu.column_name,
                        ccu.table_name AS foreign_table_name,
                        ccu.column_name AS foreign_column_name
                    FROM information_schema.table_constraints AS tc
                    JOIN information_schema.key_column_usage AS kcu
                        ON tc.constraint_name = kcu.constraint_name
                    JOIN information_schema.constraint_column_usage AS ccu
                        ON ccu.constraint_name = tc.constraint_name
                    WHERE tc.constraint_type = 'FOREIGN KEY'
                        AND tc.table_name = 'document_files'
                """)
                fks = cur.fetchall()

                assert len(fks) > 0, "Should have foreign key constraint"
                version_fk = [fk for fk in fks if fk["column_name"] == "version_id"]
                assert len(version_fk) > 0, (
                    "version_id should have FK to document_versions"
                )
                assert version_fk[0]["foreign_table_name"] == "document_versions"


@needs_database
class TestChunksEmbeddingStatus:
    """Test embedding_status column on chunks table."""

    def test_chunks_has_embedding_status(self):
        """chunks table should have embedding_status column."""
        with get_test_conn() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute("""
                    SELECT column_name, column_default
                    FROM information_schema.columns
                    WHERE table_name = 'chunks'
                    AND column_name = 'embedding_status'
                """)
                result = cur.fetchone()

                assert result is not None, "chunks should have embedding_status column"
                # Default should be 'pending'
                default_val = str(result.get("column_default", ""))
                assert (
                    "pending" in default_val
                    or result.get("column_default") == "'pending'::text"
                ), f"embedding_status should default to 'pending', got: {default_val}"

    def test_chunks_embedding_status_index(self):
        """chunks should have partial index on embedding_status."""
        with get_test_conn() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute("""
                    SELECT indexname, indexdef
                    FROM pg_indexes
                    WHERE tablename = 'chunks'
                    AND indexname = 'idx_chunks_embedding_status'
                """)
                result = cur.fetchone()

                assert result is not None, (
                    "Should have idx_chunks_embedding_status index"
                )
                assert "WHERE" in result["indexdef"].upper(), (
                    "Should be a partial index (WHERE clause)"
                )


@needs_database
class TestFileRegistryOperations:
    """Test FileRegistry CRUD operations."""

    @pytest.fixture
    def test_version(self):
        """Create a test document version."""
        project_id = str(uuid.uuid4())
        doc_id = str(uuid.uuid4())
        version_id = str(uuid.uuid4())

        with get_test_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO projects (project_id, name) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                    (project_id, "test-project"),
                )
                cur.execute(
                    "INSERT INTO documents (doc_id, project_id, title, source_type) VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
                    (doc_id, project_id, "test-doc", "mineru"),
                )
                cur.execute(
                    "INSERT INTO document_versions (version_id, doc_id, source_uri, parser_version, status) VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                    (version_id, doc_id, "test://uri", "1.0", "ready"),
                )
            conn.commit()

        yield version_id

        # Cleanup
        with get_test_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM document_files WHERE version_id = %s", (version_id,)
                )
                cur.execute(
                    "DELETE FROM document_versions WHERE version_id = %s", (version_id,)
                )
                cur.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,))
                cur.execute("DELETE FROM projects WHERE project_id = %s", (project_id,))
            conn.commit()

    def test_register_file(self, test_version):
        """register_file should create a new file record."""
        from bid_scoring.files import FileRegistry

        with get_test_conn() as conn:
            registry = FileRegistry(conn)

            file_id = registry.register_file(
                version_id=test_version,
                file_type="original_pdf",
                file_path="bids/test/version/files/original/test.pdf",
                file_name="test.pdf",
                file_size=12345,
                content_type="application/pdf",
                etag="abc123",
                metadata={"key": "value"},
            )

            assert file_id is not None

            # Verify record exists
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    "SELECT * FROM document_files WHERE file_id = %s", (file_id,)
                )
                record = cur.fetchone()

            assert str(record["version_id"]) == test_version
            assert record["file_type"] == "original_pdf"
            assert record["file_name"] == "test.pdf"
            assert record["file_size"] == 12345

    def test_get_files_by_version(self, test_version):
        """get_files_by_version should return all files for a version."""
        from bid_scoring.files import FileRegistry

        with get_test_conn() as conn:
            registry = FileRegistry(conn)

            # Register multiple files
            registry.register_file(
                version_id=test_version,
                file_type="original_pdf",
                file_path="bids/test/v1/files/original/test.pdf",
                file_name="test.pdf",
                file_size=1000,
                content_type="application/pdf",
                etag="1",
            )
            registry.register_file(
                version_id=test_version,
                file_type="markdown",
                file_path="bids/test/v1/files/parsed/full.md",
                file_name="full.md",
                file_size=500,
                content_type="text/markdown",
                etag="2",
            )

            files = registry.get_files_by_version(test_version)

            assert len(files) == 2
            file_types = {f["file_type"] for f in files}
            assert "original_pdf" in file_types
            assert "markdown" in file_types

    def test_get_original_pdf(self, test_version):
        """get_original_pdf should return the original PDF file."""
        from bid_scoring.files import FileRegistry

        with get_test_conn() as conn:
            registry = FileRegistry(conn)

            # Register files
            pdf_id = registry.register_file(
                version_id=test_version,
                file_type="original_pdf",
                file_path="bids/test/v1/files/original/test.pdf",
                file_name="test.pdf",
                file_size=1000,
                content_type="application/pdf",
                etag="1",
            )
            registry.register_file(
                version_id=test_version,
                file_type="markdown",
                file_path="bids/test/v1/files/parsed/full.md",
                file_name="full.md",
                file_size=500,
                content_type="text/markdown",
                etag="2",
            )

            pdf_record = registry.get_original_pdf(test_version)

            assert pdf_record is not None
            assert pdf_record["file_id"] == pdf_id
            assert pdf_record["file_type"] == "original_pdf"

    def test_register_file_duplicate_path(self, test_version):
        """register_file should handle duplicate paths (upsert)."""
        from bid_scoring.files import FileRegistry

        with get_test_conn() as conn:
            registry = FileRegistry(conn)

            file_path = "bids/test/v1/files/original/test.pdf"

            registry.register_file(
                version_id=test_version,
                file_type="original_pdf",
                file_path=file_path,
                file_name="test.pdf",
                file_size=1000,
                content_type="application/pdf",
                etag="1",
            )

            registry.register_file(
                version_id=test_version,
                file_type="original_pdf",
                file_path=file_path,
                file_name="test.pdf",
                file_size=2000,  # Updated size
                content_type="application/pdf",
                etag="2",
            )

            # Verify the size was updated
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT file_size FROM document_files
                    WHERE version_id = %s AND file_path = %s
                """,
                    (test_version, file_path),
                )
                record = cur.fetchone()

            assert record["file_size"] == 2000
