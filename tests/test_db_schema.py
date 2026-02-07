import psycopg
from bid_scoring.config import load_settings


def test_tables_exist():
    """Test that all required tables are created."""
    dsn = load_settings()["DATABASE_URL"]
    expected_tables = [
        "projects",
        "documents",
        "document_versions",
        "document_pages",
        "content_units",
        "chunks",
        "chunk_unit_spans",
        "hierarchical_nodes",
        "multi_vector_mappings",
        "scoring_runs",
        "scoring_results",
        "citations",
    ]
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            for table in expected_tables:
                cur.execute(
                    """
                    SELECT COUNT(*) FROM information_schema.tables 
                    WHERE table_name = %s AND table_schema = 'public'
                    """,
                    (table,),
                )
                assert cur.fetchone()[0] == 1, f"Table '{table}' does not exist"


def test_extensions_exist():
    """Test that required extensions are installed."""
    dsn = load_settings()["DATABASE_URL"]
    expected_extensions = ["pgcrypto", "vector"]
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            for ext in expected_extensions:
                cur.execute(
                    "SELECT COUNT(*) FROM pg_extension WHERE extname = %s",
                    (ext,),
                )
                assert cur.fetchone()[0] == 1, f"Extension '{ext}' is not installed"


def test_indexes_exist():
    """Test that required indexes are created."""
    dsn = load_settings()["DATABASE_URL"]
    expected_indexes = [
        "idx_chunks_text_tsv",
        "idx_chunks_embedding_hnsw",
        "idx_chunks_project_version_page",
    ]
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            for idx in expected_indexes:
                cur.execute(
                    """
                    SELECT COUNT(*) FROM pg_indexes 
                    WHERE indexname = %s AND schemaname = 'public'
                    """,
                    (idx,),
                )
                assert cur.fetchone()[0] == 1, f"Index '{idx}' does not exist"
