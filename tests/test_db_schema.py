import psycopg
from bid_scoring.config import load_settings


def test_tables_exist():
    """Test that all required tables are created."""
    dsn = load_settings()["DATABASE_URL"]
    expected_tables = [
        "projects",
        "documents",
        "document_versions",
        "chunks",
        "contextual_chunks",
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
        "idx_contextual_chunks_chunk_id",
        "idx_contextual_chunks_version_id",
        "idx_contextual_chunks_embedding_hnsw",
        "idx_contextual_chunks_version_created",
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


def test_contextual_chunks_columns():
    """Test that contextual_chunks table has correct columns."""
    dsn = load_settings()["DATABASE_URL"]
    expected_columns = {
        "contextual_id": "uuid",
        "chunk_id": "uuid",
        "version_id": "uuid",
        "original_text": "text",
        "context_prefix": "text",
        "contextualized_text": "text",
        "model_name": "text",
        "embedding_model": "text",
        "created_at": "timestamp with time zone",
        "updated_at": "timestamp with time zone",
    }
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            # Get standard columns from information_schema
            cur.execute(
                """
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'contextual_chunks' AND table_schema = 'public'
                ORDER BY ordinal_position
                """
            )
            columns = {row[0]: row[1] for row in cur.fetchall()}
            
            for col_name, expected_type in expected_columns.items():
                assert col_name in columns, f"Column '{col_name}' not found in contextual_chunks"
                assert columns[col_name] == expected_type, (
                    f"Column '{col_name}' has type '{columns[col_name]}', expected '{expected_type}'"
                )
            
            # Check embedding column separately using pg_type (vector shows as USER-DEFINED in information_schema)
            cur.execute(
                """
                SELECT pg_catalog.format_type(a.atttypid, a.atttypmod)
                FROM pg_catalog.pg_attribute a
                JOIN pg_catalog.pg_class c ON a.attrelid = c.oid
                JOIN pg_catalog.pg_namespace n ON c.relnamespace = n.oid
                WHERE c.relname = 'contextual_chunks'
                AND n.nspname = 'public'
                AND a.attname = 'embedding'
                AND a.attnum > 0
                AND NOT a.attisdropped
                """
            )
            result = cur.fetchone()
            assert result is not None, "Column 'embedding' not found in contextual_chunks"
            embedding_type = result[0]
            assert embedding_type.startswith('vector'), (
                f"Column 'embedding' has type '{embedding_type}', expected 'vector'"
            )


def test_contextual_chunks_constraints():
    """Test that contextual_chunks has correct constraints."""
    dsn = load_settings()["DATABASE_URL"]
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            # Check primary key constraint
            cur.execute(
                """
                SELECT COUNT(*) FROM information_schema.table_constraints 
                WHERE table_name = 'contextual_chunks' 
                AND constraint_type = 'PRIMARY KEY'
                AND constraint_name = 'contextual_chunks_pkey'
                """
            )
            assert cur.fetchone()[0] == 1, "Primary key constraint not found"
            
            # Check foreign key constraints
            cur.execute(
                """
                SELECT COUNT(*) FROM information_schema.table_constraints 
                WHERE table_name = 'contextual_chunks' 
                AND constraint_type = 'FOREIGN KEY'
                """
            )
            assert cur.fetchone()[0] >= 2, "Expected at least 2 foreign key constraints (chunk_id, version_id)"
            
            # Check unique constraint on chunk_id
            cur.execute(
                """
                SELECT COUNT(*) FROM information_schema.table_constraints 
                WHERE table_name = 'contextual_chunks' 
                AND constraint_type = 'UNIQUE'
                """
            )
            assert cur.fetchone()[0] >= 1, "Expected at least 1 unique constraint on chunk_id"


def test_contextual_chunks_trigger():
    """Test that contextual_chunks has the updated_at trigger."""
    dsn = load_settings()["DATABASE_URL"]
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*) FROM pg_trigger 
                WHERE tgname = 'trigger_contextual_chunks_updated_at'
                AND tgrelid = 'contextual_chunks'::regclass
                """
            )
            assert cur.fetchone()[0] == 1, "Updated_at trigger not found"


def test_search_contextual_chunks_function():
    """Test that search_contextual_chunks_by_vector function exists."""
    dsn = load_settings()["DATABASE_URL"]
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*) FROM pg_proc 
                WHERE proname = 'search_contextual_chunks_by_vector'
                """
            )
            assert cur.fetchone()[0] == 1, "search_contextual_chunks_by_vector function not found"
