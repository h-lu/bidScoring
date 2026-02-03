"""Tests for hierarchical_nodes table and related functions."""

import psycopg
import pytest
from bid_scoring.config import load_settings


class TestHierarchicalNodesTable:
    """Test hierarchical_nodes table structure and constraints."""

    @pytest.fixture
    def conn(self):
        """Provide database connection."""
        dsn = load_settings()["DATABASE_URL"]
        with psycopg.connect(dsn) as conn:
            yield conn

    def test_table_exists(self, conn):
        """Test that hierarchical_nodes table exists."""
        with conn.cursor() as cur:
            cur.execute(
                """
                select count(*) from information_schema.tables 
                where table_name = 'hierarchical_nodes' and table_schema = 'public'
                """
            )
            assert cur.fetchone()[0] == 1, "Table 'hierarchical_nodes' does not exist"

    def test_columns_exist(self, conn):
        """Test that all expected columns exist with correct types."""
        expected_columns = {
            "node_id": "uuid",
            "version_id": "uuid",
            "parent_id": "uuid",
            "level": "integer",
            "node_type": "text",
            "content": "text",
            "children_ids": "array",
            "start_chunk_id": "uuid",
            "end_chunk_id": "uuid",
            "metadata": "jsonb",
            "created_at": "timestamp with time zone",
            "updated_at": "timestamp with time zone",
        }
        
        with conn.cursor() as cur:
            # Check standard columns
            cur.execute(
                """
                select column_name, data_type 
                from information_schema.columns 
                where table_name = 'hierarchical_nodes' and table_schema = 'public'
                order by ordinal_position
                """
            )
            columns = {row[0]: row[1] for row in cur.fetchall()}
            
            for col_name, expected_type in expected_columns.items():
                assert col_name in columns, f"Column '{col_name}' not found in hierarchical_nodes"
                if expected_type == "array":
                    # Arrays show as 'ARRAY' in information_schema (case insensitive)
                    assert "array" in columns[col_name].lower(), (
                        f"Column '{col_name}' has type '{columns[col_name]}', expected 'array'"
                    )
                else:
                    assert columns[col_name] == expected_type, (
                        f"Column '{col_name}' has type '{columns[col_name]}', expected '{expected_type}'"
                    )

    def test_embedding_column_type(self, conn):
        """Test that embedding column has vector type."""
        with conn.cursor() as cur:
            cur.execute(
                """
                select pg_catalog.format_type(a.atttypid, a.atttypmod)
                from pg_catalog.pg_attribute a
                join pg_catalog.pg_class c on a.attrelid = c.oid
                join pg_catalog.pg_namespace n on c.relnamespace = n.oid
                where c.relname = 'hierarchical_nodes'
                and n.nspname = 'public'
                and a.attname = 'embedding'
                and a.attnum > 0
                and not a.attisdropped
                """
            )
            result = cur.fetchone()
            assert result is not None, "Column 'embedding' not found in hierarchical_nodes"
            embedding_type = result[0]
            assert embedding_type.startswith('vector'), (
                f"Column 'embedding' has type '{embedding_type}', expected 'vector'"
            )

    def test_children_ids_column_type(self, conn):
        """Test that children_ids column has uuid[] type."""
        with conn.cursor() as cur:
            cur.execute(
                """
                select pg_catalog.format_type(a.atttypid, a.atttypmod)
                from pg_catalog.pg_attribute a
                join pg_catalog.pg_class c on a.attrelid = c.oid
                join pg_catalog.pg_namespace n on c.relnamespace = n.oid
                where c.relname = 'hierarchical_nodes'
                and n.nspname = 'public'
                and a.attname = 'children_ids'
                and a.attnum > 0
                and not a.attisdropped
                """
            )
            result = cur.fetchone()
            assert result is not None, "Column 'children_ids' not found in hierarchical_nodes"
            column_type = result[0]
            assert column_type == 'uuid[]', (
                f"Column 'children_ids' has type '{column_type}', expected 'uuid[]'"
            )

    def test_primary_key_constraint(self, conn):
        """Test that hierarchical_nodes has primary key on node_id."""
        with conn.cursor() as cur:
            cur.execute(
                """
                select count(*) from information_schema.table_constraints 
                where table_name = 'hierarchical_nodes' 
                and constraint_type = 'PRIMARY KEY'
                and constraint_name = 'hierarchical_nodes_pkey'
                """
            )
            assert cur.fetchone()[0] == 1, "Primary key constraint not found"

    def test_foreign_key_constraints(self, conn):
        """Test that hierarchical_nodes has expected foreign key constraints."""
        with conn.cursor() as cur:
            cur.execute(
                """
                select constraint_name 
                from information_schema.table_constraints 
                where table_name = 'hierarchical_nodes' 
                and constraint_type = 'FOREIGN KEY'
                """
            )
            fk_constraints = [row[0] for row in cur.fetchall()]
            
            # Check for expected FKs (names may vary by PostgreSQL version)
            assert len(fk_constraints) >= 3, (
                f"Expected at least 3 foreign key constraints, found {len(fk_constraints)}"
            )

    def test_level_check_constraint(self, conn):
        """Test that level column has check constraint (0-3)."""
        with conn.cursor() as cur:
            cur.execute(
                """
                select count(*) from information_schema.check_constraints cc
                join information_schema.constraint_column_usage ccu 
                    on cc.constraint_name = ccu.constraint_name
                where ccu.table_name = 'hierarchical_nodes'
                and ccu.column_name = 'level'
                """
            )
            assert cur.fetchone()[0] >= 1, "Check constraint on level column not found"

    def test_node_type_check_constraint(self, conn):
        """Test that node_type has check constraint for valid values."""
        with conn.cursor() as cur:
            cur.execute(
                """
                select count(*) from information_schema.check_constraints cc
                join information_schema.constraint_column_usage ccu 
                    on cc.constraint_name = ccu.constraint_name
                where ccu.table_name = 'hierarchical_nodes'
                and ccu.column_name = 'node_type'
                """
            )
            assert cur.fetchone()[0] >= 1, "Check constraint on node_type column not found"


class TestHierarchicalNodesIndexes:
    """Test indexes on hierarchical_nodes table."""

    @pytest.fixture
    def conn(self):
        """Provide database connection."""
        dsn = load_settings()["DATABASE_URL"]
        with psycopg.connect(dsn) as conn:
            yield conn

    def test_version_id_index(self, conn):
        """Test version_id index exists."""
        with conn.cursor() as cur:
            cur.execute(
                """
                select count(*) from pg_indexes 
                where indexname = 'idx_hierarchical_nodes_version_id' and schemaname = 'public'
                """
            )
            assert cur.fetchone()[0] == 1, "Index 'idx_hierarchical_nodes_version_id' not found"

    def test_parent_id_index(self, conn):
        """Test parent_id index exists."""
        with conn.cursor() as cur:
            cur.execute(
                """
                select count(*) from pg_indexes 
                where indexname = 'idx_hierarchical_nodes_parent_id' and schemaname = 'public'
                """
            )
            assert cur.fetchone()[0] == 1, "Index 'idx_hierarchical_nodes_parent_id' not found"

    def test_version_level_index(self, conn):
        """Test version_id + level composite index exists."""
        with conn.cursor() as cur:
            cur.execute(
                """
                select count(*) from pg_indexes 
                where indexname = 'idx_hierarchical_nodes_version_level' and schemaname = 'public'
                """
            )
            assert cur.fetchone()[0] == 1, "Index 'idx_hierarchical_nodes_version_level' not found"

    def test_node_type_index(self, conn):
        """Test node_type index exists."""
        with conn.cursor() as cur:
            cur.execute(
                """
                select count(*) from pg_indexes 
                where indexname = 'idx_hierarchical_nodes_type' and schemaname = 'public'
                """
            )
            assert cur.fetchone()[0] == 1, "Index 'idx_hierarchical_nodes_type' not found"

    def test_start_chunk_index(self, conn):
        """Test start_chunk_id partial index exists."""
        with conn.cursor() as cur:
            cur.execute(
                """
                select count(*) from pg_indexes 
                where indexname = 'idx_hierarchical_nodes_start_chunk' and schemaname = 'public'
                """
            )
            assert cur.fetchone()[0] == 1, "Index 'idx_hierarchical_nodes_start_chunk' not found"

    def test_children_ids_index(self, conn):
        """Test children_ids GIN index exists."""
        with conn.cursor() as cur:
            cur.execute(
                """
                select count(*) from pg_indexes 
                where indexname = 'idx_hierarchical_nodes_children_ids' and schemaname = 'public'
                """
            )
            assert cur.fetchone()[0] == 1, "Index 'idx_hierarchical_nodes_children_ids' not found"

    def test_metadata_index(self, conn):
        """Test metadata GIN index exists."""
        with conn.cursor() as cur:
            cur.execute(
                """
                select count(*) from pg_indexes 
                where indexname = 'idx_hierarchical_nodes_metadata' and schemaname = 'public'
                """
            )
            assert cur.fetchone()[0] == 1, "Index 'idx_hierarchical_nodes_metadata' not found"

    def test_embedding_hnsw_index(self, conn):
        """Test embedding HNSW index exists."""
        with conn.cursor() as cur:
            cur.execute(
                """
                select count(*) from pg_indexes 
                where indexname = 'idx_hierarchical_nodes_embedding_hnsw' and schemaname = 'public'
                """
            )
            assert cur.fetchone()[0] == 1, "Index 'idx_hierarchical_nodes_embedding_hnsw' not found"

    def test_version_parent_index(self, conn):
        """Test version_id + parent_id composite index exists."""
        with conn.cursor() as cur:
            cur.execute(
                """
                select count(*) from pg_indexes 
                where indexname = 'idx_hierarchical_nodes_version_parent' and schemaname = 'public'
                """
            )
            assert cur.fetchone()[0] == 1, "Index 'idx_hierarchical_nodes_version_parent' not found"


class TestHierarchicalNodesTrigger:
    """Test trigger on hierarchical_nodes table."""

    @pytest.fixture
    def conn(self):
        """Provide database connection."""
        dsn = load_settings()["DATABASE_URL"]
        with psycopg.connect(dsn) as conn:
            yield conn

    def test_updated_at_trigger_exists(self, conn):
        """Test that updated_at trigger exists."""
        with conn.cursor() as cur:
            cur.execute(
                """
                select count(*) from pg_trigger 
                where tgname = 'trigger_hierarchical_nodes_updated_at'
                and tgrelid = 'hierarchical_nodes'::regclass
                """
            )
            assert cur.fetchone()[0] == 1, "Updated_at trigger not found"


class TestHierarchicalNodesFunctions:
    """Test helper functions for hierarchical_nodes."""

    @pytest.fixture
    def conn(self):
        """Provide database connection."""
        dsn = load_settings()["DATABASE_URL"]
        with psycopg.connect(dsn) as conn:
            yield conn

    def test_get_node_descendants_function(self, conn):
        """Test that get_node_descendants function exists."""
        with conn.cursor() as cur:
            cur.execute(
                "select count(*) from pg_proc where proname = 'get_node_descendants'"
            )
            assert cur.fetchone()[0] == 1, "get_node_descendants function not found"

    def test_get_node_ancestors_function(self, conn):
        """Test that get_node_ancestors function exists."""
        with conn.cursor() as cur:
            cur.execute(
                "select count(*) from pg_proc where proname = 'get_node_ancestors'"
            )
            assert cur.fetchone()[0] == 1, "get_node_ancestors function not found"

    def test_get_document_root_node_function(self, conn):
        """Test that get_document_root_node function exists."""
        with conn.cursor() as cur:
            cur.execute(
                "select count(*) from pg_proc where proname = 'get_document_root_node'"
            )
            assert cur.fetchone()[0] == 1, "get_document_root_node function not found"


class TestHierarchicalNodesComments:
    """Test that table and columns have comments."""

    @pytest.fixture
    def conn(self):
        """Provide database connection."""
        dsn = load_settings()["DATABASE_URL"]
        with psycopg.connect(dsn) as conn:
            yield conn

    def test_table_comment(self, conn):
        """Test that table has comment."""
        with conn.cursor() as cur:
            cur.execute(
                """
                select obj_description('hierarchical_nodes'::regclass, 'pg_class')
                """
            )
            comment = cur.fetchone()[0]
            assert comment is not None and len(comment) > 0, "Table comment not found"
            assert 'hichunk' in comment.lower() or 'tree' in comment.lower(), (
                "Table comment should mention HiChunk or tree"
            )

    def test_column_comments(self, conn):
        """Test that columns have comments."""
        expected_columns = [
            'node_id', 'version_id', 'parent_id', 'level', 'node_type',
            'content', 'children_ids', 'start_chunk_id', 'end_chunk_id',
            'metadata', 'embedding', 'created_at', 'updated_at'
        ]
        
        with conn.cursor() as cur:
            for col_name in expected_columns:
                cur.execute(
                    """
                    select col_description('hierarchical_nodes'::regclass, a.attnum)
                    from pg_catalog.pg_attribute a
                    where a.attrelid = 'hierarchical_nodes'::regclass
                    and a.attname = %s
                    and a.attnum > 0
                    and not a.attisdropped
                    """,
                    (col_name,)
                )
                result = cur.fetchone()
                assert result is not None and result[0] is not None, (
                    f"Comment not found for column '{col_name}'"
                )
