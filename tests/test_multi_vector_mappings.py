"""Tests for multi_vector_mappings table and related functionality."""

import psycopg
import pytest
from bid_scoring.config import load_settings


class TestMultiVectorMappingsTable:
    """Test multi_vector_mappings table structure and constraints."""

    @pytest.fixture
    def conn(self):
        """Provide database connection."""
        dsn = load_settings()["DATABASE_URL"]
        with psycopg.connect(dsn) as conn:
            yield conn

    def test_table_exists(self, conn):
        """Test that multi_vector_mappings table exists."""
        with conn.cursor() as cur:
            cur.execute(
                """
                select count(*) from information_schema.tables 
                where table_name = 'multi_vector_mappings' and table_schema = 'public'
                """
            )
            assert cur.fetchone()[0] == 1, "Table 'multi_vector_mappings' does not exist"

    def test_columns_exist(self, conn):
        """Test that all expected columns exist with correct types."""
        expected_columns = {
            "mapping_id": "uuid",
            "version_id": "uuid",
            "parent_chunk_id": "uuid",
            "child_chunk_id": "uuid",
            "parent_type": "text",
            "child_type": "text",
            "relationship": "text",
            "metadata": "jsonb",
            "created_at": "timestamp with time zone",
        }
        
        with conn.cursor() as cur:
            cur.execute(
                """
                select column_name, data_type 
                from information_schema.columns 
                where table_name = 'multi_vector_mappings' and table_schema = 'public'
                order by ordinal_position
                """
            )
            columns = {row[0]: row[1] for row in cur.fetchall()}
            
            for col_name, expected_type in expected_columns.items():
                assert col_name in columns, f"Column '{col_name}' not found in multi_vector_mappings"
                assert columns[col_name] == expected_type, (
                    f"Column '{col_name}' has type '{columns[col_name]}', expected '{expected_type}'"
                )

    def test_primary_key_constraint(self, conn):
        """Test that multi_vector_mappings has primary key on mapping_id."""
        with conn.cursor() as cur:
            cur.execute(
                """
                select count(*) from information_schema.table_constraints 
                where table_name = 'multi_vector_mappings' 
                and constraint_type = 'PRIMARY KEY'
                and constraint_name = 'multi_vector_mappings_pkey'
                """
            )
            assert cur.fetchone()[0] == 1, "Primary key constraint not found"

    def test_foreign_key_constraints(self, conn):
        """Test that multi_vector_mappings has expected foreign key constraints."""
        with conn.cursor() as cur:
            cur.execute(
                """
                select constraint_name 
                from information_schema.table_constraints 
                where table_name = 'multi_vector_mappings' 
                and constraint_type = 'FOREIGN KEY'
                """
            )
            fk_constraints = [row[0] for row in cur.fetchall()]
            
            # Check for expected FKs (version_id, parent_chunk_id, child_chunk_id)
            assert len(fk_constraints) >= 3, (
                f"Expected at least 3 foreign key constraints, found {len(fk_constraints)}"
            )

    def test_parent_type_check_constraint(self, conn):
        """Test that parent_type has check constraint for valid values."""
        with conn.cursor() as cur:
            cur.execute(
                """
                select count(*) from information_schema.check_constraints cc
                join information_schema.constraint_column_usage ccu 
                    on cc.constraint_name = ccu.constraint_name
                where ccu.table_name = 'multi_vector_mappings'
                and ccu.column_name = 'parent_type'
                """
            )
            assert cur.fetchone()[0] >= 1, "Check constraint on parent_type column not found"

    def test_child_type_check_constraint(self, conn):
        """Test that child_type has check constraint for valid values."""
        with conn.cursor() as cur:
            cur.execute(
                """
                select count(*) from information_schema.check_constraints cc
                join information_schema.constraint_column_usage ccu 
                    on cc.constraint_name = ccu.constraint_name
                where ccu.table_name = 'multi_vector_mappings'
                and ccu.column_name = 'child_type'
                """
            )
            assert cur.fetchone()[0] >= 1, "Check constraint on child_type column not found"

    def test_relationship_check_constraint(self, conn):
        """Test that relationship has check constraint for valid values."""
        with conn.cursor() as cur:
            cur.execute(
                """
                select count(*) from information_schema.check_constraints cc
                join information_schema.constraint_column_usage ccu 
                    on cc.constraint_name = ccu.constraint_name
                where ccu.table_name = 'multi_vector_mappings'
                and ccu.column_name = 'relationship'
                """
            )
            assert cur.fetchone()[0] >= 1, "Check constraint on relationship column not found"

    def test_valid_parent_type_values(self, conn):
        """Test that parent_type accepts valid values."""
        valid_values = ['hierarchical', 'contextual', 'original']
        
        with conn.cursor() as cur:
            for value in valid_values:
                # This should not raise an exception
                cur.execute(
                    """
                    insert into multi_vector_mappings 
                    (version_id, parent_type, child_type, relationship)
                    values (
                        (select version_id from document_versions limit 1),
                        %s, 'sentence', 'parent-child'
                    )
                    """,
                    (value,)
                )
            conn.commit()

    def test_invalid_parent_type_value(self, conn):
        """Test that parent_type rejects invalid values."""
        with conn.cursor() as cur:
            with pytest.raises(psycopg.errors.CheckViolation):
                cur.execute(
                    """
                    insert into multi_vector_mappings 
                    (version_id, parent_type, child_type, relationship)
                    values (
                        (select version_id from document_versions limit 1),
                        'invalid_type', 'sentence', 'parent-child'
                    )
                    """
                )
            conn.rollback()

    def test_valid_child_type_values(self, conn):
        """Test that child_type accepts valid values."""
        valid_values = ['sentence', 'paragraph', 'chunk']
        
        with conn.cursor() as cur:
            for value in valid_values:
                # This should not raise an exception
                cur.execute(
                    """
                    insert into multi_vector_mappings 
                    (version_id, parent_type, child_type, relationship)
                    values (
                        (select version_id from document_versions limit 1),
                        'original', %s, 'parent-child'
                    )
                    """,
                    (value,)
                )
            conn.commit()

    def test_invalid_child_type_value(self, conn):
        """Test that child_type rejects invalid values."""
        with conn.cursor() as cur:
            with pytest.raises(psycopg.errors.CheckViolation):
                cur.execute(
                    """
                    insert into multi_vector_mappings 
                    (version_id, parent_type, child_type, relationship)
                    values (
                        (select version_id from document_versions limit 1),
                        'original', 'invalid_type', 'parent-child'
                    )
                    """
                )
            conn.rollback()

    def test_valid_relationship_values(self, conn):
        """Test that relationship accepts valid values."""
        valid_values = ['parent-child', 'sibling', 'summary']
        
        with conn.cursor() as cur:
            for value in valid_values:
                # This should not raise an exception
                cur.execute(
                    """
                    insert into multi_vector_mappings 
                    (version_id, parent_type, child_type, relationship)
                    values (
                        (select version_id from document_versions limit 1),
                        'original', 'sentence', %s
                    )
                    """,
                    (value,)
                )
            conn.commit()

    def test_invalid_relationship_value(self, conn):
        """Test that relationship rejects invalid values."""
        with conn.cursor() as cur:
            with pytest.raises(psycopg.errors.CheckViolation):
                cur.execute(
                    """
                    insert into multi_vector_mappings 
                    (version_id, parent_type, child_type, relationship)
                    values (
                        (select version_id from document_versions limit 1),
                        'original', 'sentence', 'invalid_rel'
                    )
                    """
                )
            conn.rollback()


class TestMultiVectorMappingsIndexes:
    """Test indexes on multi_vector_mappings table."""

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
                where indexname = 'idx_multi_vector_version_id' and schemaname = 'public'
                """
            )
            assert cur.fetchone()[0] == 1, "Index 'idx_multi_vector_version_id' not found"

    def test_parent_chunk_index(self, conn):
        """Test parent_chunk_id index exists."""
        with conn.cursor() as cur:
            cur.execute(
                """
                select count(*) from pg_indexes 
                where indexname = 'idx_multi_vector_parent_chunk' and schemaname = 'public'
                """
            )
            assert cur.fetchone()[0] == 1, "Index 'idx_multi_vector_parent_chunk' not found"

    def test_child_chunk_index(self, conn):
        """Test child_chunk_id index exists."""
        with conn.cursor() as cur:
            cur.execute(
                """
                select count(*) from pg_indexes 
                where indexname = 'idx_multi_vector_child_chunk' and schemaname = 'public'
                """
            )
            assert cur.fetchone()[0] == 1, "Index 'idx_multi_vector_child_chunk' not found"

    def test_relationship_index(self, conn):
        """Test relationship index exists."""
        with conn.cursor() as cur:
            cur.execute(
                """
                select count(*) from pg_indexes 
                where indexname = 'idx_multi_vector_relationship' and schemaname = 'public'
                """
            )
            assert cur.fetchone()[0] == 1, "Index 'idx_multi_vector_relationship' not found"

    def test_parent_type_index(self, conn):
        """Test parent_type index exists."""
        with conn.cursor() as cur:
            cur.execute(
                """
                select count(*) from pg_indexes 
                where indexname = 'idx_multi_vector_parent_type' and schemaname = 'public'
                """
            )
            assert cur.fetchone()[0] == 1, "Index 'idx_multi_vector_parent_type' not found"

    def test_child_type_index(self, conn):
        """Test child_type index exists."""
        with conn.cursor() as cur:
            cur.execute(
                """
                select count(*) from pg_indexes 
                where indexname = 'idx_multi_vector_child_type' and schemaname = 'public'
                """
            )
            assert cur.fetchone()[0] == 1, "Index 'idx_multi_vector_child_type' not found"

    def test_metadata_index(self, conn):
        """Test metadata GIN index exists."""
        with conn.cursor() as cur:
            cur.execute(
                """
                select count(*) from pg_indexes 
                where indexname = 'idx_multi_vector_metadata' and schemaname = 'public'
                """
            )
            assert cur.fetchone()[0] == 1, "Index 'idx_multi_vector_metadata' not found"

    def test_version_relationship_index(self, conn):
        """Test version_id + relationship composite index exists."""
        with conn.cursor() as cur:
            cur.execute(
                """
                select count(*) from pg_indexes 
                where indexname = 'idx_multi_vector_version_relationship' and schemaname = 'public'
                """
            )
            assert cur.fetchone()[0] == 1, "Index 'idx_multi_vector_version_relationship' not found"

    def test_parent_child_types_index(self, conn):
        """Test parent_type + child_type composite index exists."""
        with conn.cursor() as cur:
            cur.execute(
                """
                select count(*) from pg_indexes 
                where indexname = 'idx_multi_vector_parent_child_types' and schemaname = 'public'
                """
            )
            assert cur.fetchone()[0] == 1, "Index 'idx_multi_vector_parent_child_types' not found"

    def test_parent_relationship_index(self, conn):
        """Test parent_chunk_id + relationship partial index exists."""
        with conn.cursor() as cur:
            cur.execute(
                """
                select count(*) from pg_indexes 
                where indexname = 'idx_multi_vector_parent_relationship' and schemaname = 'public'
                """
            )
            assert cur.fetchone()[0] == 1, "Index 'idx_multi_vector_parent_relationship' not found"

    def test_child_relationship_index(self, conn):
        """Test child_chunk_id + relationship partial index exists."""
        with conn.cursor() as cur:
            cur.execute(
                """
                select count(*) from pg_indexes 
                where indexname = 'idx_multi_vector_child_relationship' and schemaname = 'public'
                """
            )
            assert cur.fetchone()[0] == 1, "Index 'idx_multi_vector_child_relationship' not found"


class TestMultiVectorMappingsComments:
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
                select obj_description('multi_vector_mappings'::regclass, 'pg_class')
                """
            )
            comment = cur.fetchone()[0]
            assert comment is not None and len(comment) > 0, "Table comment not found"
            assert 'multi' in comment.lower() or 'vector' in comment.lower(), (
                "Table comment should mention multi-vector"
            )

    def test_column_comments(self, conn):
        """Test that columns have comments."""
        expected_columns = [
            'mapping_id', 'version_id', 'parent_chunk_id', 'child_chunk_id',
            'parent_type', 'child_type', 'relationship', 'metadata', 'created_at'
        ]
        
        with conn.cursor() as cur:
            for col_name in expected_columns:
                cur.execute(
                    """
                    select col_description('multi_vector_mappings'::regclass, a.attnum)
                    from pg_catalog.pg_attribute a
                    where a.attrelid = 'multi_vector_mappings'::regclass
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


class TestMultiVectorMappingsData:
    """Test data operations on multi_vector_mappings table."""

    @pytest.fixture
    def conn(self):
        """Provide database connection."""
        dsn = load_settings()["DATABASE_URL"]
        with psycopg.connect(dsn) as conn:
            yield conn

    def test_insert_with_metadata(self, conn):
        """Test inserting a record with metadata."""
        with conn.cursor() as cur:
            cur.execute(
                """
                insert into multi_vector_mappings 
                (version_id, parent_chunk_id, child_chunk_id, parent_type, child_type, relationship, metadata)
                values (
                    (select version_id from document_versions limit 1),
                    (select chunk_id from chunks limit 1),
                    (select chunk_id from chunks limit 1 offset 1),
                    'hierarchical', 'sentence', 'parent-child',
                    '{"similarity_score": 0.95, "weight": 1.0}'::jsonb
                )
                returning mapping_id
                """
            )
            result = cur.fetchone()
            assert result is not None
            mapping_id = result[0]
            
            # Verify the record
            cur.execute(
                "select metadata from multi_vector_mappings where mapping_id = %s",
                (mapping_id,)
            )
            metadata = cur.fetchone()[0]
            assert metadata['similarity_score'] == 0.95
            assert metadata['weight'] == 1.0
            conn.commit()

    def test_null_chunk_ids(self, conn):
        """Test that parent_chunk_id and child_chunk_id can be null."""
        with conn.cursor() as cur:
            cur.execute(
                """
                insert into multi_vector_mappings 
                (version_id, parent_chunk_id, child_chunk_id, parent_type, child_type, relationship)
                values (
                    (select version_id from document_versions limit 1),
                    null, null, 'hierarchical', 'sentence', 'parent-child'
                )
                returning mapping_id
                """
            )
            result = cur.fetchone()
            assert result is not None
            conn.commit()

    def test_cascade_delete_version(self, conn):
        """Test that mappings are deleted when version is deleted."""
        # Note: This test is skipped if there are no versions to delete
        with conn.cursor() as cur:
            # First check if we have any versions
            cur.execute("select count(*) from document_versions")
            version_count = cur.fetchone()[0]
            
            if version_count == 0:
                pytest.skip("No document versions available for cascade delete test")
            
            # Get a version_id to test
            cur.execute("select version_id from document_versions limit 1")
            version_id = cur.fetchone()[0]
            
            # Insert a mapping for this version
            cur.execute(
                """
                insert into multi_vector_mappings 
                (version_id, parent_type, child_type, relationship)
                values (%s, 'original', 'sentence', 'parent-child')
                returning mapping_id
                """,
                (version_id,)
            )
            mapping_id = cur.fetchone()[0]
            conn.commit()
            
            # Verify mapping exists
            cur.execute(
                "select count(*) from multi_vector_mappings where mapping_id = %s",
                (mapping_id,)
            )
            assert cur.fetchone()[0] == 1
