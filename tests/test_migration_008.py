"""Test migration 008: CPC Structure Rebuild"""

import psycopg
from bid_scoring.config import load_settings


def test_migration_008_columns_added():
    """测试新列已添加到 hierarchical_nodes"""
    settings = load_settings()
    
    with psycopg.connect(settings["DATABASE_URL"]) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'hierarchical_nodes'
                AND column_name IN ('heading', 'context', 'merged_chunk_count')
            """)
            columns = [row[0] for row in cur.fetchall()]
            
            assert 'heading' in columns
            assert 'context' in columns
            assert 'merged_chunk_count' in columns


def test_structure_build_status_table_exists():
    """测试状态跟踪表已创建"""
    settings = load_settings()
    
    with psycopg.connect(settings["DATABASE_URL"]) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables 
                    WHERE table_name = 'document_structure_build_status'
                )
            """)
            assert cur.fetchone()[0] is True


def test_structure_build_status_columns():
    """测试状态跟踪表有正确的列"""
    settings = load_settings()
    
    with psycopg.connect(settings["DATABASE_URL"]) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'document_structure_build_status'
                ORDER BY ordinal_position
            """)
            columns = [row[0] for row in cur.fetchall()]
            
            expected = [
                'status_id', 'version_id', 'structure_built', 
                'context_generated', 'embeddings_generated', 'raptor_built',
                'original_chunk_count', 'paragraph_count', 'section_count',
                'llm_call_count', 'errors', 'created_at', 'updated_at'
            ]
            
            for col in expected:
                assert col in columns, f"Missing column: {col}"
