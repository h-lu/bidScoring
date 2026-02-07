-- Complete database schema for CI
-- Merged from all migration files

-- Extensions
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================
-- Core Tables (from 001_init.sql)
-- ============================================

CREATE TABLE IF NOT EXISTS projects (
    project_id UUID PRIMARY KEY,
    name TEXT NOT NULL,
    owner TEXT,
    status TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS documents (
    doc_id UUID PRIMARY KEY,
    project_id UUID REFERENCES projects(project_id),
    title TEXT,
    source_type TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS document_versions (
    version_id UUID PRIMARY KEY,
    doc_id UUID REFERENCES documents(doc_id),
    source_uri TEXT,
    source_hash TEXT,
    parser_version TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    status TEXT
);

-- ============================================
-- Normalized Layer (v0.2)
-- ============================================

CREATE TABLE IF NOT EXISTS document_pages (
    version_id UUID NOT NULL REFERENCES document_versions(version_id) ON DELETE CASCADE,
    page_idx INTEGER NOT NULL,
    page_w NUMERIC,
    page_h NUMERIC,
    coord_sys TEXT NOT NULL DEFAULT 'unknown',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (version_id, page_idx)
);

CREATE INDEX IF NOT EXISTS idx_document_pages_version_page
    ON document_pages(version_id, page_idx);

CREATE TABLE IF NOT EXISTS content_units (
    unit_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    version_id UUID NOT NULL REFERENCES document_versions(version_id) ON DELETE CASCADE,
    unit_index INTEGER NOT NULL,
    unit_type TEXT NOT NULL,
    text_raw TEXT,
    text_norm TEXT,
    char_count INTEGER DEFAULT 0,
    anchor_json JSONB NOT NULL DEFAULT '{"anchors":[]}'::jsonb,
    source_element_id TEXT,
    unit_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE content_units
    DROP CONSTRAINT IF EXISTS unique_units_version_unit_index;
ALTER TABLE content_units
    ADD CONSTRAINT unique_units_version_unit_index
    UNIQUE (version_id, unit_index);

ALTER TABLE content_units
    DROP CONSTRAINT IF EXISTS unique_units_version_source_element;
ALTER TABLE content_units
    ADD CONSTRAINT unique_units_version_source_element
    UNIQUE (version_id, source_element_id);

CREATE INDEX IF NOT EXISTS idx_content_units_version_unit_index
    ON content_units(version_id, unit_index);
CREATE INDEX IF NOT EXISTS idx_content_units_version_source_element
    ON content_units(version_id, source_element_id);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id UUID PRIMARY KEY,
    project_id UUID REFERENCES projects(project_id),
    version_id UUID REFERENCES document_versions(version_id),
    source_id TEXT,
    chunk_index INT,
    page_idx INT,
    bbox JSONB,
    element_type TEXT,
    text_raw TEXT,
    text_hash TEXT,
    text_tsv TSVECTOR,
    embedding VECTOR(1536)
);

-- Add MinerU fields (from 002_add_mineru_fields.sql)
ALTER TABLE chunks 
    ADD COLUMN IF NOT EXISTS img_path TEXT,
    ADD COLUMN IF NOT EXISTS image_caption TEXT[],
    ADD COLUMN IF NOT EXISTS image_footnote TEXT[],
    ADD COLUMN IF NOT EXISTS table_body TEXT,
    ADD COLUMN IF NOT EXISTS table_caption TEXT[],
    ADD COLUMN IF NOT EXISTS table_footnote TEXT[],
    ADD COLUMN IF NOT EXISTS list_items TEXT[],
    ADD COLUMN IF NOT EXISTS sub_type TEXT,
    ADD COLUMN IF NOT EXISTS text_level INTEGER;

-- Add unique constraint (from 009_fix_mineru_import.sql)
ALTER TABLE chunks 
    DROP CONSTRAINT IF EXISTS unique_version_source;
ALTER TABLE chunks 
    ADD CONSTRAINT unique_version_source 
    UNIQUE (version_id, source_id);

CREATE INDEX IF NOT EXISTS idx_chunks_text_level 
    ON chunks(version_id, text_level) 
    WHERE text_level IS NOT NULL;

CREATE TABLE IF NOT EXISTS chunk_unit_spans (
    chunk_id UUID NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    unit_id UUID NOT NULL REFERENCES content_units(unit_id) ON DELETE CASCADE,
    unit_order INTEGER NOT NULL,
    start_char INTEGER,
    end_char INTEGER,
    PRIMARY KEY (chunk_id, unit_id),
    UNIQUE (chunk_id, unit_order)
);

CREATE INDEX IF NOT EXISTS idx_chunk_unit_spans_chunk_id ON chunk_unit_spans(chunk_id);
CREATE INDEX IF NOT EXISTS idx_chunk_unit_spans_unit_id ON chunk_unit_spans(unit_id);

CREATE TABLE IF NOT EXISTS scoring_runs (
    run_id UUID PRIMARY KEY,
    project_id UUID REFERENCES projects(project_id),
    version_id UUID REFERENCES document_versions(version_id),
    dimensions TEXT[],
    model TEXT,
    rules_version TEXT,
    params_hash TEXT,
    started_at TIMESTAMPTZ DEFAULT NOW(),
    status TEXT
);

CREATE TABLE IF NOT EXISTS scoring_results (
    result_id UUID PRIMARY KEY,
    run_id UUID REFERENCES scoring_runs(run_id),
    dimension TEXT,
    score NUMERIC,
    max_score NUMERIC,
    reasoning TEXT,
    evidence_found BOOLEAN,
    confidence TEXT
);

CREATE TABLE IF NOT EXISTS citations (
    citation_id UUID PRIMARY KEY,
    result_id UUID REFERENCES scoring_results(result_id),
    source_id TEXT,
    chunk_id UUID REFERENCES chunks(chunk_id) ON DELETE SET NULL,
    cited_text TEXT,
    unit_id UUID REFERENCES content_units(unit_id),
    quote_text TEXT,
    quote_start_char INTEGER,
    quote_end_char INTEGER,
    anchor_json JSONB,
    evidence_hash TEXT,
    verified BOOLEAN,
    match_type TEXT
);

ALTER TABLE citations
    DROP CONSTRAINT IF EXISTS citations_chunk_id_fkey;
ALTER TABLE citations
    ADD CONSTRAINT citations_chunk_id_fkey
    FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id) ON DELETE SET NULL;

ALTER TABLE citations
    ADD COLUMN IF NOT EXISTS unit_id UUID REFERENCES content_units(unit_id),
    ADD COLUMN IF NOT EXISTS quote_text TEXT,
    ADD COLUMN IF NOT EXISTS quote_start_char INTEGER,
    ADD COLUMN IF NOT EXISTS quote_end_char INTEGER,
    ADD COLUMN IF NOT EXISTS anchor_json JSONB,
    ADD COLUMN IF NOT EXISTS evidence_hash TEXT;

-- ============================================
-- Contextual Chunks (from 005_cpc_contextual_chunks.sql)
-- ============================================

CREATE TABLE IF NOT EXISTS contextual_chunks (
    contextual_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_id UUID NOT NULL UNIQUE REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    version_id UUID NOT NULL REFERENCES document_versions(version_id) ON DELETE CASCADE,
    original_text TEXT NOT NULL,
    context_prefix TEXT NOT NULL,
    contextualized_text TEXT NOT NULL,
    embedding VECTOR(1536),
    model_name TEXT NOT NULL,
    embedding_model TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_contextual_chunks_chunk_id ON contextual_chunks(chunk_id);
CREATE INDEX IF NOT EXISTS idx_contextual_chunks_version_id ON contextual_chunks(version_id);
CREATE INDEX IF NOT EXISTS idx_contextual_chunks_embedding_hnsw 
ON contextual_chunks USING hnsw(embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS idx_contextual_chunks_version_created 
ON contextual_chunks(version_id, created_at);

-- ============================================
-- Hierarchical Nodes (from 006_cpc_hierarchical_nodes.sql + 009_add_source_chunk_ids.sql + 010_small_to_big_chunking.sql)
-- ============================================

CREATE TABLE IF NOT EXISTS hierarchical_nodes (
    node_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    version_id UUID NOT NULL REFERENCES document_versions(version_id) ON DELETE CASCADE,
    parent_id UUID REFERENCES hierarchical_nodes(node_id) ON DELETE CASCADE,
    level INTEGER NOT NULL CHECK (level >= 0 AND level <= 3),
    node_type TEXT NOT NULL CHECK (node_type IN ('sentence', 'paragraph', 'section', 'document', 'chunk')),
    content TEXT NOT NULL,
    children_ids UUID[] DEFAULT '{}',
    source_chunk_ids UUID[] DEFAULT '{}',
    start_chunk_id UUID REFERENCES chunks(chunk_id) ON DELETE SET NULL,
    end_chunk_id UUID REFERENCES chunks(chunk_id) ON DELETE SET NULL,
    metadata JSONB DEFAULT '{}',
    embedding VECTOR(1536),
    content_for_embedding TEXT,
    char_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_hierarchical_nodes_version_id ON hierarchical_nodes(version_id);
CREATE INDEX IF NOT EXISTS idx_hierarchical_nodes_parent_id ON hierarchical_nodes(parent_id);
CREATE INDEX IF NOT EXISTS idx_hierarchical_nodes_version_level ON hierarchical_nodes(version_id, level);
CREATE INDEX IF NOT EXISTS idx_hierarchical_nodes_type ON hierarchical_nodes(node_type);
CREATE INDEX IF NOT EXISTS idx_hierarchical_nodes_start_chunk ON hierarchical_nodes(start_chunk_id) WHERE start_chunk_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_hierarchical_nodes_children_ids ON hierarchical_nodes USING gin(children_ids);
CREATE INDEX IF NOT EXISTS idx_hierarchical_nodes_source_chunks ON hierarchical_nodes USING gin(source_chunk_ids);
CREATE INDEX IF NOT EXISTS idx_hierarchical_nodes_metadata ON hierarchical_nodes USING gin(metadata);
CREATE INDEX IF NOT EXISTS idx_hierarchical_nodes_embedding_hnsw ON hierarchical_nodes 
USING hnsw(embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64) WHERE embedding IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_hierarchical_nodes_version_parent ON hierarchical_nodes(version_id, parent_id, level);

-- ============================================
-- Indexes (from 001_init.sql)
-- ============================================

CREATE INDEX IF NOT EXISTS idx_chunks_text_tsv ON chunks USING gin(text_tsv);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw ON chunks USING hnsw(embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_chunks_project_version_page ON chunks(project_id, version_id, page_idx);

-- ============================================
-- Functions and Triggers
-- ============================================

-- Trigger function for updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE 'plpgsql';

-- Apply triggers
DROP TRIGGER IF EXISTS trigger_contextual_chunks_updated_at ON contextual_chunks;
CREATE TRIGGER trigger_contextual_chunks_updated_at
    BEFORE UPDATE ON contextual_chunks
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS trigger_content_units_updated_at ON content_units;
CREATE TRIGGER trigger_content_units_updated_at
    BEFORE UPDATE ON content_units
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS trigger_hierarchical_nodes_updated_at ON hierarchical_nodes;
CREATE TRIGGER trigger_hierarchical_nodes_updated_at
    BEFORE UPDATE ON hierarchical_nodes
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Hierarchical node functions
CREATE OR REPLACE FUNCTION get_node_descendants(p_node_id UUID)
RETURNS TABLE (
    node_id UUID,
    parent_id UUID,
    level INTEGER,
    node_type TEXT,
    content TEXT,
    depth INTEGER
) AS $$
BEGIN
    RETURN QUERY
    WITH RECURSIVE descendants AS (
        SELECT 
            hn.node_id,
            hn.parent_id,
            hn.level,
            hn.node_type,
            hn.content,
            0 AS depth
        FROM hierarchical_nodes hn
        WHERE hn.node_id = p_node_id
        
        UNION ALL
        
        SELECT 
            hn.node_id,
            hn.parent_id,
            hn.level,
            hn.node_type,
            hn.content,
            d.depth + 1
        FROM hierarchical_nodes hn
        JOIN descendants d ON hn.parent_id = d.node_id
    )
    SELECT * FROM descendants;
END;
$$ LANGUAGE plpgsql STABLE;

CREATE OR REPLACE FUNCTION get_node_ancestors(p_node_id UUID)
RETURNS TABLE (
    node_id UUID,
    parent_id UUID,
    level INTEGER,
    node_type TEXT,
    content TEXT,
    depth INTEGER
) AS $$
BEGIN
    RETURN QUERY
    WITH RECURSIVE ancestors AS (
        SELECT 
            hn.node_id,
            hn.parent_id,
            hn.level,
            hn.node_type,
            hn.content,
            0 AS depth
        FROM hierarchical_nodes hn
        WHERE hn.node_id = p_node_id
        
        UNION ALL
        
        SELECT 
            hn.node_id,
            hn.parent_id,
            hn.level,
            hn.node_type,
            hn.content,
            a.depth - 1
        FROM hierarchical_nodes hn
        JOIN ancestors a ON hn.node_id = a.parent_id
    )
    SELECT * FROM ancestors ORDER BY depth;
END;
$$ LANGUAGE plpgsql STABLE;

CREATE OR REPLACE FUNCTION get_document_root_node(p_version_id UUID)
RETURNS TABLE (
    node_id UUID,
    content TEXT,
    metadata JSONB
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        hn.node_id,
        hn.content,
        hn.metadata
    FROM hierarchical_nodes hn
    WHERE hn.version_id = p_version_id
      AND hn.level = 3
      AND hn.node_type = 'document';
END;
$$ LANGUAGE plpgsql STABLE;

-- Vector search functions (from 003_optimize_pgvector_index.sql)
CREATE OR REPLACE FUNCTION get_recommended_ef_search(chunk_count BIGINT DEFAULT NULL)
RETURNS INTEGER AS $$
DECLARE
    count_val BIGINT;
BEGIN
    IF chunk_count IS NULL THEN
        SELECT COUNT(*) INTO count_val FROM chunks WHERE embedding IS NOT NULL;
    ELSE
        count_val := chunk_count;
    END IF;
    
    IF count_val < 10000 THEN
        RETURN 64;
    ELSIF count_val < 100000 THEN
        RETURN 100;
    ELSIF count_val < 1000000 THEN
        RETURN 150;
    ELSE
        RETURN 200;
    END IF;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION search_chunks_by_vector(
    query_vector VECTOR,
    p_version_id UUID,
    top_k INTEGER DEFAULT 10,
    min_similarity FLOAT DEFAULT 0.0
)
RETURNS TABLE (
    chunk_id UUID,
    source_id TEXT,
    page_idx INTEGER,
    element_type TEXT,
    text_raw TEXT,
    similarity FLOAT
) AS $$
DECLARE
    ef_val INTEGER;
BEGIN
    ef_val := GREATEST(get_recommended_ef_search(), top_k * 2);
    
    RETURN QUERY
    SELECT 
        c.chunk_id,
        c.source_id,
        c.page_idx,
        c.element_type,
        c.text_raw,
        1 - (c.embedding <=> query_vector)::float AS similarity
    FROM chunks c
    WHERE c.version_id = p_version_id
      AND c.embedding IS NOT NULL
      AND 1 - (c.embedding <=> query_vector)::float >= min_similarity
    ORDER BY c.embedding <=> query_vector
    LIMIT top_k;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION search_contextual_chunks_by_vector(
    query_vector VECTOR,
    p_version_id UUID,
    top_k INTEGER DEFAULT 10,
    min_similarity FLOAT DEFAULT 0.0
)
RETURNS TABLE (
    contextual_id UUID,
    chunk_id UUID,
    page_idx INTEGER,
    element_type TEXT,
    original_text TEXT,
    context_prefix TEXT,
    contextualized_text TEXT,
    similarity FLOAT
) AS $$
DECLARE
    ef_val INTEGER;
BEGIN
    ef_val := GREATEST(64, top_k * 2);
    
    RETURN QUERY
    SELECT 
        cc.contextual_id,
        cc.chunk_id,
        c.page_idx,
        c.element_type,
        cc.original_text,
        cc.context_prefix,
        cc.contextualized_text,
        1 - (cc.embedding <=> query_vector)::float AS similarity
    FROM contextual_chunks cc
    JOIN chunks c ON cc.chunk_id = c.chunk_id
    WHERE cc.version_id = p_version_id
      AND cc.embedding IS NOT NULL
      AND 1 - (cc.embedding <=> query_vector)::float >= min_similarity
    ORDER BY cc.embedding <=> query_vector
    LIMIT top_k;
END;
$$ LANGUAGE plpgsql;

-- ============================================
-- Views (from 010_small_to_big_chunking.sql)
-- ============================================

CREATE OR REPLACE VIEW v_chunks_with_sections AS
SELECT 
    c.node_id as chunk_id,
    c.metadata->>'heading' as chunk_heading,
    c.content_for_embedding as chunk_content_for_embedding,
    c.embedding as chunk_embedding,
    c.metadata as chunk_metadata,
    c.char_count as chunk_char_count,
    c.level as chunk_order,
    s.node_id as section_id,
    s.metadata->>'heading' as section_heading,
    s.content as section_content,
    s.metadata as section_metadata,
    c.version_id
FROM hierarchical_nodes c
JOIN hierarchical_nodes s ON c.parent_id = s.node_id
WHERE c.node_type = 'chunk' 
AND s.node_type = 'section';

-- ============================================
-- Multi-vector mappings table (for test compatibility)
-- ============================================

CREATE TABLE IF NOT EXISTS multi_vector_mappings (
    mapping_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_id UUID REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    query_text TEXT NOT NULL,
    embedding VECTOR(1536),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_multi_vector_mappings_chunk_id ON multi_vector_mappings(chunk_id);
CREATE INDEX IF NOT EXISTS idx_multi_vector_mappings_embedding_hnsw 
ON multi_vector_mappings USING hnsw(embedding vector_cosine_ops);

-- ============================================
-- Table and Column Comments
-- ============================================

COMMENT ON TABLE hierarchical_nodes IS 'Stores document structure as a tree for HiChunk hierarchical chunking';
COMMENT ON COLUMN hierarchical_nodes.node_id IS 'Unique identifier for the node';
COMMENT ON COLUMN hierarchical_nodes.version_id IS 'Reference to document version';
COMMENT ON COLUMN hierarchical_nodes.parent_id IS 'Reference to parent node (null for root)';
COMMENT ON COLUMN hierarchical_nodes.level IS 'Depth in tree: 0=leaf/sentence, 1=paragraph, 2=section, 3=document';
COMMENT ON COLUMN hierarchical_nodes.node_type IS 'Type of node: sentence, paragraph, section, document, chunk';
COMMENT ON COLUMN hierarchical_nodes.content IS 'Text content of this node';
COMMENT ON COLUMN hierarchical_nodes.children_ids IS 'Array of child node IDs for fast tree traversal';
COMMENT ON COLUMN hierarchical_nodes.start_chunk_id IS 'Reference to first chunk for leaf nodes (range start)';
COMMENT ON COLUMN hierarchical_nodes.end_chunk_id IS 'Reference to last chunk for leaf nodes (range end)';
COMMENT ON COLUMN hierarchical_nodes.metadata IS 'Additional metadata like headings, page numbers';
COMMENT ON COLUMN hierarchical_nodes.embedding IS 'Optional vector embedding for non-leaf nodes';
COMMENT ON COLUMN hierarchical_nodes.created_at IS 'Timestamp when the record was created';
COMMENT ON COLUMN hierarchical_nodes.updated_at IS 'Timestamp when the record was last updated';
