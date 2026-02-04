-- Migration: Small-to-Big Chunking Strategy
-- Description: Support parent-child retrieval with smart chunk merging

-- 1. Add content_for_embedding column to store processed content for vector search
ALTER TABLE hierarchical_nodes 
ADD COLUMN IF NOT EXISTS content_for_embedding TEXT;

-- 2. Add chunk boundaries for precise tracking
ALTER TABLE hierarchical_nodes 
ADD COLUMN IF NOT EXISTS char_count INTEGER DEFAULT 0;

-- 3. Create index for faster parent-child lookups
CREATE INDEX IF NOT EXISTS idx_hierarchical_nodes_parent_id 
ON hierarchical_nodes(parent_id) 
WHERE parent_id IS NOT NULL;

-- 4. Create index for node_type filtering (used in search)
CREATE INDEX IF NOT EXISTS idx_hierarchical_nodes_type 
ON hierarchical_nodes(node_type, version_id);

-- 5. Update comments
COMMENT ON COLUMN hierarchical_nodes.content_for_embedding IS 
'Processed content for embedding (may differ from content for LLM context)';

COMMENT ON COLUMN hierarchical_nodes.char_count IS 
'Character count of content_for_embedding for quick filtering';

-- 6. Add constraint to ensure content_for_embedding is set for chunk nodes
-- Note: We can't enforce this at DB level due to existing data, but application should handle it

-- 7. Create a view for convenient retrieval of chunks with parent sections
CREATE OR REPLACE VIEW v_chunks_with_sections AS
SELECT 
    c.id as chunk_id,
    c.heading as chunk_heading,
    c.content_for_embedding as chunk_content_for_embedding,
    c.embedding as chunk_embedding,
    c.metadata as chunk_metadata,
    c.char_count as chunk_char_count,
    c.order_index as chunk_order,
    s.id as section_id,
    s.heading as section_heading,
    s.content as section_content,  -- Full content for LLM
    s.metadata as section_metadata,
    s.version_id,
    s.document_id
FROM hierarchical_nodes c
JOIN hierarchical_nodes s ON c.parent_id = s.id
WHERE c.node_type = 'chunk' 
AND s.node_type = 'section';

COMMENT ON VIEW v_chunks_with_sections IS 
'Convenient view for small-to-big retrieval: chunks with their parent sections';
