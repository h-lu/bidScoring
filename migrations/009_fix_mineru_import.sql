-- Migration 009: Fix MinerU import issues
-- 1. Add unique constraint to prevent duplicate imports
-- 2. Add index for faster lookups

-- Add unique constraint on (version_id, source_id)
-- This prevents duplicate chunks from being imported
ALTER TABLE chunks 
    ADD CONSTRAINT unique_version_source 
    UNIQUE (version_id, source_id);

-- Add index for text_level lookups (used in paragraph merging)
CREATE INDEX idx_chunks_text_level 
    ON chunks(version_id, text_level) 
    WHERE text_level IS NOT NULL;

-- Add comment explaining the constraint
COMMENT ON CONSTRAINT unique_version_source ON chunks IS 
    'Prevents duplicate chunk imports from MinerU. Each source_id should be unique per version.';
