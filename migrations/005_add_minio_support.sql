-- Migration 005: Add MinIO support for file storage
-- This migration adds the document_files table to track files stored in MinIO

-- Add embedding_status column to chunks table
ALTER TABLE chunks
ADD COLUMN IF NOT EXISTS embedding_status TEXT DEFAULT 'pending';

-- Create index on embedding_status for efficient querying
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_status
ON chunks (version_id, embedding_status)
WHERE embedding_status IN ('pending', 'processing', 'failed');

-- Create document_files table
CREATE TABLE IF NOT EXISTS document_files (
    file_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    version_id UUID NOT NULL REFERENCES document_versions(version_id) ON DELETE CASCADE,
    file_type TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_name TEXT NOT NULL,
    file_size BIGINT,
    content_type TEXT,
    etag TEXT,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (version_id, file_path)
);

-- Create indexes for document_files
CREATE INDEX IF NOT EXISTS idx_document_files_version
ON document_files (version_id);

CREATE INDEX IF NOT EXISTS idx_document_files_type
ON document_files (file_type);

-- Add comment
COMMENT ON TABLE document_files IS 'Stores metadata for files stored in MinIO object storage';
COMMENT ON COLUMN document_files.file_type IS 'Type of file: original, parsed, images, markdown, etc.';
COMMENT ON COLUMN document_files.file_path IS 'Object key path in MinIO (e.g., bids/project/version/files/...)';
COMMENT ON COLUMN chunks.embedding_status IS 'Status of embedding generation: pending, processing, completed, failed';
