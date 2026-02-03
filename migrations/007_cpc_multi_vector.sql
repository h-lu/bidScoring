-- Migration: Create multi_vector_mappings table for Multi-Vector Retrieval System
-- Part of Phase 5: Multi-Vector Retrieval System implementation
-- This table stores relationships between different representations (parent-child, hierarchical, etc.)

-- Create multi_vector_mappings table
-- Supports multiple representations per document:
--   - Child chunks (small, specific)
--   - Parent chunks (large, contextual)
--   - Hierarchical nodes (structured)
create table if not exists multi_vector_mappings (
    -- Primary key
    mapping_id uuid primary key default gen_random_uuid(),
    
    -- Foreign key to document version for efficient filtering
    version_id uuid not null references document_versions(version_id) on delete cascade,
    
    -- Parent chunk reference (the larger/contextual representation)
    parent_chunk_id uuid references chunks(chunk_id) on delete cascade,
    
    -- Child chunk reference (the smaller/specific representation)
    child_chunk_id uuid references chunks(chunk_id) on delete cascade,
    
    -- Type of parent representation
    parent_type text not null check (parent_type in ('hierarchical', 'contextual', 'original')),
    
    -- Type of child representation
    child_type text not null check (child_type in ('sentence', 'paragraph', 'chunk')),
    
    -- Relationship type between parent and child
    relationship text not null check (relationship in ('parent-child', 'sibling', 'summary')),
    
    -- Additional metadata (similarity scores, weights, etc.)
    metadata jsonb default '{}',
    
    -- Timestamp
    created_at timestamptz default now()
);

-- Add comments for documentation
comment on table multi_vector_mappings is 'Stores relationships between different chunk representations for Multi-Vector Retrieval';
comment on column multi_vector_mappings.mapping_id is 'Unique identifier for the mapping record';
comment on column multi_vector_mappings.version_id is 'Reference to document version for efficient filtering';
comment on column multi_vector_mappings.parent_chunk_id is 'Reference to parent chunk (larger/contextual representation)';
comment on column multi_vector_mappings.child_chunk_id is 'Reference to child chunk (smaller/specific representation)';
comment on column multi_vector_mappings.parent_type is 'Type of parent: hierarchical, contextual, or original';
comment on column multi_vector_mappings.child_type is 'Type of child: sentence, paragraph, or chunk';
comment on column multi_vector_mappings.relationship is 'Relationship type: parent-child, sibling, or summary';
comment on column multi_vector_mappings.metadata is 'Additional metadata like similarity scores, weights, retrieval priority';
comment on column multi_vector_mappings.created_at is 'Timestamp when the mapping was created';

-- Create indexes for relationship queries

-- Primary lookup: find mappings by version
create index if not exists idx_multi_vector_version_id on multi_vector_mappings(version_id);

-- Parent-based queries: find all children of a parent chunk
create index if not exists idx_multi_vector_parent_chunk on multi_vector_mappings(parent_chunk_id);

-- Child-based queries: find all parents of a child chunk
create index if not exists idx_multi_vector_child_chunk on multi_vector_mappings(child_chunk_id);

-- Relationship type filtering: find mappings by relationship type
create index if not exists idx_multi_vector_relationship on multi_vector_mappings(relationship);

-- Parent type filtering: find mappings by parent type
create index if not exists idx_multi_vector_parent_type on multi_vector_mappings(parent_type);

-- Child type filtering: find mappings by child type
create index if not exists idx_multi_vector_child_type on multi_vector_mappings(child_type);

-- GIN index for metadata JSONB queries
create index if not exists idx_multi_vector_metadata on multi_vector_mappings using gin(metadata);

-- Composite index for common query pattern: version + relationship
-- Used when retrieving specific relationship types for a document
create index if not exists idx_multi_vector_version_relationship 
on multi_vector_mappings(version_id, relationship);

-- Composite index for parent-child traversal: parent + child types
-- Used when filtering by specific representation types
create index if not exists idx_multi_vector_parent_child_types 
on multi_vector_mappings(parent_type, child_type);

-- Composite index for efficient parent lookup with relationship
-- Used when finding children with specific relationship
create index if not exists idx_multi_vector_parent_relationship 
on multi_vector_mappings(parent_chunk_id, relationship) 
where parent_chunk_id is not null;

-- Composite index for efficient child lookup with relationship
-- Used when finding parents with specific relationship
create index if not exists idx_multi_vector_child_relationship 
on multi_vector_mappings(child_chunk_id, relationship) 
where child_chunk_id is not null;

-- Rollback statement (for reference):
-- drop table if exists multi_vector_mappings cascade;
