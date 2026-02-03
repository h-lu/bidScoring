-- Migration: Create contextual_chunks table for Anthropic-style context enhancement
-- Part of CPC (Contextual Parent-Child Chunking) implementation
-- This table stores chunks with LLM-generated context prefixes for improved retrieval

-- Create contextual_chunks table
create table if not exists contextual_chunks (
    -- Primary key
    contextual_id uuid primary key default gen_random_uuid(),
    
    -- Foreign key to original chunk (1:1 relationship)
    chunk_id uuid not null unique references chunks(chunk_id) on delete cascade,
    
    -- Foreign key to document version for efficient filtering
    version_id uuid not null references document_versions(version_id) on delete cascade,
    
    -- Original text from the chunk
    original_text text not null,
    
    -- LLM-generated context prefix describing where this chunk fits in the document
    context_prefix text not null,
    
    -- Combined text: context_prefix + original_text
    contextualized_text text not null,
    
    -- Vector embedding of contextualized_text
    embedding vector({{EMBEDDING_DIM}}),
    
    -- Metadata
    model_name text not null,  -- Which LLM model generated the context prefix
    embedding_model text,      -- Which embedding model was used (if different from model_name)
    
    -- Timestamps
    created_at timestamptz default now(),
    updated_at timestamptz default now()
);

-- Add comments for documentation
comment on table contextual_chunks is 'Stores chunks with Anthropic-style context prefixes for improved retrieval accuracy';
comment on column contextual_chunks.contextual_id is 'Unique identifier for the contextual chunk record';
comment on column contextual_chunks.chunk_id is 'Reference to the original chunk (1:1 relationship)';
comment on column contextual_chunks.version_id is 'Reference to document version for efficient filtering';
comment on column contextual_chunks.original_text is 'Original chunk text from the source document';
comment on column contextual_chunks.context_prefix is 'LLM-generated context describing where this chunk fits in the document structure';
comment on column contextual_chunks.contextualized_text is 'Combined text: context_prefix + original_text for embedding';
comment on column contextual_chunks.embedding is 'Vector embedding of contextualized_text';
comment on column contextual_chunks.model_name is 'Name of the LLM model used to generate context_prefix';
comment on column contextual_chunks.embedding_model is 'Name of the embedding model used (optional, defaults to system default)';
comment on column contextual_chunks.created_at is 'Timestamp when the record was created';
comment on column contextual_chunks.updated_at is 'Timestamp when the record was last updated';

-- Create indexes for common query patterns

-- Primary lookup: find contextual chunk by original chunk_id
create index if not exists idx_contextual_chunks_chunk_id on contextual_chunks(chunk_id);

-- Version filtering: find all contextual chunks for a document version
create index if not exists idx_contextual_chunks_version_id on contextual_chunks(version_id);

-- HNSW vector index for similarity search on contextualized embeddings
create index if not exists idx_contextual_chunks_embedding_hnsw 
on contextual_chunks 
using hnsw(embedding vector_cosine_ops)
with (m = 16, ef_construction = 64);

-- Comment on HNSW index
comment on index idx_contextual_chunks_embedding_hnsw is 'HNSW vector index for contextualized embeddings';

-- Composite index for common query pattern: filter by version, order by creation
create index if not exists idx_contextual_chunks_version_created 
on contextual_chunks(version_id, created_at);

-- Trigger function to automatically update updated_at timestamp
create or replace function update_updated_at_column()
returns trigger as $$
begin
    new.updated_at = now();
    return new;
end;
$$ language 'plpgsql';

-- Apply trigger to contextual_chunks
drop trigger if exists trigger_contextual_chunks_updated_at on contextual_chunks;
create trigger trigger_contextual_chunks_updated_at
    before update on contextual_chunks
    for each row
    execute function update_updated_at_column();

-- Create vector search function for contextual chunks
create or replace function search_contextual_chunks_by_vector(
    query_vector vector,
    p_version_id uuid,
    top_k integer default 10,
    min_similarity float default 0.0
)
returns table (
    contextual_id uuid,
    chunk_id uuid,
    page_idx integer,
    element_type text,
    original_text text,
    context_prefix text,
    contextualized_text text,
    similarity float
) as $$
declare
    ef_val integer;
begin
    -- Set ef_search based on top_k for optimal recall
    ef_val := greatest(64, top_k * 2);
    
    return query
    select 
        cc.contextual_id,
        cc.chunk_id,
        c.page_idx,
        c.element_type,
        cc.original_text,
        cc.context_prefix,
        cc.contextualized_text,
        1 - (cc.embedding <=> query_vector)::float as similarity
    from contextual_chunks cc
    join chunks c on cc.chunk_id = c.chunk_id
    where cc.version_id = p_version_id
      and cc.embedding is not null
      and 1 - (cc.embedding <=> query_vector)::float >= min_similarity
    order by cc.embedding <=> query_vector
    limit top_k;
end;
$$ language plpgsql;

comment on function search_contextual_chunks_by_vector is 'Vector similarity search on contextualized chunks with automatic ef_search optimization';

-- Rollback statement (for reference):
-- drop table if exists contextual_chunks cascade;
