-- Migration: Create hierarchical_nodes table for HiChunk tree structure
-- Part of CPC (Contextual Parent-Child Chunking) implementation
-- This table stores document structure as a tree for hierarchical chunking

-- Create hierarchical_nodes table
-- Tree structure:
--   Level 0: Leaf Nodes (Sentences)
--   Level 1: Paragraph Nodes
--   Level 2: Section Nodes
--   Level 3: Root Node (Document)
create table if not exists hierarchical_nodes (
    -- Primary key
    node_id uuid primary key default gen_random_uuid(),
    
    -- Foreign key to document version
    version_id uuid not null references document_versions(version_id) on delete cascade,
    
    -- Self-referential FK to parent node (null for root node)
    parent_id uuid references hierarchical_nodes(node_id) on delete cascade,
    
    -- Level in the tree hierarchy
    -- 0 = leaf (sentence), 1 = paragraph, 2 = section, 3 = document
    level integer not null check (level >= 0 and level <= 3),
    
    -- Node type for clarity
    node_type text not null check (node_type in ('sentence', 'paragraph', 'section', 'document')),
    
    -- Text content of this node
    content text not null,
    
    -- Array of child node IDs for faster tree traversal
    children_ids uuid[] default '{}',
    
    -- References to chunks for leaf nodes (range support)
    start_chunk_id uuid references chunks(chunk_id) on delete set null,
    end_chunk_id uuid references chunks(chunk_id) on delete set null,
    
    -- Additional metadata (headings, page numbers, etc.)
    metadata jsonb default '{}',
    
    -- Optional embedding for non-leaf nodes (aggregate embedding)
    embedding vector({{EMBEDDING_DIM}}),
    
    -- Timestamps
    created_at timestamptz default now(),
    updated_at timestamptz default now()
);

-- Add comments for documentation
comment on table hierarchical_nodes is 'Stores document structure as a tree for HiChunk hierarchical chunking';
comment on column hierarchical_nodes.node_id is 'Unique identifier for the node';
comment on column hierarchical_nodes.version_id is 'Reference to document version';
comment on column hierarchical_nodes.parent_id is 'Reference to parent node (null for root)';
comment on column hierarchical_nodes.level is 'Depth in tree: 0=leaf/sentence, 1=paragraph, 2=section, 3=document';
comment on column hierarchical_nodes.node_type is 'Type of node: sentence, paragraph, section, document';
comment on column hierarchical_nodes.content is 'Text content of this node';
comment on column hierarchical_nodes.children_ids is 'Array of child node IDs for fast tree traversal';
comment on column hierarchical_nodes.start_chunk_id is 'Reference to first chunk for leaf nodes (range start)';
comment on column hierarchical_nodes.end_chunk_id is 'Reference to last chunk for leaf nodes (range end)';
comment on column hierarchical_nodes.metadata is 'Additional metadata like headings, page numbers';
comment on column hierarchical_nodes.embedding is 'Optional vector embedding for non-leaf nodes';
comment on column hierarchical_nodes.created_at is 'Timestamp when the record was created';
comment on column hierarchical_nodes.updated_at is 'Timestamp when the record was last updated';

-- Create indexes for tree traversal queries

-- Primary lookup: find nodes by version
create index if not exists idx_hierarchical_nodes_version_id on hierarchical_nodes(version_id);

-- Tree traversal: find children of a parent
create index if not exists idx_hierarchical_nodes_parent_id on hierarchical_nodes(parent_id);

-- Level-based queries: find all nodes at specific level for a version
create index if not exists idx_hierarchical_nodes_version_level 
on hierarchical_nodes(version_id, level);

-- Type-based queries: find all nodes of specific type
create index if not exists idx_hierarchical_nodes_type on hierarchical_nodes(node_type);

-- Chunk reference lookup: find node by chunk (for leaf nodes)
create index if not exists idx_hierarchical_nodes_start_chunk 
on hierarchical_nodes(start_chunk_id) 
where start_chunk_id is not null;

-- GIN index for children_ids array (for finding parents of a node)
create index if not exists idx_hierarchical_nodes_children_ids 
on hierarchical_nodes using gin(children_ids);

-- GIN index for metadata JSONB queries
create index if not exists idx_hierarchical_nodes_metadata 
on hierarchical_nodes using gin(metadata);

-- HNSW vector index for similarity search on non-leaf embeddings
create index if not exists idx_hierarchical_nodes_embedding_hnsw 
on hierarchical_nodes 
using hnsw(embedding vector_cosine_ops)
with (m = 16, ef_construction = 64)
where embedding is not null;

comment on index idx_hierarchical_nodes_embedding_hnsw is 'HNSW vector index for hierarchical node embeddings';

-- Composite index for common pattern: version + parent (tree traversal)
create index if not exists idx_hierarchical_nodes_version_parent 
on hierarchical_nodes(version_id, parent_id, level);

-- Trigger function to automatically update updated_at timestamp
-- (reuses existing function from 005_cpc_contextual_chunks.sql if available)
create or replace function update_updated_at_column()
returns trigger as $$
begin
    new.updated_at = now();
    return new;
end;
$$ language 'plpgsql';

-- Apply trigger to hierarchical_nodes
drop trigger if exists trigger_hierarchical_nodes_updated_at on hierarchical_nodes;
create trigger trigger_hierarchical_nodes_updated_at
    before update on hierarchical_nodes
    for each row
    execute function update_updated_at_column();

-- Function to get all descendants of a node (recursive CTE)
create or replace function get_node_descendants(p_node_id uuid)
returns table (
    node_id uuid,
    parent_id uuid,
    level integer,
    node_type text,
    content text,
    depth integer
) as $$
begin
    return query
    with recursive descendants as (
        -- Base case: the node itself
        select 
            hn.node_id,
            hn.parent_id,
            hn.level,
            hn.node_type,
            hn.content,
            0 as depth
        from hierarchical_nodes hn
        where hn.node_id = p_node_id
        
        union all
        
        -- Recursive case: children of nodes in the result
        select 
            hn.node_id,
            hn.parent_id,
            hn.level,
            hn.node_type,
            hn.content,
            d.depth + 1
        from hierarchical_nodes hn
        join descendants d on hn.parent_id = d.node_id
    )
    select * from descendants;
end;
$$ language plpgsql stable;

comment on function get_node_descendants is 'Get all descendants of a node using recursive CTE';

-- Function to get all ancestors of a node (path to root)
create or replace function get_node_ancestors(p_node_id uuid)
returns table (
    node_id uuid,
    parent_id uuid,
    level integer,
    node_type text,
    content text,
    depth integer
) as $$
begin
    return query
    with recursive ancestors as (
        -- Base case: the node itself
        select 
            hn.node_id,
            hn.parent_id,
            hn.level,
            hn.node_type,
            hn.content,
            0 as depth
        from hierarchical_nodes hn
        where hn.node_id = p_node_id
        
        union all
        
        -- Recursive case: parent of nodes in the result
        select 
            hn.node_id,
            hn.parent_id,
            hn.level,
            hn.node_type,
            hn.content,
            a.depth - 1
        from hierarchical_nodes hn
        join ancestors a on hn.node_id = a.parent_id
    )
    select * from ancestors
    order by depth;  -- From root (negative) to node (0)
end;
$$ language plpgsql stable;

comment on function get_node_ancestors is 'Get all ancestors of a node (path to root) using recursive CTE';

-- Function to get root node for a document version
create or replace function get_document_root_node(p_version_id uuid)
returns table (
    node_id uuid,
    content text,
    metadata jsonb
) as $$
begin
    return query
    select 
        hn.node_id,
        hn.content,
        hn.metadata
    from hierarchical_nodes hn
    where hn.version_id = p_version_id
      and hn.level = 3
      and hn.node_type = 'document';
end;
$$ language plpgsql stable;

comment on function get_document_root_node is 'Get the root document node for a version';

-- Rollback statement (for reference):
-- drop table if exists hierarchical_nodes cascade;
