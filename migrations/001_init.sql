create extension if not exists pgcrypto;
create extension if not exists vector;

create table if not exists projects (
  project_id uuid primary key,
  name text not null,
  owner text,
  status text,
  created_at timestamptz default now()
);

create table if not exists documents (
  doc_id uuid primary key,
  project_id uuid references projects(project_id),
  title text,
  source_type text,
  created_at timestamptz default now()
);

create table if not exists document_versions (
  version_id uuid primary key,
  doc_id uuid references documents(doc_id),
  source_uri text,
  source_hash text,
  parser_version text,
  created_at timestamptz default now(),
  status text
);

create table if not exists chunks (
  chunk_id uuid primary key,
  project_id uuid references projects(project_id),
  version_id uuid references document_versions(version_id),
  source_id text,
  chunk_index int,
  page_idx int,
  bbox jsonb,
  element_type text,
  text_raw text,
  text_hash text,
  text_tsv tsvector,
  embedding vector({{EMBEDDING_DIM}})
);

create table if not exists scoring_runs (
  run_id uuid primary key,
  project_id uuid references projects(project_id),
  version_id uuid references document_versions(version_id),
  dimensions text[],
  model text,
  rules_version text,
  params_hash text,
  started_at timestamptz default now(),
  status text
);

create table if not exists scoring_results (
  result_id uuid primary key,
  run_id uuid references scoring_runs(run_id),
  dimension text,
  score numeric,
  max_score numeric,
  reasoning text,
  evidence_found boolean,
  confidence text
);

create table if not exists citations (
  citation_id uuid primary key,
  result_id uuid references scoring_results(result_id),
  source_id text,
  chunk_id uuid references chunks(chunk_id),
  cited_text text,
  verified boolean,
  match_type text
);

create index if not exists idx_chunks_text_tsv on chunks using gin(text_tsv);
-- 向量索引（择一）：HNSW 召回更好但更吃内存
create index if not exists idx_chunks_embedding_hnsw on chunks using hnsw(embedding vector_cosine_ops);
-- 向量索引（可选）：IVFFlat 构建更快、内存更省，但需调参
-- create index if not exists idx_chunks_embedding_ivfflat on chunks using ivfflat(embedding vector_cosine_ops) with (lists = 100);
create index if not exists idx_chunks_project_version_page on chunks(project_id, version_id, page_idx);
