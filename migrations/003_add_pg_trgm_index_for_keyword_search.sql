-- migrations/003_add_pg_trgm_index_for_keyword_search.sql
-- 添加 pg_trgm 扩展和 GIN 索引以优化中文关键词检索性能

-- 启用 pg_trgm 扩展（如果不存在）
-- pg_trgm 提供 trigram 匹配支持，可以为 ILIKE 查询创建 GIN 索引
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- 为 text_raw 列创建 GIN 索引以加速 ILIKE 查询
-- 这对于中文关键词检索特别重要，因为 'simple' 配置的 tsvector 不对中文分词
-- GIN 索引配合 gin_trgm_ops 操作符类可以高效支持 ILIKE '%keyword%' 查询
CREATE INDEX IF NOT EXISTS idx_chunks_text_raw_trgm 
ON chunks USING gin(text_raw gin_trgm_ops);

-- 验证信息
SELECT 
    'pg_trgm extension enabled' as status,
    (SELECT COUNT(*) FROM pg_extension WHERE extname = 'pg_trgm') as ext_count;

SELECT 
    'GIN index on text_raw created' as status,
    (SELECT COUNT(*) FROM pg_indexes 
     WHERE tablename = 'chunks' AND indexname = 'idx_chunks_text_raw_trgm') as index_count;
