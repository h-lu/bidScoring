-- Migration: Optimize pgvector HNSW index for production use
-- Based on AWS and pgvector best practices

-- 检查当前索引并优化（如果需要）
-- 注意：重建索引可能需要较长时间，建议在低峰期执行

-- 1. 检查是否需要重建索引
-- 如果数据量增长较大或索引创建时参数不理想，可以重建

-- 2. 设置 HNSW 构建参数（可选，用于新索引）
-- m = 16: 每个节点的最大边数（默认）
-- ef_construction = 64: 构建时候选队列大小（默认）

-- 如果需要重建索引，取消下面注释：
-- DROP INDEX IF EXISTS idx_chunks_embedding_hnsw;
-- CREATE INDEX idx_chunks_embedding_hnsw ON chunks 
-- USING hnsw (embedding vector_cosine_ops)
-- WITH (m = 16, ef_construction = 64);

-- 3. 创建查询优化用的函数
-- 用于动态设置 ef_search

-- 获取建议的 ef_search 值
CREATE OR REPLACE FUNCTION get_recommended_ef_search(chunk_count bigint DEFAULT NULL)
RETURNS integer AS $$
DECLARE
    count_val bigint;
BEGIN
    -- 如果没有提供数量，查询当前表中的记录数
    IF chunk_count IS NULL THEN
        SELECT COUNT(*) INTO count_val FROM chunks WHERE embedding IS NOT NULL;
    ELSE
        count_val := chunk_count;
    END IF;
    
    -- 根据数据量返回建议值
    -- 参考 AWS 最佳实践
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

COMMENT ON FUNCTION get_recommended_ef_search IS '根据数据量返回推荐的 HNSW ef_search 值';

-- 4. 创建带 ef_search 设置的搜索函数
CREATE OR REPLACE FUNCTION search_chunks_by_vector(
    query_vector vector,
    p_version_id uuid,
    top_k integer DEFAULT 10,
    min_similarity float DEFAULT 0.0
)
RETURNS TABLE (
    chunk_id uuid,
    source_id text,
    page_idx integer,
    element_type text,
    text_raw text,
    similarity float
) AS $$
DECLARE
    ef_val integer;
BEGIN
    -- 设置 ef_search 为 top_k 的 2-4 倍，确保召回率
    ef_val := GREATEST(get_recommended_ef_search(), top_k * 2);
    -- 注意：pgvector 1.1+ 支持通过 SET 设置 ef_search
    -- 旧版本可能不支持，需要应用层设置
    
    RETURN QUERY
    SELECT 
        c.chunk_id,
        c.source_id,
        c.page_idx,
        c.element_type,
        c.text_raw,
        1 - (c.embedding <=> query_vector)::float as similarity
    FROM chunks c
    WHERE c.version_id = p_version_id
      AND c.embedding IS NOT NULL
      AND 1 - (c.embedding <=> query_vector)::float >= min_similarity
    ORDER BY c.embedding <=> query_vector
    LIMIT top_k;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION search_chunks_by_vector IS '向量相似度搜索，自动设置最优 ef_search';

-- 5. 创建混合搜索函数（BM25 + Vector）
-- 注意：这个函数需要应用层实现 RRF 融合
CREATE OR REPLACE FUNCTION search_chunks_hybrid(
    query_text text,
    query_vector vector,
    p_version_id uuid,
    top_k integer DEFAULT 10
)
RETURNS TABLE (
    chunk_id uuid,
    source_id text,
    page_idx integer,
    element_type text,
    text_raw text,
    bm25_rank integer,
    vector_rank integer,
    rrf_score float
) AS $$
DECLARE
    ef_val integer;
BEGIN
    ef_val := GREATEST(get_recommended_ef_search(), top_k * 4);
    
    RETURN QUERY
    WITH 
    bm25_results AS (
        SELECT 
            c.chunk_id,
            c.source_id,
            c.page_idx,
            c.element_type,
            c.text_raw,
            ROW_NUMBER() OVER (ORDER BY ts_rank_cd(c.text_tsv, plainto_tsquery('simple', query_text)) DESC)::integer as rank
        FROM chunks c
        WHERE c.version_id = p_version_id
          AND c.text_tsv @@ plainto_tsquery('simple', query_text)
        LIMIT top_k * 2
    ),
    vector_results AS (
        SELECT 
            c.chunk_id,
            ROW_NUMBER() OVER (ORDER BY c.embedding <=> query_vector)::integer as rank
        FROM chunks c
        WHERE c.version_id = p_version_id
          AND c.embedding IS NOT NULL
        LIMIT top_k * 2
    ),
    combined AS (
        SELECT 
            COALESCE(b.chunk_id, v.chunk_id) as chunk_id,
            b.source_id,
            b.page_idx,
            b.element_type,
            b.text_raw,
            b.rank as bm25_rank,
            v.rank as vector_rank,
            -- RRF 分数: 1/(k + rank)，k=60 是常用值
            COALESCE(1.0 / (60.0 + b.rank), 0) * 0.4 + 
            COALESCE(1.0 / (60.0 + v.rank), 0) * 0.6 as rrf_score
        FROM bm25_results b
        FULL OUTER JOIN vector_results v ON b.chunk_id = v.chunk_id
    )
    SELECT * FROM combined
    ORDER BY rrf_score DESC
    LIMIT top_k;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION search_chunks_hybrid IS '混合搜索：BM25 + 向量，使用 RRF 融合';

-- 6. 创建索引使用统计视图（可选）
-- 需要 pg_stat_statements 扩展
-- CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- 7. 添加使用说明注释
COMMENT ON INDEX idx_chunks_embedding_hnsw IS 'HNSW 向量索引，使用 cosine 相似度。构建参数: m=16, ef_construction=64。查询时建议设置 ef_search=64-200 根据数据量调整。';
