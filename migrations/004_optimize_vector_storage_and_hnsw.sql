-- migrations/004_optimize_vector_storage_and_hnsw.sql
-- 优化向量存储和 HNSW 索引性能
-- 基于 Context7 pgvector 最佳实践

-- 1. 向量存储优化：将 embedding 列存储设置为 PLAIN
--    避免 TOAST 存储开销，提高查询性能
--    注意：这会增加表大小，但显著提升查询速度
DO $$
BEGIN
    -- 检查是否已经是 PLAIN 存储
    IF EXISTS (
        SELECT 1 FROM pg_attribute
        WHERE attrelid = 'chunks'::regclass
          AND attname = 'embedding'
          AND attstorage != 'p'
    ) THEN
        ALTER TABLE chunks ALTER COLUMN embedding SET STORAGE PLAIN;
        RAISE NOTICE 'Set embedding column storage to PLAIN';
    ELSE
        RAISE NOTICE 'embedding column already uses PLAIN storage or does not exist';
    END IF;
END $$;

-- 2. 重新索引 HNSW 索引以应用优化
--    在修改存储后，重建索引以确保最佳性能
REINDEX INDEX CONCURRENTLY idx_chunks_embedding_hnsw;

-- 3. 对 chunks 表进行 VACUUM ANALYZE
--    更新统计信息，帮助查询优化器做出更好的决策
VACUUM ANALYZE chunks;

-- 4. 验证优化结果
SELECT 
    'Storage optimization complete' as status,
    (SELECT attstorage FROM pg_attribute 
     WHERE attrelid = 'chunks'::regclass AND attname = 'embedding') as embedding_storage,
    (SELECT pg_size_pretty(pg_total_relation_size('chunks'))) as table_size,
    (SELECT pg_size_pretty(pg_indexes_size('chunks'))) as indexes_size;
