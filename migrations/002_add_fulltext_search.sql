-- migrations/002_add_fulltext_search.sql
-- 为 chunks 表添加全文搜索支持

-- 1. 添加 tsvector 列（如果不存在）
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'chunks' AND column_name = 'textsearch'
    ) THEN
        ALTER TABLE chunks ADD COLUMN textsearch tsvector;
        RAISE NOTICE 'Added textsearch column to chunks table';
    ELSE
        RAISE NOTICE 'textsearch column already exists';
    END IF;
END $$;

-- 2. 创建中文全文搜索配置（如果不存在）
-- 注意：PostgreSQL 默认没有中文配置，使用 simple 作为备选
DO $$
BEGIN
    BEGIN
        -- 尝试创建中文配置
        IF NOT EXISTS (
            SELECT 1 FROM pg_ts_config WHERE cfgname = 'chinese'
        ) THEN
            CREATE TEXT SEARCH CONFIGURATION chinese (COPY = simple);
            RAISE NOTICE 'Created chinese text search configuration';
        END IF;
    EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'Using simple text search configuration';
    END;
END $$;

-- 3. 更新现有数据（根据 context7 建议，使用 simple 配置更稳定）
DO $$
DECLARE
    updated_count INTEGER;
BEGIN
    UPDATE chunks 
    SET textsearch = to_tsvector('simple', COALESCE(text_raw, ''))
    WHERE textsearch IS NULL OR textsearch = ''::tsvector;
    
    GET DIAGNOSTICS updated_count = ROW_COUNT;
    RAISE NOTICE 'Updated % rows with textsearch vectors', updated_count;
END $$;

-- 4. 创建 GIN 索引（如果不存在）- 根据 context7，GIN 索引是全文搜索的关键
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes 
        WHERE indexname = 'idx_chunks_textsearch'
    ) THEN
        CREATE INDEX idx_chunks_textsearch ON chunks USING gin(textsearch);
        RAISE NOTICE 'Created GIN index idx_chunks_textsearch';
    ELSE
        RAISE NOTICE 'GIN index idx_chunks_textsearch already exists';
    END IF;
END $$;

-- 5. 创建触发器函数自动更新 tsvector
CREATE OR REPLACE FUNCTION chunks_textsearch_update()
RETURNS trigger AS $$
BEGIN
    NEW.textsearch := to_tsvector('simple', COALESCE(NEW.text_raw, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 6. 创建或替换触发器
DROP TRIGGER IF EXISTS chunks_textsearch_trigger ON chunks;
CREATE TRIGGER chunks_textsearch_trigger
    BEFORE INSERT OR UPDATE OF text_raw ON chunks
    FOR EACH ROW
    EXECUTE FUNCTION chunks_textsearch_update();

-- 7. 为 version_id 添加索引（如果不存在）以优化过滤查询
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes 
        WHERE indexname = 'idx_chunks_version_id'
    ) THEN
        CREATE INDEX idx_chunks_version_id ON chunks(version_id);
        RAISE NOTICE 'Created index idx_chunks_version_id';
    ELSE
        RAISE NOTICE 'Index idx_chunks_version_id already exists';
    END IF;
END $$;

-- 8. 为 embedding 列添加 HNSW 索引优化参数（如果尚未创建）
-- 根据 context7 建议，使用优化的 m 和 ef_construction 参数
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes 
        WHERE indexname = 'idx_chunks_embedding_hnsw'
    ) THEN
        -- 使用余弦距离操作符（适合语义相似度）
        -- m=16: 每层连接数（平衡性能和召回率）
        -- ef_construction=128: 构建时候选列表（提高索引质量）
        CREATE INDEX idx_chunks_embedding_hnsw ON chunks 
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 128);
        RAISE NOTICE 'Created HNSW index idx_chunks_embedding_hnsw with optimized parameters';
    ELSE
        RAISE NOTICE 'HNSW index already exists, skipping creation';
    END IF;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Could not create HNSW index (may require pgvector extension): %', SQLERRM;
END $$;

-- 验证信息
SELECT 
    'Migration complete' as status,
    (SELECT COUNT(*) FROM chunks) as total_chunks,
    (SELECT COUNT(*) FROM chunks WHERE textsearch IS NOT NULL) as chunks_with_textsearch,
    (SELECT COUNT(*) FROM pg_indexes WHERE tablename = 'chunks') as total_indexes;
