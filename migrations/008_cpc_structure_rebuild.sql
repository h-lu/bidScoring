-- migrations/008_cpc_structure_rebuild.sql
-- CPC 结构重建表 - 存储重建后的层次结构

-- 文档结构节点表（扩展 hierarchical_nodes 或新建）
-- 选择：扩展 existing hierarchical_nodes 表，添加新字段

-- 1. 检查并添加新列到 hierarchical_nodes（如果不存在）
DO $$
BEGIN
    -- heading 列
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name='hierarchical_nodes' AND column_name='heading') THEN
        ALTER TABLE hierarchical_nodes ADD COLUMN heading TEXT;
    END IF;
    
    -- context 列（存储生成的上下文）
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name='hierarchical_nodes' AND column_name='context') THEN
        ALTER TABLE hierarchical_nodes ADD COLUMN context TEXT;
    END IF;
    
    -- merged_chunk_count 列
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name='hierarchical_nodes' AND column_name='merged_chunk_count') THEN
        ALTER TABLE hierarchical_nodes ADD COLUMN merged_chunk_count INTEGER DEFAULT 1;
    END IF;
END $$;

-- 2. 添加/更新注释
COMMENT ON COLUMN hierarchical_nodes.heading IS '章节标题（section类型）';
COMMENT ON COLUMN hierarchical_nodes.context IS '生成的上下文描述';
COMMENT ON COLUMN hierarchical_nodes.merged_chunk_count IS '合并了多少个原始chunk';

-- 3. 创建用于结构重建处理的状态跟踪表
CREATE TABLE IF NOT EXISTS document_structure_build_status (
    status_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    version_id UUID NOT NULL REFERENCES document_versions(version_id) ON DELETE CASCADE,
    
    -- 处理状态
    structure_built BOOLEAN DEFAULT FALSE,
    context_generated BOOLEAN DEFAULT FALSE,
    embeddings_generated BOOLEAN DEFAULT FALSE,
    raptor_built BOOLEAN DEFAULT FALSE,
    
    -- 统计信息
    original_chunk_count INTEGER,
    paragraph_count INTEGER,
    section_count INTEGER,
    llm_call_count INTEGER DEFAULT 0,
    
    -- 错误信息
    errors TEXT[],
    
    -- 时间戳
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- 唯一约束
    UNIQUE(version_id)
);

-- 4. 创建索引
CREATE INDEX IF NOT EXISTS idx_structure_status_version 
    ON document_structure_build_status(version_id);

CREATE INDEX IF NOT EXISTS idx_structure_status_completed 
    ON document_structure_build_status(structure_built, context_generated);

-- 5. 触发器：自动更新 updated_at
CREATE OR REPLACE FUNCTION update_structure_status_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_structure_status_updated 
    ON document_structure_build_status;

CREATE TRIGGER trigger_structure_status_updated
    BEFORE UPDATE ON document_structure_build_status
    FOR EACH ROW
    EXECUTE FUNCTION update_structure_status_timestamp();

-- 6. 添加表注释
COMMENT ON TABLE document_structure_build_status IS '文档结构重建处理状态跟踪';

-- 7. 回滚语句（供参考）
-- ALTER TABLE hierarchical_nodes DROP COLUMN IF EXISTS heading;
-- ALTER TABLE hierarchical_nodes DROP COLUMN IF EXISTS context;
-- ALTER TABLE hierarchical_nodes DROP COLUMN IF EXISTS merged_chunk_count;
-- DROP TABLE IF EXISTS document_structure_build_status;
