-- Migration 009: Add source_chunk_ids to hierarchical_nodes
-- 添加 source_chunk_ids 字段用于追溯合并后的段落来自哪些原始 chunks

-- 1. 添加 source_chunk_ids 字段
ALTER TABLE hierarchical_nodes 
ADD COLUMN IF NOT EXISTS source_chunk_ids UUID[] DEFAULT '{}';

-- 2. 添加注释
COMMENT ON COLUMN hierarchical_nodes.source_chunk_ids IS 
'存储该节点合并的原始 chunk IDs，用于追溯来源。对于非合并节点，为空数组';

-- 3. 创建索引（用于根据 chunk_id 查找包含它的节点）
CREATE INDEX IF NOT EXISTS idx_hierarchical_nodes_source_chunks 
ON hierarchical_nodes USING GIN (source_chunk_ids);

-- 4. 添加检查约束：确保 children_ids 和 source_chunk_ids 不会同时有值
-- （只有叶子节点/段落才有 source_chunk_ids，只有非叶子节点才有 children_ids）
-- 注意：这个约束在某些边界情况下可能不适用，先不添加

-- 回滚语句（供参考）
-- ALTER TABLE hierarchical_nodes DROP COLUMN IF EXISTS source_chunk_ids;
-- DROP INDEX IF EXISTS idx_hierarchical_nodes_source_chunks;
