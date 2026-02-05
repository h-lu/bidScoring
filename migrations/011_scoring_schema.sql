-- Migration: 评分维度 Schema 支持
-- Description: 创建评分维度提取结果、证据项和审计日志表
-- Created: 2026-02-05

-- =============================================================================
-- 评分维度提取结果表
-- =============================================================================

CREATE TABLE IF NOT EXISTS bid_scoring_results (
    result_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    bid_id UUID NOT NULL,
    document_version_id UUID REFERENCES document_versions(version_id) ON DELETE CASCADE,
    dimension_id TEXT NOT NULL,
    dimension_name TEXT NOT NULL,
    weight DECIMAL(10, 2) NOT NULL CHECK (weight > 0),
    extracted_score DECIMAL(10, 2),
    final_score DECIMAL(10, 2),
    completeness_level TEXT CHECK (completeness_level IN ('complete', 'partial', 'minimal', 'empty')),
    evaluation_data JSONB DEFAULT '{}',  -- 存储评分维度的完整数据
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

COMMENT ON TABLE bid_scoring_results IS '评分维度提取结果';
COMMENT ON COLUMN bid_scoring_results.evaluation_data IS '存储 ScoringDimension 的序列化数据';

-- =============================================================================
-- 证据项表
-- =============================================================================

CREATE TABLE IF NOT EXISTS scoring_evidence (
    evidence_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    result_id UUID REFERENCES bid_scoring_results(result_id) ON DELETE CASCADE,
    field_name TEXT NOT NULL,
    field_value TEXT,
    source_text TEXT,
    page_idx INTEGER NOT NULL CHECK (page_idx >= 0),
    bbox JSONB NOT NULL,  -- BoundingBox {x1, y1, x2, y2}
    chunk_id UUID REFERENCES hierarchical_nodes(node_id),
    confidence DECIMAL(3, 2) DEFAULT 0.0 CHECK (confidence >= 0 AND confidence <= 1),
    validation_status TEXT DEFAULT 'pending' CHECK (validation_status IN ('pending', 'confirmed', 'rejected')),
    validation_notes TEXT,
    raw_value TEXT,  -- 原始文本值（用于结构化证据）
    parsed_value JSONB,  -- 解析后的结构化值 {days, hours, years, months, ...}
    evidence_type TEXT DEFAULT 'base' CHECK (evidence_type IN ('base', 'duration', 'response_time', 'warranty', 'service_fee', 'personnel')),
    extracted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    validated_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

COMMENT ON TABLE scoring_evidence IS '评分证据项，关联到原文位置';
COMMENT ON COLUMN scoring_evidence.parsed_value IS '结构化解析值，如 {days: 2, hours: null}';
COMMENT ON COLUMN scoring_evidence.evidence_type IS '证据类型：base/duration/response_time/warranty/service_fee/personnel';

-- =============================================================================
-- 多源证据冲突解决记录表
-- =============================================================================

CREATE TABLE IF NOT EXISTS evidence_field_resolutions (
    resolution_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    result_id UUID REFERENCES bid_scoring_results(result_id) ON DELETE CASCADE,
    field_name TEXT NOT NULL,
    candidates JSONB NOT NULL DEFAULT '[]',  -- 候选证据 ID 列表
    selected_evidence_id UUID REFERENCES scoring_evidence(evidence_id),
    resolution_strategy TEXT NOT NULL CHECK (resolution_strategy IN (
        'highest_confidence', 'first', 'manual', 'majority_vote', 
        'source_authority', 'temporal_recency', 'weighted_average'
    )),
    strategy_params JSONB DEFAULT '{}',  -- 策略参数，如 {authority_scores: {...}}
    has_conflict BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    resolved_by TEXT  -- 人工解决时的用户标识
);

COMMENT ON TABLE evidence_field_resolutions IS '多源证据冲突解决记录';

-- =============================================================================
-- 评分规则配置表
-- =============================================================================

CREATE TABLE IF NOT EXISTS scoring_rule_configs (
    rule_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dimension_id TEXT NOT NULL,
    rule_name TEXT NOT NULL,
    strategy_type TEXT NOT NULL CHECK (strategy_type IN ('threshold', 'range', 'composite')),
    strategy_config JSONB NOT NULL,  -- 策略配置 {threshold, operator, min_value, max_value, ...}
    score_min DECIMAL(10, 2) NOT NULL CHECK (score_min >= 0),
    score_max DECIMAL(10, 2) NOT NULL CHECK (score_max >= score_min),
    description TEXT,
    weight DECIMAL(5, 2) DEFAULT 1.0 CHECK (weight > 0),
    sequence INTEGER DEFAULT 0,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

COMMENT ON TABLE scoring_rule_configs IS '评分规则配置，支持动态规则管理';

-- =============================================================================
-- 审计日志表
-- =============================================================================

CREATE TABLE IF NOT EXISTS scoring_audit_log (
    audit_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    result_id UUID REFERENCES bid_scoring_results(result_id) ON DELETE CASCADE,
    evidence_id UUID REFERENCES scoring_evidence(evidence_id) ON DELETE SET NULL,
    action TEXT NOT NULL CHECK (action IN (
        'evidence_created', 'evidence_updated', 'evidence_confirmed', 
        'evidence_rejected', 'score_calculated', 'conflict_resolved',
        'manual_override', 'rule_applied'
    )),
    performed_by TEXT,
    performed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    old_value JSONB,
    new_value JSONB,
    reason TEXT,
    ip_address INET
);

COMMENT ON TABLE scoring_audit_log IS '评分审计日志，记录所有关键操作';

-- =============================================================================
-- 评分运行批次表（用于批量评分任务）
-- =============================================================================

CREATE TABLE IF NOT EXISTS scoring_runs (
    run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_name TEXT,
    bid_id UUID NOT NULL,
    document_version_id UUID REFERENCES document_versions(version_id) ON DELETE CASCADE,
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    total_dimensions INTEGER DEFAULT 0,
    completed_dimensions INTEGER DEFAULT 0,
    error_message TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

COMMENT ON TABLE scoring_runs IS '评分运行批次，用于跟踪批量评分任务';

-- =============================================================================
-- 索引
-- =============================================================================

-- bid_scoring_results 索引
CREATE INDEX IF NOT EXISTS idx_scoring_results_version 
    ON bid_scoring_results(document_version_id);
CREATE INDEX IF NOT EXISTS idx_scoring_results_bid 
    ON bid_scoring_results(bid_id);
CREATE INDEX IF NOT EXISTS idx_scoring_results_dimension 
    ON bid_scoring_results(dimension_id);
CREATE INDEX IF NOT EXISTS idx_scoring_results_created 
    ON bid_scoring_results(created_at DESC);

-- scoring_evidence 索引
CREATE INDEX IF NOT EXISTS idx_evidence_result 
    ON scoring_evidence(result_id);
CREATE INDEX IF NOT EXISTS idx_evidence_page 
    ON scoring_evidence(page_idx);
CREATE INDEX IF NOT EXISTS idx_evidence_field 
    ON scoring_evidence(field_name);
CREATE INDEX IF NOT EXISTS idx_evidence_status 
    ON scoring_evidence(validation_status);
CREATE INDEX IF NOT EXISTS idx_evidence_type 
    ON scoring_evidence(evidence_type);
CREATE INDEX IF NOT EXISTS idx_evidence_chunk 
    ON scoring_evidence(chunk_id);

-- evidence_field_resolutions 索引
CREATE INDEX IF NOT EXISTS idx_resolution_result 
    ON evidence_field_resolutions(result_id);
CREATE INDEX IF NOT EXISTS idx_resolution_field 
    ON evidence_field_resolutions(field_name);

-- scoring_rule_configs 索引
CREATE INDEX IF NOT EXISTS idx_rule_config_dimension 
    ON scoring_rule_configs(dimension_id);
CREATE INDEX IF NOT EXISTS idx_rule_config_active 
    ON scoring_rule_configs(is_active) WHERE is_active = TRUE;

-- scoring_audit_log 索引
CREATE INDEX IF NOT EXISTS idx_audit_result 
    ON scoring_audit_log(result_id);
CREATE INDEX IF NOT EXISTS idx_audit_action 
    ON scoring_audit_log(action);
CREATE INDEX IF NOT EXISTS idx_audit_performed_at 
    ON scoring_audit_log(performed_at DESC);

-- scoring_runs 索引
CREATE INDEX IF NOT EXISTS idx_scoring_runs_version 
    ON scoring_runs(document_version_id);
CREATE INDEX IF NOT EXISTS idx_scoring_runs_status 
    ON scoring_runs(status);

-- =============================================================================
-- 触发器：自动更新 updated_at
-- =============================================================================

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER trigger_bid_scoring_results_updated_at
    BEFORE UPDATE ON bid_scoring_results
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trigger_scoring_rule_configs_updated_at
    BEFORE UPDATE ON scoring_rule_configs
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- 触发器：审计日志自动记录
-- =============================================================================

CREATE OR REPLACE FUNCTION log_evidence_validation_change()
RETURNS TRIGGER AS $$
BEGIN
    IF OLD.validation_status IS DISTINCT FROM NEW.validation_status THEN
        INSERT INTO scoring_audit_log (
            result_id,
            evidence_id,
            action,
            old_value,
            new_value,
            performed_at
        )
        SELECT 
            result_id,
            NEW.evidence_id,
            CASE 
                WHEN NEW.validation_status = 'confirmed' THEN 'evidence_confirmed'
                WHEN NEW.validation_status = 'rejected' THEN 'evidence_rejected'
                ELSE 'evidence_updated'
            END,
            jsonb_build_object('validation_status', OLD.validation_status, 'validation_notes', OLD.validation_notes),
            jsonb_build_object('validation_status', NEW.validation_status, 'validation_notes', NEW.validation_notes),
            COALESCE(NEW.validated_at, NOW())
        FROM scoring_evidence
        WHERE evidence_id = NEW.evidence_id;
    END IF;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER trigger_evidence_validation_audit
    AFTER UPDATE ON scoring_evidence
    FOR EACH ROW
    WHEN (OLD.validation_status IS DISTINCT FROM NEW.validation_status)
    EXECUTE FUNCTION log_evidence_validation_change();

-- =============================================================================
-- 视图：便捷的评分结果查询
-- =============================================================================

CREATE OR REPLACE VIEW v_scoring_results_summary AS
SELECT 
    sr.result_id,
    sr.bid_id,
    sr.document_version_id,
    sr.dimension_id,
    sr.dimension_name,
    sr.weight,
    sr.extracted_score,
    sr.final_score,
    sr.completeness_level,
    sr.created_at,
    COUNT(se.evidence_id) AS evidence_count,
    COUNT(CASE WHEN se.validation_status = 'confirmed' THEN 1 END) AS confirmed_evidence_count,
    COUNT(CASE WHEN se.validation_status = 'rejected' THEN 1 END) AS rejected_evidence_count
FROM bid_scoring_results sr
LEFT JOIN scoring_evidence se ON sr.result_id = se.result_id
GROUP BY sr.result_id, sr.bid_id, sr.document_version_id, sr.dimension_id, 
         sr.dimension_name, sr.weight, sr.extracted_score, sr.final_score,
         sr.completeness_level, sr.created_at;

COMMENT ON VIEW v_scoring_results_summary IS '评分结果摘要视图，包含证据统计';

-- =============================================================================
-- 函数：获取投标的完整评分结果
-- =============================================================================

CREATE OR REPLACE FUNCTION get_bid_scoring_summary(p_bid_id UUID)
RETURNS TABLE (
    dimension_id TEXT,
    dimension_name TEXT,
    weight DECIMAL,
    score DECIMAL,
    completeness_level TEXT,
    evidence_count BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        sr.dimension_id,
        sr.dimension_name,
        sr.weight,
        COALESCE(sr.final_score, sr.extracted_score, 0) AS score,
        sr.completeness_level,
        COUNT(se.evidence_id) AS evidence_count
    FROM bid_scoring_results sr
    LEFT JOIN scoring_evidence se ON sr.result_id = se.result_id
    WHERE sr.bid_id = p_bid_id
    GROUP BY sr.dimension_id, sr.dimension_name, sr.weight, 
             sr.final_score, sr.extracted_score, sr.completeness_level
    ORDER BY sr.dimension_id;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_bid_scoring_summary(UUID) IS '获取投标的完整评分汇总';
