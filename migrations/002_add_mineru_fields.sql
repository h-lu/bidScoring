-- Migration: Add MineRU-specific fields to chunks table
-- 支持所有 content_list 类型 (除了 page_number)

-- 添加新列
ALTER TABLE chunks 
    ADD COLUMN IF NOT EXISTS img_path TEXT,
    ADD COLUMN IF NOT EXISTS image_caption TEXT[],
    ADD COLUMN IF NOT EXISTS image_footnote TEXT[],
    ADD COLUMN IF NOT EXISTS table_body TEXT,
    ADD COLUMN IF NOT EXISTS table_caption TEXT[],
    ADD COLUMN IF NOT EXISTS table_footnote TEXT[],
    ADD COLUMN IF NOT EXISTS list_items TEXT[],
    ADD COLUMN IF NOT EXISTS sub_type TEXT,
    ADD COLUMN IF NOT EXISTS text_level INTEGER;

-- 更新注释
COMMENT ON COLUMN chunks.img_path IS '图片/表格图片路径 (image, table)';
COMMENT ON COLUMN chunks.image_caption IS '图片标题数组 (image)';
COMMENT ON COLUMN chunks.image_footnote IS '图片脚注数组 (image)';
COMMENT ON COLUMN chunks.table_body IS '表格HTML内容 (table)';
COMMENT ON COLUMN chunks.table_caption IS '表格标题数组 (table)';
COMMENT ON COLUMN chunks.table_footnote IS '表格脚注数组 (table)';
COMMENT ON COLUMN chunks.list_items IS '列表项数组 (list)';
COMMENT ON COLUMN chunks.sub_type IS '子类型 (list: text/ref_text)';
COMMENT ON COLUMN chunks.text_level IS '文本级别 0=正文, 1=一级标题, 2=二级标题... (text)';
