# CPC 架构重构计划：从碎片优先到结构优先

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** 重构 CPC 管道，将处理顺序从"Contextual → HiChunk → RAPTOR"改为"Structure Rebuild → Contextual → RAPTOR"，解决碎片上下文质量问题并降低 LLM 成本。

**Architecture:** 
- 新架构：先合并碎片为自然段落和章节，再对完整语义单元生成上下文
- 核心技术：基于 text_level 的层次识别 + 基于语义长度的段落合并 + Parent-Child 关联
- 参考：LangChain ParentDocumentRetriever、LlamaIndex AutoMerging、Anthropic Contextual Retrieval

**Tech Stack:** Python 3.11+, PostgreSQL, pgvector, OpenAI API, scikit-learn

---

## 关键设计决策

### 1. 段落合并策略

```python
# 合并条件（同时满足才合并）
1. 当前 chunk 长度 < MIN_PARAGRAPH_LENGTH (80字符)
2. 下一个是普通文本（非标题、非表格）
3. 页码相同（跨页不合并，避免混乱）
4. 当前不以句末标点结束（.!?。！？）

# 停止合并触发条件
1. 遇到 text_level=1 的标题
2. 遇到 element_type=table/image 的特殊元素
3. 累积长度 >= MAX_PARAGRAPH_LENGTH (500字符)
4. 页码变化
```

### 2. 层次结构设计

```
Document (Level 2)
├── Section (Level 1) - 由 text_level=1 识别
│   ├── Paragraph (Level 0) - 合并后的自然段
│   │   └── Original Chunks (Level -1) - 原始碎片
│   └── Paragraph (Level 0)
├── Section (Level 1)
│   └── ...
```

### 3. 上下文生成策略

| 节点类型 | 长度 | 处理方式 | 原因 |
|---------|------|---------|------|
| 碎片 (<50字符) | 短 | 跳过 LLM，使用 Rule-based | 无完整语义 |
| 段落 (50-500字符) | 中 | LLM 生成完整上下文 | 有完整语义 |
| 章节 (>500字符) | 长 | LLM 生成摘要 | 需要概括 |

---

## Task 1: 创建 structure_rebuilder.py 核心模块

**Files:**
- Create: `bid_scoring/structure_rebuilder.py`
- Test: `tests/test_structure_rebuilder.py`

**背景知识:**
- 当前 chunks 表包含：chunk_id, text_raw, text_level, element_type, page_idx
- text_level=1 表示一级标题（如"一、投标函"）
- 有大量 <50 字符的碎片（"四"、"#"、"合同"）
- 参考：LangChain ParentDocumentRetriever、LlamaIndex HierarchicalNodeParser

**Step 1: 写测试 - 段落合并逻辑**

```python
# tests/test_structure_rebuilder.py
def test_merge_short_chunks_into_paragraph():
    """测试将短句合并为段落"""
    from bid_scoring.structure_rebuilder import ParagraphMerger
    
    chunks = [
        {"chunk_id": "1", "text_raw": "细胞和组织", "text_level": None, "page_idx": 1, "chunk_index": 0},
        {"chunk_id": "2", "text_raw": "本身会发出荧光", "text_level": None, "page_idx": 1, "chunk_index": 1},
        {"chunk_id": "3", "text_raw": "这种自体荧光会干扰观察", "text_level": None, "page_idx": 1, "chunk_index": 2},
    ]
    
    merger = ParagraphMerger(min_length=80, max_length=500)
    paragraphs = merger.merge(chunks)
    
    assert len(paragraphs) == 1
    assert "细胞和组织" in paragraphs[0]["content"]
    assert "本身会发出荧光" in paragraphs[0]["content"]
    assert paragraphs[0]["merged_count"] == 3
```

**Step 2: 运行测试，确认失败**

```bash
cd /Users/wangxq/Documents/投标分析_kimi
uv run pytest tests/test_structure_rebuilder.py::test_merge_short_chunks_into_paragraph -v
```
Expected: FAIL with "ParagraphMerger not defined"

**Step 3: 实现段落合并器**

```python
# bid_scoring/structure_rebuilder.py
"""文档结构重建器 - 从碎片还原层次结构"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any
from collections import defaultdict
import uuid

logger = logging.getLogger(__name__)

# 合并阈值配置
MIN_PARAGRAPH_LENGTH = 80      # 段落最小长度，小于此值尝试合并
MAX_PARAGRAPH_LENGTH = 500     # 段落最大长度，大于此值停止合并
SENTENCE_ENDINGS = {'。', '；', '.', ';', '！', '!', '？', '?'}


@dataclass
class RebuiltNode:
    """重建后的结构节点"""
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    node_type: str = ""           # 'document', 'section', 'paragraph'
    level: int = 0                # 0=paragraph, 1=section, 2=document
    content: str = ""             # 合并后的内容
    heading: str = ""             # 所属章节标题
    page_range: Tuple[int, int] = (0, 0)
    source_chunks: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    children: List['RebuiltNode'] = field(default_factory=list)
    context: str = ""             # 生成的上下文


class ParagraphMerger:
    """段落合并器 - 将短句合并为自然段落"""
    
    def __init__(self, min_length: int = MIN_PARAGRAPH_LENGTH, 
                 max_length: int = MAX_PARAGRAPH_LENGTH):
        self.min_length = min_length
        self.max_length = max_length
    
    def _should_merge_with_next(self, current: Dict, next_chunk: Optional[Dict]) -> bool:
        """判断当前 chunk 是否应该与下一个合并"""
        if next_chunk is None:
            return False
            
        current_text = current.get('text_raw', '').strip()
        current_len = len(current_text)
        
        # 1. 当前已是长文本，不合并
        if current_len >= self.min_length:
            return False
        
        # 2. 当前是标题，不合并
        if current.get('text_level') == 1:
            return False
        
        # 3. 下一个是标题，不合并
        if next_chunk.get('text_level') == 1:
            return False
        
        # 4. 当前以句末标点结束，可能完整，不合并
        if current_text and current_text[-1] in SENTENCE_ENDINGS:
            return False
        
        # 5. 页码变化，不合并
        if current.get('page_idx') != next_chunk.get('page_idx'):
            return False
        
        # 6. 特殊元素类型，不合并
        special_types = {'table', 'image', 'header', 'footer'}
        if current.get('element_type') in special_types:
            return False
        if next_chunk.get('element_type') in special_types:
            return False
        
        return True
    
    def merge(self, chunks: List[Dict]) -> List[Dict]:
        """将 chunks 合并为段落"""
        if not chunks:
            return []
        
        # 按页码和索引排序
        sorted_chunks = sorted(
            chunks,
            key=lambda c: (c.get('page_idx', 0), c.get('chunk_index', 0))
        )
        
        paragraphs = []
        current_buffer = []
        current_length = 0
        
        for i, chunk in enumerate(sorted_chunks):
            chunk_text = chunk.get('text_raw', '').strip()
            if not chunk_text:
                continue
            
            is_heading = chunk.get('text_level') == 1
            is_long = len(chunk_text) >= self.min_length
            
            # 情况1: 标题 - 独立成段，先结束缓冲区
            if is_heading:
                if current_buffer:
                    paragraphs.append(self._create_paragraph(current_buffer))
                    current_buffer = []
                    current_length = 0
                
                paragraphs.append({
                    'type': 'heading',
                    'content': chunk_text,
                    'level': 1,
                    'page_idx': chunk.get('page_idx', 0),
                    'source_chunks': [chunk.get('chunk_id')],
                    'is_heading': True,
                })
                continue
            
            # 情况2: 长文本 - 独立成段，先结束缓冲区
            if is_long:
                if current_buffer:
                    paragraphs.append(self._create_paragraph(current_buffer))
                    current_buffer = []
                    current_length = 0
                
                paragraphs.append({
                    'type': 'paragraph',
                    'content': chunk_text,
                    'level': 0,
                    'page_idx': chunk.get('page_idx', 0),
                    'source_chunks': [chunk.get('chunk_id')],
                    'is_heading': False,
                })
                continue
            
            # 情况3: 短文本 - 加入缓冲区
            current_buffer.append(chunk)
            current_length += len(chunk_text)
            
            # 检查是否需要结束缓冲区
            next_chunk = sorted_chunks[i + 1] if i + 1 < len(sorted_chunks) else None
            should_end = (
                current_length >= self.min_length or
                next_chunk is None or
                not self._should_merge_with_next(chunk, next_chunk)
            )
            
            if should_end:
                paragraphs.append(self._create_paragraph(current_buffer))
                current_buffer = []
                current_length = 0
        
        return paragraphs
    
    def _create_paragraph(self, chunks: List[Dict]) -> Dict:
        """从 chunk 列表创建段落"""
        contents = [c.get('text_raw', '').strip() for c in chunks]
        merged = ' '.join(contents)
        merged = ' '.join(merged.split())  # 清理多余空格
        
        page_indices = [c.get('page_idx', 0) for c in chunks]
        
        return {
            'type': 'paragraph',
            'content': merged,
            'level': 0,
            'page_idx': min(page_indices),
            'page_range': (min(page_indices), max(page_indices)),
            'source_chunks': [c.get('chunk_id') for c in chunks],
            'is_heading': False,
            'merged_count': len(chunks),
        }
```

**Step 4: 运行测试，确认通过**

```bash
uv run pytest tests/test_structure_rebuilder.py::test_merge_short_chunks_into_paragraph -v
```
Expected: PASS

**Step 5: 提交**

```bash
git add bid_scoring/structure_rebuilder.py tests/test_structure_rebuilder.py
git commit -m "feat(structure): add paragraph merger for short chunks"
```

---

## Task 2: 实现章节树构建

**Files:**
- Modify: `bid_scoring/structure_rebuilder.py` (添加 TreeBuilder 类)
- Test: `tests/test_structure_rebuilder.py` (添加测试)

**Step 1: 写测试 - 章节识别**

```python
# tests/test_structure_rebuilder.py
def test_build_section_tree_from_headings():
    """测试从标题构建章节树"""
    from bid_scoring.structure_rebuilder import TreeBuilder
    
    paragraphs = [
        {"type": "heading", "content": "一、技术规格", "level": 1, "page_idx": 1, "is_heading": True},
        {"type": "paragraph", "content": "激光共聚焦显微镜参数如下", "level": 0, "page_idx": 1, "is_heading": False},
        {"type": "paragraph", "content": "分辨率: 0.5微米", "level": 0, "page_idx": 1, "is_heading": False},
        {"type": "heading", "content": "二、商务条款", "level": 1, "page_idx": 2, "is_heading": True},
        {"type": "paragraph", "content": "质保期5年", "level": 0, "page_idx": 2, "is_heading": False},
    ]
    
    builder = TreeBuilder()
    sections = builder.build_sections(paragraphs)
    
    assert len(sections) == 2
    assert sections[0].heading == "一、技术规格"
    assert sections[1].heading == "二、商务条款"
    assert len(sections[0].children) == 2  # 两个段落
```

**Step 2: 运行测试，确认失败**

```bash
uv run pytest tests/test_structure_rebuilder.py::test_build_section_tree_from_headings -v
```
Expected: FAIL with "TreeBuilder not defined"

**Step 3: 实现树构建器**

```python
# bid_scoring/structure_rebuilder.py (添加到文件末尾)

class TreeBuilder:
    """树构建器 - 从段落构建章节层次结构"""
    
    def build_sections(self, paragraphs: List[Dict]) -> List[RebuiltNode]:
        """构建章节树"""
        sections = []
        current_section = None
        current_paragraphs = []
        
        for para in paragraphs:
            if para.get('is_heading'):
                # 保存之前的章节
                if current_section and current_paragraphs:
                    current_section.children = self._create_paragraph_nodes(current_paragraphs)
                    sections.append(current_section)
                    current_paragraphs = []
                
                # 创建新章节
                current_section = RebuiltNode(
                    node_type='section',
                    level=1,
                    heading=para['content'],
                    content=para['content'],
                    page_range=(para['page_idx'], para['page_idx']),
                    metadata={
                        'heading_level': para.get('level', 1),
                        'source_type': 'heading',
                    }
                )
            else:
                # 普通段落
                if current_section is None:
                    # 还没有遇到第一个标题，创建默认章节
                    current_section = RebuiltNode(
                        node_type='section',
                        level=1,
                        heading='文档开头',
                        content='文档开头内容',
                        page_range=(para.get('page_idx', 0), para.get('page_idx', 0)),
                        metadata={'heading_level': 0, 'is_default': True}
                    )
                current_paragraphs.append(para)
        
        # 处理最后一个章节
        if current_section:
            if current_paragraphs:
                current_section.children = self._create_paragraph_nodes(current_paragraphs)
            sections.append(current_section)
        
        return sections
    
    def _create_paragraph_nodes(self, paragraphs: List[Dict]) -> List[RebuiltNode]:
        """将段落字典转换为节点"""
        return [
            RebuiltNode(
                node_type='paragraph',
                level=0,
                content=p['content'],
                page_range=p.get('page_range', (p['page_idx'], p['page_idx'])),
                source_chunks=p.get('source_chunks', []),
                metadata={
                    'merged_count': p.get('merged_count', 1),
                    'is_heading': p.get('is_heading', False),
                }
            )
            for p in paragraphs
        ]
    
    def build_document_tree(self, sections: List[RebuiltNode], 
                           document_title: str) -> RebuiltNode:
        """构建完整文档树"""
        all_pages = []
        for section in sections:
            all_pages.extend([section.page_range[0], section.page_range[1]])
        
        return RebuiltNode(
            node_type='document',
            level=2,
            heading=document_title,
            content=document_title,
            page_range=(min(all_pages) if all_pages else 0,
                       max(all_pages) if all_pages else 0),
            children=sections,
            metadata={
                'section_count': len(sections),
                'paragraph_count': sum(len(s.children) for s in sections),
            }
        )
```

**Step 4: 运行测试，确认通过**

```bash
uv run pytest tests/test_structure_rebuilder.py::test_build_section_tree_from_headings -v
```
Expected: PASS

**Step 5: 提交**

```bash
git add bid_scoring/structure_rebuilder.py tests/test_structure_rebuilder.py
git commit -m "feat(structure): add tree builder for section hierarchy"
```

---

## Task 3: 实现上下文生成（分层策略）

**Files:**
- Modify: `bid_scoring/structure_rebuilder.py` (添加 ContextGenerator)
- Modify: `bid_scoring/contextual_retrieval.py` (添加长度检查)
- Test: `tests/test_structure_rebuilder.py`

**Step 1: 写测试 - 分层上下文生成**

```python
# tests/test_structure_rebuilder.py
def test_skip_short_chunks_for_context():
    """测试跳过短 chunk 的 LLM 上下文生成"""
    from bid_scoring.structure_rebuilder import HierarchicalContextGenerator
    
    # Mock LLM（验证不会被调用）
    class MockLLM:
        def __init__(self):
            self.call_count = 0
        def generate(self, *args, **kwargs):
            self.call_count += 1
            return "mocked context"
    
    mock_llm = MockLLM()
    generator = HierarchicalContextGenerator(llm_client=mock_llm)
    
    # 短段落（<50字符）- 不应该调用 LLM
    short_para = RebuiltNode(
        node_type='paragraph',
        level=0,
        content="细胞",
        heading="技术规格"
    )
    context = generator.generate_for_node(short_para, "显微镜文档")
    assert mock_llm.call_count == 0  # 没有调用 LLM
    assert "显微镜文档" in context  # 使用 rule-based
    
    # 中等段落（50-500字符）- 应该调用 LLM
    medium_para = RebuiltNode(
        node_type='paragraph',
        level=0,
        content="细胞和组织本身会发出荧光，这种自体荧光会干扰观察。共聚焦显微镜可以有效解决这个问题。",
        heading="技术规格"
    )
    context = generator.generate_for_node(medium_para, "显微镜文档")
    assert mock_llm.call_count == 1  # 调用了 LLM
```

**Step 2: 运行测试，确认失败**

**Step 3: 实现分层上下文生成器**

```python
# bid_scoring/structure_rebuilder.py (添加到文件末尾)

class HierarchicalContextGenerator:
    """分层上下文生成器 - 根据节点类型和长度采用不同策略"""
    
    SHORT_THRESHOLD = 50      # 短文本阈值，跳过 LLM
    MEDIUM_THRESHOLD = 500    # 中等文本阈值，正常 LLM
    
    def __init__(self, llm_client=None, document_title: str = ""):
        self.llm_client = llm_client
        self.document_title = document_title
        self.stats = {
            'short_skipped': 0,
            'medium_processed': 0,
            'long_processed': 0,
        }
    
    def generate_for_tree(self, root: RebuiltNode) -> None:
        """为整棵树生成上下文"""
        for section in root.children:
            self._generate_for_section(section)
    
    def _generate_for_section(self, section: RebuiltNode) -> None:
        """为章节及其段落生成上下文"""
        # 为段落生成上下文
        for para in section.children:
            para.context = self.generate_for_node(para, section.heading)
        
        # 为章节生成摘要（基于子段落）
        if section.children:
            combined = ' '.join([p.content for p in section.children[:3]])
            section.context = self._generate_summary(
                combined[:400],
                section.heading
            )
    
    def generate_for_node(self, node: RebuiltNode, section_title: str) -> str:
        """为单个节点生成上下文"""
        content = node.content
        content_len = len(content)
        
        # 策略1: 超短文本 (<50字符) - 跳过 LLM
        if content_len < self.SHORT_THRESHOLD:
            self.stats['short_skipped'] += 1
            return self._generate_rule_based_context(section_title)
        
        # 策略2: 中等文本 (50-500字符) - 使用 LLM
        if content_len <= self.MEDIUM_THRESHOLD:
            if self.llm_client:
                self.stats['medium_processed'] += 1
                return self._call_llm_for_context(content, section_title)
            else:
                return self._generate_rule_based_context(section_title)
        
        # 策略3: 长文本 (>500字符) - 使用摘要模式
        self.stats['long_processed'] += 1
        return self._generate_summary(content[:600], section_title)
    
    def _generate_rule_based_context(self, section_title: str) -> str:
        """基于规则的上下文（无 LLM）"""
        if section_title and section_title != "文档开头":
            return f"此内容来自《{self.document_title}》的「{section_title}」部分。"
        return f"此内容来自《{self.document_title}》。"
    
    def _call_llm_for_context(self, content: str, section_title: str) -> str:
        """调用 LLM 生成上下文"""
        if not self.llm_client:
            return self._generate_rule_based_context(section_title)
        
        prompt = f"""你是一位专业的文档分析助手。请为以下文档片段生成一段简洁的上下文描述（1-2句话）。

## 文档信息
- 文档标题：{self.document_title}
- 章节标题：{section_title}

## 需要分析的片段
```
{content}
```

## 任务要求
请生成一段上下文描述，说明：
1. 这个片段包含什么核心信息
2. 它在文档中的作用或意义

## 输出格式
直接输出上下文描述，不要添加任何解释、前缀或后缀。描述应该简洁明了（30-50字）。"""
        
        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "你是专业的文档分析助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=100
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"LLM 调用失败: {e}")
            return self._generate_rule_based_context(section_title)
    
    def _generate_summary(self, content: str, section_title: str) -> str:
        """生成摘要（用于长文本）"""
        preview = content[:150] + "..." if len(content) > 150 else content
        return f"来自《{self.document_title}》「{section_title}」：{preview}"
    
    def get_stats(self) -> Dict:
        """获取处理统计"""
        total = sum(self.stats.values())
        return {
            **self.stats,
            'total': total,
            'llm_savings_percent': (self.stats['short_skipped'] / total * 100) if total > 0 else 0
        }
```

**Step 4: 运行测试，确认通过**

**Step 5: 提交**

```bash
git add bid_scoring/structure_rebuilder.py tests/test_structure_rebuilder.py
git commit -m "feat(structure): add hierarchical context generation with length-based strategy"
```

---

## Task 4: 重构 cpc_pipeline.py 使用新架构

**Files:**
- Modify: `bid_scoring/cpc_pipeline.py`
- Modify: `tests/test_cpc_pipeline.py`

**Step 1: 修改 process_document 方法**

```python
# bid_scoring/cpc_pipeline.py 关键修改

async def process_document(
    self,
    content_list: List[Dict],
    document_title: str,
) -> ProcessResult:
    """处理文档 - 修正后的流程"""
    
    from bid_scoring.structure_rebuilder import (
        ParagraphMerger, TreeBuilder, HierarchicalContextGenerator
    )
    
    errors = []
    
    try:
        # ========== 阶段1: 结构重建 ==========
        logger.info("阶段1: 重建文档结构...")
        
        # 1.1 合并段落
        merger = ParagraphMerger()
        paragraphs = merger.merge(content_list)
        logger.info(f"  合并为 {len(paragraphs)} 个段落/标题")
        
        # 1.2 构建章节树
        tree_builder = TreeBuilder()
        sections = tree_builder.build_sections(paragraphs)
        doc_root = tree_builder.build_document_tree(sections, document_title)
        logger.info(f"  构建 {len(sections)} 个章节")
        
        # ========== 阶段2: 上下文生成（对完整段落）==========
        if self.config.enable_contextual and self.contextual_generator:
            logger.info("阶段2: 生成上下文...")
            context_gen = HierarchicalContextGenerator(
                llm_client=self.llm_client,
                document_title=document_title
            )
            context_gen.generate_for_tree(doc_root)
            
            stats = context_gen.get_stats()
            logger.info(f"  统计: {stats}")
        
        # ========== 阶段3: 存储到数据库 ==========
        logger.info("阶段3: 存储结构到数据库...")
        self._store_rebuilt_structure(doc_root)
        
        # ========== 阶段4: 可选的 RAPTOR ==========
        if self.config.enable_raptor:
            logger.info("阶段4: 构建 RAPTOR 树...")
            self._build_raptor_on_structure(doc_root)
        
        return ProcessResult(
            success=True,
            nodes_created=len(self._flatten_tree(doc_root)),
            root_node_id=doc_root.node_id,
        )
        
    except Exception as e:
        logger.error(f"处理失败: {e}")
        return ProcessResult(success=False, errors=[str(e)])
```

**Step 2: 运行集成测试**

```bash
uv run pytest tests/test_cpc_pipeline.py -v -k "test_full_pipeline"
```

**Step 3: 提交**

```bash
git add bid_scoring/cpc_pipeline.py tests/test_cpc_pipeline.py
git commit -m "refactor(cpc): restructure pipeline to use structure-first approach"
```

---

## Task 5: 数据库迁移 - 新表结构

**Files:**
- Create: `migrations/008_cpc_structure_rebuild.sql`

**Step 1: 创建新的结构表**

```sql
-- migrations/008_cpc_structure_rebuild.sql
-- CPC 结构重建表 - 存储重建后的层次结构

-- 文档结构节点表（替代/补充 hierarchical_nodes）
create table if not exists document_structure_nodes (
    node_id uuid primary key default gen_random_uuid(),
    version_id uuid not null references document_versions(version_id) on delete cascade,
    
    -- 层次信息
    level integer not null check (level in (0, 1, 2)),  -- 0=paragraph, 1=section, 2=document
    node_type text not null check (node_type in ('paragraph', 'section', 'document')),
    
    -- 内容
    content text not null,
    heading text,  -- 章节标题（section 类型）
    context text,  -- 生成的上下文
    
    -- 树结构关系
    parent_id uuid references document_structure_nodes(node_id) on delete cascade,
    source_chunk_ids uuid[] default '{}',  -- 关联的原始 chunks
    
    -- 位置信息
    start_page integer,
    end_page integer,
    
    -- 元数据
    metadata jsonb default '{}',
    
    -- 统计
    merged_chunk_count integer default 1,  -- 合并了多少个原始 chunk
    
    created_at timestamptz default now(),
    updated_at timestamptz default now()
);

-- 索引
create index if not exists idx_structure_nodes_version on document_structure_nodes(version_id);
create index if not exists idx_structure_nodes_parent on document_structure_nodes(parent_id);
create index if not exists idx_structure_nodes_level on document_structure_nodes(version_id, level);
create index if not exists idx_structure_nodes_pages on document_structure_nodes(start_page, end_page);

-- 触发器：自动更新 updated_at
create or replace function update_updated_at_column()
returns trigger as $$
begin
    new.updated_at = now();
    return new;
end;
$$ language plpgsql;

create trigger trigger_structure_nodes_updated_at
    before update on document_structure_nodes
    for each row
    execute function update_updated_at_column();

-- 注释
comment on table document_structure_nodes is '重建后的文档层次结构：document -> section -> paragraph';
```

**Step 2: 提交**

```bash
git add migrations/008_cpc_structure_rebuild.sql
git commit -m "feat(db): add document_structure_nodes table for rebuilt hierarchy"
```

---

## Task 6: 集成测试与验证

**Files:**
- Test: `tests/test_cpc_integration.py`

**Step 1: 编写端到端集成测试**

```python
# tests/test_cpc_integration.py
"""CPC 重构后的端到端集成测试"""

import pytest
from bid_scoring.cpc_pipeline import CPCPipeline, CPCPipelineConfig


@pytest.mark.asyncio
async def test_structure_first_pipeline():
    """测试结构优先的新管道"""
    
    # 模拟真实数据：包含碎片、标题、正文
    content_list = [
        # 标题
        {"chunk_id": "1", "text_raw": "一、技术规格", "text_level": 1, "page_idx": 1, "chunk_index": 0},
        # 短句碎片（应合并）
        {"chunk_id": "2", "text_raw": "细胞和组织", "text_level": None, "page_idx": 1, "chunk_index": 1},
        {"chunk_id": "3", "text_raw": "本身会发出荧光", "text_level": None, "page_idx": 1, "chunk_index": 2},
        {"chunk_id": "4", "text_raw": "这种自体荧光会干扰观察", "text_level": None, "page_idx": 1, "chunk_index": 3},
        # 长句（独立成段）
        {"chunk_id": "5", "text_raw": "共聚焦显微镜技术可以有效解决这个问题，通过激光扫描和点探测技术实现高分辨率成像。", "text_level": None, "page_idx": 1, "chunk_index": 4},
        # 另一个标题
        {"chunk_id": "6", "text_raw": "二、商务条款", "text_level": 1, "page_idx": 2, "chunk_index": 5},
        {"chunk_id": "7", "text_raw": "质保期5年，含现场支持", "text_level": None, "page_idx": 2, "chunk_index": 6},
    ]
    
    config = CPCPipelineConfig(
        enable_contextual=True,
        enable_hichunk=False,  # 新架构替代了 hichunk
        enable_raptor=False,
    )
    
    pipeline = CPCPipeline(config=config)
    result = await pipeline.process_document(
        content_list=content_list,
        document_title="测试文档"
    )
    
    assert result.success
    
    # 验证结构
    # - 应该合并碎片为1个段落
    # - 应该有2个章节
    # - 总节点数：1 document + 2 sections + 3 paragraphs = 6
    assert result.nodes_created >= 6


def test_paragraph_merging_with_real_data():
    """使用真实数据库数据测试段落合并"""
    import psycopg
    from bid_scoring.config import load_settings
    from bid_scoring.structure_rebuilder import ParagraphMerger
    
    settings = load_settings()
    
    with psycopg.connect(settings["DATABASE_URL"]) as conn:
        with conn.cursor() as cur:
            # 获取一个真实文档的 chunks
            cur.execute("""
                SELECT chunk_id, text_raw, text_level, page_idx, chunk_index
                FROM chunks
                WHERE version_id = (
                    SELECT version_id FROM document_versions LIMIT 1
                )
                ORDER BY page_idx, chunk_index
                LIMIT 50
            """)
            
            chunks = [
                {
                    "chunk_id": row[0],
                    "text_raw": row[1],
                    "text_level": row[2],
                    "page_idx": row[3],
                    "chunk_index": row[4]
                }
                for row in cur.fetchall()
            ]
    
    merger = ParagraphMerger(min_length=80, max_length=500)
    paragraphs = merger.merge(chunks)
    
    # 验证合并效果
    original_count = len(chunks)
    merged_count = len(paragraphs)
    
    print(f"\n原始 chunks: {original_count}")
    print(f"合并后 paragraphs: {merged_count}")
    print(f"压缩比: {merged_count/original_count:.1%}")
    
    # 应该显著减少数量
    assert merged_count < original_count * 0.8
    
    # 验证没有超短段落
    for para in paragraphs:
        if not para.get('is_heading'):
            assert len(para['content']) >= 20, f"段落过短: {para['content']}"
```

**Step 2: 运行集成测试**

```bash
uv run pytest tests/test_cpc_integration.py -v
```

**Step 3: 提交**

```bash
git add tests/test_cpc_integration.py
git commit -m "test(cpc): add integration tests for structure-first pipeline"
```

---

## 验收标准

### 功能验收
- [ ] 碎片正确合并为自然段落
- [ ] text_level=1 正确识别为章节标题
- [ ] 段落长度在 80-500 字符范围内
- [ ] 短文本 (<50字符) 跳过 LLM，使用 rule-based
- [ ] 长文本正常生成上下文

### 性能验收
- [ ] LLM 调用次数减少 >= 50%
- [ ] 处理时间减少 >= 30%
- [ ] 所有测试通过

### 数据验收
- [ ] document_structure_nodes 表正确存储层次结构
- [ ] 与原始 chunks 的关联正确
- [ ] 支持新旧架构并行运行

---

## 风险与回退方案

### 风险1: 合并过度导致语义丢失
**缓解**: 设置 MAX_PARAGRAPH_LENGTH=500，遇到标题强制分割

### 风险2: 旧数据处理失败
**缓解**: 保留旧代码作为 fallback，新代码标记为 experimental

### 风险3: 下游检索受影响
**缓解**: 新表结构与原 hierarchical_nodes 并存，逐步迁移

---

## 执行选择

**Plan saved to:** `docs/plans/2026-02-04-cpc-structure-rebuild-plan.md`

**Execution options:**

1. **Subagent-Driven (this session)** - 我为每个 task 调度新鲜子 agent，task 间 review，快速迭代

2. **Parallel Session (separate)** - 打开新 session 使用 executing-plans，批量执行带 checkpoint

**Which approach?** (Recommend option 1 for this critical architecture change)
