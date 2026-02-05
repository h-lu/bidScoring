"""
Citation-Aware RAG Pipeline with Precise Source Tracking
=======================================================

提供精确位置追踪和 PDF 高亮支持的 RAG Pipeline。

特性:
- 答案引用自动生成 [citation:CHUNK_ID]
- 精确位置追踪 (page_idx + bbox)
- PDF 高亮兼容的输出格式
- 支持 Small-to-Big 检索策略

Usage:
    pipeline = CitationRAGPipeline(version_id="xxx")
    result = pipeline.query("售后服务包括哪些内容？")
    # result.highlight_boxes 可用于 PDF 高亮
"""

import re
import json
import psycopg
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

from bid_scoring.config import load_settings
from bid_scoring.embeddings import embed_single_text
from bid_scoring.llm import LLMClient


@dataclass
class BoundingBox:
    """PDF 边界框坐标 (PDF 坐标系)"""
    x1: float  # 左上角 x
    y1: float  # 左上角 y
    x2: float  # 右下角 x
    y2: float  # 右下角 y
    
    def to_dict(self) -> Dict[str, float]:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}
    
    @classmethod
    def from_list(cls, bbox_list: List[float]) -> "BoundingBox":
        if len(bbox_list) >= 4:
            return cls(x1=bbox_list[0], y1=bbox_list[1], x2=bbox_list[2], y2=bbox_list[3])
        return cls(x1=0, y1=0, x2=0, y2=0)


@dataclass
class HighlightBox:
    """PDF 高亮框信息"""
    chunk_id: str           # 关联的 chunk ID
    page_idx: int          # PDF 页码 (从 0 开始)
    bbox: BoundingBox      # 边界框坐标
    text_preview: str      # 文本预览
    color: str = "yellow"  # 高亮颜色
    
    # MinerU 原始坐标 (0-1000 范围)
    raw_bbox: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "page_idx": self.page_idx,
            "bbox": self.bbox.to_dict(),
            "text_preview": self.text_preview[:100] if self.text_preview else "",
            "color": self.color,
            "raw_bbox": self.raw_bbox
        }
    
    def get_pdf_bbox(self, page_width: float, page_height: float) -> BoundingBox:
        """
        将 MinerU 坐标 (0-1000) 转换为 PDF 点坐标
        
        MinerU 的 bbox 存储在数据库中是 0-1000 范围的归一化坐标，
        需要转换为实际的 PDF 点坐标才能正确高亮。
        """
        if self.raw_bbox and len(self.raw_bbox) >= 4:
            x1 = self.raw_bbox[0] * (page_width / 1000)
            y1 = self.raw_bbox[1] * (page_height / 1000)
            x2 = self.raw_bbox[2] * (page_width / 1000)
            y2 = self.raw_bbox[3] * (page_height / 1000)
            return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
        return self.bbox


@dataclass
class SourceSpan:
    """源文本跨度信息 (用于精确引用)"""
    chunk_id: str
    source_chunk_id: str   # 原始 MinerU chunk_id
    text: str
    page_idx: int
    bbox: BoundingBox


@dataclass
class CitationContext:
    """带引用信息的上下文 (对应 section 级别)"""
    section_id: str
    heading: str
    content: str
    similarity: float
    
    # 溯源信息
    source_chunk_ids: List[str] = field(default_factory=list)
    page_range: Optional[Tuple[int, int]] = None
    
    # 精确位置 (原始 chunks)
    source_spans: List[SourceSpan] = field(default_factory=list)
    
    def to_prompt_format(self, index: int) -> str:
        """转换为 Prompt 中的引用格式"""
        return f"""
[{index}] ID: {self.section_id}
标题: {self.heading or '无标题'}
内容: {self.content[:800]}{'...' if len(self.content) > 800 else ''}
""".strip()


@dataclass
class Citation:
    """单个引用信息"""
    citation_id: str       # 引用标记，如 "[1]"
    section_id: str        # 引用的 section ID
    section_heading: str   # section 标题
    text: str             # 引用的文本片段
    highlight_boxes: List[HighlightBox] = field(default_factory=list)


@dataclass
class CitedAnswer:
    """带引用的答案 (最终输出)"""
    answer: str
    citations: List[Citation]
    highlight_boxes: List[HighlightBox]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为可序列化的字典 (用于 API 返回)"""
        return {
            "answer": self.answer,
            "citations": [
                {
                    "citation_id": c.citation_id,
                    "section_id": c.section_id,
                    "section_heading": c.section_heading,
                    "text": c.text[:200] if c.text else "",
                    "highlight_boxes": [h.to_dict() for h in c.highlight_boxes]
                }
                for c in self.citations
            ],
            "highlight_boxes": [h.to_dict() for h in self.highlight_boxes],
            "metadata": self.metadata
        }
    
    def to_json(self, indent: int = 2) -> str:
        """转换为 JSON 字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, default=str)


class CitationRetriever:
    """带精确位置追踪的检索器"""
    
    def __init__(self, version_id: str, top_k: int = 5):
        self.version_id = version_id
        self.top_k = top_k
    
    def _get_db_connection(self):
        settings = load_settings()
        return psycopg.connect(settings["DATABASE_URL"])
    
    def _fetch_chunk_spans(
        self, 
        chunk_node_ids: List[str],
        node_contents: Dict[str, str] = None
    ) -> List[SourceSpan]:
        """
        根据 hierarchical_nodes 的 chunk node IDs 获取对应的 chunks 表位置信息
        
        策略: 使用 page_range 查询对应页面的 chunks
        """
        if not chunk_node_ids:
            return []
        
        conn = self._get_db_connection()
        spans = []
        seen_chunk_ids = set()
        
        try:
            with conn.cursor() as cur:
                # 获取这些 chunk nodes 的 page_range
                placeholders = ','.join(['%s'] * len(chunk_node_ids))
                cur.execute(f"""
                    SELECT node_id, page_range, content_for_embedding
                    FROM hierarchical_nodes
                    WHERE node_id IN ({placeholders})
                      AND level = 2
                """, tuple(chunk_node_ids))
                
                node_infos = {}
                for row in cur.fetchall():
                    node_id = str(row[0])
                    page_range = row[1]
                    content = row[2]
                    node_infos[node_id] = (page_range, content)
                
                # 对每个 chunk node，查询对应页面的 chunks
                for node_id, (page_range, node_content) in node_infos.items():
                    if not page_range or len(page_range) < 2:
                        continue
                    
                    start_page, end_page = page_range[0], page_range[1]
                    
                    # 查询该页面的前 5 个 chunks（按位置排序）
                    cur.execute("""
                        SELECT chunk_id, chunk_index, page_idx, bbox, text_raw
                        FROM chunks
                        WHERE version_id = %s
                          AND page_idx >= %s 
                          AND page_idx <= %s
                          AND bbox IS NOT NULL
                          AND text_raw IS NOT NULL
                        ORDER BY chunk_index
                        LIMIT 5
                    """, (self.version_id, start_page, end_page))
                    
                    for row in cur.fetchall():
                        chunk_id = str(row[0])
                        if chunk_id in seen_chunk_ids:
                            continue
                        seen_chunk_ids.add(chunk_id)
                        
                        chunk_index, page_idx, bbox_json, text_raw = row[1], row[2], row[3], row[4]
                        
                        # 解析 bbox
                        bbox = BoundingBox.from_list(bbox_json) if bbox_json else BoundingBox(0, 0, 0, 0)
                        
                        spans.append(SourceSpan(
                            chunk_id=chunk_id,
                            source_chunk_id=chunk_id,
                            text=text_raw or "",
                            page_idx=page_idx or 0,
                            bbox=bbox
                        ))
        finally:
            conn.close()
        
        return spans
    
    def retrieve(self, query: str) -> List[CitationContext]:
        """
        检索相关 section，并获取精确位置信息
        
        流程:
        1. 嵌入 query
        2. 搜索最相似的 chunks (leaf nodes)
        3. JOIN 获取 parent sections
        4. 为每个 section 获取原始 chunk 的位置信息 (基于文本相似度匹配)
        """
        # 1. 嵌入 query
        query_embedding = embed_single_text(query)
        
        conn = self._get_db_connection()
        contexts = []
        
        try:
            with conn.cursor() as cur:
                # 2. 搜索最相似的 chunks，并 JOIN 获取 parent sections
                cur.execute("""
                    WITH ranked_chunks AS (
                        SELECT 
                            c.node_id as chunk_id,
                            c.parent_id,
                            c.heading,
                            c.content,
                            c.content_for_embedding,
                            c.source_chunk_ids,
                            c.page_range,
                            1 - (c.embedding <=> %s::vector) as similarity
                        FROM hierarchical_nodes c
                        WHERE c.version_id = %s
                            AND c.level = 2  -- leaf nodes (chunks)
                            AND c.embedding IS NOT NULL
                        ORDER BY c.embedding <=> %s::vector
                        LIMIT %s
                    )
                    SELECT DISTINCT ON (rc.parent_id)
                        s.node_id as section_id,
                        s.heading as section_heading,
                        s.content as section_content,
                        s.source_chunk_ids,
                        s.page_range,
                        rc.chunk_id as matched_chunk_id,  -- 匹配的 chunk node ID
                        rc.similarity
                    FROM ranked_chunks rc
                    JOIN hierarchical_nodes s ON rc.parent_id = s.node_id
                    WHERE s.level = 1  -- sections
                    ORDER BY rc.parent_id, rc.similarity DESC
                    LIMIT %s
                """, (query_embedding, self.version_id, 
                      query_embedding, self.top_k * 2, self.top_k))
                
                for row in cur.fetchall():
                    section_id, heading, content, source_chunk_ids, page_range, matched_chunk_id, similarity = row
                    
                    # 解析 source_chunk_ids 和 page_range
                    chunk_id_list = [str(cid) for cid in source_chunk_ids] if source_chunk_ids else []
                    page_range_tuple = tuple(page_range) if page_range else None
                    
                    # 获取精确位置信息 (基于匹配的 chunk node)
                    # 收集该 section 下所有匹配的 chunk nodes
                    source_spans = self._fetch_chunk_spans([str(matched_chunk_id)])
                    
                    context = CitationContext(
                        section_id=str(section_id),
                        heading=heading or "未命名章节",
                        content=content or "",
                        similarity=float(similarity),
                        source_chunk_ids=chunk_id_list,
                        page_range=page_range_tuple,
                        source_spans=source_spans
                    )
                    contexts.append(context)
        finally:
            conn.close()
        
        # 按相似度排序
        contexts.sort(key=lambda x: x.similarity, reverse=True)
        return contexts[:self.top_k]


class CitationRAGPipeline:
    """
    带精确位置追踪的 RAG Pipeline
    
    特点:
    - LLM 生成带 [citation:ID] 标记的答案
    - 自动提取引用并生成高亮框
    - 支持 PDF 坐标级别的高亮
    """
    
    # Citation-Aware System Prompt
    CITATION_SYSTEM_PROMPT = """你是专业的投标分析助手。请基于提供的参考资料回答问题。

重要规则:
1. **每个事实性陈述都必须标注引用**，格式: [citation:ID]
   - ID 是参考资料中标记的编号，如 [1], [2] 等
   - 引用应紧跟在相关陈述之后
2. 只使用提供的参考资料，禁止引入外部知识
3. 如果无法从资料中找到答案，说明"根据现有资料无法回答"
4. 保持回答简洁准确，优先引用相似度高的资料

引用格式示例:
- 售后服务热线是 400-650-6632 [citation:1]
- 质保期为自验收合格之日起 5 年 [citation:2][citation:3]

参考资料按相似度从高到低排列。"""

    def __init__(self, version_id: str, top_k: int = 5):
        self.version_id = version_id
        self.retriever = CitationRetriever(version_id, top_k=top_k)
        
        # 用于匹配引用标记的正则
        self.citation_pattern = re.compile(r'\[citation:(\d+)\]')
    
    def _build_user_prompt(self, query: str, contexts: List[CitationContext]) -> str:
        """构建带引用的 User Prompt"""
        contexts_text = "\n\n".join(
            ctx.to_prompt_format(i + 1) 
            for i, ctx in enumerate(contexts)
        )
        
        return f"""
参考资料:
{contexts_text}

问题: {query}

请用中文回答，并在每个事实性陈述后标注引用 [citation:ID]。"""
    
    def _extract_citations(self, answer: str, contexts: List[CitationContext]) -> List[Citation]:
        """
        从答案中提取引用标记，并生成 Citation 对象
        
        匹配 [citation:1] -> 对应 contexts[0]
        """
        citations = []
        seen_ids = set()
        
        for match in self.citation_pattern.finditer(answer):
            citation_num = int(match.group(1))
            citation_id = match.group(0)  # [citation:1]
            
            if citation_num < 1 or citation_num > len(contexts):
                continue
            if citation_id in seen_ids:
                continue
            
            seen_ids.add(citation_id)
            context = contexts[citation_num - 1]
            
            # 为每个 source_span 生成 highlight box
            highlight_boxes = []
            for span in context.source_spans:
                # 存储原始 bbox (MinerU 0-1000 格式) 用于后续 PDF 坐标转换
                raw_bbox = [span.bbox.x1, span.bbox.y1, span.bbox.x2, span.bbox.y2]
                highlight_boxes.append(HighlightBox(
                    chunk_id=span.chunk_id,
                    page_idx=span.page_idx,
                    bbox=span.bbox,
                    text_preview=span.text[:100],
                    color="yellow",
                    raw_bbox=raw_bbox
                ))
            
            citation = Citation(
                citation_id=citation_id,
                section_id=context.section_id,
                section_heading=context.heading,
                text=context.content[:300],
                highlight_boxes=highlight_boxes
            )
            citations.append(citation)
        
        return citations
    
    def _build_highlight_boxes(self, citations: List[Citation]) -> List[HighlightBox]:
        """收集所有高亮框 (去重)"""
        seen_chunks = set()
        boxes = []
        
        for citation in citations:
            for box in citation.highlight_boxes:
                if box.chunk_id not in seen_chunks:
                    seen_chunks.add(box.chunk_id)
                    boxes.append(box)
        
        # 按页码排序
        boxes.sort(key=lambda x: (x.page_idx, x.bbox.y1))
        return boxes
    
    def query(self, query: str, temperature: float = 0.3) -> CitedAnswer:
        """
        执行带精确位置追踪的 RAG 查询
        
        Args:
            query: 用户问题
            temperature: LLM 温度参数
        
        Returns:
            CitedAnswer: 包含答案、引用和高亮框
        """
        # 1. 检索相关 section
        contexts = self.retriever.retrieve(query)
        
        if not contexts:
            return CitedAnswer(
                answer="根据现有资料无法回答该问题。",
                citations=[],
                highlight_boxes=[],
                metadata={"query": query, "retrieved_count": 0}
            )
        
        # 2. 构建 Prompt
        system_prompt = self.CITATION_SYSTEM_PROMPT
        user_prompt = self._build_user_prompt(query, contexts)
        
        # 3. 调用 LLM
        llm_client = LLMClient(load_settings())
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        raw_answer = llm_client.complete(
            messages=messages,
            temperature=temperature
        )
        
        # 4. 提取引用
        citations = self._extract_citations(raw_answer, contexts)
        
        # 5. 生成高亮框
        highlight_boxes = self._build_highlight_boxes(citations)
        
        return CitedAnswer(
            answer=raw_answer,
            citations=citations,
            highlight_boxes=highlight_boxes,
            metadata={
                "query": query,
                "retrieved_count": len(contexts),
                "citation_count": len(citations),
                "highlight_count": len(highlight_boxes),
                "version_id": self.version_id,
                "timestamp": datetime.now().isoformat()
            }
        )


# 便捷函数
def query_with_citations(version_id: str, query: str, top_k: int = 5) -> CitedAnswer:
    """便捷函数：执行带引用的 RAG 查询"""
    pipeline = CitationRAGPipeline(version_id=version_id, top_k=top_k)
    return pipeline.query(query)


if __name__ == "__main__":
    # 测试
    import os
    from dotenv import load_dotenv
    load_dotenv(override=True)
    
    VERSION_ID = "9a5a0214-3b98-4a64-9194-a01648479f7a"
    
    print("🧪 测试 Citation-Aware RAG Pipeline")
    print("=" * 50)
    
    pipeline = CitationRAGPipeline(version_id=VERSION_ID, top_k=3)
    
    test_queries = [
        "售后服务包括哪些内容？",
        "质保期是多长时间？",
        "培训内容包括哪些？"
    ]
    
    for query in test_queries[:1]:  # 先测试第一个
        print(f"\n❓ 问题: {query}")
        print("-" * 50)
        
        result = pipeline.query(query)
        
        print(f"\n💡 答案:\n{result.answer}")
        print(f"\n📊 统计:")
        print(f"  - 检索到 {result.metadata['retrieved_count']} 个 sections")
        print(f"  - 生成 {result.metadata['citation_count']} 个引用")
        print(f"  - 可高亮 {len(result.highlight_boxes)} 个区域")
        
        if result.highlight_boxes:
            print(f"\n📍 高亮框预览 (前 3 个):")
            for box in result.highlight_boxes[:3]:
                print(f"  页 {box.page_idx}: bbox={box.bbox.to_dict()}, text={box.text_preview[:40]}...")
        
        print(f"\n📋 JSON 输出预览:")
        print(result.to_json()[:800] + "...")
