#!/usr/bin/env python3
"""
Citation-Aware RAG + PDF Highlight Demo
=======================================

演示带精确位置追踪的 RAG 系统，输出可用于 PDF 高亮的数据结构。

Usage:
    python scripts/demo_citation_rag.py "售后服务包括哪些内容？"
"""

import os
import sys
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(override=True)

from bid_scoring.citation_rag_pipeline import CitationRAGPipeline


def print_highlight_instructions():
    """打印 PDF 高亮使用说明"""
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                     📄 PDF 高亮数据结构说明                               ║
╚══════════════════════════════════════════════════════════════════════════╝

输出 JSON 包含以下字段:

1. answer: LLM 生成的答案（带 [citation:ID] 引用标记）

2. citations: 引用详情列表
   - citation_id: 引用标记，如 "[citation:1]"
   - section_id: section 的 UUID
   - section_heading: 章节标题
   - text: 引用的原文片段
   - highlight_boxes: 该 citation 对应的高亮框列表

3. highlight_boxes: 所有高亮框的扁平列表（去重后）
   - chunk_id: chunks 表中的 chunk UUID
   - page_idx: PDF 页码（从 0 开始）
   - bbox: 边界框坐标 {x1, y1, x2, y2}
   - text_preview: 文本预览
   - color: 高亮颜色（默认 yellow）

使用示例 (PyMuPDF):
    import fitz  # PyMuPDF
    
    doc = fitz.open("document.pdf")
    
    for box in highlight_boxes:
        page = doc[box["page_idx"]]
        rect = fitz.Rect(box["bbox"]["x1"], box["bbox"]["y1"], 
                         box["bbox"]["x2"], box["bbox"]["y2"])
        highlight = page.add_highlight_annot(rect)
    
    doc.save("highlighted.pdf")

使用示例 (PDF.js):
    // highlight_boxes 可直接用于前端高亮
    highlight_boxes.forEach(box => {
        const div = document.createElement('div');
        div.style.position = 'absolute';
        div.style.left = box.bbox.x1 + 'px';
        div.style.top = box.bbox.y1 + 'px';
        div.style.width = (box.bbox.x2 - box.bbox.x1) + 'px';
        div.style.height = (box.bbox.y2 - box.bbox.y1) + 'px';
        div.style.backgroundColor = 'rgba(255, 255, 0, 0.3)';
        pageContainer.appendChild(div);
    });
""")


def demo_query(query: str, version_id: str = None):
    """执行演示查询"""
    
    if version_id is None:
        version_id = os.getenv("TEST_VERSION_ID", "9a5a0214-3b98-4a64-9194-a01648479f7a")
    
    print(f"\n🧪 Citation-Aware RAG Demo")
    print("=" * 60)
    print(f"📄 Version ID: {version_id}")
    print(f"❓ Query: {query}")
    print("-" * 60)
    
    # 执行查询
    pipeline = CitationRAGPipeline(version_id=version_id, top_k=3)
    result = pipeline.query(query)
    
    # 显示答案
    print(f"\n💡 Answer:\n{result.answer}")
    
    # 显示统计
    print(f"\n📊 Statistics:")
    print(f"  - Retrieved sections: {result.metadata['retrieved_count']}")
    print(f"  - Citations generated: {result.metadata['citation_count']}")
    print(f"  - Highlight boxes: {len(result.highlight_boxes)}")
    
    # 显示引用详情
    print(f"\n📚 Citations:")
    for i, citation in enumerate(result.citations, 1):
        print(f"\n  [{i}] {citation.citation_id}")
        print(f"      Section: {citation.section_heading}")
        print(f"      Text: {citation.text[:100]}...")
        print(f"      Highlight boxes: {len(citation.highlight_boxes)}")
    
    # 显示高亮框预览
    print(f"\n📍 Highlight Boxes Preview (first 5):")
    for i, box in enumerate(result.highlight_boxes[:5], 1):
        print(f"  {i}. Page {box.page_idx}: bbox={box.bbox.to_dict()}")
        print(f"     Text: {box.text_preview[:60]}...")
    
    # 保存完整输出
    output_file = f"/tmp/citation_result_{query[:20].replace(' ', '_')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(result.to_json())
    print(f"\n💾 Full output saved to: {output_file}")
    
    return result


def main():
    if len(sys.argv) > 1:
        query = sys.argv[1]
    else:
        query = "售后服务包括哪些内容？"
    
    # 打印使用说明
    print_highlight_instructions()
    
    # 执行演示
    result = demo_query(query)
    
    # 输出完整 JSON
    print("\n" + "=" * 60)
    print("📋 Complete JSON Output:")
    print("=" * 60)
    print(result.to_json())


if __name__ == "__main__":
    main()
