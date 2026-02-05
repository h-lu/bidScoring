#!/usr/bin/env python3
"""
应用 RAG 高亮结果到 PDF（带坐标转换）

使用方法:
    python scripts/apply_pdf_highlight.py \
        --pdf "/path/to/document.pdf" \
        --highlight "/path/to/highlight.json" \
        --output "/path/to/output.pdf"
"""

import argparse
import json
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import fitz  # PyMuPDF
except ImportError:
    print("错误: 需要安装 PyMuPDF (pip install pymupdf)")
    sys.exit(1)


def apply_highlight_to_pdf(
    pdf_path: str,
    highlight_data: dict,
    output_path: str
):
    """
    将高亮数据应用到 PDF
    
    关键: MinerU 的 bbox 是 0-1000 范围的归一化坐标，
    需要转换为 PDF 点坐标。
    """
    print(f"📄 PDF 文件: {pdf_path}")
    print(f"📍 高亮数据: {len(highlight_data.get('highlight_boxes', []))} 个区域")
    
    # 打开 PDF
    doc = fitz.open(pdf_path)
    
    highlight_boxes = highlight_data.get('highlight_boxes', [])
    applied_count = 0
    
    for i, box_data in enumerate(highlight_boxes, 1):
        page_idx = box_data.get('page_idx', 0)
        raw_bbox = box_data.get('raw_bbox') or box_data.get('bbox')
        text_preview = box_data.get('text_preview', '')
        
        # 检查页码是否有效
        if page_idx >= len(doc):
            print(f"  ⚠️  [{i}] 跳过无效页码: {page_idx}")
            continue
        
        page = doc[page_idx]
        
        # 获取页面尺寸
        page_width = page.rect.width
        page_height = page.rect.height
        
        # 将 MinerU 坐标 (0-1000) 转换为 PDF 点坐标
        if raw_bbox and len(raw_bbox) >= 4:
            pdf_x0 = raw_bbox[0] * (page_width / 1000)
            pdf_y0 = raw_bbox[1] * (page_height / 1000)
            pdf_x1 = raw_bbox[2] * (page_width / 1000)
            pdf_y1 = raw_bbox[3] * (page_height / 1000)
        else:
            print(f"  ⚠️  [{i}] 跳过无效 bbox: {raw_bbox}")
            continue
        
        # 创建矩形并添加高亮
        rect = fitz.Rect(pdf_x0, pdf_y0, pdf_x1, pdf_y1)
        
        # 添加高亮注释
        highlight = page.add_highlight_annot(rect)
        highlight.set_info({
            "title": f"Citation {i}",
            "content": text_preview[:100]
        })
        
        # 添加红色边框用于调试（可选）
        # rect_annot = page.add_rect_annot(rect)
        # rect_annot.set_colors({"stroke": (1, 0, 0)})
        
        applied_count += 1
        
        if i <= 5:  # 只打印前 5 个
            print(f"  ✅ [{i}] 页 {page_idx + 1}: [{pdf_x0:.1f}, {pdf_y0:.1f}, {pdf_x1:.1f}, {pdf_y1:.1f}]")
            print(f"      文本: {text_preview[:50]}...")
    
    # 保存
    doc.save(output_path)
    doc.close()
    
    print(f"\n✅ 成功应用 {applied_count} 个高亮标注")
    print(f"💾 输出文件: {output_path}")
    
    return applied_count


def main():
    parser = argparse.ArgumentParser(description='应用 RAG 高亮到 PDF')
    parser.add_argument('--pdf', required=True, help='输入 PDF 文件路径')
    parser.add_argument('--highlight', required=True, help='高亮 JSON 文件路径')
    parser.add_argument('--output', required=True, help='输出 PDF 文件路径')
    
    args = parser.parse_args()
    
    # 加载高亮数据
    with open(args.highlight, 'r', encoding='utf-8') as f:
        highlight_data = json.load(f)
    
    # 应用高亮
    apply_highlight_to_pdf(args.pdf, highlight_data, args.output)
    
    # 显示答案
    if 'answer' in highlight_data:
        print("\n" + "=" * 60)
        print("💡 RAG 答案:")
        print("=" * 60)
        print(highlight_data['answer'])


if __name__ == "__main__":
    main()
