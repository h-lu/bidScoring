#!/usr/bin/env python3
"""检查PDF结构并尝试多种方法提取内容."""

import pypdf
from pathlib import Path

pdf_path = "/Users/wangxq/Documents/mineru/pdf/253135-国内公开-投标文件/0811-DSITC253135-上海悦晟生物科技有限公司投标文件.pdf"

# 检查PDF基本信息
with open(pdf_path, 'rb') as f:
    pdf_reader = pypdf.PdfReader(f)
    num_pages = len(pdf_reader.pages)
    print(f"总页数: {num_pages}")

    # 检查前几页是否有可提取的文本
    for i in range(min(5, num_pages)):
        page = pdf_reader.pages[i]
        text = page.extract_text()
        print(f"\n--- 第 {i+1} 页文本预览 ({len(text)} 字符) ---")
        print(text[:500] if text else "(无文本 - 可能是扫描版)")
