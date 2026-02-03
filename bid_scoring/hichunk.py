"""HiChunk Builder - 层次化文档分块构建器

实现基于 MineRU content_list 的4层树形结构:
- Level 0 (sentence): 叶子节点，对应 content_list 中的单个元素
- Level 1 (paragraph): 段落节点，合并相邻的句子
- Level 2 (section): 章节节点，按标题层级分组
- Level 3 (document): 文档根节点，代表整个文档

Key Design Decisions:
1. 段落边界检测: 基于 text_level 变化、页码变化、元素类型变化
2. 章节检测: 使用 text_level > 0 的元素作为章节标题
3. 元数据保留: page_idx, bbox, element_type 等
"""

from dataclasses import dataclass, field
from typing import Optional
import uuid


@dataclass
class HiChunkNode:
    """层次化文档分块节点
    
    对应数据库 hierarchical_nodes 表的结构，用于在 Python 中
    表示和操作层次化文档树。
    
    Attributes:
        node_id: 节点唯一标识符（UUID）
        level: 层级（0=sentence, 1=paragraph, 2=section, 3=document）
        node_type: 节点类型（'sentence', 'paragraph', 'section', 'document'）
        content: 节点文本内容
        parent_id: 父节点 ID（根节点为 None）
        children_ids: 子节点 ID 列表
        start_chunk_id: 起始 chunk 引用（叶子节点）
        end_chunk_id: 结束 chunk 引用（叶子节点）
        metadata: 附加元数据（页码、标题层级、边界框等）
    """
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    level: int = 0
    node_type: str = "sentence"
    content: str = ""
    parent_id: Optional[str] = None
    children_ids: list[str] = field(default_factory=list)
    start_chunk_id: Optional[str] = None
    end_chunk_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        """验证节点数据有效性"""
        if not 0 <= self.level <= 3:
            raise ValueError(f"Level must be 0-3, got {self.level}")
        valid_types = {"sentence", "paragraph", "section", "document"}
        if self.node_type not in valid_types:
            raise ValueError(f"Invalid node_type: {self.node_type}")

    def to_dict(self) -> dict:
        """转换为字典格式（用于数据库插入）"""
        return {
            "node_id": self.node_id,
            "level": self.level,
            "node_type": self.node_type,
            "content": self.content,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "start_chunk_id": self.start_chunk_id,
            "end_chunk_id": self.end_chunk_id,
            "metadata": self.metadata,
        }

    def add_child(self, child_id: str) -> None:
        """添加子节点引用"""
        if child_id not in self.children_ids:
            self.children_ids.append(child_id)

    def set_parent(self, parent_id: str) -> None:
        """设置父节点"""
        self.parent_id = parent_id


class HiChunkBuilder:
    """层次化文档分块构建器
    
    从 MineRU content_list 构建4层文档树结构。
    
    Example:
        >>> builder = HiChunkBuilder()
        >>> nodes = builder.build_hierarchy(content_list, "文档标题")
        >>> root = [n for n in nodes if n.level == 3][0]
        >>> print(f"文档包含 {len(root.children_ids)} 个章节")
    """

    # 段落边界检测配置
    PARAGRAPH_BREAK_TYPES = {"table", "image", "header", "footer", "page_number"}
    
    def __init__(self):
        """初始化构建器"""
        self.nodes: list[HiChunkNode] = []
        self._node_counter = 0

    def _generate_node_id(self) -> str:
        """生成唯一节点 ID"""
        self._node_counter += 1
        return str(uuid.uuid4())

    def _is_heading(self, item: dict) -> bool:
        """判断是否为标题元素
        
        MineRU 中 text_level > 0 表示标题层级
        - text_level=1: 一级标题
        - text_level=2: 二级标题
        - ...
        """
        return item.get("type") == "text" and item.get("text_level", 0) > 0

    def _get_heading_level(self, item: dict) -> int:
        """获取标题层级（0 表示不是标题）"""
        if self._is_heading(item):
            return item.get("text_level", 0)
        return 0

    def _should_start_new_paragraph(self, prev_item: Optional[dict], curr_item: dict) -> bool:
        """判断是否应该开始新段落
        
        段落边界条件:
        1. 前一个元素为 None（首个元素）
        2. 当前元素是标题
        3. 前一个元素是特殊类型（表格、图片等）
        4. 页码发生变化
        5. 元素类型发生变化（如从 text 到 list）
        """
        if prev_item is None:
            return True
        
        # 标题总是开启新段落
        if self._is_heading(curr_item):
            return True
        
        # 标题后面开启新段落
        if self._is_heading(prev_item):
            return True
        
        # 特殊类型元素后开启新段落
        if prev_item.get("type") in self.PARAGRAPH_BREAK_TYPES:
            return True
        
        # 特殊类型元素单独成段
        if curr_item.get("type") in self.PARAGRAPH_BREAK_TYPES:
            return True
        
        # 页码变化开启新段落
        if prev_item.get("page_idx") != curr_item.get("page_idx"):
            return True
        
        # 元素类型变化开启新段落
        if prev_item.get("type") != curr_item.get("type"):
            return True
        
        return False

    def _extract_text_from_item(self, item: dict) -> str:
        """从 content_list 项中提取文本
        
        支持多种元素类型的文本提取。
        """
        item_type = item.get("type", "")
        
        if item_type == "text":
            return item.get("text", "")
        
        elif item_type == "table":
            # 表格: caption + body + footnote
            texts = []
            if item.get("table_caption"):
                if isinstance(item["table_caption"], list):
                    texts.extend(item["table_caption"])
                else:
                    texts.append(item["table_caption"])
            if item.get("table_body"):
                # 去除 HTML 标签
                body = item["table_body"]
                if isinstance(body, str):
                    import re
                    body = re.sub(r'<[^>]+>', ' ', body)
                    body = re.sub(r'\s+', ' ', body).strip()
                texts.append(body)
            if item.get("table_footnote"):
                if isinstance(item["table_footnote"], list):
                    texts.extend(item["table_footnote"])
                else:
                    texts.append(item["table_footnote"])
            return " ".join(texts)
        
        elif item_type == "image":
            # 图片: caption + footnote
            texts = []
            if item.get("image_caption"):
                if isinstance(item["image_caption"], list):
                    texts.extend(item["image_caption"])
                else:
                    texts.append(item["image_caption"])
            if item.get("image_footnote"):
                if isinstance(item["image_footnote"], list):
                    texts.extend(item["image_footnote"])
                else:
                    texts.append(item["image_footnote"])
            return " ".join(texts)
        
        elif item_type == "list":
            # 列表: 合并所有 list_items
            if item.get("list_items"):
                return " ".join(item["list_items"])
            return ""
        
        elif item_type in ("header", "aside_text", "footer", "page_number"):
            return item.get("text", "")
        
        else:
            return item.get("text", "")

    def _create_leaf_nodes(self, content_list: list[dict]) -> list[HiChunkNode]:
        """构建 Level 0 (sentence) 叶子节点
        
        每个 content_list 项对应一个叶子节点。
        """
        leaf_nodes = []
        
        for idx, item in enumerate(content_list):
            # 跳过页码
            if item.get("type") == "page_number":
                continue
            
            text = self._extract_text_from_item(item)
            if not text.strip():
                continue
            
            node = HiChunkNode(
                node_id=self._generate_node_id(),
                level=0,
                node_type="sentence",
                content=text,
                metadata={
                    "page_idx": item.get("page_idx", 0),
                    "bbox": item.get("bbox", []),
                    "element_type": item.get("type", ""),
                    "text_level": item.get("text_level", 0),
                    "source_index": idx,
                    "sub_type": item.get("sub_type"),
                }
            )
            leaf_nodes.append(node)
        
        return leaf_nodes

    def _create_paragraph_nodes(
        self, 
        leaf_nodes: list[HiChunkNode],
        content_list: list[dict]
    ) -> list[HiChunkNode]:
        """构建 Level 1 (paragraph) 段落节点
        
        将相邻的叶子节点合并为段落，基于边界检测规则。
        """
        if not leaf_nodes:
            return []
        
        paragraph_nodes = []
        current_para_leaves: list[HiChunkNode] = []
        
        for i, leaf in enumerate(leaf_nodes):
            # 从 metadata 中获取 source_index
            source_idx = leaf.metadata.get("source_index", i)
            
            # 确定当前和上一个 content_list 项
            prev_content = content_list[source_idx - 1] if source_idx > 0 else None
            curr_content = content_list[source_idx] if source_idx < len(content_list) else None
            
            # 判断是否需要开启新段落
            if self._should_start_new_paragraph(prev_content, curr_content):
                # 保存之前的段落
                if current_para_leaves:
                    para_node = self._create_paragraph_from_leaves(current_para_leaves)
                    paragraph_nodes.append(para_node)
                current_para_leaves = []
            
            current_para_leaves.append(leaf)
        
        # 保存最后一个段落
        if current_para_leaves:
            para_node = self._create_paragraph_from_leaves(current_para_leaves)
            paragraph_nodes.append(para_node)
        
        return paragraph_nodes

    def _create_paragraph_from_leaves(self, leaves: list[HiChunkNode]) -> HiChunkNode:
        """从叶子节点列表创建段落节点"""
        # 合并文本
        texts = [leaf.content for leaf in leaves]
        combined_text = " ".join(texts)
        
        # 收集元数据
        page_indices = [leaf.metadata.get("page_idx", 0) for leaf in leaves]
        bboxes = [leaf.metadata.get("bbox", []) for leaf in leaves if leaf.metadata.get("bbox")]
        element_types = list(set(leaf.metadata.get("element_type", "") for leaf in leaves))
        
        para_node = HiChunkNode(
            node_id=self._generate_node_id(),
            level=1,
            node_type="paragraph",
            content=combined_text,
            children_ids=[leaf.node_id for leaf in leaves],
            metadata={
                "start_page": min(page_indices) if page_indices else 0,
                "end_page": max(page_indices) if page_indices else 0,
                "bboxes": bboxes,
                "element_types": element_types,
                "leaf_count": len(leaves),
            }
        )
        
        # 设置叶子节点的父节点
        for leaf in leaves:
            leaf.set_parent(para_node.node_id)
        
        return para_node

    def _create_section_nodes(
        self, 
        paragraph_nodes: list[HiChunkNode],
        content_list: list[dict]
    ) -> list[HiChunkNode]:
        """构建 Level 2 (section) 章节节点
        
        基于标题层级 (text_level) 将段落分组为章节。
        每个标题对应一个章节，该标题的段落作为章节的第一个子节点。
        """
        if not paragraph_nodes:
            return []
        
        # 提取所有标题文本及其元数据
        heading_info = {}  # text -> {level, page_idx}
        for item in content_list:
            if item.get("type") == "page_number":
                continue
            text_level = self._get_heading_level(item)
            if text_level > 0:
                heading_text = item.get("text", "")
                heading_info[heading_text] = {
                    "level": text_level,
                    "page_idx": item.get("page_idx", 0),
                }
        
        # 如果没有标题，所有段落归入一个默认章节
        if not heading_info:
            section_node = HiChunkNode(
                node_id=self._generate_node_id(),
                level=2,
                node_type="section",
                content="默认章节",
                children_ids=[p.node_id for p in paragraph_nodes],
                metadata={
                    "section_title": "默认章节",
                    "heading_level": 0,
                    "page_idx": paragraph_nodes[0].metadata.get("start_page", 0) if paragraph_nodes else 0,
                }
            )
            for para in paragraph_nodes:
                para.set_parent(section_node.node_id)
            return [section_node]
        
        # 根据标题分组段落
        section_nodes = []
        current_section_paras: list[HiChunkNode] = []
        current_section_title = "默认章节"
        current_section_level = 0
        current_section_page = 0
        
        for para in paragraph_nodes:
            # 检查该段落是否是标题
            para_content = para.content.strip()
            if para_content in heading_info:
                # 保存之前的章节（如果有段落）
                if current_section_paras:
                    section_node = HiChunkNode(
                        node_id=self._generate_node_id(),
                        level=2,
                        node_type="section",
                        content=current_section_title,
                        children_ids=[p.node_id for p in current_section_paras],
                        metadata={
                            "section_title": current_section_title,
                            "heading_level": current_section_level,
                            "page_idx": current_section_page,
                        }
                    )
                    for p in current_section_paras:
                        p.set_parent(section_node.node_id)
                    section_nodes.append(section_node)
                
                # 开始新章节
                info = heading_info[para_content]
                current_section_title = para_content
                current_section_level = info["level"]
                current_section_page = info["page_idx"]
                current_section_paras = []
            
            current_section_paras.append(para)
        
        # 保存最后一个章节
        if current_section_paras:
            section_node = HiChunkNode(
                node_id=self._generate_node_id(),
                level=2,
                node_type="section",
                content=current_section_title,
                children_ids=[p.node_id for p in current_section_paras],
                metadata={
                    "section_title": current_section_title,
                    "heading_level": current_section_level,
                    "page_idx": current_section_page,
                }
            )
            for p in current_section_paras:
                p.set_parent(section_node.node_id)
            section_nodes.append(section_node)
        
        return section_nodes

    def _create_root_node(
        self, 
        section_nodes: list[HiChunkNode], 
        document_title: str
    ) -> HiChunkNode:
        """构建 Level 3 (document) 根节点"""
        # 收集所有页面的元数据
        all_pages = set()
        for section in section_nodes:
            page_idx = section.metadata.get("page_idx", 0)
            if page_idx:
                all_pages.add(page_idx)
        
        root_node = HiChunkNode(
            node_id=self._generate_node_id(),
            level=3,
            node_type="document",
            content=document_title,
            children_ids=[s.node_id for s in section_nodes],
            metadata={
                "document_title": document_title,
                "section_count": len(section_nodes),
                "pages": sorted(list(all_pages)),
            }
        )
        
        # 设置章节的父节点
        for section in section_nodes:
            section.set_parent(root_node.node_id)
        
        return root_node

    def build_hierarchy(
        self,
        content_list: list[dict],
        document_title: str = "untitled"
    ) -> list[HiChunkNode]:
        """构建完整的层次结构
        
        从 MineRU content_list 构建4层文档树:
        1. Level 0 (sentence): 叶子节点
        2. Level 1 (paragraph): 段落节点
        3. Level 2 (section): 章节节点
        4. Level 3 (document): 文档根节点
        
        Args:
            content_list: MineRU 解析结果列表
            document_title: 文档标题
        
        Returns:
            所有节点的列表（按层级排序: 根节点 -> 章节 -> 段落 -> 叶子）
        
        Raises:
            ValueError: 如果 content_list 格式无效
        
        Example:
            >>> builder = HiChunkBuilder()
            >>> nodes = builder.build_hierarchy(content_list, "示例文档")
            >>> root = [n for n in nodes if n.level == 3][0]
            >>> print(f"文档: {root.content}")
            >>> print(f"章节数: {len(root.children_ids)}")
        """
        # 验证输入
        if not isinstance(content_list, list):
            raise ValueError("content_list must be a list")
        
        # 过滤空内容
        content_list = [
            item for item in content_list 
            if isinstance(item, dict) and self._extract_text_from_item(item).strip()
        ]
        
        if not content_list:
            # 返回空文档
            root = HiChunkNode(
                node_id=self._generate_node_id(),
                level=3,
                node_type="document",
                content=document_title,
                metadata={"document_title": document_title, "empty": True}
            )
            return [root]
        
        self.nodes = []
        
        # Step 1: 构建叶子节点 (Level 0)
        leaf_nodes = self._create_leaf_nodes(content_list)
        self.nodes.extend(leaf_nodes)
        
        if not leaf_nodes:
            # 没有有效叶子节点，返回空文档
            root = HiChunkNode(
                node_id=self._generate_node_id(),
                level=3,
                node_type="document",
                content=document_title,
                metadata={"document_title": document_title, "empty": True}
            )
            return [root]
        
        # Step 2: 构建段落节点 (Level 1)
        paragraph_nodes = self._create_paragraph_nodes(leaf_nodes, content_list)
        self.nodes.extend(paragraph_nodes)
        
        # Step 3: 构建章节节点 (Level 2)
        section_nodes = self._create_section_nodes(paragraph_nodes, content_list)
        self.nodes.extend(section_nodes)
        
        # Step 4: 构建根节点 (Level 3)
        root_node = self._create_root_node(section_nodes, document_title)
        self.nodes.append(root_node)
        
        return self.nodes

    def get_nodes_by_level(self, level: int) -> list[HiChunkNode]:
        """获取指定层级的所有节点"""
        return [n for n in self.nodes if n.level == level]

    def get_root_node(self) -> Optional[HiChunkNode]:
        """获取文档根节点"""
        for node in self.nodes:
            if node.level == 3:
                return node
        return None

    def get_tree_structure(self) -> dict:
        """获取树形结构表示（用于调试和可视化）"""
        root = self.get_root_node()
        if not root:
            return {}
        
        def build_subtree(node: HiChunkNode) -> dict:
            children = [
                build_subtree(child) 
                for child in self.nodes 
                if child.node_id in node.children_ids
            ]
            return {
                "id": node.node_id[:8],
                "type": node.node_type,
                "level": node.level,
                "content": node.content[:50] + "..." if len(node.content) > 50 else node.content,
                "children": children,
            }
        
        return build_subtree(root)
