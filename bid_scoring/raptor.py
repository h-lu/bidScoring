"""RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval)

实现基于递归聚类和摘要的文档树构建:
- Level 0: 叶子节点（原始文本块）
- Level 1+: 递归聚类 + LLM摘要生成的中间节点
- 直到单个根节点或达到最大层级

Reference: https://arxiv.org/abs/2401.18059
"""

import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from bid_scoring.embeddings import embed_texts
from bid_scoring.llm import LLMClient, load_settings


@dataclass
class RAPTORNode:
    """RAPTOR 树节点
    
    Attributes:
        node_id: 节点唯一标识符（UUID）
        level: 层级（0=叶子，1+=摘要节点）
        node_type: 节点类型（'leaf' 或 'summary'）
        content: 节点文本内容（原文或摘要）
        embedding: 文本向量（可选，用于聚类）
        parent_id: 父节点 ID（根节点为 None）
        children_ids: 子节点 ID 列表
        metadata: 附加元数据
    """
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    level: int = 0
    node_type: str = "leaf"
    content: str = ""
    embedding: Optional[list[float]] = None
    parent_id: Optional[str] = None
    children_ids: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        """验证节点数据有效性"""
        if self.level < 0:
            raise ValueError(f"Level must be >= 0, got {self.level}")
        valid_types = {"leaf", "summary"}
        if self.node_type not in valid_types:
            raise ValueError(f"Invalid node_type: {self.node_type}")

    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            "node_id": self.node_id,
            "level": self.level,
            "node_type": self.node_type,
            "content": self.content,
            "embedding": self.embedding,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "metadata": self.metadata,
        }

    def add_child(self, child_id: str) -> None:
        """添加子节点引用"""
        if child_id not in self.children_ids:
            self.children_ids.append(child_id)

    def set_parent(self, parent_id: str) -> None:
        """设置父节点"""
        self.parent_id = parent_id


class RAPTORBuilder:
    """RAPTOR 树构建器
    
    通过递归聚类和摘要生成构建语义树结构。
    
    Example:
        >>> builder = RAPTORBuilder()
        >>> chunks = ["文本块1", "文本块2", "文本块3"]
        >>> nodes = builder.build_tree(chunks)
        >>> root = builder.get_root_node()
        >>> print(f"树深度: {root.level + 1}")
    """

    # 默认配置
    DEFAULT_MAX_LEVELS = 5
    DEFAULT_CLUSTER_SIZE = 10
    DEFAULT_MIN_CLUSTER_SIZE = 2
    DEFAULT_SUMMARY_MAX_TOKENS = 512

    def __init__(
        self,
        max_levels: int = DEFAULT_MAX_LEVELS,
        cluster_size: int = DEFAULT_CLUSTER_SIZE,
        min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
        summary_max_tokens: int = DEFAULT_SUMMARY_MAX_TOKENS,
        llm_client: Optional[LLMClient] = None,
    ):
        """初始化 RAPTOR 构建器
        
        Args:
            max_levels: 最大层级数（防止无限递归）
            cluster_size: 目标聚类大小
            min_cluster_size: 最小聚类大小（少于此时停止）
            summary_max_tokens: 摘要最大 token 数
            llm_client: LLM 客户端（为 None 时自动创建）
        """
        self.max_levels = max_levels
        self.cluster_size = cluster_size
        self.min_cluster_size = min_cluster_size
        self.summary_max_tokens = summary_max_tokens
        self.nodes: list[RAPTORNode] = []
        self._node_map: dict[str, RAPTORNode] = {}
        self._llm_client = llm_client

    def _get_llm_client(self) -> LLMClient:
        """获取 LLM 客户端（延迟初始化）"""
        if self._llm_client is None:
            settings = load_settings()
            self._llm_client = LLMClient(settings)
        return self._llm_client

    def _generate_node_id(self) -> str:
        """生成唯一节点 ID"""
        return str(uuid.uuid4())

    def _calculate_num_clusters(self, num_nodes: int) -> int:
        """计算聚类数量
        
        根据节点数量动态计算合适的聚类数:
        - 节点数 <= min_cluster_size: 返回 1（不聚类）
        - 否则: 按 cluster_size 计算，但至少 2 个聚类
        
        Args:
            num_nodes: 节点数量
            
        Returns:
            聚类数量
        """
        if num_nodes <= self.min_cluster_size:
            return 1
        
        # 计算聚类数，确保至少有 2 个
        num_clusters = max(2, num_nodes // self.cluster_size)
        
        # 聚类数不能超过节点数
        return min(num_clusters, num_nodes)

    def _cluster_nodes(
        self, 
        nodes: list[RAPTORNode]
    ) -> list[list[RAPTORNode]]:
        """对节点进行聚类
        
        使用 KMeans 聚类算法，基于节点 embedding 进行分组。
        
        Args:
            nodes: 需要聚类的节点列表
            
        Returns:
            聚类结果，每个子列表是一个聚类
        """
        if len(nodes) < 2:
            return [nodes] if nodes else []
        
        # 检查是否有 embedding
        nodes_with_embedding = [n for n in nodes if n.embedding is not None]
        if len(nodes_with_embedding) < 2:
            # 没有 embedding，全部放入一个聚类
            return [nodes]
        
        # 准备数据
        embeddings = np.array([n.embedding for n in nodes_with_embedding])
        
        # 计算聚类数
        num_clusters = self._calculate_num_clusters(len(nodes_with_embedding))
        
        if num_clusters == 1:
            return [nodes_with_embedding]
        
        # KMeans 聚类
        try:
            from sklearn.cluster import KMeans
            
            kmeans = KMeans(
                n_clusters=num_clusters, 
                random_state=42,
                n_init=10,
            )
            labels = kmeans.fit_predict(embeddings)
            
            # 按标签分组
            clusters: list[list[RAPTORNode]] = [[] for _ in range(num_clusters)]
            for node, label in zip(nodes_with_embedding, labels):
                clusters[label].append(node)
            
            return clusters
            
        except Exception as e:
            # 聚类失败，返回单个大聚类
            return [nodes_with_embedding]

    def _summarize_cluster(
        self, 
        cluster_nodes: list[RAPTORNode],
        level: int
    ) -> str:
        """为聚类生成摘要
        
        使用 LLM 对聚类中的文本进行摘要生成。
        
        Args:
            cluster_nodes: 聚类中的节点列表
            level: 当前层级
            
        Returns:
            生成的摘要文本
        """
        if not cluster_nodes:
            return ""
        
        # 合并聚类中的文本
        texts = []
        for node in cluster_nodes:
            if node.content.strip():
                texts.append(node.content.strip())
        
        if not texts:
            return ""
        
        combined_text = "\n\n".join(texts)
        
        # 构建提示词
        if level == 1:
            prompt = f"""请对以下文本进行摘要，提炼关键信息：

{combined_text}

摘要要求：
1. 保留核心观点和关键细节
2. 长度控制在 200-300 字
3. 使用简洁、连贯的语言

摘要："""
        else:
            prompt = f"""请对以下高级摘要进行整合，生成更高层级的主题摘要：

{combined_text}

摘要要求：
1. 提炼共同主题和核心观点
2. 长度控制在 150-250 字
3. 保持抽象性和概括性

主题摘要："""
        
        try:
            client = self._get_llm_client()
            messages = [
                {"role": "system", "content": "你是一个专业的文本摘要专家。"},
                {"role": "user", "content": prompt},
            ]
            
            summary = client.complete(
                messages=messages,
                temperature=0.3,
                max_tokens=self.summary_max_tokens,
            )
            
            return summary.strip()
            
        except Exception as e:
            # 摘要生成失败，返回简单拼接
            return combined_text[:500] + "..." if len(combined_text) > 500 else combined_text

    def _create_summary_node(
        self, 
        cluster_nodes: list[RAPTORNode],
        level: int
    ) -> RAPTORNode:
        """从聚类创建摘要节点
        
        Args:
            cluster_nodes: 聚类中的节点列表
            level: 当前层级
            
        Returns:
            新建的摘要节点
        """
        # 生成摘要
        summary = self._summarize_cluster(cluster_nodes, level)
        
        # 创建节点
        summary_node = RAPTORNode(
            node_id=self._generate_node_id(),
            level=level,
            node_type="summary",
            content=summary,
            children_ids=[n.node_id for n in cluster_nodes],
            metadata={
                "cluster_size": len(cluster_nodes),
                "source_levels": list(set(n.level for n in cluster_nodes)),
            }
        )
        
        # 设置子节点的父节点
        for node in cluster_nodes:
            node.set_parent(summary_node.node_id)
        
        return summary_node

    def _generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """为文本列表生成 embedding
        
        Args:
            texts: 文本列表
            
        Returns:
            embedding 列表
        """
        if not texts:
            return []
        
        try:
            return embed_texts(texts, batch_size=50)
        except Exception as e:
            # embedding 生成失败，返回 None
            return [None] * len(texts)

    def build_tree(self, chunks: list[str]) -> list[RAPTORNode]:
        """构建 RAPTOR 树
        
        从文本块列表递归构建语义树结构。
        
        流程:
        1. Level 0: 创建叶子节点
        2. Level 1+: 递归聚类 → 摘要 → 创建父节点
        3. 直到只剩一个节点或达到 max_levels
        
        Args:
            chunks: 文本块列表（每个块是一个字符串）
            
        Returns:
            所有树节点的列表
            
        Raises:
            ValueError: 如果 chunks 数量 < 2
            
        Example:
            >>> builder = RAPTORBuilder()
            >>> chunks = ["段落1", "段落2", "段落3", "段落4"]
            >>> nodes = builder.build_tree(chunks)
            >>> print(f"总节点数: {len(nodes)}")
        """
        # 输入验证
        if not isinstance(chunks, list):
            raise ValueError("chunks must be a list")
        
        if len(chunks) < 2:
            raise ValueError(f"At least 2 chunks required, got {len(chunks)}")
        
        # 过滤空文本
        valid_chunks = [c for c in chunks if c and c.strip()]
        if len(valid_chunks) < 2:
            raise ValueError(f"At least 2 non-empty chunks required, got {len(valid_chunks)}")
        
        # 清空之前的节点
        self.nodes = []
        self._node_map = {}
        
        # Step 1: 创建 Level 0 叶子节点
        leaf_nodes = []
        for chunk in valid_chunks:
            node = RAPTORNode(
                node_id=self._generate_node_id(),
                level=0,
                node_type="leaf",
                content=chunk,
                metadata={"source": "original_chunk"},
            )
            leaf_nodes.append(node)
            self.nodes.append(node)
            self._node_map[node.node_id] = node
        
        # 为叶子节点生成 embedding
        leaf_texts = [n.content for n in leaf_nodes]
        leaf_embeddings = self._generate_embeddings(leaf_texts)
        for node, emb in zip(leaf_nodes, leaf_embeddings):
            node.embedding = emb
        
        # Step 2: 递归构建上层节点
        current_level_nodes = leaf_nodes
        current_level = 0
        
        while (
            len(current_level_nodes) > 1 
            and current_level < self.max_levels - 1
        ):
            current_level += 1
            
            # 聚类
            clusters = self._cluster_nodes(current_level_nodes)
            
            if len(clusters) == 1 and len(clusters[0]) == len(current_level_nodes):
                # 聚类没有产生有效分组，停止
                break
            
            # 为每个聚类创建摘要节点
            next_level_nodes = []
            for cluster in clusters:
                if len(cluster) >= 1:
                    summary_node = self._create_summary_node(cluster, current_level)
                    next_level_nodes.append(summary_node)
                    self.nodes.append(summary_node)
                    self._node_map[summary_node.node_id] = summary_node
            
            if len(next_level_nodes) == len(current_level_nodes):
                # 没有有效压缩，停止
                break
            
            # 为摘要节点生成 embedding（用于下一轮聚类）
            summary_texts = [n.content for n in next_level_nodes]
            summary_embeddings = self._generate_embeddings(summary_texts)
            for node, emb in zip(next_level_nodes, summary_embeddings):
                node.embedding = emb
            
            current_level_nodes = next_level_nodes
        
        # Step 3: 如果还有多个节点，创建一个根节点
        if len(current_level_nodes) > 1:
            root_node = self._create_summary_node(current_level_nodes, current_level + 1)
            self.nodes.append(root_node)
            self._node_map[root_node.node_id] = root_node
        
        return self.nodes

    def get_root_node(self) -> Optional[RAPTORNode]:
        """获取树的根节点（最高层级）"""
        if not self.nodes:
            return None
        
        max_level = max(n.level for n in self.nodes)
        root_candidates = [n for n in self.nodes if n.level == max_level]
        
        # 返回第一个（通常只有一个根）
        return root_candidates[0] if root_candidates else None

    def get_nodes_by_level(self, level: int) -> list[RAPTORNode]:
        """获取指定层级的所有节点"""
        return [n for n in self.nodes if n.level == level]

    def get_leaf_nodes(self) -> list[RAPTORNode]:
        """获取所有叶子节点"""
        return [n for n in self.nodes if n.node_type == "leaf"]

    def get_tree_structure(self) -> dict:
        """获取树形结构表示（用于调试和可视化）"""
        root = self.get_root_node()
        if not root:
            return {}
        
        def build_subtree(node: RAPTORNode) -> dict:
            children = [
                build_subtree(self._node_map[child_id])
                for child_id in node.children_ids
                if child_id in self._node_map
            ]
            return {
                "id": node.node_id[:8],
                "type": node.node_type,
                "level": node.level,
                "content": node.content[:50] + "..." if len(node.content) > 50 else node.content,
                "children": children,
            }
        
        return build_subtree(root)

    def get_tree_stats(self) -> dict:
        """获取树的统计信息"""
        if not self.nodes:
            return {
                "total_nodes": 0,
                "max_level": 0,
                "leaf_count": 0,
                "summary_count": 0,
            }
        
        levels = [n.level for n in self.nodes]
        return {
            "total_nodes": len(self.nodes),
            "max_level": max(levels),
            "level_distribution": {
                level: len(self.get_nodes_by_level(level))
                for level in range(max(levels) + 1)
            },
            "leaf_count": len(self.get_leaf_nodes()),
            "summary_count": len([n for n in self.nodes if n.node_type == "summary"]),
        }
