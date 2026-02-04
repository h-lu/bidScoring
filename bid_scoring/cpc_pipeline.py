"""CPC (Contextual Parent-Child) Pipeline - 完整上下文父子分块管道

整合所有 Multi-Vector Retrieval 组件:
- Contextual Retrieval (Task 1): LLM 生成上下文前缀
- HiChunk (Task 2): 层次化文档分块
- RAPTOR (Task 3): 递归聚类摘要树
- Late Chunking (Task 4): 长上下文感知向量
- Multi-Vector Retrieval (Task 5): 父子块联合检索

Usage:
    >>> pipeline = CPCPipeline()
    >>> result = await pipeline.process_document(
    ...     content_list=mineru_output,
    ...     document_title="投标文件",
    ...     project_id=uuid,
    ...     document_id=uuid,
    ...     version_id=uuid,
    ... )
    >>> results = await pipeline.retrieve("查询内容", version_id=uuid)
"""

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import psycopg

from bid_scoring.config import load_settings
from bid_scoring.contextual_retrieval import ContextualRetrievalGenerator
from bid_scoring.embeddings import embed_texts, get_embedding_client, get_embedding_config
from bid_scoring.hichunk import HiChunkBuilder
from bid_scoring.ingest import ingest_content_list
from bid_scoring.late_chunking import LateChunkingEncoder, estimate_token_boundaries
from bid_scoring.llm import LLMClient, get_llm_client
from bid_scoring.multi_vector_retrieval import MultiVectorRetriever
from bid_scoring.raptor import RAPTORBuilder

logger = logging.getLogger(__name__)


@dataclass
class CPCPipelineConfig:
    """CPC Pipeline 配置"""

    # 组件开关
    enable_contextual: bool = True
    enable_hichunk: bool = True
    enable_raptor: bool = True
    enable_late_chunking: bool = False  # 默认关闭，需要特定模型支持

    # Contextual Retrieval 配置
    contextual_model: str = "gpt-4"
    contextual_temperature: float = 0.0
    contextual_max_tokens: int = 200

    # HiChunk 配置
    hichunk_max_level: int = 3  # 0-3

    # RAPTOR 配置
    raptor_max_levels: int = 5
    raptor_cluster_size: int = 10
    raptor_min_cluster_size: int = 2
    raptor_summary_max_tokens: int = 512

    # Late Chunking 配置
    late_chunking_model: str = "jina-embeddings-v2-base-zh"
    late_chunking_pooling: str = "mean"

    # Retrieval 配置
    retrieval_mode: str = "hybrid"
    retrieval_top_k: int = 5
    retrieval_rerank: bool = True

    # 数据库配置
    database_url: Optional[str] = None

    def __post_init__(self):
        if self.database_url is None:
            self.database_url = load_settings()["DATABASE_URL"]


@dataclass
class ProcessResult:
    """文档处理结果"""

    success: bool
    version_id: str
    chunks_count: int = 0
    contextual_chunks_count: int = 0
    hierarchical_nodes_count: int = 0
    raptor_nodes_count: int = 0
    multi_vector_mappings_count: int = 0
    message: str = ""
    errors: list[str] = field(default_factory=list)


class CPCPipeline:
    """CPC (Contextual Parent-Child) 管道

    提供统一的文档处理和检索接口，整合所有 Multi-Vector Retrieval 组件。

    Args:
        config: Pipeline 配置 (默认使用 CPCPipelineConfig())
        llm_client: LLM 客户端 (可选，默认自动创建)
        embedding_client: Embedding 客户端 (可选，默认自动创建)

    Example:
        >>> pipeline = CPCPipeline()
        >>> # 处理文档
        >>> result = await pipeline.process_document(
        ...     content_list=content_list,
        ...     document_title="示例文档",
        ...     project_id=project_uuid,
        ...     document_id=doc_uuid,
        ...     version_id=version_uuid,
        ... )
        >>> # 检索
        >>> results = await pipeline.retrieve("查询", version_id=version_uuid)
    """

    def __init__(
        self,
        config: Optional[CPCPipelineConfig] = None,
        llm_client: Optional[LLMClient] = None,
        embedding_client: Optional[Any] = None,
    ):
        self.config = config or CPCPipelineConfig()
        self._llm_client = llm_client
        self._embedding_client = embedding_client
        self._contextual_generator: Optional[ContextualRetrievalGenerator] = None
        self._late_chunking_encoder: Optional[LateChunkingEncoder] = None
        self._retriever: Optional[MultiVectorRetriever] = None

    def _get_llm_client(self) -> LLMClient:
        """获取 LLM 客户端 (延迟初始化)"""
        if self._llm_client is None:
            settings = load_settings()
            self._llm_client = LLMClient(settings)
        return self._llm_client

    def _get_embedding_client(self) -> Any:
        """获取 Embedding 客户端 (延迟初始化)"""
        if self._embedding_client is None:
            self._embedding_client = get_embedding_client()
        return self._embedding_client

    def _get_contextual_generator(self) -> ContextualRetrievalGenerator:
        """获取 Contextual Retrieval 生成器"""
        if self._contextual_generator is None:
            self._contextual_generator = ContextualRetrievalGenerator(
                client=self._get_embedding_client(),
                model=self.config.contextual_model,
                temperature=self.config.contextual_temperature,
                max_tokens=self.config.contextual_max_tokens,
            )
        return self._contextual_generator

    def _get_late_chunking_encoder(self) -> LateChunkingEncoder:
        """获取 Late Chunking 编码器"""
        if self._late_chunking_encoder is None:
            self._late_chunking_encoder = LateChunkingEncoder(
                model=self.config.late_chunking_model,
                pooling_strategy=self.config.late_chunking_pooling,
                fallback_to_standard=True,
            )
        return self._late_chunking_encoder

    def _get_retriever(self) -> MultiVectorRetriever:
        """获取 Multi-Vector Retriever"""
        if self._retriever is None:
            self._retriever = MultiVectorRetriever(
                dsn=self.config.database_url,
            )
        return self._retriever

    def _get_connection(self) -> psycopg.Connection:
        """获取数据库连接"""
        return psycopg.connect(self.config.database_url)

    async def process_document(
        self,
        content_list: list[dict],
        document_title: str,
        project_id: str | uuid.UUID,
        document_id: str | uuid.UUID,
        version_id: str | uuid.UUID,
        source_type: str = "mineru",
        source_uri: Optional[str] = None,
    ) -> ProcessResult:
        """处理文档通过完整 CPC 管道

        执行步骤:
        1. 文档入库 (ingest_content_list)
        2. 生成 Contextual Chunks (可选)
        3. 构建 HiChunk 层次结构 (可选)
        4. 构建 RAPTOR 树 (可选)
        5. 创建 Multi-Vector 映射
        6. 生成 Embeddings

        Args:
            content_list: MineRU content_list 格式的内容列表
            document_title: 文档标题
            project_id: 项目 UUID
            document_id: 文档 UUID
            version_id: 版本 UUID
            source_type: 来源类型 (默认 mineru)
            source_uri: 源文件路径 (可选)

        Returns:
            ProcessResult 包含处理结果和统计信息
        """
        errors: list[str] = []
        chunks_count = 0
        contextual_chunks_count = 0
        hierarchical_nodes_count = 0
        raptor_nodes_count = 0
        multi_vector_mappings_count = 0

        try:
            with self._get_connection() as conn:
                # Step 1: 文档入库
                logger.info(f"[CPC Pipeline] Step 1: Ingesting document {document_id}")
                try:
                    ingest_stats = ingest_content_list(
                        conn=conn,
                        project_id=str(project_id),
                        document_id=str(document_id),
                        version_id=str(version_id),
                        content_list=content_list,
                        document_title=document_title,
                        source_type=source_type,
                        source_uri=source_uri,
                    )
                    chunks_count = ingest_stats["total_chunks"]
                    logger.info(f"[CPC Pipeline] Ingested {chunks_count} chunks")
                except Exception as e:
                    error_msg = f"Document ingestion failed: {e}"
                    logger.error(f"[CPC Pipeline] {error_msg}")
                    errors.append(error_msg)
                    return ProcessResult(
                        success=False,
                        version_id=str(version_id),
                        errors=errors,
                        message="Failed at document ingestion stage",
                    )

                # Step 2: Contextual Retrieval (可选)
                if self.config.enable_contextual and chunks_count > 0:
                    logger.info("[CPC Pipeline] Step 2: Generating contextual chunks")
                    try:
                        contextual_chunks_count = await self._build_contextual_chunks(
                            conn, str(version_id), document_title
                        )
                        logger.info(
                            f"[CPC Pipeline] Created {contextual_chunks_count} contextual chunks"
                        )
                    except Exception as e:
                        error_msg = f"Contextual chunk generation failed: {e}"
                        logger.warning(f"[CPC Pipeline] {error_msg}")
                        errors.append(error_msg)

                # Step 3: HiChunk 层次结构 (可选)
                if self.config.enable_hichunk and chunks_count > 0:
                    logger.info("[CPC Pipeline] Step 3: Building hierarchical nodes")
                    try:
                        hierarchical_nodes_count = await self._build_hierarchical_nodes(
                            conn, str(version_id), content_list, document_title
                        )
                        logger.info(
                            f"[CPC Pipeline] Created {hierarchical_nodes_count} hierarchical nodes"
                        )
                    except Exception as e:
                        error_msg = f"Hierarchical node building failed: {e}"
                        logger.warning(f"[CPC Pipeline] {error_msg}")
                        errors.append(error_msg)

                # Step 4: RAPTOR 树 (可选)
                if self.config.enable_raptor and chunks_count > 0:
                    logger.info("[CPC Pipeline] Step 4: Building RAPTOR tree")
                    try:
                        raptor_nodes_count = await self._build_raptor_nodes(
                            conn, str(version_id)
                        )
                        logger.info(
                            f"[CPC Pipeline] Created {raptor_nodes_count} RAPTOR nodes"
                        )
                    except Exception as e:
                        error_msg = f"RAPTOR tree building failed: {e}"
                        logger.warning(f"[CPC Pipeline] {error_msg}")
                        errors.append(error_msg)

                # Step 5: Multi-Vector 映射
                if chunks_count > 0:
                    logger.info("[CPC Pipeline] Step 5: Creating multi-vector mappings")
                    try:
                        multi_vector_mappings_count = await self._build_multi_vector_mappings(
                            conn, str(version_id)
                        )
                        logger.info(
                            f"[CPC Pipeline] Created {multi_vector_mappings_count} mappings"
                        )
                    except Exception as e:
                        error_msg = f"Multi-vector mapping creation failed: {e}"
                        logger.warning(f"[CPC Pipeline] {error_msg}")
                        errors.append(error_msg)

                # Step 6: 生成 Embeddings
                logger.info("[CPC Pipeline] Step 6: Generating embeddings")
                try:
                    await self._generate_embeddings(conn, str(version_id))
                    logger.info("[CPC Pipeline] Embeddings generated")
                except Exception as e:
                    error_msg = f"Embedding generation failed: {e}"
                    logger.warning(f"[CPC Pipeline] {error_msg}")
                    errors.append(error_msg)

                conn.commit()

            success = len(errors) == 0 or chunks_count > 0
            message = (
                f"Document processing completed with {len(errors)} warnings"
                if errors
                else "Document processing completed successfully"
            )

            return ProcessResult(
                success=success,
                version_id=str(version_id),
                chunks_count=chunks_count,
                contextual_chunks_count=contextual_chunks_count,
                hierarchical_nodes_count=hierarchical_nodes_count,
                raptor_nodes_count=raptor_nodes_count,
                multi_vector_mappings_count=multi_vector_mappings_count,
                message=message,
                errors=errors,
            )

        except Exception as e:
            error_msg = f"Pipeline execution failed: {e}"
            logger.error(f"[CPC Pipeline] {error_msg}")
            errors.append(error_msg)
            return ProcessResult(
                success=False,
                version_id=str(version_id),
                errors=errors,
                message="Pipeline execution failed",
            )

    async def _build_contextual_chunks(
        self,
        conn: psycopg.Connection,
        version_id: str,
        document_title: str,
    ) -> int:
        """构建 Contextual Chunks

        为每个 chunk 生成 LLM 上下文前缀并存储。
        """
        generator = self._get_contextual_generator()
        count = 0

        # 获取所有 chunks
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT chunk_id, text_raw, element_type
                FROM chunks
                WHERE version_id = %s AND text_raw IS NOT NULL AND text_raw != ''
                """,
                (version_id,),
            )
            rows = cur.fetchall()

        if not rows:
            return 0

        # 批量生成上下文
        chunk_contexts = []
        for chunk_id, text_raw, element_type in rows:
            context = generator.generate_context(
                chunk_text=text_raw,
                document_title=document_title,
                section_title=element_type,
            )
            contextualized_text = f"{context}\n\n{text_raw}"
            chunk_contexts.append({
                "chunk_id": chunk_id,
                "original_text": text_raw,
                "context_prefix": context,
                "contextualized_text": contextualized_text,
            })

        # 生成 embeddings
        texts_to_embed = [c["contextualized_text"] for c in chunk_contexts]
        embeddings = embed_texts(texts_to_embed)

        # 插入数据库
        with conn.cursor() as cur:
            for chunk_data, embedding in zip(chunk_contexts, embeddings):
                cur.execute(
                    """
                    INSERT INTO contextual_chunks (
                        chunk_id, version_id, original_text, context_prefix,
                        contextualized_text, embedding, model_name, embedding_model
                    ) VALUES (%s, %s, %s, %s, %s, %s::vector, %s, %s)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        context_prefix = EXCLUDED.context_prefix,
                        contextualized_text = EXCLUDED.contextualized_text,
                        embedding = EXCLUDED.embedding,
                        updated_at = now()
                    """,
                    (
                        chunk_data["chunk_id"],
                        version_id,
                        chunk_data["original_text"],
                        chunk_data["context_prefix"],
                        chunk_data["contextualized_text"],
                        embedding,
                        self.config.contextual_model,
                        get_embedding_config()["model"],
                    ),
                )
                count += 1

        return count

    async def _build_hierarchical_nodes(
        self,
        conn: psycopg.Connection,
        version_id: str,
        content_list: list[dict],
        document_title: str,
    ) -> int:
        """构建层次化节点 (HiChunk)

        从 content_list 构建4层树形结构并存储。
        """
        builder = HiChunkBuilder()
        nodes = builder.build_hierarchy(content_list, document_title)

        # 获取 chunk_id 映射 (通过 source_id)
        chunk_id_map: dict[str, str] = {}
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT source_id, chunk_id
                FROM chunks
                WHERE version_id = %s
                """,
                (version_id,),
            )
            for source_id, chunk_id in cur.fetchall():
                chunk_id_map[source_id] = str(chunk_id)

        # 插入层次化节点
        count = 0
        with conn.cursor() as cur:
            for node in nodes:
                # 映射 chunk 引用
                start_chunk_id = None
                end_chunk_id = None
                if node.start_chunk_id and node.start_chunk_id in chunk_id_map:
                    start_chunk_id = chunk_id_map[node.start_chunk_id]
                if node.end_chunk_id and node.end_chunk_id in chunk_id_map:
                    end_chunk_id = chunk_id_map[node.end_chunk_id]

                cur.execute(
                    """
                    INSERT INTO hierarchical_nodes (
                        version_id, parent_id, level, node_type, content,
                        children_ids, start_chunk_id, end_chunk_id, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                    RETURNING node_id
                    """,
                    (
                        version_id,
                        node.parent_id,
                        node.level,
                        node.node_type,
                        node.content,
                        node.children_ids,
                        start_chunk_id,
                        end_chunk_id,
                        node.metadata,
                    ),
                )
                if cur.fetchone():
                    count += 1

        return count

    async def _build_raptor_nodes(
        self,
        conn: psycopg.Connection,
        version_id: str,
    ) -> int:
        """构建 RAPTOR 树节点

        对文档 chunks 进行递归聚类和摘要。
        """
        # 获取所有 chunks 文本
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT chunk_id, text_raw
                FROM chunks
                WHERE version_id = %s AND text_raw IS NOT NULL AND text_raw != ''
                ORDER BY chunk_index
                """,
                (version_id,),
            )
            rows = cur.fetchall()

        if len(rows) < 2:
            return 0

        chunks = [text for _, text in rows]
        chunk_ids = [cid for cid, _ in rows]

        # 构建 RAPTOR 树
        builder = RAPTORBuilder(
            max_levels=self.config.raptor_max_levels,
            cluster_size=self.config.raptor_cluster_size,
            min_cluster_size=self.config.raptor_min_cluster_size,
            summary_max_tokens=self.config.raptor_summary_max_tokens,
            llm_client=self._get_llm_client(),
        )

        try:
            nodes = builder.build_tree(chunks)
        except ValueError:
            #  insufficient chunks
            return 0

        # 插入 RAPTOR 节点 (使用 hierarchical_nodes 表，添加 raptor 标记)
        count = 0
        with conn.cursor() as cur:
            for node in nodes:
                metadata = {
                    **node.metadata,
                    "raptor": True,
                    "source_chunk_ids": chunk_ids if node.level == 0 else [],
                }

                cur.execute(
                    """
                    INSERT INTO hierarchical_nodes (
                        version_id, level, node_type, content,
                        children_ids, metadata, embedding
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
                    ON CONFLICT DO NOTHING
                    """,
                    (
                        version_id,
                        node.level + 4,  # RAPTOR 节点使用 level 4+ 避免与 HiChunk 冲突
                        "raptor_" + node.node_type,
                        node.content,
                        node.children_ids,
                        metadata,
                        node.embedding,
                    ),
                )
                count += 1

        return count

    async def _build_multi_vector_mappings(
        self,
        conn: psycopg.Connection,
        version_id: str,
    ) -> int:
        """构建 Multi-Vector 映射

        创建 parent-child chunk 之间的关系映射。
        """
        count = 0

        with conn.cursor() as cur:
            # 策略 1: 基于 element_type 的段落分组
            # 相邻的同类型 chunks 组合成 parent
            cur.execute(
                """
                SELECT chunk_id, element_type, chunk_index
                FROM chunks
                WHERE version_id = %s
                ORDER BY chunk_index
                """,
                (version_id,),
            )
            rows = cur.fetchall()

            if len(rows) < 2:
                return 0

            # 简单的分组策略：每 3-5 个 chunks 组成一个 parent
            group_size = min(5, max(2, len(rows) // 3))
            groups = [
                rows[i : i + group_size] for i in range(0, len(rows), group_size)
            ]

            for group in groups:
                if len(group) < 2:
                    continue

                parent_id = group[0][0]  # 第一个作为 parent
                children_ids = [row[0] for row in group[1:]]

                for child_id in children_ids:
                    cur.execute(
                        """
                        INSERT INTO multi_vector_mappings (
                            version_id, parent_chunk_id, child_chunk_id,
                            parent_type, child_type, relationship
                        ) VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT DO NOTHING
                        """,
                        (
                            version_id,
                            parent_id,
                            child_id,
                            "original",
                            "chunk",
                            "parent-child",
                        ),
                    )
                    count += 1

            # 策略 2: Contextual chunks 作为 parent
            cur.execute(
                """
                SELECT cc.chunk_id, cc.contextual_id
                FROM contextual_chunks cc
                WHERE cc.version_id = %s
                """,
                (version_id,),
            )
            contextual_rows = cur.fetchall()

            for chunk_id, contextual_id in contextual_rows:
                cur.execute(
                    """
                    INSERT INTO multi_vector_mappings (
                        version_id, parent_chunk_id, child_chunk_id,
                        parent_type, child_type, relationship, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                    """,
                    (
                        version_id,
                        chunk_id,  # original chunk as parent
                        chunk_id,  # same chunk as child (self-reference for contextual)
                        "contextual",
                        "chunk",
                        "summary",
                        {"contextual_id": str(contextual_id)},
                    ),
                )
                count += 1

        return count

    async def _generate_embeddings(
        self,
        conn: psycopg.Connection,
        version_id: str,
    ) -> None:
        """为没有 embedding 的 chunks 生成向量"""
        with conn.cursor() as cur:
            # 获取没有 embedding 的 chunks
            cur.execute(
                """
                SELECT chunk_id, text_raw
                FROM chunks
                WHERE version_id = %s AND embedding IS NULL AND text_raw IS NOT NULL
                """,
                (version_id,),
            )
            rows = cur.fetchall()

            if not rows:
                return

            chunk_ids = [row[0] for row in rows]
            texts = [row[1] for row in rows]

            # 批量生成 embeddings
            embeddings = embed_texts(texts, show_progress=True)

            # 更新数据库
            for chunk_id, embedding in zip(chunk_ids, embeddings):
                cur.execute(
                    """
                    UPDATE chunks
                    SET embedding = %s::vector
                    WHERE chunk_id = %s
                    """,
                    (embedding, chunk_id),
                )

    async def retrieve(
        self,
        query: str,
        version_id: Optional[str] = None,
        retrieval_mode: Optional[str] = None,
        top_k: Optional[int] = None,
        rerank: Optional[bool] = None,
        return_parents: bool = True,
    ) -> list[dict[str, Any]]:
        """执行检索

        使用 Multi-Vector Retriever 进行统一检索。

        Args:
            query: 查询文本
            version_id: 版本 ID 过滤 (可选)
            retrieval_mode: 检索模式 (可选，默认使用配置)
            top_k: 返回结果数量 (可选)
            rerank: 是否重排序 (可选)
            return_parents: 是否返回 parent chunks (默认 True)

        Returns:
            检索结果列表，每个结果包含 chunk 信息和元数据
        """
        retriever = self._get_retriever()

        mode = retrieval_mode or self.config.retrieval_mode
        k = top_k or self.config.retrieval_top_k
        do_rerank = rerank if rerank is not None else self.config.retrieval_rerank

        try:
            results = await retriever.retrieve(
                query=query,
                retrieval_mode=mode,
                top_k=k,
                rerank=do_rerank,
                version_id=version_id,
                return_parents=return_parents,
            )
            return results
        except Exception as e:
            logger.error(f"[CPC Pipeline] Retrieval failed: {e}")
            return []

    async def build_indices(self, version_id: Optional[str] = None) -> dict[str, Any]:
        """构建或重建索引

        Args:
            version_id: 特定版本 ID (可选，为 None 时重建所有)

        Returns:
            构建结果统计
        """
        results = {
            "success": True,
            "rebuilt_embeddings": 0,
            "rebuilt_contextual": 0,
            "errors": [],
        }

        try:
            with self._get_connection() as conn:
                # 重建 contextual chunks embeddings
                if self.config.enable_contextual:
                    with conn.cursor() as cur:
                        where_clause = "AND version_id = %s" if version_id else ""
                        params = (version_id,) if version_id else ()

                        cur.execute(
                            f"""
                            SELECT contextual_id, contextualized_text
                            FROM contextual_chunks
                            WHERE embedding IS NULL {where_clause}
                            """,
                            params,
                        )
                        rows = cur.fetchall()

                        if rows:
                            texts = [row[1] for row in rows]
                            embeddings = embed_texts(texts)

                            for (contextual_id, _), embedding in zip(rows, embeddings):
                                cur.execute(
                                    """
                                    UPDATE contextual_chunks
                                    SET embedding = %s::vector
                                    WHERE contextual_id = %s
                                    """,
                                    (embedding, contextual_id),
                                )
                            results["rebuilt_contextual"] = len(rows)

                conn.commit()

        except Exception as e:
            results["success"] = False
            results["errors"].append(str(e))

        return results

    def get_stats(self, version_id: str) -> dict[str, Any]:
        """获取文档处理统计信息

        Args:
            version_id: 版本 ID

        Returns:
            统计信息字典
        """
        stats = {
            "version_id": version_id,
            "chunks": 0,
            "contextual_chunks": 0,
            "hierarchical_nodes": 0,
            "multi_vector_mappings": 0,
        }

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    # Chunks count
                    cur.execute(
                        "SELECT COUNT(*) FROM chunks WHERE version_id = %s",
                        (version_id,),
                    )
                    stats["chunks"] = cur.fetchone()[0]

                    # Contextual chunks count
                    cur.execute(
                        "SELECT COUNT(*) FROM contextual_chunks WHERE version_id = %s",
                        (version_id,),
                    )
                    stats["contextual_chunks"] = cur.fetchone()[0]

                    # Hierarchical nodes count
                    cur.execute(
                        "SELECT COUNT(*) FROM hierarchical_nodes WHERE version_id = %s",
                        (version_id,),
                    )
                    stats["hierarchical_nodes"] = cur.fetchone()[0]

                    # Multi-vector mappings count
                    cur.execute(
                        "SELECT COUNT(*) FROM multi_vector_mappings WHERE version_id = %s",
                        (version_id,),
                    )
                    stats["multi_vector_mappings"] = cur.fetchone()[0]

        except Exception as e:
            logger.error(f"[CPC Pipeline] Failed to get stats: {e}")
            stats["error"] = str(e)

        return stats


# 便捷函数

async def process_document_with_cpc(
    content_list: list[dict],
    document_title: str,
    project_id: str | uuid.UUID,
    document_id: str | uuid.UUID,
    version_id: str | uuid.UUID,
    config: Optional[CPCPipelineConfig] = None,
) -> ProcessResult:
    """使用 CPC Pipeline 处理文档的便捷函数

    Args:
        content_list: MineRU content_list
        document_title: 文档标题
        project_id: 项目 UUID
        document_id: 文档 UUID
        version_id: 版本 UUID
        config: Pipeline 配置 (可选)

    Returns:
        ProcessResult 处理结果
    """
    pipeline = CPCPipeline(config=config)
    return await pipeline.process_document(
        content_list=content_list,
        document_title=document_title,
        project_id=project_id,
        document_id=document_id,
        version_id=version_id,
    )


async def retrieve_with_cpc(
    query: str,
    version_id: Optional[str] = None,
    config: Optional[CPCPipelineConfig] = None,
    **kwargs,
) -> list[dict[str, Any]]:
    """使用 CPC Pipeline 检索的便捷函数

    Args:
        query: 查询文本
        version_id: 版本 ID 过滤
        config: Pipeline 配置 (可选)
        **kwargs: 传递给 retrieve() 的额外参数

    Returns:
        检索结果列表
    """
    pipeline = CPCPipeline(config=config)
    return await pipeline.retrieve(query, version_id=version_id, **kwargs)
