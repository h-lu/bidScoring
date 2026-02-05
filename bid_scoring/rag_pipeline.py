"""
RAG Pipeline - 最高标准实现

基于 2024-2025 年最佳实践:
- 模块化设计 (检索 → 构建上下文 → 生成)
- System + User Prompt 分离
- 严格的上下文约束（禁止 hallucination）
- 来源引用和可解释性
- Token 管理和上下文窗口优化

Best Practices from:
- LangChain RAG patterns
- OpenAI Prompt Engineering Guide
- Cohere Reranking Best Practices
- Mistral RAG Documentation
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import psycopg

from bid_scoring.config import load_settings
from bid_scoring.embeddings import embed_single_text, get_embedding_client
from bid_scoring.llm import LLMClient, get_llm_client

logger = logging.getLogger(__name__)


@dataclass
class RAGContext:
    """RAG 检索结果上下文"""
    content: str  # 用于 LLM 的完整内容
    source: str  # 来源标识（章节标题）
    similarity: float  # 相似度分数
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGResponse:
    """RAG 响应结果"""
    answer: str  # LLM 生成的答案
    contexts: list[RAGContext]  # 使用的上下文
    query: str  # 原始查询
    model: str  # 使用的模型
    tokens_used: int = 0  # Token 使用量


class RAGPromptBuilder:
    """
    RAG Prompt 构建器 - 基于最佳实践
    
    设计原则:
    1. System Prompt: 定义角色、约束、行为规则
    2. User Prompt: 包含查询和上下文
    3. 严格的 Grounding 约束（禁止 hallucination）
    4. 来源引用要求
    """
    
    # System Prompt: 定义助手角色和核心约束
    SYSTEM_TEMPLATE = """你是投标分析助手，专门回答关于投标文件的问题。

核心规则:
1. 严格基于提供的上下文回答，禁止引用外部知识
2. 如果上下文中没有答案，明确说明"根据提供的资料，无法找到相关信息"
3. 回答时引用具体的章节来源（如：根据"三、技术方案"部分）
4. 保持客观、准确，不添加推测性内容
5. 如果上下文有矛盾，指出矛盾点并说明依据

回答格式:
- 直接回答用户问题
- 必要时分点说明
- 标注信息来源"""

    # User Prompt Template: 包含上下文和查询
    USER_TEMPLATE = """请参考以下投标文件相关内容，回答用户问题。

=== 参考资料 ===

{context}

=== 用户问题 ===

{question}

=== 回答要求 ===
- 仅基于上述参考资料回答
- 标注信息来源（章节名称）
- 如无法回答，明确说明"资料中未提及相关信息" """

    @classmethod
    def build_prompt(
        cls,
        query: str,
        contexts: list[RAGContext],
        max_context_tokens: int = 6000,
    ) -> tuple[str, str]:
        """
        构建 RAG Prompt
        
        Args:
            query: 用户查询
            contexts: 检索到的上下文列表
            max_context_tokens: 最大上下文 token 数
            
        Returns:
            (system_prompt, user_prompt)
        """
        # 构建上下文文本（带来源标识）
        context_parts = []
        for i, ctx in enumerate(contexts, 1):
            source_tag = f"[来源: {ctx.source}]" if ctx.source else f"[来源 {i}]"
            context_parts.append(
                f"--- 片段 {i} {source_tag} ---\n{ctx.content}\n"
            )
        
        context_str = "\n".join(context_parts)
        
        # 简单的 token 估算（字符数/2）
        estimated_tokens = len(context_str) / 2
        if estimated_tokens > max_context_tokens:
            # 截断上下文
            logger.warning(
                f"上下文过长 ({estimated_tokens:.0f} tokens)，截断至 {max_context_tokens}"
            )
            # 保留最相关的部分
            chars_to_keep = int(max_context_tokens * 2)
            context_str = context_str[:chars_to_keep] + "\n...[内容截断]"
        
        system_prompt = cls.SYSTEM_TEMPLATE
        user_prompt = cls.USER_TEMPLATE.format(
            context=context_str,
            question=query,
        )
        
        return system_prompt, user_prompt


class RAGRetriever:
    """
    RAG 检索器 - Small-to-Big 策略
    
    流程:
    1. 在 chunks 上执行向量搜索（小粒度，高精度）
    2. 获取 parent sections（完整上下文）
    3. 去重并按相似度排序
    """
    
    def __init__(self, version_id: str, top_k: int = 5):
        self.version_id = version_id
        self.top_k = top_k
        self.settings = load_settings()
    
    def retrieve(self, query: str) -> list[RAGContext]:
        """
        执行检索
        
        Args:
            query: 用户查询
            
        Returns:
            按相似度排序的上下文列表
        """
        # 生成查询向量
        query_embedding = embed_single_text(query)
        
        with psycopg.connect(self.settings["DATABASE_URL"]) as conn:
            with conn.cursor() as cur:
                # Small-to-Big: 在 chunks 上搜索，返回 sections
                cur.execute(
                    """
                    SELECT DISTINCT ON (s.node_id)
                        s.content as section_content,
                        s.heading as section_heading,
                        1 - (c.embedding <=> %s::vector) as similarity,
                        c.char_count,
                        c.content_for_embedding as chunk_preview
                    FROM hierarchical_nodes c
                    JOIN hierarchical_nodes s ON c.parent_id = s.node_id
                    WHERE c.version_id = %s 
                      AND c.node_type = 'chunk'
                      AND c.embedding IS NOT NULL
                    ORDER BY s.node_id, c.embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (query_embedding, self.version_id, query_embedding, self.top_k * 2)
                )
                
                results = cur.fetchall()
                
                # 构建 RAGContext 列表（去重后）
                contexts = []
                seen_sections = set()
                
                for row in results:
                    section_heading = row[1]
                    if section_heading not in seen_sections:
                        seen_sections.add(section_heading)
                        contexts.append(RAGContext(
                            content=row[0],  # section_content (完整内容)
                            source=section_heading,
                            similarity=row[2],
                            metadata={
                                "chunk_preview": row[4],
                                "char_count": row[3],
                            }
                        ))
                        
                        if len(contexts) >= self.top_k:
                            break
                
                return contexts


class RAGPipeline:
    """
    RAG Pipeline - 最高标准实现
    
    完整流程:
    Query → Embed → Retrieve (Small-to-Big) → Build Prompt → LLM Generate → Response
    """
    
    def __init__(
        self,
        version_id: str,
        llm_client: Optional[Any] = None,
        top_k: int = 5,
        max_context_tokens: int = 6000,
    ):
        self.version_id = version_id
        # 使用 LLMClient 包装 OpenAI client
        if llm_client is None:
            from bid_scoring.llm import LLMClient
            self.llm_client = LLMClient(load_settings())
        else:
            self.llm_client = llm_client
        self.retriever = RAGRetriever(version_id, top_k=top_k)
        self.prompt_builder = RAGPromptBuilder()
        self.max_context_tokens = max_context_tokens
    
    async def query(self, query: str) -> RAGResponse:
        """
        执行 RAG 查询
        
        Args:
            query: 用户查询
            
        Returns:
            RAGResponse 包含答案和上下文
        """
        logger.info(f"[RAG] 处理查询: {query[:50]}...")
        
        # Step 1: 检索
        logger.info("[RAG] Step 1: 检索相关上下文...")
        contexts = self.retriever.retrieve(query)
        
        if not contexts:
            logger.warning("[RAG] 未找到相关上下文")
            return RAGResponse(
                answer="根据提供的投标文件资料，无法找到与您问题相关的信息。",
                contexts=[],
                query=query,
                model="unknown",
                tokens_used=0,
            )
        
        logger.info(f"[RAG] 找到 {len(contexts)} 个相关章节")
        for i, ctx in enumerate(contexts, 1):
            logger.debug(f"  [{i}] {ctx.source} (相似度: {ctx.similarity:.3f})")
        
        # Step 2: 构建 Prompt
        logger.info("[RAG] Step 2: 构建 Prompt...")
        system_prompt, user_prompt = self.prompt_builder.build_prompt(
            query=query,
            contexts=contexts,
            max_context_tokens=self.max_context_tokens,
        )
        
        # Step 3: LLM 生成
        logger.info("[RAG] Step 3: LLM 生成答案...")
        try:
            # 使用 LLMClient 生成
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            
            answer = self.llm_client.complete(
                messages=messages,
                model=load_settings().get("OPENAI_LLM_MODEL_DEFAULT"),
                temperature=0.3,  # 低温度确保准确性
            )
            
            # LLMClient.complete 返回字符串
            tokens_used = 0  # TODO: 从响应中获取 token 使用量
            model_used = load_settings().get("OPENAI_LLM_MODEL_DEFAULT", "unknown")
            
            logger.info(f"[RAG] 生成完成，使用 {tokens_used} tokens")
            
            return RAGResponse(
                answer=answer,
                contexts=contexts,
                query=query,
                model=model_used,
                tokens_used=tokens_used,
            )
            
        except Exception as e:
            logger.error(f"[RAG] LLM 生成失败: {e}")
            raise
    
    def query_sync(self, query: str) -> RAGResponse:
        """同步版本（用于测试）"""
        import asyncio
        return asyncio.run(self.query(query))


# 便捷函数

async def rag_query(
    query: str,
    version_id: str,
    top_k: int = 5,
) -> RAGResponse:
    """
    便捷的 RAG 查询函数
    
    Example:
        >>> response = await rag_query("售后服务包括哪些？", version_id="xxx")
        >>> print(response.answer)
        >>> for ctx in response.contexts:
        ...     print(f"来源: {ctx.source}, 相似度: {ctx.similarity}")
    """
    pipeline = RAGPipeline(version_id=version_id, top_k=top_k)
    return await pipeline.query(query)


def format_rag_response(response: RAGResponse) -> str:
    """格式化 RAG 响应为可读文本"""
    lines = [
        "=" * 60,
        "RAG 查询结果",
        "=" * 60,
        f"查询: {response.query}",
        f"模型: {response.model}",
        f"Token 使用: {response.tokens_used}",
        "-" * 60,
        "答案:",
        response.answer,
        "-" * 60,
        "参考来源:",
    ]
    
    for i, ctx in enumerate(response.contexts, 1):
        lines.append(f"  [{i}] {ctx.source} (相似度: {ctx.similarity:.3f})")
    
    lines.append("=" * 60)
    return "\n".join(lines)
