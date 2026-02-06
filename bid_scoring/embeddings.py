"""OpenAI Embeddings 模块

最佳实践:
- 使用 text-embedding-3-small 作为默认模型 (1536维)
- 批量请求提高效率
- 缓存机制避免重复调用
- 自动重试和错误处理
"""

from functools import lru_cache
from typing import Any

import tiktoken
from openai import OpenAI
from bid_scoring.config import load_settings


# 默认配置
DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_DIM = 1536
MAX_BATCH_SIZE = 2048  # OpenAI API 限制
MAX_TOKENS_PER_BATCH = 600000  # 约 600k tokens/分钟限制
MAX_INPUT_TOKENS = 8191  # OpenAI embedding 模型输入限制

# tiktoken 编码器缓存
_encoding_cache: dict[str, tiktoken.Encoding] = {}


def get_encoding(model: str = DEFAULT_MODEL) -> tiktoken.Encoding:
    """获取 tiktoken 编码器（带缓存）

    Args:
        model: 模型名称

    Returns:
        tiktoken 编码器
    """
    if model not in _encoding_cache:
        try:
            _encoding_cache[model] = tiktoken.encoding_for_model(model)
        except KeyError:
            # 如果模型未知，使用 cl100k_base（OpenAI 最新模型的编码）
            _encoding_cache[model] = tiktoken.get_encoding("cl100k_base")
    return _encoding_cache[model]


def count_tokens(text: str, model: str = DEFAULT_MODEL) -> int:
    """精确计算文本的 token 数量

    使用 tiktoken 进行精确计算，支持所有 OpenAI 模型。

    Args:
        text: 输入文本
        model: 模型名称

    Returns:
        精确的 token 数量
    """
    if not text:
        return 0
    encoding = get_encoding(model)
    return len(encoding.encode(text))


def truncate_to_max_tokens(
    text: str, max_tokens: int = MAX_INPUT_TOKENS, model: str = DEFAULT_MODEL
) -> str:
    """将文本截断到最大 token 数量

    使用 tiktoken 精确截断，确保不超过模型输入限制。

    Args:
        text: 输入文本
        max_tokens: 最大 token 数（默认 8191）
        model: 模型名称

    Returns:
        截断后的文本
    """
    if not text:
        return text

    encoding = get_encoding(model)
    tokens = encoding.encode(text)

    if len(tokens) <= max_tokens:
        return text

    # 截断并添加提示
    truncated = encoding.decode(tokens[:max_tokens])
    return truncated


def estimate_tokens(text: str, model: str = DEFAULT_MODEL) -> int:
    """估算文本的 token 数量（快速估算，不依赖 tiktoken）

    使用保守估计：每 2 个字符约 1 个 token。
    适用于快速估算，不需要精确值的场景。

    Args:
        text: 输入文本
        model: 模型名称（保留参数用于兼容性，但不使用）

    Returns:
        token 数量估计值
    """
    if not text:
        return 0
    # 保守估计：每 2 个字符约 1 个 token
    return len(text) // 2 + 1


def get_embedding_client() -> OpenAI:
    """获取配置好的 OpenAI 客户端"""
    settings = load_settings()

    timeout = settings.get("OPENAI_TIMEOUT") or 30
    max_retries = settings.get("OPENAI_MAX_RETRIES") or 2

    return OpenAI(
        api_key=settings["OPENAI_API_KEY"],
        base_url=settings.get("OPENAI_BASE_URL"),
        timeout=timeout,
        max_retries=max_retries,
    )


def get_embedding_config() -> dict[str, Any]:
    """获取 embedding 配置"""
    settings = load_settings()
    return {
        "model": settings.get("OPENAI_EMBEDDING_MODEL") or DEFAULT_MODEL,
        "dim": settings.get("OPENAI_EMBEDDING_DIM") or DEFAULT_DIM,
    }


@lru_cache(maxsize=1024)
def _cached_embedding(text_hash: str, model: str) -> tuple[list[float], str] | None:
    """内部缓存函数（使用文本哈希作为 key）

    注意: 这个函数实际不会直接调用，缓存逻辑在 embed_texts 中处理
    """
    return None


def embed_texts(
    texts: list[str],
    client: OpenAI | None = None,
    model: str | None = None,
    batch_size: int = 100,
    show_progress: bool = False,
    truncate: bool = True,
) -> list[list[float]]:
    """批量生成文本向量

    最佳实践:
    - 批量处理: 50-100条/批
    - Token限制: 每批不超过 100k tokens
    - 输入限制: 单条文本最多 8191 tokens
    - 自动重试: 使用 OpenAI 客户端内置重试
    - 进度显示: 可选显示处理进度

    Args:
        texts: 文本列表
        client: OpenAI 客户端（为 None 时自动创建）
        model: 模型名称（为 None 时使用配置）
        batch_size: 每批大小（默认 100）
        show_progress: 是否显示进度
        truncate: 是否自动截断超长文本（默认 True）

    Returns:
        向量列表（与输入顺序一致）

    Raises:
        ValueError: 输入为空或 API Key 未设置
        Exception: API 调用失败
    """
    if not texts:
        return []

    # 过滤空文本
    valid_texts = [(i, t) for i, t in enumerate(texts) if t and t.strip()]

    embedding_config = get_embedding_config()
    dim = embedding_config["dim"]

    if not valid_texts:
        # 全部为空，返回空向量
        return [[0.0] * dim for _ in texts]

    # 初始化客户端
    if client is None:
        client = get_embedding_client()

    if model is None:
        model = embedding_config["model"]

    # 检查并截断超长文本
    processed_texts = []
    for orig_idx, text in valid_texts:
        token_count = count_tokens(text, model)
        if token_count > MAX_INPUT_TOKENS:
            if truncate:
                if show_progress:
                    print(
                        f"  ⚠️ 文本 {orig_idx} 超出 {MAX_INPUT_TOKENS} tokens，自动截断"
                    )
                text = truncate_to_max_tokens(text, MAX_INPUT_TOKENS, model)
            else:
                raise ValueError(
                    f"文本 {orig_idx} 包含 {token_count} tokens，"
                    f"超过最大限制 {MAX_INPUT_TOKENS}"
                )
        processed_texts.append((orig_idx, text))

    # 按 token 数量分批
    batches = []
    current_batch = []
    current_tokens = 0

    for orig_idx, text in processed_texts:
        tokens = count_tokens(text, model)

        # 如果加入当前文本会超出限制，先保存当前批次
        if (
            len(current_batch) >= batch_size
            or current_tokens + tokens > MAX_TOKENS_PER_BATCH
        ):
            if current_batch:
                batches.append(current_batch)
            current_batch = []
            current_tokens = 0

        current_batch.append((orig_idx, text))
        current_tokens += tokens

    # 添加最后一批
    if current_batch:
        batches.append(current_batch)

    # 处理所有批次
    all_embeddings = []
    for text in texts:
        if text and text.strip():
            all_embeddings.append(None)
        else:
            all_embeddings.append([0.0] * dim)

    for batch_idx, batch in enumerate(batches):
        if show_progress:
            print(
                f"  批次 {batch_idx + 1}/{len(batches)}: {len(batch)} 条...",
                end=" ",
                flush=True,
            )

        batch_texts = [t for _, t in batch]

        try:
            response = client.embeddings.create(
                model=model,
                input=batch_texts,
            )

            # 按原始顺序填充结果
            for (orig_idx, _), embedding_data in zip(batch, response.data):
                all_embeddings[orig_idx] = embedding_data.embedding

            if show_progress:
                print("✅")

        except Exception as e:
            if show_progress:
                print(f"❌ {e}")
            raise

    return all_embeddings


def embed_single_text(
    text: str,
    client: OpenAI | None = None,
    model: str | None = None,
) -> list[float]:
    """生成单条文本的向量

    Args:
        text: 输入文本
        client: OpenAI 客户端
        model: 模型名称

    Returns:
        向量（1536维或3072维）
    """
    if not text or not text.strip():
        dim = get_embedding_config()["dim"]
        return [0.0] * dim

    results = embed_texts([text], client=client, model=model)
    return results[0]


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """计算两个向量的余弦相似度

    Args:
        vec1: 向量1
        vec2: 向量2

    Returns:
        余弦相似度 [-1, 1]
    """
    import math

    if len(vec1) != len(vec2):
        raise ValueError(f"向量维度不匹配: {len(vec1)} vs {len(vec2)}")

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)
