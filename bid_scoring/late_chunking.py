"""Late Chunking Encoder - 长上下文感知向量生成

实现基于 Late Chunking 的向量化方法:
1. 对完整文本进行一次性编码，获取 token-level embeddings
2. 根据 chunk 边界对 token embeddings 进行池化
3. 生成具有上下文感知的 chunk 向量

Reference: https://arxiv.org/abs/2409.04701
Reference: https://github.com/jina-ai/late-chunking

最佳实践:
- 使用支持长上下文的 embedding 模型（如 jina-embeddings-v2，支持 8k tokens）
- Mean pooling 作为默认池化策略
- 失败时自动回退到标准 embedding
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Protocol, Tuple, Union

from bid_scoring.embeddings import embed_texts, get_embedding_config

logger = logging.getLogger(__name__)


# 默认配置
DEFAULT_LATE_CHUNKING_MODEL = "jina-embeddings-v2-base-zh"
DEFAULT_MAX_LENGTH = 8192  # jina-embeddings-v2 支持 8k tokens
DEFAULT_POOLING = "mean"


class TokenizerProtocol(Protocol):
    """Tokenizer 协议定义"""
    
    def encode(self, text: str) -> List[int]:
        """将文本编码为 token IDs"""
        ...
    
    def decode(self, token_ids: List[int]) -> str:
        """将 token IDs 解码为文本"""
        ...
    
    def __len__(self) -> int:
        """返回词表大小"""
        ...


class EmbeddingModelProtocol(Protocol):
    """Embedding 模型协议定义"""
    
    def encode(
        self, 
        texts: Union[str, List[str]], 
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> Union[List[float], List[List[float]]]:
        """编码文本为向量"""
        ...


@dataclass
class LateChunkingResult:
    """Late Chunking 编码结果
    
    Attributes:
        chunk_embeddings: 每个 chunk 的向量列表
        full_embedding: 完整文本的向量（可选）
        token_embeddings: 所有 token 的向量（可选，用于调试）
        token_boundaries: 每个 chunk 对应的 token 边界
        pooling_strategy: 使用的池化策略
    """
    chunk_embeddings: List[List[float]]
    full_embedding: Optional[List[float]] = None
    token_embeddings: Optional[List[List[float]]] = None
    token_boundaries: List[Tuple[int, int]] = field(default_factory=list)
    pooling_strategy: str = "mean"


class LateChunkingEncoder:
    """Late Chunking 编码器
    
    实现长上下文感知向量生成，通过以下步骤:
    1. 对完整文本进行 tokenization
    2. 获取 token-level embeddings（需要模型支持）
    3. 根据 chunk 边界进行池化
    
    Example:
        >>> encoder = LateChunkingEncoder()
        >>> full_text = "这是第一段。这是第二段。这是第三段。"
        >>> boundaries = [(0, 6), (6, 12), (12, 18)]  # token 边界
        >>> result = encoder.encode_with_late_chunking(full_text, boundaries)
        >>> print(f"生成 {len(result.chunk_embeddings)} 个 chunk 向量")
    
    Args:
        model: Embedding 模型名称或实例
        tokenizer: Tokenizer 实例（可选，模型自带则不需要）
        max_length: 最大序列长度
        pooling_strategy: 池化策略 ("mean", "max", "cls")
        fallback_to_standard: 失败时是否回退到标准 embedding
    """

    def __init__(
        self,
        model: Optional[Union[str, EmbeddingModelProtocol]] = None,
        tokenizer: Optional[TokenizerProtocol] = None,
        max_length: int = DEFAULT_MAX_LENGTH,
        pooling_strategy: str = DEFAULT_POOLING,
        fallback_to_standard: bool = True,
    ):
        self.model = model or DEFAULT_LATE_CHUNKING_MODEL
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pooling_strategy = pooling_strategy
        self.fallback_to_standard = fallback_to_standard
        
        # 内部状态
        self._model_instance: Optional[EmbeddingModelProtocol] = None
        self._tokenizer_instance: Optional[TokenizerProtocol] = None
        
        # 验证池化策略
        valid_strategies = {"mean", "max", "cls", "sum"}
        if pooling_strategy not in valid_strategies:
            raise ValueError(f"Invalid pooling_strategy: {pooling_strategy}. "
                           f"Must be one of {valid_strategies}")

    def _get_model(self) -> EmbeddingModelProtocol:
        """获取或初始化模型实例"""
        if self._model_instance is not None:
            return self._model_instance
        
        if isinstance(self.model, str):
            # 尝试加载模型
            try:
                self._model_instance = self._load_model(self.model)
            except Exception as e:
                logger.warning(f"Failed to load model {self.model}: {e}")
                raise RuntimeError(f"Cannot load embedding model: {e}")
        else:
            self._model_instance = self.model
        
        return self._model_instance

    def _load_model(self, model_name: str) -> EmbeddingModelProtocol:
        """加载 embedding 模型
        
        支持多种模型:
        - jina-embeddings-v2 系列（推荐，原生支持 token-level）
        - 其他通过 transformers 加载的模型
        """
        # 首先尝试加载 sentence-transformers 格式
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading model via sentence-transformers: {model_name}")
            model = SentenceTransformer(model_name, trust_remote_code=True)
            return _SentenceTransformerWrapper(model)
        except ImportError:
            logger.debug("sentence-transformers not installed")
        except Exception as e:
            logger.debug(f"Failed to load with sentence-transformers: {e}")
        
        # 尝试直接通过 transformers 加载
        try:
            from transformers import AutoModel, AutoTokenizer
            
            logger.info(f"Loading model via transformers: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            
            self._tokenizer_instance = _TransformersTokenizerWrapper(tokenizer)
            return _TransformersModelWrapper(model, tokenizer)
        except ImportError:
            logger.debug("transformers not installed")
        except Exception as e:
            logger.debug(f"Failed to load with transformers: {e}")
        
        raise RuntimeError(
            f"Cannot load model {model_name}. "
            "Please install sentence-transformers or transformers."
        )

    def _get_tokenizer(self) -> TokenizerProtocol:
        """获取或初始化 tokenizer"""
        if self._tokenizer_instance is not None:
            return self._tokenizer_instance
        
        # 如果模型已经加载，尝试从中获取 tokenizer
        if isinstance(self.model, str) and "jina" in self.model.lower():
            try:
                from transformers import AutoTokenizer
                self._tokenizer_instance = _TransformersTokenizerWrapper(
                    AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)
                )
                return self._tokenizer_instance
            except Exception as e:
                logger.warning(f"Failed to load tokenizer: {e}")
        
        # 使用简单的字符级 tokenizer 作为后备
        logger.warning("Using fallback character-based tokenizer")
        self._tokenizer_instance = _CharacterTokenizer()
        return self._tokenizer_instance

    def _tokenize(self, text: str) -> Tuple[List[int], List[str]]:
        """将文本 tokenize
        
        Returns:
            (token_ids, tokens) 元组
        """
        tokenizer = self._get_tokenizer()
        token_ids = tokenizer.encode(text)
        
        # 截断到最大长度
        if len(token_ids) > self.max_length:
            logger.warning(f"Text length {len(token_ids)} exceeds max_length {self.max_length}, truncating")
            token_ids = token_ids[:self.max_length]
        
        # 尝试获取 tokens（用于调试）
        try:
            if hasattr(tokenizer, 'convert_ids_to_tokens'):
                tokens = tokenizer.convert_ids_to_tokens(token_ids)
            else:
                tokens = [str(tid) for tid in token_ids]
        except Exception:
            tokens = [str(tid) for tid in token_ids]
        
        return token_ids, tokens

    def _get_token_embeddings(self, text: str) -> List[List[float]]:
        """获取文本的 token-level embeddings
        
        这是 Late Chunking 的核心，需要模型输出 token-level embeddings。
        不是所有模型都支持，需要检查模型能力。
        
        Args:
            text: 输入文本
            
        Returns:
            每个 token 的向量列表
        """
        model = self._get_model()
        
        # 检查模型是否支持 token-level embeddings
        if hasattr(model, 'encode'):
            try:
                # 尝试调用模型获取详细输出
                result = model.encode(text, output_value="token_embeddings")
                if result is not None:
                    return result
            except (TypeError, AttributeError):
                pass
        
        # 对于 transformers 模型，需要前向传播获取 hidden states
        if hasattr(model, 'forward') or hasattr(model, '__call__'):
            return self._get_transformers_token_embeddings(text)
        
        raise NotImplementedError(
            "Model does not support token-level embeddings. "
            "Please use a model that supports this feature, such as jina-embeddings-v2."
        )

    def _get_transformers_token_embeddings(self, text: str) -> List[List[float]]:
        """使用 transformers 模型获取 token embeddings"""
        import torch
        
        tokenizer = self._get_tokenizer()
        model = self._get_model()
        
        # Tokenize
        if hasattr(tokenizer, '_tokenizer'):
            # 包装过的 tokenizer
            encodings = tokenizer._tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            )
        else:
            encodings = tokenizer.encode(text)
            if isinstance(encodings, list):
                encodings = {"input_ids": torch.tensor([encodings])}
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**encodings, output_hidden_states=True)
            # 使用最后一层 hidden state
            hidden_states = outputs.hidden_states[-1]
            # 移除 batch 维度
            token_embeddings = hidden_states[0].cpu().numpy().tolist()
        
        return token_embeddings

    def _pool_embeddings(
        self, 
        token_embeddings: List[List[float]], 
        start_idx: int, 
        end_idx: int,
    ) -> List[float]:
        """池化指定范围内的 token embeddings
        
        Args:
            token_embeddings: 所有 token 的向量
            start_idx: 起始索引（包含）
            end_idx: 结束索引（不包含）
            
        Returns:
            池化后的向量
        """
        if start_idx >= end_idx:
            # 空范围，返回零向量
            dim = len(token_embeddings[0]) if token_embeddings else 768
            return [0.0] * dim
        
        # 确保索引在有效范围内
        start_idx = max(0, start_idx)
        end_idx = min(len(token_embeddings), end_idx)
        
        chunk_embs = token_embeddings[start_idx:end_idx]
        
        if self.pooling_strategy == "mean":
            return self._mean_pooling(chunk_embs)
        elif self.pooling_strategy == "max":
            return self._max_pooling(chunk_embs)
        elif self.pooling_strategy == "sum":
            return self._sum_pooling(chunk_embs)
        elif self.pooling_strategy == "cls":
            # CLS token 通常在位置 0
            return chunk_embs[0] if chunk_embs else [0.0] * len(token_embeddings[0])
        else:
            return self._mean_pooling(chunk_embs)

    def _mean_pooling(self, embeddings: List[List[float]]) -> List[float]:
        """Mean pooling"""
        if not embeddings:
            return []
        dim = len(embeddings[0])
        result = []
        for i in range(dim):
            values = [emb[i] for emb in embeddings]
            result.append(sum(values) / len(values))
        return result

    def _max_pooling(self, embeddings: List[List[float]]) -> List[float]:
        """Max pooling"""
        if not embeddings:
            return []
        dim = len(embeddings[0])
        return [max(emb[i] for emb in embeddings) for i in range(dim)]

    def _sum_pooling(self, embeddings: List[List[float]]) -> List[float]:
        """Sum pooling"""
        if not embeddings:
            return []
        dim = len(embeddings[0])
        return [sum(emb[i] for emb in embeddings) for i in range(dim)]

    def encode_with_late_chunking(
        self,
        full_text: str,
        chunk_boundaries: List[Tuple[int, int]],
    ) -> LateChunkingResult:
        """使用 Late Chunking 编码文本
        
        这是核心方法，实现以下步骤:
        1. 对完整文本进行 tokenization
        2. 获取 token-level embeddings
        3. 根据 chunk_boundaries 进行池化
        
        Args:
            full_text: 完整文本
            chunk_boundaries: Chunk 边界列表，每个元素是 (start_token_idx, end_token_idx)
                            其中索引对应 token 位置
                            
        Returns:
            LateChunkingResult 包含每个 chunk 的向量
            
        Raises:
            ValueError: 输入参数无效
            RuntimeError: 编码过程中发生错误且无法回退
            
        Example:
            >>> encoder = LateChunkingEncoder()
            >>> text = "第一段内容。第二段内容。第三段内容。"
            >>> # 假设每个句号分隔一个 chunk，token 位置分别是 0-4, 5-9, 10-14
            >>> boundaries = [(0, 5), (5, 10), (10, 15)]
            >>> result = encoder.encode_with_late_chunking(text, boundaries)
        """
        if not full_text or not full_text.strip():
            raise ValueError("full_text cannot be empty")
        
        if not chunk_boundaries:
            raise ValueError("chunk_boundaries cannot be empty")
        
        # 验证边界
        for i, (start, end) in enumerate(chunk_boundaries):
            if start < 0 or end < 0:
                raise ValueError(f"Invalid boundary at index {i}: ({start}, {end}), indices must be non-negative")
            if start >= end:
                raise ValueError(f"Invalid boundary at index {i}: ({start}, {end}), start must be less than end")
        
        try:
            # 1. Tokenize 完整文本
            token_ids, tokens = self._tokenize(full_text)
            
            # 2. 获取 token embeddings
            token_embeddings = self._get_token_embeddings(full_text)
            
            # 3. 按边界池化
            chunk_embeddings = []
            valid_boundaries = []
            
            for start_idx, end_idx in chunk_boundaries:
                # 确保边界在有效范围内
                if start_idx >= len(token_embeddings):
                    logger.warning(f"Boundary ({start_idx}, {end_idx}) exceeds token count {len(token_embeddings)}, skipping")
                    continue
                
                # 截断到有效范围
                actual_end = min(end_idx, len(token_embeddings))
                
                pooled_emb = self._pool_embeddings(token_embeddings, start_idx, actual_end)
                chunk_embeddings.append(pooled_emb)
                valid_boundaries.append((start_idx, actual_end))
            
            # 4. 生成完整文本的 embedding（用于参考）
            full_embedding = self._pool_embeddings(token_embeddings, 0, len(token_embeddings))
            
            return LateChunkingResult(
                chunk_embeddings=chunk_embeddings,
                full_embedding=full_embedding,
                token_embeddings=token_embeddings if False else None,  # 默认不保存，节省内存
                token_boundaries=valid_boundaries,
                pooling_strategy=self.pooling_strategy,
            )
            
        except Exception as e:
            logger.error(f"Late chunking encoding failed: {e}")
            
            if self.fallback_to_standard:
                logger.warning("Falling back to standard embedding")
                return self._fallback_encode(full_text, chunk_boundaries)
            else:
                raise RuntimeError(f"Late chunking failed and no fallback: {e}")

    def _fallback_encode(
        self, 
        full_text: str, 
        chunk_boundaries: List[Tuple[int, int]],
    ) -> LateChunkingResult:
        """回退到标准 embedding
        
        当 Late Chunking 失败时，独立编码每个 chunk。
        这会丢失上下文信息，但保证可用性。
        """
        # 从边界推断 chunk 文本
        token_ids, _ = self._tokenize(full_text)
        tokenizer = self._get_tokenizer()
        
        chunk_texts = []
        for start_idx, end_idx in chunk_boundaries:
            if start_idx >= len(token_ids):
                chunk_texts.append("")
                continue
            
            chunk_token_ids = token_ids[start_idx:min(end_idx, len(token_ids))]
            try:
                chunk_text = tokenizer.decode(chunk_token_ids)
            except Exception:
                # 如果 decode 失败，直接截取字符串
                char_start = start_idx * 2  # 粗略估计
                char_end = min(end_idx * 2, len(full_text))
                chunk_text = full_text[char_start:char_end]
            
            chunk_texts.append(chunk_text)
        
        # 使用标准 embedding
        embeddings = embed_texts(chunk_texts)
        
        return LateChunkingResult(
            chunk_embeddings=embeddings,
            full_embedding=embed_texts([full_text])[0] if full_text else None,
            token_boundaries=chunk_boundaries,
            pooling_strategy="fallback_standard",
        )

    def encode_text_to_boundaries(
        self,
        full_text: str,
        chunk_texts: List[str],
    ) -> LateChunkingResult:
        """从 chunk 文本列表自动推断边界并编码
        
        这是一个便捷方法，用户不需要手动计算 token 边界。
        
        Args:
            full_text: 完整文本
            chunk_texts: 每个 chunk 的文本列表
            
        Returns:
            LateChunkingResult
            
        Example:
            >>> encoder = LateChunkingEncoder()
            >>> full_text = "第一段。第二段。第三段。"
            >>> chunks = ["第一段。", "第二段。", "第三段。"]
            >>> result = encoder.encode_text_to_boundaries(full_text, chunks)
        """
        tokenizer = self._get_tokenizer()
        
        # 计算每个 chunk 的 token 边界
        boundaries = []
        current_pos = 0
        
        for chunk_text in chunk_texts:
            chunk_tokens = tokenizer.encode(chunk_text)
            start_idx = current_pos
            end_idx = current_pos + len(chunk_tokens)
            boundaries.append((start_idx, end_idx))
            current_pos = end_idx
        
        return self.encode_with_late_chunking(full_text, boundaries)

    def is_available(self) -> bool:
        """检查编码器是否可用（依赖是否安装）"""
        try:
            self._get_model()
            return True
        except Exception as e:
            logger.debug(f"LateChunkingEncoder not available: {e}")
            return False


# 模型包装器

class _SentenceTransformerWrapper:
    """SentenceTransformer 模型包装器"""
    
    def __init__(self, model):
        self.model = model
        self._has_token_embeddings = hasattr(model, 'encode') and 'output_value' in \
            model.encode.__code__.co_varnames
    
    def encode(self, text, **kwargs):
        return self.model.encode(text, **kwargs)
    
    def __call__(self, **kwargs):
        # 用于 transformers 风格调用
        import torch
        
        if hasattr(self.model, 'forward'):
            return self.model.forward(**kwargs)
        
        # 默认行为：返回输入
        class DummyOutput:
            def __init__(self):
                self.hidden_states = [kwargs.get('input_ids')]
        return DummyOutput()


class _TransformersModelWrapper:
    """Transformers 模型包装器"""
    
    def __init__(self, model, tokenizer):
        self._model = model
        self._tokenizer = tokenizer
    
    def __call__(self, **kwargs):
        import torch
        
        # 处理输入
        if 'input_ids' in kwargs:
            input_ids = kwargs['input_ids']
            if isinstance(input_ids, list):
                input_ids = torch.tensor([input_ids])
            elif not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor(input_ids)
            
            attention_mask = kwargs.get('attention_mask')
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            elif isinstance(attention_mask, list):
                attention_mask = torch.tensor([attention_mask])
            
            # 确保在正确的设备上
            device = next(self._model.parameters()).device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            return self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=kwargs.get('output_hidden_states', False),
            )
        
        return self._model(**kwargs)


class _TransformersTokenizerWrapper:
    """Transformers Tokenizer 包装器"""
    
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
    
    def encode(self, text: str) -> List[int]:
        result = self._tokenizer.encode(text, add_special_tokens=True)
        # 确保返回的是 list of ints
        if hasattr(result, 'tolist'):
            return result.tolist()
        return list(result)
    
    def decode(self, token_ids: List[int]) -> str:
        return self._tokenizer.decode(token_ids, skip_special_tokens=True)
    
    def convert_ids_to_tokens(self, token_ids: List[int]) -> List[str]:
        return self._tokenizer.convert_ids_to_tokens(token_ids)
    
    def __len__(self) -> int:
        return len(self._tokenizer)


class _CharacterTokenizer:
    """简单的字符级 Tokenizer（作为后备）"""
    
    def encode(self, text: str) -> List[int]:
        # 每 2 个字符作为一个 token（粗略估计）
        return list(range(0, len(text), 2))
    
    def decode(self, token_ids: List[int]) -> str:
        # 无法真正解码，返回占位符
        return ""
    
    def __len__(self) -> int:
        return 65536  # 假设最大字符数


def estimate_token_boundaries(
    full_text: str,
    chunk_texts: List[str],
    chars_per_token: int = 2,
) -> List[Tuple[int, int]]:
    """估算 chunk 的 token 边界
    
    当没有精确 tokenizer 时，使用字符数估算 token 位置。
    
    Args:
        full_text: 完整文本
        chunk_texts: 每个 chunk 的文本
        chars_per_token: 每个 token 平均字符数（中文约 1.5，英文约 4）
        
    Returns:
        Token 边界列表
        
    Example:
        >>> full_text = "这是第一段。这是第二段。"
        >>> chunks = ["这是第一段。", "这是第二段。"]
        >>> boundaries = estimate_token_boundaries(full_text, chunks, chars_per_token=2)
        >>> print(boundaries)  # [(0, 4), (4, 8)]
    """
    boundaries = []
    current_pos = 0
    
    for chunk_text in chunk_texts:
        # 估算 token 数
        token_count = len(chunk_text) // chars_per_token + 1
        start_idx = current_pos
        end_idx = current_pos + token_count
        boundaries.append((start_idx, end_idx))
        current_pos = end_idx
    
    return boundaries


def create_late_chunking_encoder(
    model_name: Optional[str] = None,
    pooling: str = "mean",
    fallback: bool = True,
) -> LateChunkingEncoder:
    """创建 Late Chunking 编码器的工厂函数
    
    Args:
        model_name: 模型名称，默认使用 jina-embeddings-v2-base-zh
        pooling: 池化策略
        fallback: 失败时是否回退
        
    Returns:
        LateChunkingEncoder 实例
        
    Example:
        >>> encoder = create_late_chunking_encoder("jina-embeddings-v2-base-zh")
        >>> result = encoder.encode_with_late_chunking(text, boundaries)
    """
    model = model_name or DEFAULT_LATE_CHUNKING_MODEL
    
    return LateChunkingEncoder(
        model=model,
        pooling_strategy=pooling,
        fallback_to_standard=fallback,
    )
