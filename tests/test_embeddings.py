"""Tests for embedding module.

Best practices for text embeddings in RAG systems:
1. Use tiktoken for accurate token counting
2. Handle token limit (8191 tokens per input for text-embedding-3)
3. Batch processing for efficiency (max 2048 inputs per request)
4. Persistent caching to avoid redundant API calls
5. Vector storage and similarity search with pgvector
"""

import pytest
from unittest.mock import Mock, patch


class TestTokenEstimation:
    """Test token estimation functions."""

    def test_estimate_tokens_empty(self):
        """Test token estimation for empty string."""
        from bid_scoring.embeddings import estimate_tokens

        assert estimate_tokens("") == 0
        assert estimate_tokens(None) == 0

    def test_estimate_tokens_chinese(self):
        """Test token estimation for Chinese text.

        Chinese typically uses ~1.5 chars per token.
        Our conservative estimate uses 2 chars per token.
        """
        from bid_scoring.embeddings import estimate_tokens

        text = "这是一个测试句子"
        estimated = estimate_tokens(text)

        # Conservative estimate: len // 2 + 1
        assert estimated == len(text) // 2 + 1
        assert estimated > 0

    def test_estimate_tokens_english(self):
        """Test token estimation for English text.

        English typically uses ~4 chars per token.
        Our conservative estimate uses 2 chars per token.
        """
        from bid_scoring.embeddings import estimate_tokens

        text = "This is a test sentence for token estimation."
        estimated = estimate_tokens(text)

        assert estimated == len(text) // 2 + 1


class TestEmbeddingClient:
    """Test embedding client configuration."""

    def test_get_embedding_client(self):
        """Test that embedding client can be created."""
        from bid_scoring.embeddings import get_embedding_client

        # This will fail if OPENAI_API_KEY is not set
        try:
            client = get_embedding_client()
            assert client is not None
        except Exception as e:
            pytest.skip(f"OpenAI client creation failed: {e}")

    def test_get_embedding_config(self):
        """Test embedding configuration."""
        from bid_scoring.embeddings import get_embedding_config

        config = get_embedding_config()

        assert "model" in config
        assert "dim" in config
        assert config["dim"] in [1536, 3072]  # text-embedding-3-small or large


class TestEmbedTexts:
    """Test batch text embedding."""

    @patch("bid_scoring.embeddings.OpenAI")
    def test_embed_texts_empty_list(self, mock_openai):
        """Test embedding empty list returns empty list."""
        from bid_scoring.embeddings import embed_texts

        result = embed_texts([])
        assert result == []

    @patch("bid_scoring.embeddings.OpenAI")
    def test_embed_texts_all_empty_strings(self, mock_openai):
        """Test embedding list of empty strings returns zero vectors."""
        from bid_scoring.embeddings import embed_texts, DEFAULT_DIM

        texts = ["", "   ", "\n"]
        result = embed_texts(texts)

        assert len(result) == 3
        for vec in result:
            assert len(vec) == DEFAULT_DIM
            assert all(v == 0.0 for v in vec)

    @patch("bid_scoring.embeddings.OpenAI")
    def test_embed_texts_mixed_empty_and_valid(self, mock_openai):
        """Test embedding list with both empty and valid texts."""
        from bid_scoring.embeddings import embed_texts, DEFAULT_DIM

        # Mock the API response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1] * DEFAULT_DIM),
        ]
        mock_client.embeddings.create.return_value = mock_response

        texts = ["", "valid text", ""]
        result = embed_texts(texts, client=mock_client)

        assert len(result) == 3
        # The valid text should be embedded, empty texts handled appropriately
        # Note: current implementation may return None for empty positions
        assert result[1] == [0.1] * DEFAULT_DIM

    @patch("bid_scoring.embeddings.OpenAI")
    def test_embed_texts_mixed_empty_and_valid_fills_zero_vectors(self, mock_openai):
        """Test mixed empty inputs return zero vectors for empty entries."""
        from bid_scoring.embeddings import embed_texts, DEFAULT_DIM

        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * DEFAULT_DIM)]
        mock_client.embeddings.create.return_value = mock_response

        texts = ["", "valid text", "  "]
        result = embed_texts(texts, client=mock_client)

        assert len(result) == 3
        assert result[1] == [0.1] * DEFAULT_DIM
        assert result[0] == [0.0] * DEFAULT_DIM
        assert result[2] == [0.0] * DEFAULT_DIM

    def test_embed_texts_all_empty_uses_config_dim(self, monkeypatch):
        """Test all empty inputs use config dim even with explicit model."""
        import bid_scoring.embeddings as embeddings

        monkeypatch.setattr(
            embeddings,
            "get_embedding_config",
            lambda: {"model": "text-embedding-3-large", "dim": 3072},
        )
        texts = ["", "   "]

        result = embeddings.embed_texts(texts, model="text-embedding-3-large")

        assert len(result) == 2
        assert len(result[0]) == 3072
        assert len(result[1]) == 3072

    @patch("bid_scoring.embeddings.OpenAI")
    def test_embed_texts_batching(self, mock_openai):
        """Test that texts are properly batched."""
        from bid_scoring.embeddings import embed_texts, DEFAULT_DIM

        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1] * DEFAULT_DIM),
            Mock(embedding=[0.2] * DEFAULT_DIM),
        ]
        mock_client.embeddings.create.return_value = mock_response

        texts = ["text1", "text2"]
        result = embed_texts(texts, client=mock_client, batch_size=10)

        assert len(result) == 2
        # Verify API was called once for small batch
        mock_client.embeddings.create.assert_called_once()

    @patch("bid_scoring.embeddings.OpenAI")
    def test_embed_texts_respects_batch_size(self, mock_openai):
        """Test that batch_size parameter is respected."""
        from bid_scoring.embeddings import embed_texts, DEFAULT_DIM

        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * DEFAULT_DIM)]
        mock_client.embeddings.create.return_value = mock_response

        texts = ["text"] * 5
        embed_texts(texts, client=mock_client, batch_size=2)

        # Should be called 3 times: 2+2+1
        assert mock_client.embeddings.create.call_count == 3


class TestEmbedSingleText:
    """Test single text embedding."""

    @patch("bid_scoring.embeddings.embed_texts")
    def test_embed_single_text_empty(self, mock_embed_texts):
        """Test embedding empty string returns zero vector."""
        from bid_scoring.embeddings import embed_single_text, DEFAULT_DIM

        result = embed_single_text("")

        assert len(result) == DEFAULT_DIM
        assert all(v == 0.0 for v in result)
        mock_embed_texts.assert_not_called()

    @patch("bid_scoring.embeddings.embed_texts")
    def test_embed_single_text_valid(self, mock_embed_texts):
        """Test embedding valid text."""
        from bid_scoring.embeddings import embed_single_text, DEFAULT_DIM

        mock_embed_texts.return_value = [[0.1] * DEFAULT_DIM]

        result = embed_single_text("test text")

        assert result == [0.1] * DEFAULT_DIM
        mock_embed_texts.assert_called_once()


class TestCosineSimilarity:
    """Test cosine similarity calculation."""

    def test_cosine_similarity_identical_vectors(self):
        """Test identical vectors have similarity 1.0."""
        from bid_scoring.embeddings import cosine_similarity

        vec = [1.0, 2.0, 3.0]
        result = cosine_similarity(vec, vec)

        assert result == pytest.approx(1.0, abs=1e-6)

    def test_cosine_similarity_opposite_vectors(self):
        """Test opposite vectors have similarity -1.0."""
        from bid_scoring.embeddings import cosine_similarity

        vec1 = [1.0, 2.0, 3.0]
        vec2 = [-1.0, -2.0, -3.0]
        result = cosine_similarity(vec1, vec2)

        assert result == pytest.approx(-1.0, abs=1e-6)

    def test_cosine_similarity_orthogonal_vectors(self):
        """Test orthogonal vectors have similarity 0.0."""
        from bid_scoring.embeddings import cosine_similarity

        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        result = cosine_similarity(vec1, vec2)

        assert result == pytest.approx(0.0, abs=1e-6)

    def test_cosine_similarity_different_dimensions(self):
        """Test that different dimensions raise error."""
        from bid_scoring.embeddings import cosine_similarity

        vec1 = [1.0, 2.0, 3.0]
        vec2 = [1.0, 2.0]

        with pytest.raises(ValueError):
            cosine_similarity(vec1, vec2)

    def test_cosine_similarity_zero_vector(self):
        """Test that zero vector returns 0.0."""
        from bid_scoring.embeddings import cosine_similarity

        vec1 = [0.0, 0.0, 0.0]
        vec2 = [1.0, 2.0, 3.0]
        result = cosine_similarity(vec1, vec2)

        assert result == 0.0


class TestEmbeddingIntegration:
    """Integration tests for embedding workflow."""

    def test_embed_real_text(self):
        """Test embedding real text (requires API key)."""
        from bid_scoring.embeddings import embed_texts, DEFAULT_DIM

        texts = [
            "这是一个测试句子。",
            "This is a test sentence.",
            "Hello, world!",
        ]

        results = embed_texts(texts)

        assert len(results) == 3
        for vec in results:
            assert len(vec) == DEFAULT_DIM
            # Check that it's not all zeros
            assert any(v != 0.0 for v in vec)

    def test_embedding_consistency(self):
        """Test that same text produces similar embedding (within tolerance)."""
        from bid_scoring.embeddings import embed_single_text

        text = "Consistency test"

        vec1 = embed_single_text(text)
        vec2 = embed_single_text(text)

        # Embeddings should be very similar (cosine similarity close to 1)
        # but may not be exactly equal due to API non-determinism
        assert len(vec1) == len(vec2)

        # Calculate cosine similarity
        import math
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        cosine_sim = dot_product / (magnitude1 * magnitude2)

        # Cosine similarity should be very close to 1.0 (within 0.1%)
        assert cosine_sim > 0.999, f"Cosine similarity {cosine_sim} is not close enough to 1.0"


class TestEmbeddingBestPractices:
    """Tests for embedding best practices."""

    def test_token_limit_awareness(self):
        """Test that we handle token limits properly.

        OpenAI embedding models have a limit of 8191 tokens per input.
        We should either:
        1. Truncate long texts
        2. Split into chunks
        3. Raise an error
        """
        # This is a design decision - current implementation doesn't check
        # We should add this protection
        pass

    def test_batch_size_limits(self):
        """Test that batch size respects OpenAI limits.

        Max 2048 inputs per request.
        """
        from bid_scoring.embeddings import MAX_BATCH_SIZE

        assert MAX_BATCH_SIZE <= 2048

    def test_embedding_dimension_consistency(self):
        """Test that embedding dimension is consistent."""
        from bid_scoring.embeddings import DEFAULT_DIM, DEFAULT_MODEL

        # text-embedding-3-small: 1536 dimensions
        # text-embedding-3-large: 3072 dimensions

        if "small" in DEFAULT_MODEL:
            assert DEFAULT_DIM == 1536
        elif "large" in DEFAULT_MODEL:
            assert DEFAULT_DIM == 3072
