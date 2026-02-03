"""Tests for Late Chunking Encoder

Test coverage:
- Late chunking encoding with different scenarios
- Pooling strategies (mean, max, sum, cls)
- Fallback behavior when late chunking fails
- Edge cases (empty input, invalid boundaries, etc.)
- Token boundary estimation
"""

import pytest
from unittest.mock import MagicMock, patch

from bid_scoring.late_chunking import (
    LateChunkingEncoder,
    LateChunkingResult,
    _CharacterTokenizer,
    estimate_token_boundaries,
    create_late_chunking_encoder,
    DEFAULT_LATE_CHUNKING_MODEL,
    DEFAULT_MAX_LENGTH,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_model():
    """Create a mock embedding model that returns fixed embeddings"""
    model = MagicMock()
    # Return predictable embeddings: embedding[i] = [i * 0.1] * dim
    def mock_encode(text, **kwargs):
        if isinstance(text, list):
            return [[0.1 * i] * 768 for i in range(len(text))]
        return [0.0] * 768
    model.encode = mock_encode
    return model


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer"""
    tokenizer = MagicMock()
    # Simple tokenizer: each char is a token
    tokenizer.encode = lambda text: list(range(len(text)))
    tokenizer.decode = lambda ids: "".join(chr(ord('a') + (i % 26)) for i in ids)
    tokenizer.convert_ids_to_tokens = lambda ids: [f"t{i}" for i in ids]
    tokenizer.__len__ = lambda self: 1000
    return tokenizer


@pytest.fixture
def simple_encoder(mock_model, mock_tokenizer):
    """Create a simple encoder with mock dependencies"""
    return LateChunkingEncoder(
        model=mock_model,
        tokenizer=mock_tokenizer,
        max_length=100,
        pooling_strategy="mean",
        fallback_to_standard=True,
    )


# ============================================================================
# Test LateChunkingResult
# ============================================================================

class TestLateChunkingResult:
    """Test LateChunkingResult dataclass"""
    
    def test_result_creation(self):
        """Test creating a LateChunkingResult"""
        embeddings = [[0.1, 0.2], [0.3, 0.4]]
        result = LateChunkingResult(
            chunk_embeddings=embeddings,
            full_embedding=[0.2, 0.3],
            token_boundaries=[(0, 2), (2, 4)],
            pooling_strategy="mean",
        )
        
        assert result.chunk_embeddings == embeddings
        assert result.full_embedding == [0.2, 0.3]
        assert result.token_boundaries == [(0, 2), (2, 4)]
        assert result.pooling_strategy == "mean"
    
    def test_result_defaults(self):
        """Test LateChunkingResult default values"""
        result = LateChunkingResult(chunk_embeddings=[[0.1]])
        
        assert result.full_embedding is None
        assert result.token_embeddings is None
        assert result.token_boundaries == []
        assert result.pooling_strategy == "mean"


# ============================================================================
# Test LateChunkingEncoder Initialization
# ============================================================================

class TestLateChunkingEncoderInit:
    """Test LateChunkingEncoder initialization"""
    
    def test_default_initialization(self):
        """Test default initialization"""
        encoder = LateChunkingEncoder()
        
        assert encoder.model == DEFAULT_LATE_CHUNKING_MODEL
        assert encoder.tokenizer is None
        assert encoder.max_length == DEFAULT_MAX_LENGTH
        assert encoder.pooling_strategy == "mean"
        assert encoder.fallback_to_standard is True
    
    def test_custom_initialization(self):
        """Test custom initialization"""
        encoder = LateChunkingEncoder(
            model="custom-model",
            max_length=512,
            pooling_strategy="max",
            fallback_to_standard=False,
        )
        
        assert encoder.model == "custom-model"
        assert encoder.max_length == 512
        assert encoder.pooling_strategy == "max"
        assert encoder.fallback_to_standard is False
    
    def test_invalid_pooling_strategy(self):
        """Test invalid pooling strategy raises error"""
        with pytest.raises(ValueError, match="Invalid pooling_strategy"):
            LateChunkingEncoder(pooling_strategy="invalid")
    
    def test_valid_pooling_strategies(self):
        """Test all valid pooling strategies are accepted"""
        for strategy in ["mean", "max", "sum", "cls"]:
            encoder = LateChunkingEncoder(pooling_strategy=strategy)
            assert encoder.pooling_strategy == strategy


# ============================================================================
# Test Pooling Strategies
# ============================================================================

class TestPoolingStrategies:
    """Test different pooling strategies"""
    
    def test_mean_pooling(self, simple_encoder):
        """Test mean pooling via encoder method"""
        embeddings = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        result = simple_encoder._mean_pooling(embeddings)
        
        assert result == [3.0, 4.0]  # (1+3+5)/3, (2+4+6)/3
    
    def test_mean_pooling_empty(self, simple_encoder):
        """Test mean pooling with empty list"""
        result = simple_encoder._mean_pooling([])
        assert result == []
    
    def test_max_pooling(self, simple_encoder):
        """Test max pooling"""
        embeddings = [[1.0, 2.0], [3.0, 1.0], [2.0, 4.0]]
        result = simple_encoder._max_pooling(embeddings)
        
        assert result == [3.0, 4.0]
    
    def test_max_pooling_empty(self, simple_encoder):
        """Test max pooling with empty list"""
        result = simple_encoder._max_pooling([])
        assert result == []
    
    def test_sum_pooling(self, simple_encoder):
        """Test sum pooling"""
        embeddings = [[1.0, 2.0], [3.0, 4.0]]
        result = simple_encoder._sum_pooling(embeddings)
        
        assert result == [4.0, 6.0]
    
    def test_sum_pooling_empty(self, simple_encoder):
        """Test sum pooling with empty list"""
        result = simple_encoder._sum_pooling([])
        assert result == []
    
    def test_pooling_integration(self, simple_encoder):
        """Test pooling through encoder"""
        # Create encoder with mean pooling
        encoder = LateChunkingEncoder(
            model=simple_encoder._model_instance,
            tokenizer=simple_encoder._tokenizer_instance,
            pooling_strategy="mean",
        )
        
        # Mock token embeddings
        token_embeddings = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        
        result = encoder._pool_embeddings(token_embeddings, 0, 2)
        assert result == [2.0, 3.0]  # mean of first two


# ============================================================================
# Test Token Boundary Estimation
# ============================================================================

class TestEstimateTokenBoundaries:
    """Test token boundary estimation"""
    
    def test_basic_estimation(self):
        """Test basic boundary estimation"""
        full_text = "这是第一段。这是第二段。"
        chunks = ["这是第一段。", "这是第二段。"]
        
        boundaries = estimate_token_boundaries(full_text, chunks, chars_per_token=2)
        
        # First chunk: 6 chars / 2 = 3 tokens -> +1 for ceiling = 4
        # Second chunk: 6 chars / 2 = 3 tokens -> +1 for ceiling = 4
        assert boundaries == [(0, 4), (4, 8)]  # +1 for ceiling
    
    def test_empty_chunks(self):
        """Test with empty chunks list"""
        boundaries = estimate_token_boundaries("text", [])
        assert boundaries == []
    
    def test_single_chunk(self):
        """Test with single chunk"""
        boundaries = estimate_token_boundaries("hello world", ["hello world"], chars_per_token=2)
        assert boundaries == [(0, 6)]  # 11 chars / 2 + 1 = 6


# ============================================================================
# Test Character Tokenizer
# ============================================================================

class TestCharacterTokenizer:
    """Test the fallback character tokenizer"""
    
    def test_encode(self):
        """Test encoding"""
        tokenizer = _CharacterTokenizer()
        tokens = tokenizer.encode("hello")
        
        # Every 2 chars is a token
        assert tokens == [0, 2, 4]
    
    def test_decode(self):
        """Test decode returns empty string"""
        tokenizer = _CharacterTokenizer()
        result = tokenizer.decode([0, 1, 2])
        
        assert result == ""
    
    def test_len(self):
        """Test tokenizer length"""
        tokenizer = _CharacterTokenizer()
        assert len(tokenizer) == 65536


# ============================================================================
# Test Encode with Late Chunking
# ============================================================================

class TestEncodeWithLateChunking:
    """Test the main encode_with_late_chunking method"""
    
    def test_empty_text_raises_error(self):
        """Test empty text raises ValueError"""
        encoder = LateChunkingEncoder()
        
        with pytest.raises(ValueError, match="full_text cannot be empty"):
            encoder.encode_with_late_chunking("", [(0, 1)])
        
        with pytest.raises(ValueError, match="full_text cannot be empty"):
            encoder.encode_with_late_chunking("   ", [(0, 1)])
    
    def test_empty_boundaries_raises_error(self):
        """Test empty boundaries raises ValueError"""
        encoder = LateChunkingEncoder()
        
        with pytest.raises(ValueError, match="chunk_boundaries cannot be empty"):
            encoder.encode_with_late_chunking("some text", [])
    
    def test_invalid_boundaries_negative(self):
        """Test negative boundary indices raise error"""
        encoder = LateChunkingEncoder()
        
        with pytest.raises(ValueError, match="Invalid boundary"):
            encoder.encode_with_late_chunking("text", [(-1, 2)])
    
    def test_invalid_boundaries_start_gte_end(self):
        """Test start >= end raises error"""
        encoder = LateChunkingEncoder()
        
        with pytest.raises(ValueError, match="start must be less than end"):
            encoder.encode_with_late_chunking("text", [(2, 2)])
        
        with pytest.raises(ValueError, match="start must be less than end"):
            encoder.encode_with_late_chunking("text", [(3, 2)])
    
    def test_valid_input_returns_result(self, simple_encoder):
        """Test valid input returns LateChunkingResult"""
        # Mock the embed_texts function to avoid API calls
        with patch('bid_scoring.late_chunking.embed_texts') as mock_embed:
            mock_embed.return_value = [[0.1] * 768, [0.2] * 768]
            
            result = simple_encoder.encode_with_late_chunking(
                "hello world",
                [(0, 3), (3, 6)]
            )
            
            assert isinstance(result, LateChunkingResult)
            assert len(result.chunk_embeddings) == 2
    
    def test_boundary_truncation(self, simple_encoder):
        """Test boundaries are truncated to token count"""
        # Create mock token embeddings
        simple_encoder._tokenizer_instance = _CharacterTokenizer()
        
        # Mock _get_token_embeddings to return controlled embeddings
        with patch.object(simple_encoder, '_get_token_embeddings') as mock_get:
            mock_get.return_value = [[0.1] * 768] * 5  # 5 tokens
            
            result = simple_encoder.encode_with_late_chunking(
                "hello world test",
                [(0, 3), (3, 10)]  # Second boundary exceeds token count
            )
            
            # Second chunk should be truncated to (3, 5)
            assert len(result.chunk_embeddings) == 2
            assert result.token_boundaries[1] == (3, 5)


# ============================================================================
# Test Fallback Behavior
# ============================================================================

class TestFallbackBehavior:
    """Test fallback to standard embedding"""
    
    def test_fallback_when_late_chunking_fails(self):
        """Test fallback when late chunking fails"""
        encoder = LateChunkingEncoder(fallback_to_standard=True)
        
        # Mock _get_token_embeddings to raise an error (after tokenizer is set)
        with patch.object(encoder, '_get_token_embeddings', side_effect=Exception("Model failed")):
            with patch.object(encoder, '_get_tokenizer') as mock_get_tokenizer:
                # Set up a simple mock tokenizer
                mock_tokenizer = MagicMock()
                mock_tokenizer.encode = lambda text: list(range(0, len(text), 2))
                mock_tokenizer.decode = lambda ids: "mock"
                mock_get_tokenizer.return_value = mock_tokenizer
                
                with patch('bid_scoring.late_chunking.embed_texts') as mock_embed:
                    mock_embed.return_value = [[0.1, 0.2], [0.3, 0.4]]
                    
                    result = encoder.encode_with_late_chunking(
                        "hello world",
                        [(0, 3), (3, 6)]
                    )
                    
                    assert isinstance(result, LateChunkingResult)
                    assert result.pooling_strategy == "fallback_standard"
                    mock_embed.assert_called()
    
    def test_no_fallback_raises_error(self):
        """Test no fallback raises RuntimeError"""
        encoder = LateChunkingEncoder(fallback_to_standard=False)
        
        with patch.object(encoder, '_tokenize', side_effect=Exception("Failed")):
            with pytest.raises(RuntimeError, match="Late chunking failed"):
                encoder.encode_with_late_chunking("text", [(0, 1)])
    
    def test_fallback_encode(self, simple_encoder):
        """Test _fallback_encode method"""
        with patch('bid_scoring.late_chunking.embed_texts') as mock_embed:
            mock_embed.return_value = [[0.1, 0.2], [0.3, 0.4]]
            
            result = simple_encoder._fallback_encode(
                "hello world",
                [(0, 3), (3, 6)]
            )
            
            assert isinstance(result, LateChunkingResult)
            assert len(result.chunk_embeddings) == 2
            assert result.pooling_strategy == "fallback_standard"


# ============================================================================
# Test Encode Text to Boundaries
# ============================================================================

class TestEncodeTextToBoundaries:
    """Test encode_text_to_boundaries convenience method"""
    
    def test_convenience_method(self, simple_encoder):
        """Test convenience method infers boundaries"""
        simple_encoder._tokenizer_instance = _CharacterTokenizer()
        
        with patch.object(simple_encoder, '_get_token_embeddings') as mock_get:
            mock_get.return_value = [[0.1] * 768] * 20
            
            result = simple_encoder.encode_text_to_boundaries(
                "hello world foo bar",
                ["hello", "world", "foo bar"]
            )
            
            assert isinstance(result, LateChunkingResult)
            assert len(result.chunk_embeddings) == 3


# ============================================================================
# Test Create Encoder Factory
# ============================================================================

class TestCreateLateChunkingEncoder:
    """Test create_late_chunking_encoder factory function"""
    
    def test_default_factory(self):
        """Test factory with default parameters"""
        encoder = create_late_chunking_encoder()
        
        assert encoder.model == DEFAULT_LATE_CHUNKING_MODEL
        assert encoder.pooling_strategy == "mean"
        assert encoder.fallback_to_standard is True
    
    def test_custom_factory(self):
        """Test factory with custom parameters"""
        encoder = create_late_chunking_encoder(
            model_name="custom-model",
            pooling="max",
            fallback=False,
        )
        
        assert encoder.model == "custom-model"
        assert encoder.pooling_strategy == "max"
        assert encoder.fallback_to_standard is False


# ============================================================================
# Test Integration Scenarios
# ============================================================================

class TestIntegrationScenarios:
    """Test real-world integration scenarios"""
    
    def test_multiple_chunks(self, simple_encoder):
        """Test encoding multiple chunks"""
        simple_encoder._tokenizer_instance = _CharacterTokenizer()
        
        with patch.object(simple_encoder, '_get_token_embeddings') as mock_get:
            # Create predictable embeddings: each token has embedding [token_idx * 0.1]
            token_embeddings = [[i * 0.1] * 768 for i in range(10)]
            mock_get.return_value = token_embeddings
            
            result = simple_encoder.encode_with_late_chunking(
                "This is a longer text with multiple segments",
                [(0, 3), (3, 6), (6, 9)]
            )
            
            assert len(result.chunk_embeddings) == 3
            # Each chunk should have pooled embeddings
            assert len(result.chunk_embeddings[0]) == 768
    
    def test_overlapping_boundaries(self, simple_encoder):
        """Test with overlapping chunk boundaries"""
        simple_encoder._tokenizer_instance = _CharacterTokenizer()
        
        with patch.object(simple_encoder, '_get_token_embeddings') as mock_get:
            mock_get.return_value = [[0.1 * i] * 768 for i in range(10)]
            
            # Overlapping: (0,4), (2,6), (4,8)
            result = simple_encoder.encode_with_late_chunking(
                "Overlapping chunks test",
                [(0, 4), (2, 6), (4, 8)]
            )
            
            assert len(result.chunk_embeddings) == 3
    
    def test_single_chunk_boundary(self, simple_encoder):
        """Test with single chunk boundary"""
        simple_encoder._tokenizer_instance = _CharacterTokenizer()
        
        with patch.object(simple_encoder, '_get_token_embeddings') as mock_get:
            mock_get.return_value = [[0.1] * 768] * 5
            
            result = simple_encoder.encode_with_late_chunking(
                "Single chunk",
                [(0, 5)]
            )
            
            assert len(result.chunk_embeddings) == 1
            assert result.full_embedding is not None


# ============================================================================
# Test Model Loading
# ============================================================================

class TestModelLoading:
    """Test model loading functionality"""
    
    def test_is_available_with_mock(self, simple_encoder):
        """Test is_available returns True with mock model"""
        assert simple_encoder.is_available() is True
    
    def test_is_available_without_dependencies(self):
        """Test is_available returns False when dependencies missing"""
        encoder = LateChunkingEncoder(model="nonexistent-model-12345")
        
        # Should return False because model cannot be loaded
        assert encoder.is_available() is False


# ============================================================================
# Test Error Handling
# ============================================================================

class TestErrorHandling:
    """Test error handling in various scenarios"""
    
    def test_pooling_with_empty_embeddings(self, simple_encoder):
        """Test pooling with empty embeddings returns zero vector"""
        result = simple_encoder._pool_embeddings([], 0, 0)
        
        # Should return zero vector with default dimension
        assert result == [0.0] * 768
    
    def test_pooling_with_negative_indices(self, simple_encoder):
        """Test pooling with negative start index"""
        embeddings = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        
        result = simple_encoder._pool_embeddings(embeddings, -1, 2)
        # Should clamp to 0
        assert result == [2.0, 3.0]  # mean of [1,2] and [3,4]
    
    def test_boundary_exceeds_token_count_warning(self, simple_encoder, caplog):
        """Test warning when boundary exceeds token count"""
        simple_encoder._tokenizer_instance = _CharacterTokenizer()
        
        with patch.object(simple_encoder, '_get_token_embeddings') as mock_get:
            mock_get.return_value = [[0.1] * 768] * 3  # Only 3 tokens
            
            with caplog.at_level("WARNING"):
                result = simple_encoder.encode_with_late_chunking(
                    "short",
                    [(0, 2), (5, 10)]  # Second boundary exceeds 3 tokens
                )
                
                # Should warn about exceeding token count
                assert "exceeds token count" in caplog.text
                # Should only return 1 chunk (the second was skipped)
                assert len(result.chunk_embeddings) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
