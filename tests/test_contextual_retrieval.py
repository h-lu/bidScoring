"""Tests for contextual retrieval generator."""

import pytest
from unittest.mock import Mock, MagicMock

from bid_scoring.contextual_retrieval import (
    ContextualRetrievalGenerator,
    _build_prompt,
    _generate_rule_based_context,
)


class TestBuildPrompt:
    """Test prompt building functionality."""

    def test_build_prompt_basic(self):
        """Should build prompt with basic information."""
        prompt = _build_prompt(
            chunk_text="Test chunk content",
            document_title="Test Document",
        )
        assert "Test Document" in prompt
        assert "Test chunk content" in prompt
        assert "Document Title:" in prompt

    def test_build_prompt_with_section(self):
        """Should include section title when provided."""
        prompt = _build_prompt(
            chunk_text="Test chunk",
            document_title="Test Doc",
            section_title="Section 1",
        )
        assert "Section 1" in prompt
        assert "Section Title:" in prompt

    def test_build_prompt_with_surrounding_chunks(self):
        """Should include surrounding context when provided."""
        prompt = _build_prompt(
            chunk_text="Test chunk",
            document_title="Test Doc",
            surrounding_chunks=["Previous chunk", "Next chunk"],
        )
        assert "Surrounding Context:" in prompt
        assert "Context chunk 1:" in prompt
        assert "Context chunk 2:" in prompt

    def test_build_prompt_surrounding_chunks_truncated(self):
        """Should truncate surrounding chunks to reasonable length."""
        long_chunk = "x" * 500
        prompt = _build_prompt(
            chunk_text="Test",
            document_title="Doc",
            surrounding_chunks=[long_chunk],
        )
        # Should truncate to 200 chars + "..."
        assert "..." in prompt

    def test_build_prompt_surrounding_chunks_limit(self):
        """Should limit to max 2 surrounding chunks."""
        prompt = _build_prompt(
            chunk_text="Test",
            document_title="Doc",
            surrounding_chunks=["Chunk 1", "Chunk 2", "Chunk 3", "Chunk 4"],
        )
        assert "Context chunk 1:" in prompt
        assert "Context chunk 2:" in prompt
        # Should not have chunk 3
        assert "Context chunk 3:" not in prompt


class TestGenerateRuleBasedContext:
    """Test rule-based context generation."""

    def test_with_section_title(self):
        """Should include section title when provided."""
        context = _generate_rule_based_context(
            document_title="Test Doc",
            section_title="Section 1",
        )
        assert "Test Doc" in context
        assert "Section 1" in context

    def test_without_section_title(self):
        """Should work without section title."""
        context = _generate_rule_based_context(
            document_title="Test Doc",
        )
        assert "Test Doc" in context
        assert "section" not in context.lower()


class TestContextualRetrievalGeneratorInit:
    """Test generator initialization."""

    def test_init_with_defaults(self):
        """Should initialize with default values."""
        mock_client = Mock()
        generator = ContextualRetrievalGenerator(mock_client)

        assert generator.client == mock_client
        assert generator.model == "gpt-4"
        assert generator.temperature == 0.0
        assert generator.max_tokens == 200

    def test_init_with_custom_values(self):
        """Should accept custom configuration."""
        mock_client = Mock()
        generator = ContextualRetrievalGenerator(
            client=mock_client,
            model="gpt-3.5-turbo",
            temperature=0.5,
            max_tokens=100,
        )

        assert generator.model == "gpt-3.5-turbo"
        assert generator.temperature == 0.5
        assert generator.max_tokens == 100


class TestGenerateContext:
    """Test context generation with mocked LLM."""

    def test_successful_generation(self):
        """Should return LLM-generated context on success."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content="  Generated context about chunk  "))
        ]
        mock_client.chat.completions.create.return_value = mock_response

        generator = ContextualRetrievalGenerator(mock_client)
        context = generator.generate_context(
            chunk_text="Test chunk content",
            document_title="Test Document",
            section_title="Section 1",
        )

        assert "Generated context" in context
        mock_client.chat.completions.create.assert_called_once()

        # Verify the call arguments
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "gpt-4"
        assert call_args.kwargs["temperature"] == 0.0
        assert call_args.kwargs["max_tokens"] == 200

    def test_generation_with_all_parameters(self):
        """Should pass all parameters to LLM correctly."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Context"))]
        mock_client.chat.completions.create.return_value = mock_response

        generator = ContextualRetrievalGenerator(mock_client)
        context = generator.generate_context(
            chunk_text="Chunk text",
            document_title="Doc Title",
            section_title="Section Title",
            surrounding_chunks=["Prev chunk", "Next chunk"],
        )

        # Verify the prompt includes all context
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        user_message = messages[1]["content"]

        assert "Doc Title" in user_message
        assert "Section Title" in user_message
        assert "Surrounding Context:" in user_message
        assert "Chunk text" in user_message

    def test_fallback_on_empty_response(self):
        """Should fallback to rule-based when LLM returns empty."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="   "))]
        mock_client.chat.completions.create.return_value = mock_response

        generator = ContextualRetrievalGenerator(mock_client)
        context = generator.generate_context(
            chunk_text="Test chunk",
            document_title="Test Doc",
            section_title="Section 1",
        )

        # Should fallback to rule-based
        assert "Test Doc" in context
        assert "Section 1" in context

    def test_fallback_on_short_response(self):
        """Should fallback to rule-based when LLM returns too short."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Hi"))]
        mock_client.chat.completions.create.return_value = mock_response

        generator = ContextualRetrievalGenerator(mock_client)
        context = generator.generate_context(
            chunk_text="Test chunk",
            document_title="Test Doc",
        )

        # Should fallback to rule-based (short responses are invalid)
        assert "Test Doc" in context

    def test_fallback_on_exception(self):
        """Should fallback to rule-based when LLM raises exception."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        generator = ContextualRetrievalGenerator(mock_client)
        context = generator.generate_context(
            chunk_text="Test chunk",
            document_title="Test Doc",
            section_title="Section 1",
        )

        # Should fallback to rule-based
        assert "Test Doc" in context
        assert "Section 1" in context

    def test_fallback_without_section(self):
        """Should fallback correctly without section title."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        generator = ContextualRetrievalGenerator(mock_client)
        context = generator.generate_context(
            chunk_text="Test chunk",
            document_title="Test Doc",
        )

        assert "Test Doc" in context
        assert "section" not in context.lower()


class TestGenerateContextBatch:
    """Test batch context generation."""

    def test_batch_generation(self):
        """Should generate contexts for multiple chunks."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Generated context"))]
        mock_client.chat.completions.create.return_value = mock_response

        generator = ContextualRetrievalGenerator(mock_client)
        chunks = [
            {
                "chunk_text": "Chunk 1",
                "document_title": "Doc 1",
                "section_title": "Section 1",
            },
            {
                "chunk_text": "Chunk 2",
                "document_title": "Doc 2",
            },
        ]

        contexts = generator.generate_context_batch(chunks)

        assert len(contexts) == 2
        assert all("Generated context" in ctx for ctx in contexts)

    def test_batch_with_mixed_results(self):
        """Should handle mix of successful and failed generations."""
        mock_client = Mock()

        def side_effect(*args, **kwargs):
            mock_response = Mock()
            # First call succeeds, second fails
            if mock_client.chat.completions.create.call_count == 1:
                mock_response.choices = [
                    Mock(message=Mock(content="  Generated context  "))
                ]
            else:
                raise Exception("API Error")
            return mock_response

        mock_client.chat.completions.create.side_effect = side_effect

        generator = ContextualRetrievalGenerator(mock_client)
        chunks = [
            {"chunk_text": "Chunk 1", "document_title": "Doc 1"},
            {"chunk_text": "Chunk 2", "document_title": "Doc 2"},
        ]

        contexts = generator.generate_context_batch(chunks)

        assert len(contexts) == 2
        assert "Generated context" in contexts[0]
        assert "Doc 2" in contexts[1]  # Fallback for second chunk

    def test_empty_batch(self):
        """Should handle empty batch."""
        mock_client = Mock()
        generator = ContextualRetrievalGenerator(mock_client)

        contexts = generator.generate_context_batch([])

        assert contexts == []


class TestIntegrationStyle:
    """Integration-style tests matching the spec requirements."""

    def test_contextual_generation(self):
        """Test matching the spec example - verifies successful LLM generation."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                message=Mock(
                    content="该内容讨论共聚焦显微镜的技术规格。This content discusses technical specifications for confocal microscopy."
                )
            )
        ]
        mock_client.chat.completions.create.return_value = mock_response

        generator = ContextualRetrievalGenerator(mock_client)
        context = generator.generate_context(
            chunk_text="细胞和组织本身会发出荧光",
            document_title="共聚焦显微镜投标文件",
            section_title="技术规格",
        )

        # Verify LLM-generated context is returned (contains content from mock)
        assert "技术规格" in context or "共聚焦" in context

    def test_prompt_contains_all_info(self):
        """Verify that the prompt contains all necessary information."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Context"))]
        mock_client.chat.completions.create.return_value = mock_response

        generator = ContextualRetrievalGenerator(mock_client)
        generator.generate_context(
            chunk_text="细胞和组织本身会发出荧光",
            document_title="共聚焦显微镜投标文件",
            section_title="技术规格",
        )

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        user_content = messages[1]["content"]

        assert "共聚焦显微镜投标文件" in user_content
        assert "技术规格" in user_content
        assert "细胞和组织本身会发出荧光" in user_content
