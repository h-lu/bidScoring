# tests/test_llm_models.py
"""Tests for LLM client and models."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from bid_scoring.llm import LLMClient, get_model_for_task, ScoreResult, Citation


class TestGetModelForTask:
    """Test get_model_for_task function."""

    def test_returns_default_model_when_no_task_models(self):
        """Should return default model when no task-specific models configured."""
        settings = {
            "OPENAI_LLM_MODEL_DEFAULT": "gpt-4",
            "OPENAI_LLM_MODELS": {}
        }
        model = get_model_for_task(settings, "scoring")
        assert model == "gpt-4"

    def test_returns_task_specific_model_when_configured(self):
        """Should return task-specific model when configured."""
        settings = {
            "OPENAI_LLM_MODEL_DEFAULT": "gpt-4",
            "OPENAI_LLM_MODELS": {"scoring": "gpt-4-turbo"}
        }
        model = get_model_for_task(settings, "scoring")
        assert model == "gpt-4-turbo"

    def test_returns_default_when_task_not_found(self):
        """Should return default when specific task not in config."""
        settings = {
            "OPENAI_LLM_MODEL_DEFAULT": "gpt-4",
            "OPENAI_LLM_MODELS": {"other": "gpt-3.5"}
        }
        model = get_model_for_task(settings, "scoring")
        assert model == "gpt-4"


class TestLLMClientInit:
    """Test LLMClient initialization."""

    def test_init_with_api_key(self):
        """Should initialize with provided settings."""
        settings = {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_BASE_URL": "https://api.example.com",
            "OPENAI_TIMEOUT": 30.0,
            "OPENAI_MAX_RETRIES": 3,
            "OPENAI_LLM_MODEL_DEFAULT": "gpt-4",
            "OPENAI_LLM_MODELS": {}
        }
        client = LLMClient(settings)
        assert client.settings == settings

    def test_init_without_api_key_raises(self):
        """Should raise ValueError when API key not provided."""
        settings = {
            "OPENAI_API_KEY": None,
            "OPENAI_LLM_MODEL_DEFAULT": "gpt-4",
            "OPENAI_LLM_MODELS": {}
        }
        with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
            LLMClient(settings)


class TestScoreResult:
    """Test ScoreResult dataclass."""

    def test_score_result_creation(self):
        """Should create ScoreResult with all fields."""
        citation = Citation(source_number=1, cited_text="test", supports_claim="yes")
        result = ScoreResult(
            dimension="培训方案",
            score=8,
            max_score=10,
            reasoning="Good training plan",
            citations=[citation],
            evidence_found=True
        )
        assert result.dimension == "培训方案"
        assert result.score == 8
        assert result.max_score == 10
        assert result.evidence_found is True

    def test_score_result_from_dict(self):
        """Should create ScoreResult from dictionary."""
        data = {
            "dimension": "培训方案",
            "score": 8,
            "max_score": 10,
            "reasoning": "Good",
            "citations": [{"source_number": 1, "cited_text": "test", "supports_claim": "yes"}],
            "evidence_found": True
        }
        result = ScoreResult.from_dict(data)
        assert result.dimension == "培训方案"
        assert result.score == 8
        assert len(result.citations) == 1
