# tests/test_scoring_llm.py
"""Tests for scoring module with LLM integration."""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open

from bid_scoring.scoring import ScoringEngine, load_scoring_rules, build_scoring_request
from bid_scoring.llm import ScoreResult, Citation


class TestLoadScoringRules:
    """Test load_scoring_rules function."""

    def test_load_rules_from_yaml(self, tmp_path):
        """Should load scoring rules from YAML file."""
        rules_file = tmp_path / "test_rules.yaml"
        rules_file.write_text("""
dimensions:
  - name: 培训方案
    max_score: 10
    keywords: ["培训"]
    rules:
      - condition: "详细"
        score_range: [9, 10]
""")
        rules = load_scoring_rules(str(rules_file))
        assert "dimensions" in rules
        assert len(rules["dimensions"]) == 1
        assert rules["dimensions"][0]["name"] == "培训方案"

    def test_file_not_found_raises(self):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_scoring_rules("/nonexistent/rules.yaml")


class TestScoringEngineInit:
    """Test ScoringEngine initialization."""

    def test_init_with_llm_client(self):
        """Should initialize with LLM client."""
        mock_client = Mock()
        engine = ScoringEngine(llm_client=mock_client)
        assert engine.llm_client == mock_client


class TestScoringEngineScoreDimension:
    """Test scoring dimension with mocked LLM."""

    def test_score_dimension_returns_result(self):
        """Should return ScoreResult for valid input."""
        mock_llm = Mock()
        mock_response = {
            "dimension": "培训方案",
            "score": 8,
            "max_score": 10,
            "reasoning": "培训方案完整",
            "citations": [
                {"source_number": 1, "cited_text": "培训时间：2天", "supports_claim": "培训时间明确"}
            ],
            "evidence_found": True
        }
        mock_llm.complete_with_schema.return_value = mock_response

        engine = ScoringEngine(llm_client=mock_llm)
        
        evidence = [{"source_number": 1, "text": "培训时间：2天，含安装培训"}]
        result = engine.score_dimension("培训方案", evidence, max_score=10)

        assert isinstance(result, ScoreResult)
        assert result.dimension == "培训方案"
        assert result.score == 8
        assert result.evidence_found is True

    def test_score_dimension_handles_validation_error(self):
        """Should handle validation errors gracefully."""
        mock_llm = Mock()
        mock_llm.complete_with_schema.side_effect = ValueError("Invalid response")

        engine = ScoringEngine(llm_client=mock_llm)
        evidence = [{"source_number": 1, "text": "some text"}]

        result = engine.score_dimension("培训方案", evidence, max_score=10)

        assert result.dimension == "培训方案"
        assert result.score == 0
        assert result.evidence_found is False
        assert "Error" in result.reasoning


class TestScoringEngineBatchScore:
    """Test batch scoring functionality."""

    def test_batch_score_returns_multiple_results(self):
        """Should return multiple ScoreResults."""
        mock_llm = Mock()
        mock_response = [
            {
                "dimension": "培训方案",
                "score": 8,
                "max_score": 10,
                "reasoning": "培训方案完整",
                "citations": [{"source_number": 1, "cited_text": "test", "supports_claim": "yes"}],
                "evidence_found": True
            },
            {
                "dimension": "技术支持",
                "score": 7,
                "max_score": 10,
                "reasoning": "支持良好",
                "citations": [{"source_number": 2, "cited_text": "支持", "supports_claim": "yes"}],
                "evidence_found": True
            }
        ]
        mock_llm.complete_with_schema.return_value = mock_response

        engine = ScoringEngine(llm_client=mock_llm)
        dimensions = [
            ("培训方案", [{"source_number": 1, "text": "test"}]),
            ("技术支持", [{"source_number": 2, "text": "support"}])
        ]
        
        results = engine.batch_score(dimensions)

        assert len(results) == 2
        assert results[0].dimension == "培训方案"
        assert results[1].dimension == "技术支持"


class TestBuildScoringRequest:
    """Test build_scoring_request function."""

    def test_returns_dict_with_required_keys(self):
        """Should return dict with model, input, and response_format."""
        schema = {
            "type": "object",
            "properties": {"dimension": {"type": "string"}},
            "required": ["dimension"]
        }
        with patch("builtins.open", mock_open(read_data=json.dumps(schema))):
            with patch("bid_scoring.scoring.Path") as mock_path:
                mock_path.return_value.read_text.return_value = json.dumps(schema)
                with patch("bid_scoring.scoring.select_llm_model") as mock_select:
                    mock_select.return_value = "gpt-4"
                    
                    result = build_scoring_request(
                        dimension="培训方案",
                        max_score=10,
                        evidence=["培训时间：2天"],
                        rules=["详细说明培训内容"]
                    )
        
        assert "model" in result
        assert "input" in result
        assert "response_format" in result
        assert result["model"] == "gpt-4"
        assert result["response_format"]["type"] == "json_schema"

    def test_includes_dimension_and_max_score_in_prompt(self):
        """Should include dimension and max_score in the prompt."""
        schema = {"type": "object", "properties": {}}
        with patch("bid_scoring.scoring.Path") as mock_path:
            mock_path.return_value.read_text.return_value = json.dumps(schema)
            with patch("bid_scoring.scoring.select_llm_model") as mock_select:
                mock_select.return_value = "gpt-4"
                
                result = build_scoring_request(
                    dimension="技术支持",
                    max_score=20,
                    evidence=["提供24小时支持"]
                )
        
        prompt = result["input"]
        assert "技术支持" in prompt
        assert "20" in prompt

    def test_formats_evidence_with_numbers(self):
        """Should format evidence with numbered references."""
        schema = {"type": "object", "properties": {}}
        with patch("bid_scoring.scoring.Path") as mock_path:
            mock_path.return_value.read_text.return_value = json.dumps(schema)
            with patch("bid_scoring.scoring.select_llm_model") as mock_select:
                mock_select.return_value = "gpt-4"
                
                result = build_scoring_request(
                    dimension="培训方案",
                    max_score=10,
                    evidence=["证据1", "证据2", "证据3"]
                )
        
        prompt = result["input"]
        assert "[1] 证据1" in prompt
        assert "[2] 证据2" in prompt
        assert "[3] 证据3" in prompt

    def test_handles_empty_evidence(self):
        """Should handle empty evidence list."""
        schema = {"type": "object", "properties": {}}
        with patch("bid_scoring.scoring.Path") as mock_path:
            mock_path.return_value.read_text.return_value = json.dumps(schema)
            with patch("bid_scoring.scoring.select_llm_model") as mock_select:
                mock_select.return_value = "gpt-4"
                
                result = build_scoring_request(
                    dimension="培训方案",
                    max_score=10,
                    evidence=[]
                )
        
        prompt = result["input"]
        assert "（无证据）" in prompt

    def test_handles_none_rules(self):
        """Should handle None rules."""
        schema = {"type": "object", "properties": {}}
        with patch("bid_scoring.scoring.Path") as mock_path:
            mock_path.return_value.read_text.return_value = json.dumps(schema)
            with patch("bid_scoring.scoring.select_llm_model") as mock_select:
                mock_select.return_value = "gpt-4"
                
                result = build_scoring_request(
                    dimension="培训方案",
                    max_score=10,
                    evidence=["证据"],
                    rules=None
                )
        
        prompt = result["input"]
        assert "（无明确规则，需谨慎评分）" in prompt

    def test_includes_rules_in_prompt(self):
        """Should include rules in the prompt."""
        schema = {"type": "object", "properties": {}}
        with patch("bid_scoring.scoring.Path") as mock_path:
            mock_path.return_value.read_text.return_value = json.dumps(schema)
            with patch("bid_scoring.scoring.select_llm_model") as mock_select:
                mock_select.return_value = "gpt-4"
                
                result = build_scoring_request(
                    dimension="培训方案",
                    max_score=10,
                    evidence=["证据"],
                    rules=["规则1", "规则2"]
                )
        
        prompt = result["input"]
        assert "规则1" in prompt
        assert "规则2" in prompt

    def test_response_format_has_json_schema(self):
        """Should include proper JSON schema in response_format."""
        schema = {"type": "object", "properties": {"test": {"type": "string"}}}
        with patch("bid_scoring.scoring.Path") as mock_path:
            mock_path.return_value.read_text.return_value = json.dumps(schema)
            with patch("bid_scoring.scoring.select_llm_model") as mock_select:
                mock_select.return_value = "gpt-4"
                
                result = build_scoring_request(
                    dimension="培训方案",
                    max_score=10,
                    evidence=["证据"]
                )
        
        response_format = result["response_format"]
        assert response_format["type"] == "json_schema"
        assert response_format["json_schema"]["name"] == "bid_score"
        assert response_format["json_schema"]["strict"] is True
        assert response_format["json_schema"]["schema"] == schema
