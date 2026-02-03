# bid_scoring/scoring.py
"""Scoring engine for bid evaluation."""

import json
import os
from pathlib import Path
from typing import Any, Optional

import yaml

from bid_scoring.llm import LLMClient, ScoreResult, get_model_for_task, select_llm_model


def load_scoring_rules(rules_path: str) -> dict:
    """Load scoring rules from YAML file.
    
    Args:
        rules_path: Path to the scoring rules YAML file
        
    Returns:
        Dictionary containing scoring rules
        
    Raises:
        FileNotFoundError: If rules file does not exist
    """
    if not os.path.exists(rules_path):
        raise FileNotFoundError(f"Scoring rules file not found: {rules_path}")
    
    with open(rules_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class ScoringEngine:
    """Engine for scoring bid documents across dimensions."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        rules_path: Optional[str] = None
    ):
        """Initialize scoring engine.
        
        Args:
            llm_client: LLM client for scoring (optional)
            rules_path: Path to scoring rules YAML (optional)
        """
        self.llm_client = llm_client
        self.rules_path = rules_path or "references/scoring_rules.yaml"
        self._rules: Optional[dict] = None

    @property
    def rules(self) -> dict:
        """Lazy loading of scoring rules."""
        if self._rules is None:
            self._rules = load_scoring_rules(self.rules_path)
        return self._rules

    def _load_output_schema(self, batch: bool = False) -> dict:
        """Load output JSON schema.
        
        Args:
            batch: Whether to load batch schema
            
        Returns:
            JSON schema dictionary
        """
        schema_filename = "output_schema_batch.json" if batch else "output_schema.json"
        schema_path = os.path.join("references", schema_filename)
        
        # Default schema if file doesn't exist
        default_schema = {
            "type": "object",
            "properties": {
                "dimension": {"type": "string"},
                "score": {"type": "number"},
                "max_score": {"type": "number"},
                "reasoning": {"type": "string"},
                "citations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source_number": {"type": "integer"},
                            "cited_text": {"type": "string"},
                            "supports_claim": {"type": "string"}
                        },
                        "required": ["source_number", "cited_text", "supports_claim"]
                    }
                },
                "evidence_found": {"type": "boolean"}
            },
            "required": ["dimension", "score", "max_score", "reasoning", "citations", "evidence_found"]
        }
        
        if not os.path.exists(schema_path):
            return default_schema
        
        with open(schema_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_scoring_prompt(
        self,
        dimension: str,
        evidence: list[dict],
        max_score: float,
        rules: Optional[dict] = None
    ) -> str:
        """Build the scoring prompt for LLM.
        
        Args:
            dimension: Dimension name to score
            evidence: List of evidence items with source_number and text
            max_score: Maximum possible score
            rules: Optional scoring rules for this dimension
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""请对投标文档的"{dimension}"维度进行评分。

评分标准（满分 {max_score} 分）：
"""
        
        # Add dimension-specific rules if available
        if rules and "dimensions" in rules:
            for dim in rules["dimensions"]:
                if dim["name"] == dimension:
                    prompt += f"\n维度：{dim['name']}（满分 {dim.get('max_score', max_score)} 分）\n"
                    prompt += f"关键词：{', '.join(dim.get('keywords', []))}\n"
                    prompt += "评分规则：\n"
                    for rule in dim.get("rules", []):
                        score_range = rule.get("score_range", [0, max_score])
                        prompt += f"  - {rule['condition']}: {score_range[0]}-{score_range[1]} 分\n"
                    break
        
        prompt += f"\n证据内容：\n"
        for item in evidence:
            source = item.get("source_number", 0)
            text = item.get("text", "")
            prompt += f"\n[{source}] {text}\n"
        
        prompt += """
请根据证据内容，给出：
1. 评分分数（数字）
2. 评分理由（详细说明）
3. 引用证据（指出支持评分的具体原文片段）
4. 是否找到相关证据

输出必须符合JSON格式，包含 dimension、score、max_score、reasoning、citations、evidence_found 字段。
"""
        return prompt

    def score_dimension(
        self,
        dimension: str,
        evidence: list[dict],
        max_score: float = 10.0,
        model: Optional[str] = None
    ) -> ScoreResult:
        """Score a single dimension.
        
        Args:
            dimension: Dimension name to score
            evidence: List of evidence items
            max_score: Maximum possible score
            model: Specific model to use
            
        Returns:
            ScoreResult with scoring details
        """
        if self.llm_client is None:
            # Return empty result if no LLM client
            return ScoreResult(
                dimension=dimension,
                score=0.0,
                max_score=max_score,
                reasoning="No LLM client configured",
                citations=[],
                evidence_found=False
            )
        
        try:
            schema = self._load_output_schema(batch=False)
            prompt = self._build_scoring_prompt(dimension, evidence, max_score, self.rules)
            
            messages = [
                {"role": "system", "content": "你是专业的投标评分专家。所有结论必须引用原文片段。"},
                {"role": "user", "content": prompt}
            ]
            
            model = model or get_model_for_task(self.llm_client.settings, "scoring")
            
            response = self.llm_client.complete_with_schema(
                messages=messages,
                schema=schema,
                model=model,
                temperature=0.0
            )
            
            # Ensure dimension is set correctly
            response["dimension"] = dimension
            response["max_score"] = max_score
            
            return ScoreResult.from_dict(response)
            
        except Exception as e:
            # Return error result on failure
            return ScoreResult(
                dimension=dimension,
                score=0.0,
                max_score=max_score,
                reasoning=f"Error during scoring: {str(e)}",
                citations=[],
                evidence_found=False
            )

    def batch_score(
        self,
        dimensions: list[tuple[str, list[dict]]],
        max_score: float = 10.0,
        model: Optional[str] = None
    ) -> list[ScoreResult]:
        """Score multiple dimensions in batch.
        
        Args:
            dimensions: List of (dimension_name, evidence) tuples
            max_score: Maximum possible score for each dimension
            model: Specific model to use
            
        Returns:
            List of ScoreResults
        """
        if self.llm_client is None:
            return [
                ScoreResult(
                    dimension=dim,
                    score=0.0,
                    max_score=max_score,
                    reasoning="No LLM client configured",
                    citations=[],
                    evidence_found=False
                )
                for dim, _ in dimensions
            ]
        
        try:
            schema = self._load_output_schema(batch=True)
            
            prompts = []
            for dimension, evidence in dimensions:
                prompt = self._build_scoring_prompt(dimension, evidence, max_score, self.rules)
                prompts.append(f"## {dimension}\n{prompt}")
            
            combined_prompt = "\n\n".join(prompts)
            combined_prompt += "\n\n请对每个维度分别评分，返回一个JSON数组，每个元素包含该维度的评分结果。"
            
            messages = [
                {"role": "system", "content": "你是专业的投标评分专家。所有结论必须引用原文片段。"},
                {"role": "user", "content": combined_prompt}
            ]
            
            model = model or get_model_for_task(self.llm_client.settings, "scoring")
            
            response = self.llm_client.complete_with_schema(
                messages=messages,
                schema=schema,
                model=model,
                temperature=0.0
            )
            
            # Handle both single result and array
            if isinstance(response, list):
                results = []
                for i, (dimension, _) in enumerate(dimensions):
                    if i < len(response):
                        item = response[i]
                        item["dimension"] = dimension
                        item["max_score"] = max_score
                        results.append(ScoreResult.from_dict(item))
                    else:
                        results.append(ScoreResult(
                            dimension=dimension,
                            score=0.0,
                            max_score=max_score,
                            reasoning="No response for this dimension",
                            citations=[],
                            evidence_found=False
                        ))
                return results
            else:
                # Single result case - score only first dimension
                response["dimension"] = dimensions[0][0]
                response["max_score"] = max_score
                return [ScoreResult.from_dict(response)]
                
        except Exception as e:
            # Return error results on failure
            return [
                ScoreResult(
                    dimension=dim,
                    score=0.0,
                    max_score=max_score,
                    reasoning=f"Error during batch scoring: {str(e)}",
                    citations=[],
                    evidence_found=False
                )
                for dim, _ in dimensions
            ]


def build_scoring_request(
    dimension: str,
    max_score: int,
    evidence: list[str],
    rules: list[str] | None = None,
) -> dict:
    """Build a scoring request for LLM.
    
    Args:
        dimension: Dimension name to score
        max_score: Maximum possible score
        evidence: List of evidence text strings
        rules: Optional list of scoring rules
        
    Returns:
        Dictionary with model, input, and response_format for LLM call
    """
    schema = json.loads(Path("references/output_schema.json").read_text(encoding="utf-8"))
    evidence_lines = [f"[{i + 1}] {text}" for i, text in enumerate(evidence)]
    evidence_block = "\n".join(evidence_lines) if evidence_lines else "（无证据）"
    rules_block = "\n- ".join(rules) if rules else "（无明确规则，需谨慎评分）"
    prompt = (
        "你是投标评分专家，只能依据给定证据评分。\n"
        "请严格输出符合 JSON Schema 的结果，只输出 JSON。\n"
        "\n"
        f"评分维度：{dimension}\n"
        f"满分：{max_score}\n"
        "评分规则：\n"
        f"- {rules_block}\n"
        "\n"
        "证据（按编号引用）：\n"
        f"{evidence_block}\n"
        "\n"
        "输出要求：\n"
        "- 输出字段必须包含：dimension, score, max_score, reasoning, citations, evidence_found\n"
        "- score 为 0 到 max_score 的数值\n"
        "- evidence_found 无证据则为 false；有有效证据才可为 true\n"
        "- citations 为数组；每条引用必须包含 source_number（证据编号）、cited_text（原文片段）、supports_claim\n"
        "- cited_text 必须是对应证据的子串\n"
        "- 无证据时 citations 为空数组\n"
        "\n"
        "示例输出（仅示例，不代表最终评分）：\n"
        "{\n"
        '  "dimension": "培训方案",\n'
        '  "score": 8,\n'
        '  "max_score": 10,\n'
        '  "reasoning": "证据[1]表明包含培训时长与内容，因此方案较完整。",\n'
        '  "citations": [\n'
        "    {\n"
        '      "source_number": 1,\n'
        '      "cited_text": "培训时间：2天，含安装培训、操作培训",\n'
        '      "supports_claim": "包含培训时长与内容"\n'
        "    }\n"
        "  ],\n"
        '  "evidence_found": true\n'
        "}\n"
    )
    return {
        "model": select_llm_model("scoring"),
        "input": prompt,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "bid_score",
                "schema": schema,
                "strict": True,
            },
        },
    }
