from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_POLICY_PATH_ENV = "BID_SCORING_AGENT_MCP_POLICY_PATH"


@dataclass(frozen=True)
class AgentScoringPolicy:
    constraints: list[str]
    risk_rules: dict[str, str]
    output_schema_hint: str
    tool_calling_required: bool

    def tool_calling_system_prompt(self) -> str:
        constraints = "；".join(self.constraints)
        risk = "；".join(
            [
                f"high={self.risk_rules.get('high', '')}",
                f"medium={self.risk_rules.get('medium', '')}",
                f"low={self.risk_rules.get('low', '')}",
            ]
        )
        workflow = (
            "必须先调用工具 retrieve_dimension_evidence 获取证据，再输出最终评分 JSON。"
            if self.tool_calling_required
            else "可直接输出评分 JSON。"
        )
        return (
            "你是评标专家。"
            f"{workflow}"
            f"规则：{constraints}。"
            "若证据不足，请明确说明“证据不足”。"
            f"风险判定：{risk}。"
            f"输出严格 JSON：{self.output_schema_hint}。"
            "dimensions 是对象，key 为维度名，value 含 score/risk_level/summary。"
        )

    def bulk_system_prompt(self) -> str:
        constraints = "；".join(self.constraints)
        risk = "；".join(
            [
                f"high={self.risk_rules.get('high', '')}",
                f"medium={self.risk_rules.get('medium', '')}",
                f"low={self.risk_rules.get('low', '')}",
            ]
        )
        return (
            "你是评标专家。"
            "你将收到按维度聚合后的证据，请仅基于这些证据评分。"
            f"规则：{constraints}。"
            "若证据不足，请明确说明“证据不足”。"
            f"风险判定：{risk}。"
            f"输出严格 JSON：{self.output_schema_hint}。"
            "dimensions 是对象，key 为维度名，value 含 score/risk_level/summary。"
        )


def load_agent_scoring_policy(path: str | Path | None = None) -> AgentScoringPolicy:
    policy_path = _resolve_policy_path(path)
    if not policy_path.exists():
        return _default_policy()
    raw = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return _default_policy()

    constraints = _normalize_str_list(raw.get("constraints"))
    risk_rules_raw = raw.get("risk_rules")
    risk_rules: dict[str, str] = {}
    if isinstance(risk_rules_raw, dict):
        for key in ("high", "medium", "low"):
            value = risk_rules_raw.get(key)
            if isinstance(value, str) and value.strip():
                risk_rules[key] = value.strip()

    output_raw = raw.get("output")
    output_schema_hint = "{overall_score,risk_level,total_risks,total_benefits,recommendations,dimensions}"
    if isinstance(output_raw, dict):
        candidate = output_raw.get("schema_hint")
        if isinstance(candidate, str) and candidate.strip():
            output_schema_hint = candidate.strip()

    workflow_raw = raw.get("workflow")
    tool_calling_required = True
    if isinstance(workflow_raw, dict):
        flag = workflow_raw.get("tool_calling_required")
        if isinstance(flag, bool):
            tool_calling_required = flag

    return AgentScoringPolicy(
        constraints=constraints or _default_policy().constraints,
        risk_rules=risk_rules or _default_policy().risk_rules,
        output_schema_hint=output_schema_hint,
        tool_calling_required=tool_calling_required,
    )


def _resolve_policy_path(path: str | Path | None) -> Path:
    if path is not None:
        return Path(path)
    env_value = os.getenv(_POLICY_PATH_ENV)
    if env_value:
        return Path(env_value)
    return Path(__file__).resolve().parents[3] / "config" / "agent_scoring_policy.yaml"


def _normalize_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    output: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        normalized = item.strip()
        if not normalized:
            continue
        output.append(normalized)
    return output


def _default_policy() -> AgentScoringPolicy:
    return AgentScoringPolicy(
        constraints=[
            "必须仅基于给定证据评分",
            "禁止使用外部知识和杜撰",
            "如证据不足，明确说明“证据不足”",
            "risk_level 仅允许 low/medium/high",
        ],
        risk_rules={
            "high": "存在重大合规/履约风险，或高风险证据明显多于优势",
            "medium": "风险与优势并存，关键条款需澄清",
            "low": "证据完整，主要条款清晰且风险可控",
        },
        output_schema_hint=(
            "{overall_score,risk_level,total_risks,total_benefits,recommendations,dimensions}"
        ),
        tool_calling_required=True,
    )
