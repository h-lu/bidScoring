from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bid_scoring.policy import (
    PolicyLoadError,
    load_policy_bundle_from_artifact,
    load_policy_bundle_from_env,
)


@dataclass(frozen=True)
class AgentScoringPolicy:
    constraints: list[str]
    risk_rules: dict[str, str]
    output_schema_hint: str
    tool_calling_required: bool
    required_tools: list[str]
    max_turns_default: int
    retrieval_default_mode: str
    retrieval_default_top_k: int
    retrieval_dimension_overrides: dict[str, dict[str, Any]]
    retrieval_evaluation_thresholds: dict[str, dict[str, float]]
    evidence_default_min_citations: int
    evidence_require_page_idx: bool
    evidence_require_bbox: bool
    evidence_require_quote: bool

    def resolve_dimension_defaults(
        self,
        dimension: str,
        *,
        fallback_mode: str,
        fallback_top_k: int,
    ) -> tuple[str, int]:
        override = self.retrieval_dimension_overrides.get(dimension, {})
        mode = override.get("mode") if isinstance(override, dict) else None
        top_k = override.get("top_k") if isinstance(override, dict) else None
        resolved_mode = (
            mode if mode in {"hybrid", "keyword", "vector"} else fallback_mode
        )
        resolved_top_k = (
            int(top_k) if isinstance(top_k, int) and top_k > 0 else fallback_top_k
        )
        return resolved_mode, resolved_top_k

    def tool_calling_system_prompt(self) -> str:
        constraints = "；".join(self.constraints)
        required_tools = ", ".join(self.required_tools)
        risk = "；".join(
            [
                f"high={self.risk_rules.get('high', '')}",
                f"medium={self.risk_rules.get('medium', '')}",
                f"low={self.risk_rules.get('low', '')}",
            ]
        )
        workflow = (
            f"必须先调用工具 {required_tools} 获取证据，再输出最终评分 JSON。"
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
    try:
        if path is not None:
            bundle = load_policy_bundle_from_artifact(Path(path))
        else:
            bundle = load_policy_bundle_from_env()
    except PolicyLoadError:
        return _default_policy()

    return AgentScoringPolicy(
        constraints=list(bundle.constraints),
        risk_rules=dict(bundle.risk_rules),
        output_schema_hint=bundle.output.schema_hint,
        tool_calling_required=bundle.workflow.tool_calling_required,
        required_tools=list(bundle.workflow.required_tools),
        max_turns_default=int(bundle.workflow.max_turns_default),
        retrieval_default_mode=bundle.retrieval.default_mode,
        retrieval_default_top_k=int(bundle.retrieval.default_top_k),
        retrieval_dimension_overrides={
            key: {"mode": value.mode, "top_k": value.top_k}
            for key, value in bundle.retrieval.dimension_overrides.items()
        },
        retrieval_evaluation_thresholds={
            method: dict(metrics)
            for method, metrics in bundle.retrieval.evaluation_thresholds.items()
        },
        evidence_default_min_citations=bundle.evidence_gate.default_min_citations,
        evidence_require_page_idx=bundle.evidence_gate.require_page_idx,
        evidence_require_bbox=bundle.evidence_gate.require_bbox,
        evidence_require_quote=bundle.evidence_gate.require_quote,
    )


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
        required_tools=["retrieve_dimension_evidence"],
        max_turns_default=8,
        retrieval_default_mode="hybrid",
        retrieval_default_top_k=8,
        retrieval_dimension_overrides={},
        retrieval_evaluation_thresholds={},
        evidence_default_min_citations=1,
        evidence_require_page_idx=True,
        evidence_require_bbox=True,
        evidence_require_quote=True,
    )
