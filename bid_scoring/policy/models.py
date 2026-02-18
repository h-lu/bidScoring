from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PolicyMeta:
    pack_id: str
    overlay: str | None
    version: str
    policy_hash: str | None = None


@dataclass(frozen=True)
class WorkflowPolicy:
    tool_calling_required: bool
    max_turns_default: int
    required_tools: list[str]


@dataclass(frozen=True)
class ScoringPolicy:
    baseline_score: float
    min_score: float
    max_score: float
    positive_evidence_delta_range: tuple[float, float]
    risk_evidence_delta_range: tuple[float, float]


@dataclass(frozen=True)
class OutputPolicy:
    strict_json: bool
    schema_hint: str


@dataclass(frozen=True)
class RetrievalOverride:
    mode: str | None = None
    top_k: int | None = None


@dataclass(frozen=True)
class RetrievalPolicy:
    default_mode: str
    default_top_k: int
    dimension_overrides: dict[str, RetrievalOverride]
    evaluation_thresholds: dict[str, dict[str, float]]


@dataclass(frozen=True)
class EvidenceGatePolicy:
    default_min_citations: int
    require_page_idx: bool
    require_bbox: bool
    require_quote: bool


@dataclass(frozen=True)
class PolicyBundle:
    meta: PolicyMeta
    constraints: list[str]
    workflow: WorkflowPolicy
    scoring: ScoringPolicy
    risk_rules: dict[str, str]
    output: OutputPolicy
    retrieval: RetrievalPolicy
    evidence_gate: EvidenceGatePolicy

    def as_runtime_dict(self) -> dict[str, Any]:
        return {
            "meta": {
                "pack_id": self.meta.pack_id,
                "overlay": self.meta.overlay,
                "version": self.meta.version,
                "policy_hash": self.meta.policy_hash,
            },
            "constraints": list(self.constraints),
            "workflow": {
                "tool_calling_required": self.workflow.tool_calling_required,
                "max_turns_default": self.workflow.max_turns_default,
                "required_tools": list(self.workflow.required_tools),
            },
            "scoring": {
                "baseline_score": self.scoring.baseline_score,
                "min_score": self.scoring.min_score,
                "max_score": self.scoring.max_score,
                "positive_evidence_delta_range": list(
                    self.scoring.positive_evidence_delta_range
                ),
                "risk_evidence_delta_range": list(
                    self.scoring.risk_evidence_delta_range
                ),
            },
            "risk_rules": dict(self.risk_rules),
            "output": {
                "strict_json": self.output.strict_json,
                "schema_hint": self.output.schema_hint,
            },
            "retrieval": {
                "default_mode": self.retrieval.default_mode,
                "default_top_k": self.retrieval.default_top_k,
                "dimension_overrides": {
                    key: {"mode": value.mode, "top_k": value.top_k}
                    for key, value in self.retrieval.dimension_overrides.items()
                },
                "evaluation_thresholds": {
                    method: dict(metrics)
                    for method, metrics in self.retrieval.evaluation_thresholds.items()
                },
            },
            "evidence_gate": {
                "default_min_citations": self.evidence_gate.default_min_citations,
                "require_page_idx": self.evidence_gate.require_page_idx,
                "require_bbox": self.evidence_gate.require_bbox,
                "require_quote": self.evidence_gate.require_quote,
            },
        }
