from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

from .models import (
    EvidenceGatePolicy,
    OutputPolicy,
    PolicyBundle,
    PolicyMeta,
    RetrievalOverride,
    RetrievalPolicy,
    ScoringPolicy,
    WorkflowPolicy,
)

DEFAULT_POLICY_PACK = "cn_medical_v1"
_PACK_ENV = "BID_SCORING_POLICY_PACK"
_OVERLAY_ENV = "BID_SCORING_POLICY_OVERLAY"
_PACKS_ROOT_ENV = "BID_SCORING_POLICY_PACKS_ROOT"
_ARTIFACT_ENV = "BID_SCORING_POLICY_ARTIFACT"


class PolicyLoadError(ValueError):
    """Raised when policy config is invalid or incomplete."""


def load_policy_bundle(
    pack_id: str | None = None,
    *,
    overlay_name: str | None = None,
    packs_root: Path | None = None,
) -> PolicyBundle:
    root = packs_root or _resolve_default_packs_root()
    selected_pack = (pack_id or os.getenv(_PACK_ENV) or DEFAULT_POLICY_PACK).strip()
    if not selected_pack:
        raise PolicyLoadError("policy pack id is required")

    pack_dir = root / selected_pack
    manifest_path = pack_dir / "manifest.yaml"
    manifest = _load_yaml_object(manifest_path)

    declared_pack = _required_str(manifest, "pack_id")
    version = _required_str(manifest, "version")
    base_file = _required_str(manifest, "base_file")
    overlays = _required_str_list(manifest, "overlays")
    default_overlay = manifest.get("default_overlay")
    if default_overlay is not None and not isinstance(default_overlay, str):
        raise PolicyLoadError("manifest.default_overlay must be string when provided")

    base_payload = _load_yaml_object(pack_dir / base_file)

    selected_overlay = (
        overlay_name
        or os.getenv(_OVERLAY_ENV)
        or (default_overlay if isinstance(default_overlay, str) else None)
    )
    overlay_payload: dict[str, Any] = {}
    if selected_overlay:
        overlay_file = (
            selected_overlay
            if selected_overlay.endswith(".yaml")
            else f"{selected_overlay}.yaml"
        )
        if overlay_file not in overlays:
            raise PolicyLoadError(
                f"overlay '{selected_overlay}' not declared in manifest.overlays"
            )
        overlay_raw = _load_yaml_object(pack_dir / "overlays" / overlay_file)
        overlay_policy = overlay_raw.get("policy")
        if overlay_policy is None:
            overlay_policy = {}
        if not isinstance(overlay_policy, dict):
            raise PolicyLoadError("overlay.policy must be an object")
        overlay_payload = overlay_policy
        selected_overlay = Path(overlay_file).stem
    else:
        selected_overlay = None

    merged = _deep_merge(base_payload, overlay_payload)
    return _to_policy_bundle(
        merged,
        meta=PolicyMeta(
            pack_id=declared_pack,
            overlay=selected_overlay,
            version=version,
            policy_hash=None,
        ),
    )


def load_policy_bundle_from_env() -> PolicyBundle:
    artifact_path = os.getenv(_ARTIFACT_ENV)
    if artifact_path:
        return load_policy_bundle_from_artifact(Path(artifact_path))
    return load_policy_bundle()


def load_policy_bundle_from_artifact(path: Path) -> PolicyBundle:
    if not path.exists():
        raise PolicyLoadError(f"policy artifact not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PolicyLoadError(f"invalid policy artifact json: {path}") from exc
    if not isinstance(payload, dict):
        raise PolicyLoadError("policy artifact root must be object")

    meta_payload = payload.get("meta")
    if not isinstance(meta_payload, dict):
        raise PolicyLoadError("policy artifact meta must be object")
    meta = PolicyMeta(
        pack_id=_required_str(meta_payload, "pack_id"),
        overlay=_optional_str(meta_payload, "overlay"),
        version=_required_str(meta_payload, "version"),
        policy_hash=_optional_str(meta_payload, "policy_hash"),
    )
    core_payload = dict(payload)
    core_payload.pop("meta", None)
    return _to_policy_bundle(core_payload, meta=meta)


def _to_policy_bundle(payload: dict[str, Any], *, meta: PolicyMeta) -> PolicyBundle:
    constraints = _required_str_list(payload, "constraints")
    workflow_raw = _required_dict(payload, "workflow")
    scoring_raw = _required_dict(payload, "scoring")
    risk_raw = _required_dict(payload, "risk_rules")
    output_raw = _required_dict(payload, "output")
    retrieval_raw = _required_dict(payload, "retrieval")
    gate_raw = _required_dict(payload, "evidence_gate")

    workflow = WorkflowPolicy(
        tool_calling_required=_required_bool(workflow_raw, "tool_calling_required"),
        max_turns_default=_required_int(workflow_raw, "max_turns_default", minimum=1),
        required_tools=_required_str_list(workflow_raw, "required_tools"),
    )

    scoring = ScoringPolicy(
        baseline_score=_required_float(scoring_raw, "baseline_score"),
        min_score=_required_float(scoring_raw, "min_score"),
        max_score=_required_float(scoring_raw, "max_score"),
        positive_evidence_delta_range=_required_range_tuple(
            scoring_raw, "positive_evidence_delta_range"
        ),
        risk_evidence_delta_range=_required_range_tuple(
            scoring_raw, "risk_evidence_delta_range"
        ),
    )

    risk_rules = {
        "high": _required_str(risk_raw, "high"),
        "medium": _required_str(risk_raw, "medium"),
        "low": _required_str(risk_raw, "low"),
    }

    output = OutputPolicy(
        strict_json=_required_bool(output_raw, "strict_json"),
        schema_hint=_required_str(output_raw, "schema_hint"),
    )

    retrieval_overrides_raw = retrieval_raw.get("dimension_overrides", {})
    if not isinstance(retrieval_overrides_raw, dict):
        raise PolicyLoadError("retrieval.dimension_overrides must be object")
    dimension_overrides: dict[str, RetrievalOverride] = {}
    for key, value in retrieval_overrides_raw.items():
        if not isinstance(key, str):
            raise PolicyLoadError("retrieval.dimension_overrides keys must be string")
        if not isinstance(value, dict):
            raise PolicyLoadError(
                f"retrieval.dimension_overrides['{key}'] must be object"
            )
        mode = value.get("mode")
        if mode is not None and mode not in {"hybrid", "keyword", "vector"}:
            raise PolicyLoadError(
                f"retrieval.dimension_overrides['{key}'].mode invalid: {mode}"
            )
        top_k = value.get("top_k")
        if top_k is not None:
            if not isinstance(top_k, int) or top_k < 1 or top_k > 100:
                raise PolicyLoadError(
                    f"retrieval.dimension_overrides['{key}'].top_k must be 1..100"
                )
        dimension_overrides[key] = RetrievalOverride(mode=mode, top_k=top_k)

    retrieval = RetrievalPolicy(
        default_mode=_required_mode(retrieval_raw, "default_mode"),
        default_top_k=_required_int(retrieval_raw, "default_top_k", minimum=1),
        dimension_overrides=dimension_overrides,
        evaluation_thresholds=_optional_metric_thresholds(
            retrieval_raw, "evaluation_thresholds"
        ),
    )

    evidence_gate = EvidenceGatePolicy(
        default_min_citations=_required_int(
            gate_raw, "default_min_citations", minimum=1
        ),
        require_page_idx=_required_bool(gate_raw, "require_page_idx"),
        require_bbox=_required_bool(gate_raw, "require_bbox"),
        require_quote=_required_bool(gate_raw, "require_quote"),
    )

    return PolicyBundle(
        meta=meta,
        constraints=constraints,
        workflow=workflow,
        scoring=scoring,
        risk_rules=risk_rules,
        output=output,
        retrieval=retrieval,
        evidence_gate=evidence_gate,
    )


def _resolve_default_packs_root() -> Path:
    env_root = os.getenv(_PACKS_ROOT_ENV)
    if env_root:
        return Path(env_root)
    return Path(__file__).resolve().parents[2] / "config" / "policy" / "packs"


def _load_yaml_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise PolicyLoadError(f"policy file not found: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise PolicyLoadError(f"invalid yaml: {path}") from exc
    if not isinstance(payload, dict):
        raise PolicyLoadError(f"yaml root must be object: {path}")
    return payload


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
            continue
        merged[key] = value
    return merged


def _required_dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise PolicyLoadError(f"{key} must be object")
    return value


def _required_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PolicyLoadError(f"{key} must be non-empty string")
    return value.strip()


def _optional_str(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise PolicyLoadError(f"{key} must be string when provided")
    normalized = value.strip()
    return normalized or None


def _required_bool(payload: dict[str, Any], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise PolicyLoadError(f"{key} must be boolean")
    return value


def _required_int(payload: dict[str, Any], key: str, *, minimum: int) -> int:
    value = payload.get(key)
    if not isinstance(value, int):
        raise PolicyLoadError(f"{key} must be integer")
    if value < minimum:
        raise PolicyLoadError(f"{key} must be >= {minimum}")
    return value


def _required_float(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)):
        raise PolicyLoadError(f"{key} must be number")
    return float(value)


def _required_range_tuple(payload: dict[str, Any], key: str) -> tuple[float, float]:
    value = payload.get(key)
    if not isinstance(value, list) or len(value) != 2:
        raise PolicyLoadError(f"{key} must be [min,max] numeric list")
    first, second = value
    if not isinstance(first, (int, float)) or not isinstance(second, (int, float)):
        raise PolicyLoadError(f"{key} values must be numeric")
    return float(first), float(second)


def _required_mode(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if value not in {"hybrid", "keyword", "vector"}:
        raise PolicyLoadError(f"{key} must be one of hybrid|keyword|vector")
    return str(value)


def _required_str_list(payload: dict[str, Any], key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list) or not value:
        raise PolicyLoadError(f"{key} must be non-empty string list")
    output: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise PolicyLoadError(f"{key} must contain non-empty strings")
        output.append(item.strip())
    return output


def _optional_metric_thresholds(
    payload: dict[str, Any],
    key: str,
) -> dict[str, dict[str, float]]:
    value = payload.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise PolicyLoadError(f"{key} must be object when provided")

    allowed_methods = {"hybrid", "keyword", "vector"}
    output: dict[str, dict[str, float]] = {}
    for method, metrics in value.items():
        if method not in allowed_methods:
            raise PolicyLoadError(
                f"{key} method must be one of hybrid|keyword|vector: {method}"
            )
        if not isinstance(metrics, dict) or not metrics:
            raise PolicyLoadError(f"{key}.{method} must be non-empty object")
        metric_output: dict[str, float] = {}
        for metric, threshold in metrics.items():
            if not isinstance(metric, str) or not metric.strip():
                raise PolicyLoadError(f"{key}.{method} metric name must be string")
            if not isinstance(threshold, (int, float)):
                raise PolicyLoadError(
                    f"{key}.{method}.{metric} threshold must be numeric"
                )
            threshold_value = float(threshold)
            if threshold_value < 0:
                raise PolicyLoadError(f"{key}.{method}.{metric} threshold must be >= 0")
            metric_output[metric.strip()] = threshold_value
        output[method] = metric_output
    return output
