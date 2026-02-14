from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "compare_scoring_runs.py"
)
_SPEC = importlib.util.spec_from_file_location("compare_scoring_runs", _SCRIPT_PATH)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

summarize_run_output = _MODULE.summarize_run_output
compare_run_outputs = _MODULE.compare_run_outputs


def test_summarize_run_output_extracts_core_metrics():
    payload = {
        "status": "completed",
        "warnings": ["mineru_bypassed"],
        "scoring": {
            "overall_score": 82.5,
            "risk_level": "medium",
            "total_risks": 12,
            "total_benefits": 21,
            "chunks_analyzed": 50,
            "evidence_warnings": ["citation_missing_bbox"],
            "dimensions": {
                "warranty": {"score": 86.0},
                "delivery": {"score": 79.0},
            },
        },
        "traceability": {
            "status": "verified_with_warnings",
            "citation_count_total": 40,
            "citation_count_traceable": 38,
            "citation_coverage_ratio": 0.95,
        },
        "observability": {"scoring_backend": "hybrid"},
    }

    summary = summarize_run_output(payload)

    assert summary["status"] == "completed"
    assert summary["overall_score"] == 82.5
    assert summary["risk_level"] == "medium"
    assert summary["traceability_status"] == "verified_with_warnings"
    assert summary["coverage_ratio"] == 0.95
    assert summary["dimension_scores"]["warranty"] == 86.0
    assert summary["evidence_warning_codes"] == ["citation_missing_bbox"]


def test_compare_run_outputs_computes_deltas_and_warning_diff():
    baseline = {
        "status": "completed",
        "warnings": ["mineru_bypassed"],
        "scoring": {
            "overall_score": 80.0,
            "risk_level": "medium",
            "dimensions": {
                "warranty": {"score": 80.0},
                "delivery": {"score": 78.0},
            },
        },
        "traceability": {
            "status": "verified",
            "citation_count_total": 20,
            "citation_count_traceable": 20,
            "citation_coverage_ratio": 1.0,
        },
        "observability": {"scoring_backend": "analyzer"},
    }
    candidate = {
        "status": "completed",
        "warnings": ["mineru_bypassed", "agent_mcp_fallback"],
        "scoring": {
            "overall_score": 84.0,
            "risk_level": "low",
            "dimensions": {
                "warranty": {"score": 86.0},
                "delivery": {"score": 81.0},
            },
        },
        "traceability": {
            "status": "verified",
            "citation_count_total": 25,
            "citation_count_traceable": 24,
            "citation_coverage_ratio": 0.96,
        },
        "observability": {"scoring_backend": "hybrid"},
    }

    comparison = compare_run_outputs(baseline=baseline, candidate=candidate)

    assert comparison["delta"]["overall_score"] == 4.0
    assert comparison["delta"]["coverage_ratio"] == -0.04
    assert comparison["delta"]["citation_count_total"] == 5
    assert comparison["delta"]["dimension_scores"]["warranty"] == 6.0
    assert comparison["warnings_added"] == ["agent_mcp_fallback"]
    assert comparison["warnings_removed"] == []
