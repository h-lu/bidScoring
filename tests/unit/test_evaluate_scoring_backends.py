from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "evaluate_scoring_backends.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "evaluate_scoring_backends", _SCRIPT_PATH
)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

evaluate_thresholds = _MODULE.evaluate_thresholds
summarize_backend_result = _MODULE.summarize_backend_result


def test_summarize_backend_result_counts_citations_and_timings():
    payload = {
        "status": "completed",
        "warnings": ["mineru_bypassed"],
        "scoring": {
            "overall_score": 80.0,
            "risk_level": "low",
            "chunks_analyzed": 10,
            "evidence_citations": {
                "warranty": [{"chunk_id": "c1"}, {"chunk_id": "c2"}],
                "delivery": [{"chunk_id": "c3"}],
            },
        },
        "traceability": {
            "status": "verified",
            "citation_coverage_ratio": 1.0,
        },
        "observability": {"timings_ms": {"scoring": 120, "total": 600}},
    }

    summary = summarize_backend_result(payload)

    assert summary["status"] == "completed"
    assert summary["citation_total"] == 3
    assert summary["traceability_status"] == "verified"
    assert summary["coverage_ratio"] == 1.0
    assert summary["timings_ms"]["scoring"] == 120


def test_evaluate_thresholds_flags_relation_and_forbidden_warning():
    summaries = {
        "analyzer": {
            "status": "completed",
            "overall_score": 80.0,
            "citation_total": 5,
            "traceability_status": "verified",
            "coverage_ratio": 1.0,
            "warnings": [],
        },
        "agent-mcp": {
            "status": "completed",
            "overall_score": 90.0,
            "citation_total": 5,
            "traceability_status": "verified",
            "coverage_ratio": 1.0,
            "warnings": [],
        },
        "hybrid": {
            "status": "completed",
            "overall_score": 95.0,
            "citation_total": 5,
            "traceability_status": "verified",
            "coverage_ratio": 1.0,
            "warnings": ["no_evidence_citations"],
        },
    }
    thresholds = {
        "per_backend": {
            "analyzer": {"expected_status": "completed", "min_citation_total": 1},
            "agent-mcp": {"expected_status": "completed", "min_citation_total": 1},
            "hybrid": {"expected_status": "completed", "min_citation_total": 1},
        },
        "global": {"forbidden_warning_codes": ["no_evidence_citations"]},
        "relations": {"hybrid_score_between_analyzer_and_agent": True},
    }

    violations = evaluate_thresholds(summaries=summaries, thresholds=thresholds)

    metrics = {(item["scope"], item["metric"]) for item in violations}
    assert ("hybrid", "warnings") in metrics
    assert ("relations", "hybrid_score_between_analyzer_and_agent") in metrics
