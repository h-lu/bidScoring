#!/usr/bin/env python3
"""Evaluate retrieval summary against policy-configured thresholds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bid_scoring.policy.loader import (  # noqa: E402
    load_policy_bundle,
    load_policy_bundle_from_artifact,
)
from bid_scoring.retrieval import (  # noqa: E402
    evaluate_retrieval_summary,
    extract_retrieval_thresholds,
)


def _load_summary(path: Path) -> dict[str, dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("summary payload must be object")

    summary = payload.get("summary", payload)
    if not isinstance(summary, dict):
        raise ValueError("summary must be object")

    normalized: dict[str, dict[str, float]] = {}
    for method, metrics in summary.items():
        if not isinstance(method, str) or not isinstance(metrics, dict):
            continue
        method_metrics: dict[str, float] = {}
        for metric, value in metrics.items():
            if not isinstance(metric, str):
                continue
            if not isinstance(value, (int, float)):
                continue
            method_metrics[metric] = float(value)
        normalized[method] = method_metrics
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check retrieval eval summary against policy thresholds"
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        required=True,
        help="JSON from evaluate_hybrid_search_gold.py --output",
    )
    parser.add_argument("--policy-pack", type=str, default=None)
    parser.add_argument("--policy-overlay", type=str, default=None)
    parser.add_argument("--policy-packs-root", type=Path, default=None)
    parser.add_argument("--policy-artifact", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fail-on-violations", action="store_true")
    args = parser.parse_args()

    if args.policy_artifact is not None:
        bundle = load_policy_bundle_from_artifact(args.policy_artifact)
    else:
        bundle = load_policy_bundle(
            pack_id=args.policy_pack,
            overlay_name=args.policy_overlay,
            packs_root=args.policy_packs_root,
        )

    summary = _load_summary(args.summary_file)
    thresholds = extract_retrieval_thresholds(bundle)
    violations = evaluate_retrieval_summary(summary, thresholds)

    payload: dict[str, Any] = {
        "policy_pack": bundle.meta.pack_id,
        "policy_overlay": bundle.meta.overlay,
        "policy_version": bundle.meta.version,
        "thresholds": thresholds,
        "violations": [
            {
                "method": item.method,
                "metric": item.metric,
                "actual": item.actual,
                "minimum": item.minimum,
            }
            for item in violations
        ],
        "ok": len(violations) == 0,
    }

    rendered = json.dumps(payload, ensure_ascii=False)
    print(rendered)
    if args.output is not None:
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if violations and args.fail_on_violations:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
