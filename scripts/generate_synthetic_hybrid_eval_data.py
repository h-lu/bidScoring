#!/usr/bin/env python3
"""Generate synthetic medical bid retrieval evaluation assets.

This module is intentionally thin: the generator implementation lives in
`bid_scoring/synthetic_eval/` (kept < 500 LOC per file).

Outputs:
- content_list.synthetic_bidder_{A|B|C}.json
- queries.json
- qrels.source_id.{A|B|C}.jsonl
- README.md
- multi_version_manifest.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bid_scoring.synthetic_eval.dataset import (
    DEFAULT_OUTPUT_DIR,
    generate_all,
    validate,
)
from bid_scoring.synthetic_eval.profiles import DEFAULT_SCENARIOS, SCENARIO_PROFILES


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate synthetic hybrid retrieval evaluation data"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="output directory",
    )
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIO_PROFILES),
        help="generate only one scenario",
    )
    parser.add_argument(
        "--all-scenarios",
        action="store_true",
        help="generate A/B/C scenarios",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate existing files only",
    )
    args = parser.parse_args()

    scenarios = (
        DEFAULT_SCENARIOS if args.all_scenarios or not args.scenario else (args.scenario,)
    )

    if args.validate_only:
        summary = validate(args.output_dir, scenarios=scenarios)
        for name, (c, q, r) in summary.items():
            print(f"{name}: content={c}, queries={q}, qrels={r}")
        print("Validation passed")
        return 0

    generate_all(args.output_dir, scenarios=scenarios)
    summary = validate(args.output_dir, scenarios=scenarios)
    for name, (c, q, r) in summary.items():
        print(f"{name}: content={c}, queries={q}, qrels={r}")
    print(f"Generated assets at: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
