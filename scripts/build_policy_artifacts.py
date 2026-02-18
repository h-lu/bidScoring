from __future__ import annotations

import argparse
import json
from pathlib import Path

from bid_scoring.policy import compile_policy_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile policy pack/overlay into runtime artifacts."
    )
    parser.add_argument("--pack", default="cn_medical_v1")
    parser.add_argument("--overlay", default="strict_traceability")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--packs-root", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    artifacts = compile_policy_artifacts(
        pack_id=args.pack,
        overlay_name=args.overlay,
        output_root=args.output_root,
        packs_root=args.packs_root,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "policy_hash": artifacts.policy_hash,
                "runtime_policy_path": str(artifacts.runtime_policy_path),
                "agent_prompt_path": str(artifacts.agent_prompt_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
