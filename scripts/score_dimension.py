#!/usr/bin/env python3
"""Script to score a single dimension from command line."""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bid_scoring.config import load_settings
from bid_scoring.llm import LLMClient
from bid_scoring.scoring import ScoringEngine


def main():
    parser = argparse.ArgumentParser(description="Score a bid document dimension")
    parser.add_argument("dimension", help="Dimension name to score")
    parser.add_argument("--evidence", "-e", required=True, help="JSON file with evidence items")
    parser.add_argument("--max-score", "-m", type=float, default=10.0, help="Maximum score")
    parser.add_argument("--rules", "-r", default="references/scoring_rules.yaml", help="Scoring rules file")
    parser.add_argument("--model", help="Model to use for scoring")
    parser.add_argument("--output", "-o", help="Output file (default: stdout)")
    
    args = parser.parse_args()
    
    # Load evidence
    with open(args.evidence, "r", encoding="utf-8") as f:
        evidence = json.load(f)
    
    # Load settings and create LLM client
    settings = load_settings()
    
    try:
        llm_client = LLMClient(settings)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Create scoring engine and score
    engine = ScoringEngine(llm_client=llm_client, rules_path=args.rules)
    result = engine.score_dimension(
        dimension=args.dimension,
        evidence=evidence,
        max_score=args.max_score,
        model=args.model
    )
    
    # Output result
    output = {
        "dimension": result.dimension,
        "score": result.score,
        "max_score": result.max_score,
        "reasoning": result.reasoning,
        "citations": [
            {
                "source_number": c.source_number,
                "cited_text": c.cited_text,
                "supports_claim": c.supports_claim
            }
            for c in result.citations
        ],
        "evidence_found": result.evidence_found
    }
    
    json_output = json.dumps(output, ensure_ascii=False, indent=2)
    
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(json_output)
        print(f"Result written to {args.output}")
    else:
        print(json_output)


if __name__ == "__main__":
    main()
