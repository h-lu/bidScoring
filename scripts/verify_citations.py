# scripts/verify_citations.py
import json
import sys

from bid_scoring.verify import verify_citation


def main():
    payload = json.loads(input())
    result = verify_citation(payload["cited_text"], payload["original_text"])
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
