---
description: "Run agent-team comparison across multiple version_id values"
argument-hint: "<version_id_A> <version_id_B> [version_id_C ...]"
---

# Bid Review Team (Multi Version Compare)

Use `bid-review-orchestrator` to compare bidders for:

`$ARGUMENTS`

Execution rules:
1. Require at least 2 `version_id` values.
2. Score each bidder with the same rubric and process.
3. Use `compare_across_versions` for key clause cross-checks.
4. Preserve traceability evidence for every conclusion.
5. Return one strict JSON object with ranking and warnings.

Output must include:
- `ranking`
- `bidders[]` with dimension scores and evidence
- `cross_bid_findings`
- `warnings`
