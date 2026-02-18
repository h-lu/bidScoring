---
description: "Run full agent-team bid review for one version_id (evidence -> scoring -> traceability)"
argument-hint: "<version_id> [bidder_name] [project_name]"
---

# Bid Review Team (Single Version)

Use `bid-review-orchestrator` to run a complete review for:

`$ARGUMENTS`

Execution rules:
1. If `version_id` is missing, ask for it first.
2. Use agent team flow:
   - `bid-evidence-retriever`
   - `bid-scoring-evaluator`
   - `bid-traceability-auditor`
3. Use `bid-scoring` MCP tools before scoring.
4. Enforce strict evidence traceability:
   - `version_id/chunk_id/page_idx/bbox/quote`
5. Return one strict JSON object only (no markdown wrapper).

If MCP tool validation fails, retry with corrected parameter types:
- arrays must be native arrays, not quoted strings.
