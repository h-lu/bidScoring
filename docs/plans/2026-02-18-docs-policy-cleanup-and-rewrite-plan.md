# Docs + Policy Cleanup and Rewrite Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate stale policy references, restore retrieval policy gate execution with a real summary artifact, and rewrite the documentation corpus into a single consistent architecture narrative.

**Architecture:** Keep runtime code behavior unchanged; clean policy/source-of-truth references in `.claude` docs, generate retrieval evaluation summary artifact from existing synthetic dataset and DB version, and rewrite markdown docs around one canonical flow (`run-prod` first, `run-e2e` advanced). Preserve contract-critical files (`prompt.md`) so sync gate remains green.

**Tech Stack:** Python 3.11+, pytest, uv, argparse scripts, markdown docs

---

### Task 1: Remove stale policy-file references in Claude docs

**Files:**
- Modify: `.claude/skills/bid-analyze/rubric.md`
- Modify: `.claude/agents/bid-team-scoring.md`

**Step 1: Write the failing check expectation**
- Define expected behavior: no references to `config/agent_scoring_policy.yaml` remain in `.claude` docs.

**Step 2: Run search to verify current failure**
- Run: `rg -n "config/agent_scoring_policy\.yaml" .claude -S`
- Expected: finds at least 2 matches.

**Step 3: Apply minimal edits**
- Replace stale source path with `config/policy/packs/cn_medical_v1/*` and compiled artifact reference.

**Step 4: Re-run search to verify pass**
- Run: `rg -n "config/agent_scoring_policy\.yaml" .claude -S`
- Expected: no matches.

### Task 2: Generate retrieval eval summary artifact and run policy gate

**Files:**
- Create/Update artifact: `data/eval/hybrid_medical_synthetic/eval_summary.json`

**Step 1: Run retrieval evaluator to generate summary**
- Run:
  `uv run python scripts/evaluate_hybrid_search_gold.py --version-id <version_id> --queries-file data/eval/hybrid_medical_synthetic/queries.json --qrels-file data/eval/hybrid_medical_synthetic/qrels.source_id.A.jsonl --thresholds-file data/eval/hybrid_medical_synthetic/retrieval_baseline.thresholds.json --output data/eval/hybrid_medical_synthetic/eval_summary.json`

**Step 2: Verify summary file exists and has `summary` payload**
- Run: `jq 'keys' data/eval/hybrid_medical_synthetic/eval_summary.json`
- Expected: contains `summary`.

**Step 3: Run retrieval policy gate on generated summary**
- Run:
  `uv run python scripts/evaluate_retrieval_policy_gate.py --summary-file data/eval/hybrid_medical_synthetic/eval_summary.json --policy-pack cn_medical_v1 --policy-overlay strict_traceability --fail-on-violations`
- Expected: exits 0 and prints `"ok": true`.

### Task 3: Rewrite documentation corpus with one canonical story

**Files:**
- Modify: `README.md`
- Modify: `docs/usage.md`
- Modify: `docs/guides/*.md` (except archived historical reports if any are intentionally preserved)
- Modify: `.claude/README.md`
- Modify: `.claude/skills/bid-analyze/{SKILL.md,workflow.md,rubric.md,examples.md,prompt.md}`
- Modify: `.claude/agents/*.md`
- Modify: `.claude/commands/*.md`
- Modify: `mcp_servers/{README.md,CLAUDE_CODE_GUIDE.md}`
- Modify: `data/eval/*/README.md`

**Step 1: Define canonical doc contract**
- Production primary path: `run-prod`.
- Advanced/debug path: `run-e2e`.
- Policy source of truth: policy pack + overlay (+ optional artifact).
- Verification gates and required commands fixed in one place.

**Step 2: Rewrite docs file-by-file**
- Preserve examples that are executable in this repo.
- Remove stale or contradictory statements.
- Keep prompt contract fields required by sync checker.

**Step 3: Validate internal consistency by grep**
- Run:
  - `rg -n "agent_scoring_policy\.yaml|旧|待定|TODO" README.md docs .claude mcp_servers data/eval -S`
  - `rg -n "run-prod|run-e2e|BID_SCORING_POLICY_PACK|BID_SCORING_POLICY_OVERLAY" README.md docs .claude -S`

**Step 4: Run policy/doc sync + focused tests**
- Run:
  - `uv run python scripts/check_skill_policy_sync.py --policy-pack cn_medical_v1 --policy-overlay strict_traceability --fail-on-violations`
  - `uv run pytest tests/unit/test_skill_policy_sync.py -q`

### Task 4: Full verification before completion

**Files:**
- No new files; verification only.

**Step 1: Run essential regression set**
- Run:
  - `uv run pytest tests/unit/policy -q`
  - `uv run pytest tests/unit/pipeline/test_scoring_agent_policy.py -q`
  - `uv run pytest tests/unit/test_skill_policy_sync.py -q`

**Step 2: Run end-to-end gate commands**
- Run:
  - `uv run python scripts/build_policy_artifacts.py --pack cn_medical_v1 --overlay strict_traceability`
  - `uv run python scripts/check_skill_policy_sync.py --fail-on-violations`
  - `uv run python scripts/evaluate_retrieval_policy_gate.py --summary-file data/eval/hybrid_medical_synthetic/eval_summary.json --policy-pack cn_medical_v1 --policy-overlay strict_traceability --fail-on-violations`

**Step 3: Capture final status with concrete evidence**
- Report command outputs, failures (if any), and next actions.
