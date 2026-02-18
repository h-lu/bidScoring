# Policy-as-Config Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace scattered scoring policy rules with a single configurable policy bundle system that drives runtime scoring prompts, agent skill prompt content, and validation checks.

**Architecture:** Introduce a new `bid_scoring.policy` module that loads/validates policy packs (`base + overlay`), compiles deterministic artifacts, and exposes one typed policy contract for runtime usage. Keep pipeline execution unchanged but switch policy ingestion to the new contract.

**Tech Stack:** Python 3.11+, PyYAML, pytest, existing bid_scoring pipeline + .claude skill files

---

### Task 1: Add policy pack files and schema

**Files:**
- Create: `config/policy/schema/policy_bundle.schema.json`
- Create: `config/policy/packs/cn_medical_v1/manifest.yaml`
- Create: `config/policy/packs/cn_medical_v1/base.yaml`
- Create: `config/policy/packs/cn_medical_v1/overlays/strict_traceability.yaml`
- Create: `config/policy/packs/cn_medical_v1/overlays/fast_eval.yaml`

**Step 1: Write the failing tests**
- Add tests expecting valid policy pack structure and overlay merge behavior.

**Step 2: Run test to verify it fails**
- Run: `uv run pytest tests/unit/policy/test_policy_loader.py -q`
- Expected: FAIL (module/files not found)

**Step 3: Add schema + base/overlay config files**
- Add strict, normalized fields:
  - `constraints`, `workflow`, `scoring`, `risk_rules`, `output`, `retrieval`, `evidence_gate`.

**Step 4: Run test to verify file-level contract**
- Run: `uv run pytest tests/unit/policy/test_policy_loader.py -q`
- Expected: still FAIL until loader exists.

---

### Task 2: Implement policy loader/merger/validator

**Files:**
- Create: `bid_scoring/policy/models.py`
- Create: `bid_scoring/policy/loader.py`
- Create: `bid_scoring/policy/__init__.py`
- Create: `tests/unit/policy/test_policy_loader.py`

**Step 1: Write the failing tests**
- Validate:
  - load default pack and overlay
  - deep-merge semantics for overlay
  - required-field validation errors

**Step 2: Run test to verify it fails**
- Run: `uv run pytest tests/unit/policy/test_policy_loader.py -q`
- Expected: FAIL

**Step 3: Write minimal implementation**
- Typed dataclasses for policy structures.
- Loader that reads manifest/base/overlay and validates keys + value types.

**Step 4: Run test to verify it passes**
- Run: `uv run pytest tests/unit/policy/test_policy_loader.py -q`
- Expected: PASS

---

### Task 3: Implement artifact compiler and build script

**Files:**
- Create: `bid_scoring/policy/compiler.py`
- Create: `scripts/build_policy_artifacts.py`
- Create: `tests/unit/policy/test_policy_compiler.py`

**Step 1: Write the failing tests**
- Expect compiler to output:
  - `runtime_policy.json`
  - `agent_prompt.md`
  - stable `policy_hash`

**Step 2: Run test to verify it fails**
- Run: `uv run pytest tests/unit/policy/test_policy_compiler.py -q`
- Expected: FAIL

**Step 3: Write minimal implementation**
- Compile from loader result into deterministic JSON/Markdown artifacts.

**Step 4: Run test to verify it passes**
- Run: `uv run pytest tests/unit/policy/test_policy_compiler.py -q`
- Expected: PASS

---

### Task 4: Switch runtime scoring policy ingestion to policy bundle

**Files:**
- Modify: `bid_scoring/pipeline/application/scoring_agent_policy.py`
- Modify: `tests/unit/pipeline/test_scoring_agent_policy.py`

**Step 1: Write the failing tests**
- Add tests for new loading path:
  - from policy bundle artifact
  - from policy pack + overlay env vars

**Step 2: Run test to verify it fails**
- Run: `uv run pytest tests/unit/pipeline/test_scoring_agent_policy.py -q`
- Expected: FAIL

**Step 3: Write minimal implementation**
- Resolve policy from new source first.
- Build prompts from compiled policy fields.

**Step 4: Run test to verify it passes**
- Run: `uv run pytest tests/unit/pipeline/test_scoring_agent_policy.py -q`
- Expected: PASS

---

### Task 5: Align .claude prompt + sync checker to generated artifacts

**Files:**
- Modify: `.claude/skills/bid-analyze/prompt.md`
- Modify: `scripts/check_skill_policy_sync.py`
- Modify: `tests/unit/test_skill_policy_sync.py`

**Step 1: Write the failing tests**
- Assert checker can validate against policy bundle or compiled runtime artifact.

**Step 2: Run test to verify it fails**
- Run: `uv run pytest tests/unit/test_skill_policy_sync.py -q`
- Expected: FAIL

**Step 3: Write minimal implementation**
- Checker accepts `--policy-pack/--policy-overlay` and validates generated prompt contract.
- Update prompt template to declare generated source of truth.

**Step 4: Run test to verify it passes**
- Run: `uv run pytest tests/unit/test_skill_policy_sync.py -q`
- Expected: PASS

---

### Task 6: End-to-end verification and documentation

**Files:**
- Modify: `docs/usage.md`
- Modify: `README.md`

**Step 1: Run focused regression suite**
- Run:
  - `uv run pytest tests/unit/policy -q`
  - `uv run pytest tests/unit/pipeline/test_scoring_agent_policy.py -q`
  - `uv run pytest tests/unit/test_skill_policy_sync.py -q`

**Step 2: Run full unit suite**
- Run: `uv run pytest -q`
- Expected: PASS

**Step 3: Verify artifacts build command**
- Run: `uv run python scripts/build_policy_artifacts.py --pack cn_medical_v1 --overlay strict_traceability`
- Expected: artifacts generated under `artifacts/policy/...`

**Step 4: Update docs**
- Document new configuration and build flow.

