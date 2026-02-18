# Changelog

All notable changes to this project are documented here.

## [Unreleased]

### Added
- Policy-as-config architecture:
  - `config/policy/packs/<pack_id>/base.yaml`
  - `config/policy/packs/<pack_id>/overlays/*.yaml`
  - `scripts/build_policy_artifacts.py`
- Skill/policy sync gate:
  - `scripts/check_skill_policy_sync.py`
  - `tests/unit/test_skill_policy_sync.py`
- Retrieval policy gate:
  - `scripts/evaluate_retrieval_policy_gate.py`
- Production-first CLI:
  - `bid-pipeline run-prod`
  - stable defaults: `hybrid` + `cn_medical_v1/strict_traceability`
- Run comparison utility:
  - `scripts/compare_scoring_runs.py`
  - `data/eval/scoring_compare/runs/*`

### Changed
- E2E output contract now emphasizes three blocks:
  - `scoring`
  - `traceability`
  - `observability`
- Agent scoring policy loading now reads policy bundle/artifact first.
- Documentation set rewritten to a single canonical narrative.

### Removed
- Legacy policy file references (`config/agent_scoring_policy.yaml`) from active `.claude` docs.

### Notes
- Current recommended execution path is `run-prod`.
- `run-e2e` remains available for advanced experiments and diagnostics.
