from __future__ import annotations

from pathlib import Path


def test_schema_contains_source_artifacts_and_citation_warning_fields():
    sql = Path("migrations/000_init.sql").read_text(encoding="utf-8")
    assert "CREATE TABLE IF NOT EXISTS source_artifacts" in sql
    assert "evidence_status TEXT NOT NULL DEFAULT 'verified'" in sql
    assert "warning_codes TEXT[] NOT NULL DEFAULT '{}'" in sql

