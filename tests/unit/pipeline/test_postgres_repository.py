from __future__ import annotations

from bid_scoring.pipeline.infrastructure.postgres_repository import (
    PostgresPipelineRepository,
)


class _Cursor:
    def __init__(self):
        self.calls = []

    def execute(self, sql, params):  # pragma: no cover
        self.calls.append((sql, params))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _Conn:
    def __init__(self):
        self.cursor_instance = _Cursor()
        self.commits = 0

    def cursor(self):
        return self.cursor_instance

    def commit(self):
        self.commits += 1


def test_record_source_artifact_executes_upsert_and_commits():
    conn = _Conn()
    repo = PostgresPipelineRepository(conn)

    repo.record_source_artifact(
        version_id="v-1",
        source_uri="minio://x/y.pdf",
        parser_version="pipeline-v1",
        file_sha256="abc",
        page_count=10,
    )

    assert len(conn.cursor_instance.calls) == 1
    assert conn.commits == 1


def test_hash_bytes_is_deterministic():
    value = b"hello"
    assert PostgresPipelineRepository.hash_bytes(value) == PostgresPipelineRepository.hash_bytes(value)

