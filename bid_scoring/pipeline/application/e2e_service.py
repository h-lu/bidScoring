from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

from bid_scoring.pipeline.application.scoring_provider import (
    ScoringProvider,
    ScoringRequest,
)
from bid_scoring.pipeline.application.traceability import build_traceability_summary


@dataclass
class E2ERunRequest:
    """Input contract for end-to-end pipeline execution."""

    project_id: str
    document_id: str
    version_id: str
    document_title: str = "untitled"
    source_uri: str | None = None
    parser_version: str | None = "pipeline-v1"
    bidder_name: str = "Unknown"
    project_name: str = "Unknown Project"
    dimensions: list[str] | None = None
    content_list_path: Path | None = None
    mineru_output_dir: Path | None = None
    pdf_path: Path | None = None
    mineru_parser: str | None = None
    build_embeddings: bool = True
    scoring_backend: str = "analyzer"
    hybrid_primary_weight: float | None = None


@dataclass(frozen=True)
class LoadedContent:
    """Normalized payload from input-source adapters."""

    content_list: list[dict[str, Any]]
    source_uri: str
    parser_version: str | None
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class E2ERunResult:
    """Result for end-to-end pipeline execution."""

    status: str
    project_id: str
    document_id: str
    version_id: str
    warnings: list[str] = field(default_factory=list)
    ingest: dict[str, Any] = field(default_factory=dict)
    embeddings: dict[str, Any] = field(default_factory=dict)
    scoring: dict[str, Any] = field(default_factory=dict)
    traceability: dict[str, Any] = field(default_factory=dict)
    observability: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "project_id": self.project_id,
            "document_id": self.document_id,
            "version_id": self.version_id,
            "warnings": list(self.warnings),
            "ingest": dict(self.ingest),
            "embeddings": dict(self.embeddings),
            "scoring": dict(self.scoring),
            "traceability": dict(self.traceability),
            "observability": dict(self.observability),
        }


class ContentSource(Protocol):
    def load(self, request: E2ERunRequest) -> LoadedContent: ...


class IngestApplicationService(Protocol):
    def ingest_content_list(
        self,
        *,
        project_id: str,
        document_id: str,
        version_id: str,
        content_list: list[dict[str, Any]],
        document_title: str = "untitled",
        source_uri: str | None = None,
        parser_version: str | None = "pipeline-v1",
    ) -> Any: ...


class EmbeddingBuilder(Protocol):
    def build_embeddings(self, *, version_id: str, conn: Any) -> dict[str, Any]: ...


class E2EPipelineService:
    """Orchestrates load -> ingest -> embedding -> scoring pipeline."""

    def __init__(
        self,
        *,
        content_source: ContentSource,
        pipeline_service: IngestApplicationService,
        scoring_provider: ScoringProvider,
        index_builder: EmbeddingBuilder | None = None,
    ) -> None:
        self._content_source = content_source
        self._pipeline_service = pipeline_service
        self._index_builder = index_builder
        self._scoring_provider = scoring_provider

    def run(self, request: E2ERunRequest, conn: Any | None = None) -> E2ERunResult:
        total_started = perf_counter()
        timings_ms: dict[str, int] = {}

        stage_started = perf_counter()
        loaded = self._content_source.load(request)
        timings_ms["load"] = _elapsed_ms(stage_started)
        warnings = _merge_unique_warnings([], loaded.warnings)

        stage_started = perf_counter()
        ingest_summary = self._pipeline_service.ingest_content_list(
            project_id=request.project_id,
            document_id=request.document_id,
            version_id=request.version_id,
            content_list=loaded.content_list,
            document_title=request.document_title,
            source_uri=request.source_uri or loaded.source_uri,
            parser_version=request.parser_version or loaded.parser_version,
        )
        timings_ms["ingest"] = _elapsed_ms(stage_started)
        ingest_result = _normalize_ingest_summary(ingest_summary)

        stage_started = perf_counter()
        embeddings_result = {"status": "skipped"}
        if request.build_embeddings:
            if self._index_builder is None:
                embeddings_result = {
                    "status": "warning",
                    "warning": "index_builder_not_configured",
                }
                warnings = _merge_unique_warnings(
                    warnings, ["index_builder_not_configured"]
                )
            else:
                if conn is None:
                    raise RuntimeError(
                        "Database connection is required when build_embeddings is True"
                    )
                embeddings_result = self._index_builder.build_embeddings(
                    version_id=request.version_id,
                    conn=conn,
                )
        timings_ms["embeddings"] = _elapsed_ms(stage_started)

        stage_started = perf_counter()
        scoring = self._scoring_provider.score(
            ScoringRequest(
                version_id=request.version_id,
                bidder_name=request.bidder_name,
                project_name=request.project_name,
                dimensions=request.dimensions,
            )
        )
        timings_ms["scoring"] = _elapsed_ms(stage_started)
        scoring_result = scoring.as_dict()
        traceability = build_traceability_summary(scoring_result)
        warnings = _merge_unique_warnings(warnings, scoring.evidence_warnings)
        warnings = _merge_unique_warnings(warnings, scoring.warnings)
        warnings = _merge_unique_warnings(
            warnings,
            [w for w in traceability.get("warnings", []) if isinstance(w, str)],
        )

        timings_ms["total"] = _elapsed_ms(total_started)

        return E2ERunResult(
            status="completed",
            project_id=request.project_id,
            document_id=request.document_id,
            version_id=request.version_id,
            warnings=warnings,
            ingest=ingest_result,
            embeddings=embeddings_result,
            scoring=scoring_result,
            traceability=traceability,
            observability={
                "timings_ms": timings_ms,
                "scoring_backend": request.scoring_backend,
                "embeddings_enabled": request.build_embeddings,
            },
        )


def _normalize_ingest_summary(summary: Any) -> dict[str, Any]:
    if isinstance(summary, dict):
        return dict(summary)

    return {
        "status": getattr(summary, "status", "completed"),
        "project_id": getattr(summary, "project_id", None),
        "document_id": getattr(summary, "document_id", None),
        "version_id": getattr(summary, "version_id", None),
        "chunks_imported": getattr(summary, "chunks_imported", 0),
        "source_uri": getattr(summary, "source_uri", None),
        "warnings": list(getattr(summary, "warnings", [])),
    }


def _merge_unique_warnings(
    existing: list[str], additions: list[str] | None = None
) -> list[str]:
    merged = list(existing)
    seen = set(merged)
    for warning in additions or []:
        if warning in seen:
            continue
        seen.add(warning)
        merged.append(warning)
    return merged


def _elapsed_ms(started_at: float) -> int:
    return max(0, int((perf_counter() - started_at) * 1000))
