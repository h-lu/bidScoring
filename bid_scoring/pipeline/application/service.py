from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol

from bid_scoring.citations_v2 import verify_citation
from bid_scoring.pipeline.domain.models import EvidenceWarning
from bid_scoring.pipeline.domain.verification import CitationVerifier


VerifyCitationFn = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class CitationEvaluationSummary:
    """Summary for citation verification stage."""

    status: str
    total_citations: int
    verified_citations: int
    unverifiable_citations: int
    warning_codes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class IngestSummary:
    """Summary for ingest stage."""

    status: str
    project_id: str
    document_id: str
    version_id: str
    chunks_imported: int
    source_uri: str | None = None
    warnings: list[str] = field(default_factory=list)


class PipelineRepository(Protocol):
    """Repository contract for persistence operations."""

    def persist_content_list(
        self,
        *,
        project_id: str,
        document_id: str,
        version_id: str,
        content_list: list[dict[str, Any]],
        document_title: str,
        source_uri: str | None,
        parser_version: str | None,
    ) -> dict[str, Any]: ...

    def record_source_artifact(
        self,
        *,
        version_id: str,
        source_uri: str,
        parser_version: str | None,
        file_sha256: str | None = None,
        page_count: int | None = None,
    ) -> None: ...


class PipelineService:
    """Evidence-first application service."""

    def __init__(
        self,
        verify_citation_fn: VerifyCitationFn = verify_citation,
        verifier: CitationVerifier | None = None,
        repository: PipelineRepository | None = None,
    ):
        self._verify_citation_fn = verify_citation_fn
        self._verifier = verifier or CitationVerifier()
        self._repository = repository

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
    ) -> IngestSummary:
        if self._repository is None:
            raise RuntimeError("Pipeline repository is required for ingest operations")

        ingest_result = self._repository.persist_content_list(
            project_id=project_id,
            document_id=document_id,
            version_id=version_id,
            content_list=content_list,
            document_title=document_title,
            source_uri=source_uri,
            parser_version=parser_version,
        )

        if source_uri:
            self._repository.record_source_artifact(
                version_id=version_id,
                source_uri=source_uri,
                parser_version=parser_version,
                file_sha256=None,
                page_count=None,
            )

        return IngestSummary(
            status="completed",
            project_id=project_id,
            document_id=document_id,
            version_id=version_id,
            chunks_imported=int(ingest_result.get("total_chunks", 0)),
            source_uri=source_uri,
            warnings=[],
        )

    def evaluate_citations(
        self,
        conn: Any,
        citation_ids: Iterable[str],
    ) -> CitationEvaluationSummary:
        total = 0
        verified = 0
        unverifiable = 0
        warning_codes: list[str] = []
        seen_codes: set[str] = set()

        for citation_id in citation_ids:
            total += 1
            verification = self._verify_citation_fn(conn, citation_id=citation_id)
            assessment = self._verifier.assess(verification)

            if assessment.evidence_status == "verified":
                verified += 1
            else:
                unverifiable += 1
                for warn in assessment.warnings:
                    _append_warning_code(warn, warning_codes, seen_codes)

        return CitationEvaluationSummary(
            status="completed",
            total_citations=total,
            verified_citations=verified,
            unverifiable_citations=unverifiable,
            warning_codes=warning_codes,
        )


def _append_warning_code(
    warning: EvidenceWarning, warning_codes: list[str], seen_codes: set[str]
) -> None:
    if warning.code in seen_codes:
        return
    seen_codes.add(warning.code)
    warning_codes.append(warning.code)
