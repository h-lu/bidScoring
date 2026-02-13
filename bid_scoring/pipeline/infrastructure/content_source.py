from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from bid_scoring.pipeline.application.e2e_service import E2ERunRequest, LoadedContent
from bid_scoring.pipeline.infrastructure.mineru_adapter import (
    load_content_list_from_output,
    parse_pdf_with_mineru,
    resolve_content_list_path,
)


class ContextListSource:
    """Loads content list directly from JSON file (bypass MinerU)."""

    def load(self, request: E2ERunRequest) -> LoadedContent:
        path = request.content_list_path
        if path is None:
            raise ValueError("content_list_path is required for ContextListSource")

        payload = _load_json_list(path)
        return LoadedContent(
            content_list=payload,
            source_uri=request.source_uri or str(path),
            parser_version=request.parser_version or "context-list-v1",
            warnings=["mineru_bypassed"],
        )


class MinerUOutputSource:
    """Loads MinerU output from output directory containing content_list.json."""

    def load(self, request: E2ERunRequest) -> LoadedContent:
        output_dir = request.mineru_output_dir
        if output_dir is None:
            raise ValueError("mineru_output_dir is required for MinerUOutputSource")

        payload = load_content_list_from_output(output_dir)
        warnings: list[str] = []
        if not payload:
            warnings.append("empty_content_list")

        return LoadedContent(
            content_list=payload,
            source_uri=request.source_uri or str(output_dir / "content_list.json"),
            parser_version=request.parser_version or "mineru-output-v1",
            warnings=warnings,
        )


class PdfMinerUAdapter:
    """Direct PDF -> MinerU execution adapter."""

    def __init__(
        self,
        parse_pdf_fn: Callable[..., Path] | None = None,
    ) -> None:
        self._parse_pdf_fn = parse_pdf_fn or parse_pdf_with_mineru

    def load(self, request: E2ERunRequest) -> LoadedContent:
        pdf_path = request.pdf_path
        if pdf_path is None:
            raise ValueError("pdf_path is required for PdfMinerUAdapter")

        output_dir = self._parse_pdf_fn(
            pdf_path,
            parser_mode=request.mineru_parser,
        )
        payload = load_content_list_from_output(output_dir)
        warnings: list[str] = []
        if not payload:
            warnings.append("empty_content_list")

        resolved_content_path = resolve_content_list_path(output_dir)
        return LoadedContent(
            content_list=payload,
            source_uri=request.source_uri
            or str(resolved_content_path or (output_dir / "content_list.json")),
            parser_version=request.parser_version or "mineru-direct-v1",
            warnings=warnings,
        )


class AutoContentSource:
    """Selects content loader based on request fields."""

    def __init__(
        self,
        *,
        context_source: ContextListSource | None = None,
        mineru_output_source: MinerUOutputSource | None = None,
        pdf_source: PdfMinerUAdapter | None = None,
    ) -> None:
        self._context_source = context_source or ContextListSource()
        self._mineru_output_source = mineru_output_source or MinerUOutputSource()
        self._pdf_source = pdf_source or PdfMinerUAdapter()

    def load(self, request: E2ERunRequest) -> LoadedContent:
        if request.content_list_path is not None:
            return self._context_source.load(request)
        if request.mineru_output_dir is not None:
            return self._mineru_output_source.load(request)
        if request.pdf_path is not None:
            return self._pdf_source.load(request)
        raise ValueError(
            "One input is required: --context-list/--content-list, "
            "--mineru-output-dir, or --pdf-path"
        )


def _load_json_list(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must be a JSON list")
    return payload
