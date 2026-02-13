from __future__ import annotations

from typing import Any, Callable

from psycopg.rows import dict_row


class ChunkRetriever:
    """Retrieve candidate chunks for dimension analysis."""

    def __init__(self, conn: Any, limit_per_dimension: int = 50) -> None:
        self._conn = conn
        self._limit_per_dimension = int(limit_per_dimension)

    def search_chunks(self, version_id: str, keywords: list[str]) -> list[dict[str, Any]]:
        with self._conn.cursor(row_factory=dict_row) as cur:
            conditions = " OR ".join(["text_raw ILIKE %s"] * len(keywords))
            params = [f"%{kw}%" for kw in keywords]

            cur.execute(
                f"""
                SELECT chunk_id, page_idx, text_raw, bbox, element_type
                FROM chunks
                WHERE version_id = %s
                AND ({conditions})
                LIMIT {self._limit_per_dimension}
                """,
                [version_id] + params,
            )
            return cur.fetchall()

    @staticmethod
    def collect_evidence_warnings(chunks: list[dict[str, Any]]) -> list[str]:
        warnings: list[str] = []
        for chunk in chunks:
            bbox = chunk.get("bbox")
            if not bbox:
                warnings.append("missing_bbox")
        if warnings:
            return sorted(set(warnings))
        return []


class McpChunkRetriever:
    """Retriever adapter backed by retrieval MCP implementation."""

    def __init__(
        self,
        *,
        retrieve_fn: Callable[..., dict[str, Any]] | None = None,
        top_k: int = 50,
        mode: str = "hybrid",
    ) -> None:
        self._retrieve_fn = retrieve_fn
        self._top_k = int(top_k)
        self._mode = mode
        self._last_retrieval_warnings: list[str] = []

    def search_chunks(self, version_id: str, keywords: list[str]) -> list[dict[str, Any]]:
        retrieve_fn = self._retrieve_fn or _default_retrieve_fn
        query = " ".join(dict.fromkeys(keywords))

        response = retrieve_fn(
            version_id=version_id,
            query=query,
            top_k=self._top_k,
            mode=self._mode,
            keywords=keywords,
            include_text=True,
            include_diagnostics=False,
        )
        self._last_retrieval_warnings = list(response.get("warnings", []))

        chunks: list[dict[str, Any]] = []
        for item in response.get("results", []):
            evidence_status = item.get("evidence_status")
            is_verifiable = _is_verifiable_for_scoring(item)
            chunks.append(
                {
                    "chunk_id": item.get("chunk_id"),
                    "page_idx": item.get("page_idx"),
                    "text_raw": item.get("text", ""),
                    "bbox": item.get("bbox"),
                    "element_type": item.get("element_type"),
                    "warnings": list(item.get("warnings", [])),
                    "evidence_status": evidence_status,
                    "is_verifiable": is_verifiable,
                }
            )

        return chunks

    def collect_evidence_warnings(self, chunks: list[dict[str, Any]]) -> list[str]:
        warnings = set(self._last_retrieval_warnings)

        for chunk in chunks:
            for code in chunk.get("warnings", []):
                warnings.add(code)

            if chunk.get("bbox") is None:
                warnings.add("missing_bbox")
            if chunk.get("evidence_status") not in {"verified", "verified_with_warnings"}:
                warnings.add("unverifiable_evidence_for_scoring")

        return sorted(warnings)


def _is_verifiable_for_scoring(item: dict[str, Any]) -> bool:
    evidence_status = item.get("evidence_status")
    if evidence_status not in {"verified", "verified_with_warnings"}:
        return False

    bbox = item.get("bbox")
    return isinstance(bbox, list) and len(bbox) == 4


def _default_retrieve_fn(**kwargs: Any) -> dict[str, Any]:
    from mcp_servers.retrieval_server import retrieve_impl

    return retrieve_impl(**kwargs)
