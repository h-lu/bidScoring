from __future__ import annotations

import json
import logging
from typing import Dict, List

from .types import EvidenceUnit, MergedChunk, RetrievalResult

logger = logging.getLogger(__name__)


def fetch_chunks(
    retriever: object, merged_results: List[MergedChunk]
) -> List[RetrievalResult]:
    """Fetch full chunk rows and attach v0.2 unit-level evidence when available."""
    if not merged_results:
        return []

    chunk_ids = [doc_id for doc_id, _, _ in merged_results]
    scores_dict = {
        doc_id: (rrf_score, sources) for doc_id, rrf_score, sources in merged_results
    }

    try:
        with retriever._get_connection() as conn:  # type: ignore[attr-defined]
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        c.chunk_id::text, c.text_raw, c.page_idx, c.embedding,
                        c.bbox, c.element_type, dp.coord_sys
                    FROM chunks c
                    LEFT JOIN document_pages dp
                        ON c.version_id = dp.version_id AND c.page_idx = dp.page_idx
                    WHERE c.chunk_id = ANY(%s::uuid[])
                    """,
                    (chunk_ids,),
                )
                rows = {row[0]: row for row in cur.fetchall()}

                evidence_by_chunk: Dict[str, List[EvidenceUnit]] = {}
                try:
                    cur.execute(
                        """
                        SELECT
                            s.chunk_id::text,
                            cu.unit_id::text,
                            cu.unit_index,
                            cu.unit_type,
                            cu.text_raw,
                            cu.anchor_json,
                            s.unit_order,
                            s.start_char,
                            s.end_char
                        FROM chunk_unit_spans s
                        JOIN content_units cu ON cu.unit_id = s.unit_id
                        WHERE s.chunk_id = ANY(%s::uuid[])
                        ORDER BY s.chunk_id, s.unit_order
                        """,
                        (chunk_ids,),
                    )
                    for (
                        chunk_id,
                        unit_id,
                        unit_index,
                        unit_type,
                        text_raw,
                        anchor_json,
                        unit_order,
                        start_char,
                        end_char,
                    ) in cur.fetchall():
                        if isinstance(anchor_json, str):
                            anchor_json = json.loads(anchor_json)

                        evidence_by_chunk.setdefault(chunk_id, []).append(
                            EvidenceUnit(
                                unit_id=str(unit_id),
                                unit_index=int(unit_index),
                                unit_type=str(unit_type),
                                text=text_raw or "",
                                anchor_json=anchor_json,
                                unit_order=int(unit_order or 0),
                                start_char=int(start_char)
                                if start_char is not None
                                else None,
                                end_char=int(end_char)
                                if end_char is not None
                                else None,
                            )
                        )
                except Exception as e:
                    logger.debug(
                        "Skip unit evidence attachment (v0.2 tables missing?): %s",
                        e,
                        exc_info=True,
                    )

                results: List[RetrievalResult] = []
                for chunk_id in chunk_ids:
                    if chunk_id not in rows:
                        continue

                    row = rows[chunk_id]
                    rrf_score, sources = scores_dict[chunk_id]

                    source_types = list(sources.keys())
                    if len(source_types) == 2:
                        source = "hybrid"
                    elif "vector" in source_types:
                        source = "vector"
                    elif "keyword" in source_types:
                        source = "keyword"
                    else:
                        source = "unknown"

                    vector_score = sources.get("vector", {}).get("score")
                    keyword_score = sources.get("keyword", {}).get("score")

                    results.append(
                        RetrievalResult(
                            chunk_id=row[0],
                            text=row[1] or "",
                            page_idx=row[2] or 0,
                            score=rrf_score,
                            source=source,
                            vector_score=vector_score,
                            keyword_score=keyword_score,
                            embedding=row[3] if row[3] else None,
                            evidence_units=evidence_by_chunk.get(chunk_id, []),
                            bbox=row[4] if row[4] else None,
                            element_type=row[5] if row[5] else None,
                            coord_system=row[6] if row[6] else None,
                        )
                    )

                return results
    except Exception as e:
        logger.error("Failed to fetch chunks: %s", e, exc_info=True)
        return []
