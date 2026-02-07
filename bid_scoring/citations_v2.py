from __future__ import annotations

import hashlib
import json
from typing import Any

from bid_scoring.anchors_v2 import canonical_json


def _as_obj(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return json.loads(value)
    return value


def compute_evidence_hash(*, quote_text: str, unit_hash: str, anchor_json: Any) -> str:
    """Compute a stable evidence hash for v0.2 citations.

    Contract:
    - Evidence must remain verifiable even if the index layer (chunks) is rebuilt.
    - Hash binds: quote_text + unit_hash + anchor_json.
    """
    payload = "\n".join(
        [
            quote_text or "",
            unit_hash or "",
            canonical_json(_as_obj(anchor_json) or {}),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_citation(conn, *, citation_id: str) -> dict[str, Any]:
    """Verify a citation against the current normalized content_units truth."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                c.citation_id::text,
                c.unit_id::text,
                c.quote_text,
                c.evidence_hash,
                c.anchor_json,
                cu.unit_hash,
                cu.anchor_json
            FROM citations c
            JOIN content_units cu ON cu.unit_id = c.unit_id
            WHERE c.citation_id = %s
            """,
            (citation_id,),
        )
        row = cur.fetchone()

    if not row:
        return {"ok": False, "reason": "not_found", "citation_id": citation_id}

    (
        _cid,
        unit_id,
        quote_text,
        evidence_hash,
        citation_anchor_json,
        unit_hash,
        unit_anchor_json,
    ) = row

    expected = compute_evidence_hash(
        quote_text=quote_text or "",
        unit_hash=unit_hash or "",
        anchor_json=unit_anchor_json,
    )
    hash_ok = bool(evidence_hash) and evidence_hash == expected

    citation_anchor_obj = _as_obj(citation_anchor_json)
    unit_anchor_obj = _as_obj(unit_anchor_json)
    anchor_ok = citation_anchor_obj is None or canonical_json(
        citation_anchor_obj
    ) == canonical_json(unit_anchor_obj)

    return {
        "ok": bool(hash_ok and anchor_ok),
        "citation_id": citation_id,
        "unit_id": unit_id,
        "hash_ok": hash_ok,
        "anchor_ok": anchor_ok,
        "expected_evidence_hash": expected,
        "actual_evidence_hash": evidence_hash,
    }
