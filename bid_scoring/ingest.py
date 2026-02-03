import hashlib
import json


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ingest_content_list(
    conn,
    project_id: str,
    document_id: str,
    version_id: str,
    content_list: list[dict],
    document_title: str = "untitled",
    source_type: str = "mineru",
    source_uri: str | None = None,
    parser_version: str | None = None,
    status: str = "ready",
) -> None:
    rows = []
    for i, item in enumerate(content_list):
        if item.get("type") not in ["text", "table"]:
            continue
        text = (item.get("text") or "").strip()
        if not text:
            continue
        rows.append((
            project_id,
            version_id,
            f"chunk_{i:04d}",
            i,
            item.get("page_idx", 0),
            json.dumps(item.get("bbox", [])),
            item.get("type"),
            text,
            _hash_text(text),
        ))
    with conn.cursor() as cur:
        cur.execute(
            "insert into projects (project_id, name) values (%s, %s) on conflict do nothing",
            (project_id, f"project-{project_id[:8]}"),
        )
        cur.execute(
            "insert into documents (doc_id, project_id, title, source_type) values (%s, %s, %s, %s) on conflict do nothing",
            (document_id, project_id, document_title, source_type),
        )
        cur.execute(
            """
            insert into document_versions (version_id, doc_id, source_uri, source_hash, parser_version, status)
            values (%s, %s, %s, %s, %s, %s)
            on conflict do nothing
            """,
            (version_id, document_id, source_uri, None, parser_version, status),
        )
        cur.executemany(
            """
            insert into chunks (
              chunk_id, project_id, version_id, source_id, chunk_index, page_idx,
              bbox, element_type, text_raw, text_hash, text_tsv
            )
            values (gen_random_uuid(), %s, %s, %s, %s, %s, %s, %s, %s, %s, to_tsvector('simple', %s))
            """,
            [(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[7]) for r in rows],
        )
    conn.commit()
