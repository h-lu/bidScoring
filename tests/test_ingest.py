import json
from pathlib import Path
import psycopg
from bid_scoring.config import load_settings
from bid_scoring.ingest import (
    ingest_content_list,
    _extract_text_from_item,
    _prepare_chunk_data,
)


def test_ingest_inserts_chunks():
    data = json.loads(
        Path("tests/fixtures/sample_content_list.json").read_text(encoding="utf-8")
    )
    dsn = load_settings()["DATABASE_URL"]
    project_id = "00000000-0000-0000-0000-000000000001"
    document_id = "00000000-0000-0000-0000-000000000003"
    version_id = "00000000-0000-0000-0000-000000000002"
    with psycopg.connect(dsn) as conn:
        ingest_content_list(
            conn,
            project_id=project_id,
            document_id=document_id,
            version_id=version_id,
            document_title="示例文档",
            source_type="mineru",
            content_list=data,
        )
        with conn.cursor() as cur:
            cur.execute(
                "select count(*) from documents where doc_id = %s", (document_id,)
            )
            assert cur.fetchone()[0] == 1
            cur.execute(
                "select count(*) from document_versions where version_id = %s",
                (version_id,),
            )
            assert cur.fetchone()[0] == 1
            cur.execute("select count(*) from chunks")
            assert cur.fetchone()[0] > 0


def test_extract_text_from_item_accepts_string_captions():
    item = {
        "type": "image",
        "image_caption": "Figure 1: Sample",
        "image_footnote": "Credit: Author",
    }
    text = _extract_text_from_item(item)
    assert "Figure 1: Sample" in text
    assert "Credit: Author" in text


def test_prepare_chunk_data_normalizes_caption_lists():
    item = {
        "type": "table",
        "table_caption": "Table 1: Caption",
        "table_footnote": "Footnote",
        "table_body": "<tr><td>Cell</td></tr>",
    }
    data = _prepare_chunk_data(item, 0)
    assert data["table_caption"] == ["Table 1: Caption"]
    assert data["table_footnote"] == ["Footnote"]
