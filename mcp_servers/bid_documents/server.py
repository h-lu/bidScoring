# mcp_servers/bid_documents/server.py
from fastmcp import FastMCP
import psycopg
from pgvector.psycopg import register_vector
from bid_scoring.config import load_settings
from bid_scoring.search import rrf_fuse
from bid_scoring.embeddings import embed_texts

mcp = FastMCP("bid-documents")


@mcp.tool()
def search_chunks(query: str, document_id: str, top_k: int = 10, filters: dict | None = None):
    dsn = load_settings()["DATABASE_URL"]
    filters = filters or {}
    version_id = filters.get("version_id")
    with psycopg.connect(dsn) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            # 可按索引类型调参（择一）
            # cur.execute("SET LOCAL hnsw.ef_search = 64")
            # cur.execute("SET LOCAL ivfflat.probes = 10")
            if version_id is None:
                cur.execute(
                    """
                    select version_id
                    from document_versions
                    where doc_id = %s
                    order by created_at desc
                    limit 1
                    """,
                    (document_id,),
                )
                row = cur.fetchone()
                if row is None:
                    return {"query": query, "results": []}
                version_id = row[0]
            query_vec = embed_texts([query])[0]
            cur.execute(
                """
                select source_id, text_raw, page_idx, bbox
                from chunks
                where version_id = %s
                  and text_tsv @@ plainto_tsquery('simple', %s)
                limit %s
                """,
                (version_id, query, top_k * 2),
            )
            bm25_rows = cur.fetchall()
            bm25 = [(r[0], i) for i, r in enumerate(bm25_rows)]
            cur.execute(
                """
                select source_id, text_raw, page_idx, bbox
                from chunks
                where version_id = %s
                order by embedding <-> %s
                limit %s
                """,
                (version_id, query_vec, top_k * 2),
            )
            vec_rows = cur.fetchall()
            vec = [(r[0], i) for i, r in enumerate(vec_rows)]
            fused_ids = rrf_fuse(bm25, vec)
    return {"query": query, "results": fused_ids[:top_k]}


@mcp.tool()
def get_document_info(document_id: str):
    dsn = load_settings()["DATABASE_URL"]
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "select doc_id, title, source_type from documents where doc_id = %s",
                (document_id,),
            )
            row = cur.fetchone()
            if row is None:
                return {"found": False}
            return {"found": True, "doc_id": row[0], "title": row[1], "source_type": row[2]}


@mcp.tool()
def get_page_content(document_id: str, page_idx: int):
    dsn = load_settings()["DATABASE_URL"]
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                select c.text_raw, c.bbox
                from chunks c
                join document_versions dv on c.version_id = dv.version_id
                where dv.doc_id = %s and c.page_idx = %s
                limit 200
                """,
                (document_id, page_idx),
            )
            return [{"text": r[0], "bbox": r[1]} for r in cur.fetchall()]


if __name__ == "__main__":
    mcp.run()
