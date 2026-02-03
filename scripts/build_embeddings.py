# scripts/build_embeddings.py
import psycopg
from pgvector.psycopg import register_vector
from bid_scoring.config import load_settings
from bid_scoring.embeddings import embed_texts


def main():
    dsn = load_settings()["DATABASE_URL"]
    with psycopg.connect(dsn) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("select chunk_id, text_raw from chunks where embedding is null limit 200")
            rows = cur.fetchall()
        if not rows:
            return
        ids = [r[0] for r in rows]
        texts = [r[1] for r in rows]
        vecs = embed_texts(texts)
        with conn.cursor() as cur:
            cur.executemany(
                "update chunks set embedding = %s where chunk_id = %s",
                [(v, i) for v, i in zip(vecs, ids)]
            )
        conn.commit()


if __name__ == "__main__":
    main()
