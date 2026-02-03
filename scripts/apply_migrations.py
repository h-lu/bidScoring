import pathlib
import psycopg
from bid_scoring.config import load_settings


def main():
    settings = load_settings()
    dsn = settings["DATABASE_URL"]
    dim = settings["OPENAI_EMBEDDING_DIM"]
    if not dim:
        raise ValueError("OPENAI_EMBEDDING_DIM is required")
    template = pathlib.Path("migrations/001_init.sql").read_text(encoding="utf-8")
    sql = template.replace("{{EMBEDDING_DIM}}", str(dim))
    statements = [s.strip() for s in sql.split(";") if s.strip()]
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            for stmt in statements:
                cur.execute(stmt)
        conn.commit()


if __name__ == "__main__":
    main()
