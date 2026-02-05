import pathlib
import psycopg
from bid_scoring.config import load_settings


def main():
    settings = load_settings()
    dsn = settings["DATABASE_URL"]
    dim = settings["OPENAI_EMBEDDING_DIM"]
    if not dim:
        raise ValueError("OPENAI_EMBEDDING_DIM is required")
    template = pathlib.Path("migrations/000_init.sql").read_text(encoding="utf-8")
    sql = template  # 000_init.sql 已包含固定的 1536 维向量定义，无需替换
    # 使用 psycopg.sql 模块执行整个 SQL 脚本，正确处理函数定义
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()


if __name__ == "__main__":
    main()
