import pathlib
import psycopg
from bid_scoring.config import load_settings


def validate_embedding_dim(dim: int | None) -> int:
    if not dim:
        raise ValueError("OPENAI_EMBEDDING_DIM is required and must be 1536")
    if dim != 1536:
        raise ValueError("OPENAI_EMBEDDING_DIM must be 1536 for current schema")
    return dim


def main():
    settings = load_settings()
    dsn = settings["DATABASE_URL"]
    # 验证嵌入维度配置正确
    validate_embedding_dim(settings["OPENAI_EMBEDDING_DIM"])
    template = pathlib.Path("migrations/000_init.sql").read_text(encoding="utf-8")
    sql = template  # 000_init.sql 已包含固定的 1536 维向量定义，无需替换
    # 使用 psycopg.sql 模块执行整个 SQL 脚本，正确处理函数定义
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()


if __name__ == "__main__":
    main()
