# bid_scoring/config.py
import os
from dotenv import load_dotenv


def load_settings() -> dict:
    """加载应用配置

    Returns:
        配置字典，包含数据库和 OpenAI API 相关配置
    """
    load_dotenv(override=True)  # Allow .env to override existing env vars
    return {
        "DATABASE_URL": os.getenv(
            "DATABASE_URL", "postgresql://localhost:5432/bid_scoring"
        ),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "OPENAI_BASE_URL": os.getenv("OPENAI_BASE_URL"),
        "OPENAI_TIMEOUT": float(os.getenv("OPENAI_TIMEOUT", "0") or 0),
        "OPENAI_MAX_RETRIES": int(os.getenv("OPENAI_MAX_RETRIES", "0") or 0),
        "OPENAI_EMBEDDING_MODEL": os.getenv("OPENAI_EMBEDDING_MODEL"),
        "OPENAI_EMBEDDING_DIM": int(os.getenv("OPENAI_EMBEDDING_DIM", "0")),
    }
