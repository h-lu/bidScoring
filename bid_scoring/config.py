# bid_scoring/config.py
import os
from dotenv import load_dotenv


def _load_task_models(prefix: str, default_key: str) -> dict:
    models: dict[str, str] = {}
    for key, value in os.environ.items():
        if key.startswith(prefix) and key != default_key and value:
            task = key[len(prefix):].lower()
            models[task] = value
    return models


def load_settings() -> dict:
    load_dotenv()
    default_llm = os.getenv("OPENAI_LLM_MODEL_DEFAULT") or os.getenv("OPENAI_LLM_MODEL")
    return {
        "DATABASE_URL": os.getenv("DATABASE_URL", "postgresql://localhost:5432/bid_scoring"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "OPENAI_BASE_URL": os.getenv("OPENAI_BASE_URL"),
        "OPENAI_LLM_MODEL_DEFAULT": default_llm,
        "OPENAI_LLM_MODELS": _load_task_models("OPENAI_LLM_MODEL_", "OPENAI_LLM_MODEL_DEFAULT"),
        "OPENAI_TIMEOUT": float(os.getenv("OPENAI_TIMEOUT", "0") or 0),
        "OPENAI_MAX_RETRIES": int(os.getenv("OPENAI_MAX_RETRIES", "0") or 0),
        "OPENAI_EMBEDDING_MODEL": os.getenv("OPENAI_EMBEDDING_MODEL"),
        "OPENAI_EMBEDDING_DIM": int(os.getenv("OPENAI_EMBEDDING_DIM", "0")),
    }
