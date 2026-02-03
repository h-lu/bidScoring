# tests/test_config.py
from bid_scoring.config import load_settings


def test_load_settings_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENAI_LLM_MODEL_DEFAULT", "gpt-4o-mini")
    monkeypatch.setenv("OPENAI_LLM_MODEL_SCORING", "gpt-4o")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("OPENAI_EMBEDDING_DIM", "1536")
    settings = load_settings()
    assert settings["OPENAI_BASE_URL"] == "https://api.openai.com/v1"
    assert settings["OPENAI_LLM_MODEL_DEFAULT"] == "gpt-4o-mini"
    assert settings["OPENAI_LLM_MODELS"]["scoring"] == "gpt-4o"
    assert settings["OPENAI_EMBEDDING_MODEL"] == "text-embedding-3-small"
    assert settings["OPENAI_EMBEDDING_DIM"] == 1536
