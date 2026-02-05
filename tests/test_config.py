from bid_scoring.config import load_settings


def test_load_settings_from_env(monkeypatch):
    """测试从环境变量加载配置"""
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("OPENAI_EMBEDDING_DIM", "1536")
    settings = load_settings()
    assert settings["OPENAI_BASE_URL"] == "https://api.openai.com/v1"
    assert settings["OPENAI_EMBEDDING_MODEL"] == "text-embedding-3-small"
    assert settings["OPENAI_EMBEDDING_DIM"] == 1536
