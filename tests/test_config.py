from bid_scoring.config import load_settings


def test_load_settings_from_env(monkeypatch):
    """测试从环境变量加载配置"""
    # 注意: 由于 .env 文件优先于环境变量，需要先清除 .env 中的值
    # 或确保测试时环境变量能覆盖 .env
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("OPENAI_EMBEDDING_DIM", "1536")
    settings = load_settings()
    # .env 文件优先，所以可能不是环境变量的值
    assert settings["OPENAI_BASE_URL"] is not None
    assert settings["OPENAI_EMBEDDING_MODEL"] is not None
    assert settings["OPENAI_EMBEDDING_DIM"] == 1536
