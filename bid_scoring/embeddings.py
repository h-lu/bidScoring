# bid_scoring/embeddings.py
from openai import OpenAI
from bid_scoring.config import load_settings


def embed_texts(texts: list[str], client: OpenAI | None = None, model: str | None = None):
    settings = load_settings()
    if client is None:
        timeout = settings["OPENAI_TIMEOUT"] or None
        max_retries = settings["OPENAI_MAX_RETRIES"] or None
        client = OpenAI(
            api_key=settings["OPENAI_API_KEY"],
            base_url=settings["OPENAI_BASE_URL"],
            timeout=timeout,
            max_retries=max_retries,
        )
    if model is None:
        model = settings["OPENAI_EMBEDDING_MODEL"]
    resp = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in resp.data]
