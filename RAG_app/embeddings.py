from collections.abc import Iterable


class GeminiEmbedder:
    def __init__(self, api_key: str | None, model: str):
        if not api_key:
            raise RuntimeError("Missing GEMINI_API_KEY. Set it in environment or .env.")
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError("Missing dependency: install google-genai from requirements-rag.txt.") from exc

        self.model = model
        self.client = genai.Client(api_key=api_key)

    def embed_texts(self, texts: Iterable[str]) -> list[list[float]]:
        texts = list(texts)
        if not texts:
            return []
        result = self.client.models.embed_content(model=self.model, contents=texts)
        embeddings = getattr(result, "embeddings", result)
        return [_embedding_values(item) for item in embeddings]

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


def _embedding_values(item: object) -> list[float]:
    if isinstance(item, dict):
        values = item.get("values") or item.get("embedding") or item.get("value")
    else:
        values = getattr(item, "values", None) or getattr(item, "embedding", None)
    if values is None:
        raise RuntimeError(f"Cannot extract embedding values from: {type(item)!r}")
    return [float(value) for value in values]
