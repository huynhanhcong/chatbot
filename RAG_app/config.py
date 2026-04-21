import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    root: Path
    entities_path: Path
    chunks_path: Path
    qa_path: Path
    index_dir: Path
    bm25_path: Path
    chunk_bm25_path: Path
    qdrant_url: str
    qdrant_path: Path | None
    qdrant_collection: str
    chunk_qdrant_collection: str
    gemini_api_key: str | None
    gemini_generation_model: str
    gemini_embedding_model: str
    redis_url: str | None
    use_redis: bool
    rag_use_chunk_index: bool
    router_llm_enabled: bool
    reranker_provider: str
    cache_ttl_seconds: int


def load_settings() -> Settings:
    load_dotenv(ROOT / ".env")

    index_dir = ROOT / "Data_RAG" / "index"
    return Settings(
        root=ROOT,
        entities_path=ROOT / "Data_RAG" / "entities" / "hanhphuc_entities.jsonl",
        chunks_path=ROOT / "Data_RAG" / "retrieval" / "hanhphuc_retrieval_chunks.jsonl",
        qa_path=ROOT / "Data_RAG" / "qa" / "hanhphuc_rag_qa_200.jsonl",
        index_dir=index_dir,
        bm25_path=index_dir / "hanhphuc_bm25.json",
        chunk_bm25_path=index_dir / "hanhphuc_chunks_bm25.json",
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        qdrant_path=Path(os.environ["QDRANT_PATH"]) if os.getenv("QDRANT_PATH") else None,
        qdrant_collection=os.getenv("QDRANT_COLLECTION", "hanhphuc_hospital_rag"),
        chunk_qdrant_collection=os.getenv(
            "CHUNK_QDRANT_COLLECTION",
            "hanhphuc_hospital_rag_chunks",
        ),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        gemini_generation_model=os.getenv("GEMINI_GENERATION_MODEL", "gemini-2.5-flash"),
        gemini_embedding_model=os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"),
        redis_url=os.getenv("REDIS_URL"),
        use_redis=_env_bool("USE_REDIS", default=False),
        rag_use_chunk_index=_env_bool("RAG_USE_CHUNK_INDEX", default=False),
        router_llm_enabled=_env_bool("ROUTER_LLM_ENABLED", default=False),
        reranker_provider=os.getenv("RERANKER_PROVIDER", "llm"),
        cache_ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", "900")),
    )


def _env_bool(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
