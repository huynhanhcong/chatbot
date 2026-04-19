import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    root: Path
    entities_path: Path
    qa_path: Path
    index_dir: Path
    bm25_path: Path
    qdrant_url: str
    qdrant_path: Path | None
    qdrant_collection: str
    gemini_api_key: str | None
    gemini_generation_model: str
    gemini_embedding_model: str


def load_settings() -> Settings:
    load_dotenv(ROOT / ".env")

    index_dir = ROOT / "Data_RAG" / "index"
    return Settings(
        root=ROOT,
        entities_path=ROOT / "Data_RAG" / "entities" / "hanhphuc_entities.jsonl",
        qa_path=ROOT / "Data_RAG" / "qa" / "hanhphuc_rag_qa_200.jsonl",
        index_dir=index_dir,
        bm25_path=index_dir / "hanhphuc_bm25.json",
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        qdrant_path=Path(os.environ["QDRANT_PATH"]) if os.getenv("QDRANT_PATH") else None,
        qdrant_collection=os.getenv("QDRANT_COLLECTION", "hanhphuc_hospital_rag"),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        gemini_generation_model=os.getenv("GEMINI_GENERATION_MODEL", "gemini-2.5-flash"),
        gemini_embedding_model=os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"),
    )
