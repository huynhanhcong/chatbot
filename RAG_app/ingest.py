from .bm25_store import BM25Store
from .config import Settings
from .data_loader import load_hanhphuc_documents
from .embeddings import GeminiEmbedder
from .vector_store import QdrantVectorStore


def ingest_hanhphuc(settings: Settings, recreate: bool = False, batch_size: int = 16) -> dict:
    documents = load_hanhphuc_documents(settings.entities_path)
    if len(documents) != 102:
        raise ValueError(f"Expected 102 Hanh Phuc documents, got {len(documents)}")

    settings.index_dir.mkdir(parents=True, exist_ok=True)
    bm25 = BM25Store.from_documents(documents)
    bm25.save(settings.bm25_path)

    embedder = GeminiEmbedder(settings.gemini_api_key, settings.gemini_embedding_model)
    vector_store = QdrantVectorStore(
        settings.qdrant_url,
        settings.qdrant_collection,
        path=str(settings.qdrant_path) if settings.qdrant_path else None,
    )

    vectors: list[list[float]] = []
    for start in range(0, len(documents), batch_size):
        batch = documents[start : start + batch_size]
        vectors.extend(embedder.embed_texts([doc.text for doc in batch]))

    if not vectors:
        raise RuntimeError("No vectors generated.")
    if recreate or not vector_store.collection_exists():
        vector_store.recreate_collection(vector_size=len(vectors[0]))

    vector_store.upload(documents, vectors)
    return {
        "documents": len(documents),
        "vectors": len(vectors),
        "vector_size": len(vectors[0]),
        "qdrant_collection": settings.qdrant_collection,
        "qdrant_count": vector_store.count(),
        "bm25_path": str(settings.bm25_path),
    }
