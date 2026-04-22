from .bm25_store import BM25Store
from .config import Settings
from .data_loader import load_hanhphuc_chunk_documents, load_hanhphuc_documents
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
    entity_stats = _embed_and_upload(
        documents=documents,
        embedder=embedder,
        settings=settings,
        collection=settings.qdrant_collection,
        recreate=recreate,
        batch_size=batch_size,
    )

    chunk_stats = {}
    if settings.chunks_path.exists():
        chunk_documents = load_hanhphuc_chunk_documents(settings.chunks_path)
        chunk_bm25 = BM25Store.from_documents(chunk_documents)
        chunk_bm25.save(settings.chunk_bm25_path)
        chunk_stats = _embed_and_upload(
            documents=chunk_documents,
            embedder=embedder,
            settings=settings,
            collection=settings.chunk_qdrant_collection,
            recreate=recreate,
            batch_size=batch_size,
        )

    return {
        "documents": len(documents),
        "vectors": entity_stats["vectors"],
        "vector_size": entity_stats["vector_size"],
        "qdrant_collection": settings.qdrant_collection,
        "qdrant_count": entity_stats["qdrant_count"],
        "bm25_path": str(settings.bm25_path),
        "chunk_documents": chunk_stats.get("documents", 0),
        "chunk_vectors": chunk_stats.get("vectors", 0),
        "chunk_qdrant_collection": settings.chunk_qdrant_collection,
        "chunk_qdrant_count": chunk_stats.get("qdrant_count", 0),
        "chunk_bm25_path": str(settings.chunk_bm25_path),
    }


def _embed_and_upload(
    *,
    documents: list,
    embedder: GeminiEmbedder,
    settings: Settings,
    collection: str,
    recreate: bool,
    batch_size: int,
) -> dict:
    vector_store = QdrantVectorStore(
        settings.qdrant_url,
        collection,
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
    qdrant_count = vector_store.count()
    close = getattr(vector_store.client, "close", None)
    if callable(close):
        close()
    return {
        "documents": len(documents),
        "vectors": len(vectors),
        "vector_size": len(vectors[0]),
        "qdrant_count": qdrant_count,
    }
