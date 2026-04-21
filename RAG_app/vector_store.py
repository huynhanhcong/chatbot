import uuid

from .models import RagDocument, SearchResult


class QdrantVectorStore:
    def __init__(self, url: str, collection: str, path: str | None = None):
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise RuntimeError("Missing dependency: install qdrant-client from requirements-rag.txt.") from exc

        self.collection = collection
        self.client = QdrantClient(path=path) if path else QdrantClient(url=url)

    def recreate_collection(self, vector_size: int) -> None:
        from qdrant_client import models

        self.client.recreate_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )

    def collection_exists(self) -> bool:
        try:
            self.client.get_collection(collection_name=self.collection)
        except Exception:
            return False
        return True

    def upload(self, documents: list[RagDocument], vectors: list[list[float]], batch_size: int = 64) -> None:
        from qdrant_client import models

        points = [
            models.PointStruct(
                id=_point_id(doc.id),
                vector=vector,
                payload=doc.payload,
            )
            for doc, vector in zip(documents, vectors, strict=True)
        ]
        for start in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=self.collection,
                points=points[start : start + batch_size],
                wait=True,
            )

    def search(self, query_vector: list[float], limit: int = 30) -> list[SearchResult]:
        response = self._query_points(query_vector=query_vector, limit=limit)
        points = getattr(response, "points", response)
        results: list[SearchResult] = []
        for point in points:
            payload = getattr(point, "payload", {}) or {}
            score = float(getattr(point, "score", 0.0) or 0.0)
            results.append(
                SearchResult(
                    id=payload.get("entity_id", str(getattr(point, "id", ""))),
                    parent_id=payload.get("parent_id"),
                    score=score,
                    vector_score=score,
                    title=payload.get("title", ""),
                    entity_type=payload.get("entity_type", ""),
                    category=payload.get("category"),
                    chunk_type=payload.get("chunk_type"),
                    price_vnd=payload.get("price_vnd"),
                    source_url=payload.get("source_url"),
                    text=payload.get("text", ""),
                    payload=payload,
                )
            )
        return results

    def count(self) -> int:
        result = self.client.count(collection_name=self.collection, exact=True)
        return int(getattr(result, "count", 0))

    def _query_points(self, query_vector: list[float], limit: int):
        if hasattr(self.client, "query_points"):
            return self.client.query_points(
                collection_name=self.collection,
                query=query_vector,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
        return self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )


def _point_id(entity_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"hanhphuc:{entity_id}"))
