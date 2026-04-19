from pathlib import Path

from .config import Settings
from .data_loader import read_jsonl
from .pipeline import RagPipeline


def evaluate(settings: Settings, qa_path: Path, limit: int | None = None) -> dict:
    pipeline = RagPipeline(settings)
    rows = list(read_jsonl(qa_path))
    if limit:
        rows = rows[:limit]

    total = 0
    retrieval_hits = 0
    answers_with_source = 0
    unsupported = 0

    for row in rows:
        total += 1
        answer = pipeline.answer(row["question"])
        expected_id = row.get("entity_id")
        if expected_id and expected_id in answer.used_context_ids[:5]:
            retrieval_hits += 1
        if answer.sources:
            answers_with_source += 1
        if "chưa tìm thấy thông tin phù hợp" in answer.answer.lower():
            unsupported += 1

    denominator = total or 1
    return {
        "qa_path": str(qa_path),
        "total": total,
        "retrieval_hit_at_5": retrieval_hits / denominator,
        "answer_has_source_rate": answers_with_source / denominator,
        "unsupported_answer_rate": unsupported / denominator,
    }
