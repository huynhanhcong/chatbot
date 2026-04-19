import json
from pathlib import Path
from typing import Iterable

from .models import RagDocument
from .text import clean_text, flatten_list, format_price


REQUIRED_ENTITY_FIELDS = {"entity_id", "entity_type", "canonical_name", "source_url", "search_text"}


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc


def validate_entity(record: dict, line_number: int | None = None) -> None:
    missing = [field for field in REQUIRED_ENTITY_FIELDS if not clean_text(record.get(field))]
    if missing:
        where = f" line {line_number}" if line_number else ""
        raise ValueError(f"Entity{where} missing required fields: {', '.join(missing)}")
    if record["entity_type"] not in {"doctor", "package"}:
        raise ValueError(f"Unsupported entity_type: {record['entity_type']}")


def _entity_category(record: dict) -> str:
    if record.get("entity_type") == "doctor":
        return clean_text(record.get("department") or record.get("category"))
    return clean_text(record.get("category") or record.get("department"))


def _document_text(record: dict) -> str:
    parts = [
        clean_text(record.get("canonical_name")),
        _entity_category(record),
        clean_text(record.get("summary_text") or record.get("summary")),
    ]

    list_fields = [
        "aliases",
        "subspecialties",
        "audience",
        "conditions_treated",
        "conditions",
        "procedures_performed",
        "procedures",
        "service_lines",
        "clinical_focus",
        "recommended_for",
        "education",
        "includes",
        "preparation",
        "terms",
        "related_doctors",
        "related_packages",
    ]
    for field in list_fields:
        items = flatten_list(record.get(field))
        if items:
            parts.append("; ".join(items))

    price = format_price(record.get("price_vnd"))
    if price:
        parts.append(f"Giá: {price}")

    parts.append(clean_text(record.get("search_text")))
    return "\n".join(part for part in parts if part)


def entity_to_document(record: dict) -> RagDocument:
    category = _entity_category(record)
    text = _document_text(record)
    return RagDocument(
        id=record["entity_id"],
        entity_type=record["entity_type"],
        title=clean_text(record["canonical_name"]),
        category=category or None,
        source_url=clean_text(record["source_url"]),
        text=text,
        search_text=clean_text(record.get("search_text") or text),
        payload={
            "source": "hanhphuc",
            "entity_id": record["entity_id"],
            "entity_type": record["entity_type"],
            "title": clean_text(record["canonical_name"]),
            "category": category or None,
            "source_url": clean_text(record["source_url"]),
            "price_vnd": record.get("price_vnd"),
            "text": text,
            "search_text": clean_text(record.get("search_text") or text),
            "raw": record,
        },
    )


def load_hanhphuc_documents(path: Path) -> list[RagDocument]:
    documents: list[RagDocument] = []
    for line_number, record in enumerate(read_jsonl(path), start=1):
        validate_entity(record, line_number)
        documents.append(entity_to_document(record))
    return documents
