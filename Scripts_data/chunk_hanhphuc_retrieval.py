import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RETRIEVAL_DIR = ROOT / "Data_RAG" / "retrieval"
SOURCE_JSONL = RETRIEVAL_DIR / "hanhphuc_retrieval.jsonl"
CHUNKS_JSONL = RETRIEVAL_DIR / "hanhphuc_retrieval_chunks.jsonl"
CHUNKS_CSV = RETRIEVAL_DIR / "hanhphuc_retrieval_chunks.csv"

MAX_CHUNK_CHARS = 900
MIN_CHUNK_CHARS = 180


def clean_text(value):
    if value is None:
        return ""
    return " ".join(str(value).replace("\n", " ").split())


def load_json_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [clean_text(item) for item in value if clean_text(item)]
    text = clean_text(value)
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [text]
    if isinstance(parsed, list):
        return [clean_text(item) for item in parsed if clean_text(item)]
    return [text]


def read_records():
    with SOURCE_JSONL.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def price_text(record):
    price = record.get("price_vnd")
    if price is None:
        return ""
    try:
        price = int(float(price))
    except (TypeError, ValueError):
        return ""
    return f"Giá: {price:,} VND".replace(",", ".")


def chunk_prefix(record, chunk_type):
    parts = [
        f"Loại dữ liệu: {'Bác sĩ' if record['entity_type'] == 'doctor' else 'Gói khám'}",
        f"Tên: {record['title']}",
        f"Nhóm/Chuyên môn: {record['category']}",
    ]
    price = price_text(record)
    if price:
        parts.append(price)
    parts.append(f"Mục thông tin: {chunk_type}")
    return "\n".join(parts)


def make_text(record, chunk_type, body):
    prefix = chunk_prefix(record, chunk_type)
    return f"{prefix}\nNội dung: {body}".strip()


def split_long_text(text, max_chars=MAX_CHUNK_CHARS):
    text = clean_text(text)
    if len(text) <= max_chars:
        return [text] if text else []

    sentences = []
    buffer = ""
    for char in text:
        buffer += char
        if char in ".!?…":
            sentence = buffer.strip()
            if sentence:
                sentences.append(sentence)
            buffer = ""
    if buffer.strip():
        sentences.append(buffer.strip())

    chunks = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        chunks.append(current)

    # Fall back to hard split if a single sentence is too long.
    hard_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            hard_chunks.append(chunk)
            continue
        words = chunk.split()
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if current and len(candidate) > max_chars:
                hard_chunks.append(current)
                current = word
            else:
                current = candidate
        if current:
            hard_chunks.append(current)

    return hard_chunks


def split_list_items(items, max_chars=MAX_CHUNK_CHARS):
    items = [clean_text(item) for item in items if clean_text(item)]
    if not items:
        return []

    chunks = []
    current = []
    current_len = 0
    for item in items:
        item_len = len(item) + 4
        if current and current_len + item_len > max_chars:
            chunks.append(current)
            current = [item]
            current_len = item_len
        else:
            current.append(item)
            current_len += item_len
    if current:
        chunks.append(current)

    # Merge tiny tail chunks when safe.
    merged = []
    for chunk in chunks:
        body = "; ".join(chunk)
        if merged and len(body) < MIN_CHUNK_CHARS:
            previous = merged[-1]
            candidate = previous + chunk
            if len("; ".join(candidate)) <= max_chars:
                merged[-1] = candidate
                continue
        merged.append(chunk)
    return ["; ".join(chunk) for chunk in merged]


def build_chunk(record, chunk_type, body, local_index):
    text = make_text(record, chunk_type, body)
    return {
        "chunk_id": f"{record['id']}__{chunk_type.lower().replace(' ', '_')}__{local_index:02d}",
        "parent_id": record["id"],
        "entity_type": record["entity_type"],
        "title": record["title"],
        "category": record["category"],
        "chunk_type": chunk_type,
        "chunk_index": local_index,
        "chunk_count": None,
        "price_vnd": record.get("price_vnd"),
        "source_url": record.get("source_url"),
        "text": text,
        "search_text": text,
        "metadata": {
            "source_file": SOURCE_JSONL.name,
            "parent_summary": clean_text(record.get("summary")),
        },
    }


def chunks_for_record(record):
    chunks = []

    summary = clean_text(record.get("summary"))
    if summary:
        for idx, part in enumerate(split_long_text(summary), start=1):
            chunks.append(build_chunk(record, "summary", part, idx))

    facts = load_json_list(record.get("facts_json"))
    if facts:
        for idx, part in enumerate(split_list_items(facts), start=1):
            chunks.append(build_chunk(record, "facts", part, idx))

    services = load_json_list(record.get("services_json"))
    if services:
        for idx, part in enumerate(split_list_items(services), start=1):
            chunks.append(build_chunk(record, "services", part, idx))

    preparation = load_json_list(record.get("preparation_json"))
    if preparation:
        for idx, part in enumerate(split_list_items(preparation), start=1):
            chunks.append(build_chunk(record, "preparation", part, idx))

    terms = load_json_list(record.get("terms_json"))
    if terms:
        for idx, part in enumerate(split_list_items(terms), start=1):
            chunks.append(build_chunk(record, "terms", part, idx))

    if not chunks:
        fallback = clean_text(record.get("search_text"))
        for idx, part in enumerate(split_long_text(fallback), start=1):
            chunks.append(build_chunk(record, "fallback", part, idx))

    total = len(chunks)
    for index, chunk in enumerate(chunks, start=1):
        chunk["global_parent_chunk_index"] = index
        chunk["chunk_count"] = total
    return chunks


def write_outputs(chunks):
    with CHUNKS_JSONL.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    fieldnames = [
        "chunk_id",
        "parent_id",
        "entity_type",
        "title",
        "category",
        "chunk_type",
        "chunk_index",
        "global_parent_chunk_index",
        "chunk_count",
        "price_vnd",
        "source_url",
        "text",
        "search_text",
        "metadata",
    ]
    with CHUNKS_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for chunk in chunks:
            row = dict(chunk)
            row["metadata"] = json.dumps(row["metadata"], ensure_ascii=False)
            writer.writerow(row)


def validate(chunks):
    if not chunks:
        raise ValueError("No chunks generated")

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Duplicate chunk_id found")

    empty = [chunk["chunk_id"] for chunk in chunks if not clean_text(chunk.get("text"))]
    if empty:
        raise ValueError(f"Empty chunk text found: {empty[:5]}")


def main():
    chunks = []
    parent_count = 0
    for record in read_records():
        parent_count += 1
        chunks.extend(chunks_for_record(record))

    validate(chunks)
    write_outputs(chunks)

    by_type = {}
    for chunk in chunks:
        by_type[chunk["chunk_type"]] = by_type.get(chunk["chunk_type"], 0) + 1

    print(
        json.dumps(
            {
                "parents": parent_count,
                "chunks": len(chunks),
                "by_chunk_type": by_type,
                "jsonl": str(CHUNKS_JSONL),
                "csv": str(CHUNKS_CSV),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
