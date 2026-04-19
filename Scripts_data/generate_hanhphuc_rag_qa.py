import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_JSONL = ROOT / "Data_RAG" / "retrieval" / "hanhphuc_retrieval.jsonl"
OUTPUT_DIR = ROOT / "Data_RAG" / "qa"
OUTPUT_JSONL = OUTPUT_DIR / "hanhphuc_rag_qa_100.jsonl"
OUTPUT_PREVIEW = OUTPUT_DIR / "hanhphuc_rag_qa_100_preview.txt"

TARGET_QA_COUNT = 100
CREATED_AT = "2026-04-19"


def clean_text(value):
    if value is None:
        return ""
    return " ".join(str(value).replace("\n", " ").split())


def parse_json_list(value):
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


def format_price(price):
    if price is None:
        return "chưa có thông tin giá trong dữ liệu hiện tại"
    try:
        value = int(float(price))
    except (TypeError, ValueError):
        return "chưa có thông tin giá trong dữ liệu hiện tại"
    return f"{value:,}đ".replace(",", ".")


def format_items(items, limit=None):
    items = [clean_text(item) for item in items if clean_text(item)]
    if limit:
        items = items[:limit]
    if not items:
        return "chưa có thông tin trong dữ liệu hiện tại"
    return "; ".join(items)


def read_records():
    records = []
    with SOURCE_JSONL.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def make_qa(record, question, answer, qa_type, grounding_fields):
    return {
        "id": None,
        "question": question,
        "answer": answer,
        "qa_type": qa_type,
        "entity_id": record["id"],
        "entity_type": record["entity_type"],
        "title": record["title"],
        "category": record["category"],
        "source_url": record.get("source_url"),
        "grounding_fields": grounding_fields,
        "created_at": CREATED_AT,
        "search_text": clean_text(f"{question} {answer}"),
    }


def doctor_qa(record):
    facts = parse_json_list(record.get("facts_json"))
    summary = clean_text(record.get("summary"))
    answer_parts = [
        f"{record['title']} thuộc nhóm chuyên môn/vai trò: {record['category']}.",
    ]
    if summary:
        answer_parts.append(summary)
    elif facts:
        answer_parts.append(format_items(facts, limit=4))
    answer = " ".join(answer_parts)

    return make_qa(
        record,
        f"{record['title']} chuyên về lĩnh vực gì?",
        answer,
        "doctor_profile",
        ["title", "category", "summary", "facts_json"],
    )


def package_qa(record):
    services = parse_json_list(record.get("services_json"))
    preparation = parse_json_list(record.get("preparation_json"))
    terms = parse_json_list(record.get("terms_json"))
    summary = clean_text(record.get("summary"))
    price = format_price(record.get("price_vnd"))

    answer_parts = [
        f"{record['title']} thuộc nhóm {record['category']}.",
        f"Giá gói: {price}.",
    ]
    if summary:
        answer_parts.append(summary)
    if services:
        answer_parts.append(f"Dịch vụ trong gói gồm: {format_items(services, limit=12)}.")
    if preparation:
        answer_parts.append(f"Chuẩn bị trước khi khám: {format_items(preparation, limit=8)}.")
    if terms:
        answer_parts.append(f"Điều kiện/điều khoản chính: {format_items(terms, limit=5)}.")

    return make_qa(
        record,
        f"{record['title']} bao gồm gì và giá bao nhiêu?",
        " ".join(answer_parts),
        "package_overview",
        ["title", "category", "price_vnd", "summary", "services_json", "preparation_json", "terms_json"],
    )


def validate(qas):
    if len(qas) != TARGET_QA_COUNT:
        raise ValueError(f"Expected {TARGET_QA_COUNT} Q&A, got {len(qas)}")

    ids = [qa["id"] for qa in qas]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate Q&A ids found")

    for qa in qas:
        if not qa["question"].strip() or not qa["answer"].strip():
            raise ValueError(f"Empty question/answer found: {qa['id']}")
        if not qa["source_url"]:
            raise ValueError(f"Missing source_url: {qa['id']}")


def write_jsonl(qas):
    with OUTPUT_JSONL.open("w", encoding="utf-8") as handle:
        for qa in qas:
            handle.write(json.dumps(qa, ensure_ascii=False) + "\n")


def write_preview(qas):
    with OUTPUT_PREVIEW.open("w", encoding="utf-8") as handle:
        for qa in qas[:20]:
            handle.write(f"{qa['id']}\n")
            handle.write(f"Q: {qa['question']}\n")
            handle.write(f"A: {qa['answer']}\n")
            handle.write(f"Source: {qa['source_url']}\n")
            handle.write("-" * 100 + "\n")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = read_records()
    doctors = [record for record in records if record.get("entity_type") == "doctor"]
    packages = [record for record in records if record.get("entity_type") == "package"]

    # Prioritize all package records because they carry price/services fields that users often ask about.
    selected_records = packages + doctors
    qas = []
    for record in selected_records[:TARGET_QA_COUNT]:
        if record["entity_type"] == "package":
            qas.append(package_qa(record))
        else:
            qas.append(doctor_qa(record))

    for index, qa in enumerate(qas, start=1):
        qa["id"] = f"hanhphuc_rag_qa_{index:03d}"

    validate(qas)
    write_jsonl(qas)
    write_preview(qas)

    stats = {
        "source": str(SOURCE_JSONL),
        "output": str(OUTPUT_JSONL),
        "preview": str(OUTPUT_PREVIEW),
        "qa_count": len(qas),
        "package_qa": sum(1 for qa in qas if qa["entity_type"] == "package"),
        "doctor_qa": sum(1 for qa in qas if qa["entity_type"] == "doctor"),
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
