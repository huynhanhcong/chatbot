import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_JSONL = ROOT / "Data_RAG" / "retrieval" / "hanhphuc_retrieval.jsonl"
EXISTING_JSONL = ROOT / "Data_RAG" / "qa" / "hanhphuc_rag_qa_100.jsonl"
OUTPUT_DIR = ROOT / "Data_RAG" / "qa"
OUTPUT_EXTRA_JSONL = OUTPUT_DIR / "hanhphuc_rag_qa_extra_100.jsonl"
OUTPUT_COMBINED_JSONL = OUTPUT_DIR / "hanhphuc_rag_qa_200.jsonl"
OUTPUT_PREVIEW = OUTPUT_DIR / "hanhphuc_rag_qa_extra_100_preview.txt"

TARGET_EXTRA_COUNT = 100
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
    cleaned = [clean_text(item) for item in items if clean_text(item)]
    if limit is not None:
        cleaned = cleaned[:limit]
    if not cleaned:
        return "chưa có thông tin trong dữ liệu hiện tại"
    return "; ".join(cleaned)


def read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def make_qa(record, question, answer, qa_type, grounding_fields):
    return {
        "id": None,
        "question": clean_text(question),
        "answer": clean_text(answer),
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


def doctor_experience_qa(record):
    facts = parse_json_list(record.get("facts_json"))
    summary = clean_text(record.get("summary"))
    answer_parts = [
        f"Theo dữ liệu Hạnh Phúc, {record['title']} thuộc nhóm chuyên môn/vai trò: {record['category']}."
    ]
    if facts:
        answer_parts.append(f"Các thông tin chính gồm: {format_items(facts, limit=6)}.")
    elif summary:
        answer_parts.append(summary)
    else:
        answer_parts.append("Dữ liệu hiện tại chưa có mô tả chi tiết ngoài tên và chuyên môn.")

    return make_qa(
        record,
        f"Kinh nghiệm và thế mạnh của {record['title']} là gì?",
        " ".join(answer_parts),
        "doctor_experience_strength",
        ["title", "category", "summary", "facts_json"],
    )


def package_detail_qa(record):
    services = parse_json_list(record.get("services_json"))
    preparation = parse_json_list(record.get("preparation_json"))
    terms = parse_json_list(record.get("terms_json"))
    price = format_price(record.get("price_vnd"))

    if preparation:
        question = f"Cần chuẩn bị gì trước khi thực hiện {record['title']}?"
        answer = (
            f"{record['title']} thuộc nhóm {record['category']} và có giá {price}. "
            f"Chuẩn bị trước khi thực hiện gồm: {format_items(preparation, limit=10)}."
        )
        if terms:
            answer += f" Điều kiện/điều khoản liên quan: {format_items(terms, limit=5)}."
        return make_qa(
            record,
            question,
            answer,
            "package_preparation_terms",
            ["title", "category", "price_vnd", "preparation_json", "terms_json"],
        )

    if terms:
        question = f"Điều kiện và điều khoản của {record['title']} là gì?"
        answer = (
            f"{record['title']} thuộc nhóm {record['category']} và có giá {price}. "
            f"Điều kiện/điều khoản chính gồm: {format_items(terms, limit=8)}."
        )
        return make_qa(
            record,
            question,
            answer,
            "package_terms",
            ["title", "category", "price_vnd", "terms_json"],
        )

    question = f"{record['title']} gồm những hạng mục hoặc dịch vụ nào?"
    answer = (
        f"{record['title']} thuộc nhóm {record['category']} và có giá {price}. "
        f"Dịch vụ trong gói gồm: {format_items(services, limit=14)}."
    )
    return make_qa(
        record,
        question,
        answer,
        "package_services",
        ["title", "category", "price_vnd", "services_json"],
    )


def build_extra_qas(records, existing_qas):
    existing_questions = {clean_text(qa.get("question")).lower() for qa in existing_qas}
    doctors = [record for record in records if record.get("entity_type") == "doctor"]
    packages = [record for record in records if record.get("entity_type") == "package"]

    candidates = [doctor_experience_qa(record) for record in doctors]
    candidates.extend(package_detail_qa(record) for record in packages)

    extra = []
    seen_questions = set(existing_questions)
    for qa in candidates:
        key = qa["question"].lower()
        if key in seen_questions:
            continue
        seen_questions.add(key)
        extra.append(qa)
        if len(extra) == TARGET_EXTRA_COUNT:
            break

    if len(extra) != TARGET_EXTRA_COUNT:
        raise ValueError(f"Expected {TARGET_EXTRA_COUNT} extra Q&A, got {len(extra)}")

    for index, qa in enumerate(extra, start=101):
        qa["id"] = f"hanhphuc_rag_qa_{index:03d}"
    return extra


def validate(qas, expected_count):
    if len(qas) != expected_count:
        raise ValueError(f"Expected {expected_count} Q&A, got {len(qas)}")

    ids = [qa["id"] for qa in qas]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate Q&A ids found")

    questions = [clean_text(qa["question"]).lower() for qa in qas]
    if len(questions) != len(set(questions)):
        raise ValueError("Duplicate questions found")

    for qa in qas:
        if not qa["question"].strip() or not qa["answer"].strip():
            raise ValueError(f"Empty question/answer found: {qa['id']}")
        if not qa["source_url"]:
            raise ValueError(f"Missing source_url: {qa['id']}")


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


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
    records = read_jsonl(SOURCE_JSONL)
    existing_qas = read_jsonl(EXISTING_JSONL)
    extra_qas = build_extra_qas(records, existing_qas)
    combined_qas = existing_qas + extra_qas

    validate(extra_qas, TARGET_EXTRA_COUNT)
    validate(combined_qas, len(existing_qas) + TARGET_EXTRA_COUNT)

    write_jsonl(OUTPUT_EXTRA_JSONL, extra_qas)
    write_jsonl(OUTPUT_COMBINED_JSONL, combined_qas)
    write_preview(extra_qas)

    stats = {
        "source": str(SOURCE_JSONL),
        "existing": str(EXISTING_JSONL),
        "extra_output": str(OUTPUT_EXTRA_JSONL),
        "combined_output": str(OUTPUT_COMBINED_JSONL),
        "preview": str(OUTPUT_PREVIEW),
        "extra_count": len(extra_qas),
        "combined_count": len(combined_qas),
        "extra_package_qa": sum(1 for qa in extra_qas if qa["entity_type"] == "package"),
        "extra_doctor_qa": sum(1 for qa in extra_qas if qa["entity_type"] == "doctor"),
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
