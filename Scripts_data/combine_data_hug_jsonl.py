import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JSONL_DIR = ROOT / "Data_Hug" / "jsonl"

CHAT_JSONL = JSONL_DIR / "data_vietnamese-medical-chat-data.jsonl"
QA_JSONL = JSONL_DIR / "data_vietnamese-medical-qa.jsonl"
COMBINED_JSONL = JSONL_DIR / "data_vietnamese-medical-combined.jsonl"


REQUIRED_FIELDS = {
    "id",
    "source_dataset",
    "record_type",
    "messages",
    "turn_count",
    "search_text",
}


def iter_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            missing = REQUIRED_FIELDS - set(record)
            if missing:
                raise ValueError(f"{path}:{line_number} missing fields: {sorted(missing)}")
            yield record


def main():
    counts = {}
    total = 0

    with COMBINED_JSONL.open("w", encoding="utf-8") as output:
        for source_path in [CHAT_JSONL, QA_JSONL]:
            count = 0
            for record in iter_jsonl(source_path):
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
            counts[source_path.name] = count
            total += count

    counts[COMBINED_JSONL.name] = total
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
