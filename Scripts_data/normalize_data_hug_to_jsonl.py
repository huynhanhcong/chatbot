import ast
import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_HUG_DIR = ROOT / "Data_Hug"
OUTPUT_DIR = DATA_HUG_DIR / "jsonl"

CHAT_SOURCE = DATA_HUG_DIR / "data_vietnamese-medical-chat-data.csv"
QA_SOURCE = DATA_HUG_DIR / "data_vietnamese-medical-qa.csv"

CHAT_OUTPUT = OUTPUT_DIR / "data_vietnamese-medical-chat-data.jsonl"
QA_OUTPUT = OUTPUT_DIR / "data_vietnamese-medical-qa.jsonl"

ZERO_WIDTH_CHARS = "\u200b\u200c\u200d\ufeff"


def clean_text(value):
    if pd.isna(value):
        return ""

    text = str(value)
    for char in ZERO_WIDTH_CHARS:
        text = text.replace(char, "")

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def slugify(text):
    text = clean_text(text).lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def parse_conversation(text):
    text = clean_text(text)
    text = re.sub(r"}\s+{", "}, {", text)
    data = ast.literal_eval(text)

    if not isinstance(data, list):
        raise ValueError("Conversation is not a list")

    messages = []
    for item in data:
        if not isinstance(item, dict):
            continue
        role = clean_text(item.get("role"))
        content = clean_text(item.get("content"))
        if role not in {"user", "assistant", "system"} or not content:
            continue
        messages.append({"role": role, "content": content})

    if not messages:
        raise ValueError("Conversation contains no valid messages")

    return messages


def strip_question_boilerplate(text):
    text = clean_text(text)
    if not text:
        return ""

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    cleaned = []

    greeting_patterns = [
        r"^chào bác sĩ[,!:]*$",
        r"^thưa bác sĩ[,!:]*$",
        r"^bác sĩ cho em hỏi[,!:]*$",
        r"^bác sĩ cho cháu hỏi[,!:]*$",
        r"^bác sĩ cho con hỏi[,!:]*$",
    ]
    thanks_patterns = [
        r"^em cảm ơn.*$",
        r"^cháu cảm ơn.*$",
        r"^con cảm ơn.*$",
        r"^xin cảm ơn.*$",
        r"^cảm ơn bác sĩ.*$",
    ]

    for line in lines:
        lowered = line.lower()
        if any(re.match(pattern, lowered) for pattern in greeting_patterns):
            continue
        if any(re.match(pattern, lowered) for pattern in thanks_patterns):
            continue
        cleaned.append(line)

    text = clean_text("\n".join(cleaned))
    text = re.sub(r"(?i)[, ]*(em|cháu|con|xin)?\s*cảm ơn( bác sĩ)?[.! ]*$", "", text).strip()
    return clean_text(text)


def strip_answer_boilerplate(text):
    text = clean_text(text)
    if not text:
        return ""

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    cleaned = []

    greeting_patterns = [
        r"^chào bạn[,!:]*$",
        r"^chào em[,!:]*$",
        r"^chào bạn!$",
        r"^chào em!$",
        r"^cảm ơn bạn đã tin tưởng ai-doctor.*$",
        r"^cảm ơn em đã tin tưởng ai-doctor.*$",
        r"^để trả lời câu hỏi trên, bác sĩ xin giải đáp như sau:?$",
    ]
    closing_patterns = [
        r"^trân trọng[.!]*$",
        r"^trân trọng!$",
        r"^thân ái.*$",
    ]

    for line in lines:
        lowered = line.lower()
        if any(re.match(pattern, lowered) for pattern in greeting_patterns):
            continue
        if any(re.match(pattern, lowered) for pattern in closing_patterns):
            continue
        cleaned.append(line)

    text = clean_text("\n".join(cleaned))
    text = re.sub(r"(?i)[, ]*trân trọng[.! ]*$", "", text).strip()
    return clean_text(text)


def build_search_text(parts):
    values = []
    for part in parts:
        if isinstance(part, list):
            values.extend(clean_text(item) for item in part if clean_text(item))
            continue
        text = clean_text(part)
        if text:
            values.append(text)
    return " ".join(values)


def write_jsonl(records, output_path):
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_chat_dataset():
    df = pd.read_csv(CHAT_SOURCE)
    records = []
    parse_errors = 0

    for idx, row in df.iterrows():
        raw_text = row.iloc[0]
        try:
            messages = parse_conversation(raw_text)
        except Exception:
            parse_errors += 1
            continue

        user_turn_count = sum(1 for item in messages if item["role"] == "user")
        assistant_turn_count = sum(1 for item in messages if item["role"] == "assistant")
        first_user = next((item["content"] for item in messages if item["role"] == "user"), "")
        first_assistant = next((item["content"] for item in messages if item["role"] == "assistant"), "")
        search_text = build_search_text([item["content"] for item in messages])

        records.append(
            {
                "id": f"hug_chat_{idx + 1:06d}",
                "source_dataset": CHAT_SOURCE.name,
                "record_type": "conversation",
                "messages": messages,
                "turn_count": len(messages),
                "user_turn_count": user_turn_count,
                "assistant_turn_count": assistant_turn_count,
                "first_user_message": first_user,
                "first_assistant_message": first_assistant,
                "search_text": search_text,
            }
        )

    write_jsonl(records, CHAT_OUTPUT)
    return {
        "rows_in_csv": len(df),
        "rows_written": len(records),
        "parse_errors": parse_errors,
        "output": str(CHAT_OUTPUT),
    }


def normalize_qa_dataset():
    df = pd.read_csv(QA_SOURCE)
    question_col = df.columns[1]
    answer_col = df.columns[0]

    records = []
    for idx, row in df.iterrows():
        question_raw = clean_text(row[question_col])
        answer_raw = clean_text(row[answer_col])

        question_clean = strip_question_boilerplate(question_raw)
        answer_clean = strip_answer_boilerplate(answer_raw)

        messages = []
        if question_clean:
            messages.append({"role": "user", "content": question_clean})
        if answer_clean:
            messages.append({"role": "assistant", "content": answer_clean})

        records.append(
            {
                "id": f"hug_qa_{idx + 1:06d}",
                "source_dataset": QA_SOURCE.name,
                "record_type": "qa",
                "question_raw": question_raw,
                "answer_raw": answer_raw,
                "question_clean": question_clean,
                "answer_clean": answer_clean,
                "messages": messages,
                "turn_count": len(messages),
                "search_text": build_search_text([question_clean, answer_clean]),
            }
        )

    write_jsonl(records, QA_OUTPUT)
    return {
        "rows_in_csv": len(df),
        "rows_written": len(records),
        "output": str(QA_OUTPUT),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    chat_stats = normalize_chat_dataset()
    qa_stats = normalize_qa_dataset()

    print(json.dumps({"chat": chat_stats, "qa": qa_stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
