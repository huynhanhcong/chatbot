import hashlib
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT_JSONL = ROOT / "Data_Hug" / "jsonl" / "data_vietnamese-medical-combined.jsonl"
OUTPUT_DIR = ROOT / "Data_RAG" / "retrieval"
OUTPUT_JSONL = OUTPUT_DIR / "vietnamese_medical_combined_chunks.jsonl"
OUTPUT_MANIFEST = OUTPUT_DIR / "vietnamese_medical_combined_chunks_manifest.json"

MAX_CHUNK_WORDS = 380
TARGET_CHUNK_WORDS = 300
OVERLAP_WORDS = 35
MIN_CHUNK_WORDS = 18
MAX_QUESTION_CONTEXT_WORDS = 120
MAX_PREVIOUS_CONTEXT_WORDS = 50
MAX_KEYWORDS = 14
CREATED_AT = "2026-04-19"


VI_STOPWORDS = {
    "a", "ai", "anh", "bạn", "bà", "bác", "bằng", "bị", "biết", "bệnh", "bên", "bởi",
    "các", "cái", "cần", "chỉ", "cho", "chúng", "chưa", "chữa", "có", "còn", "của",
    "cùng", "cũng", "đã", "đang", "đây", "để", "đến", "đi", "điều", "đó", "được",
    "gì", "hay", "hơn", "khi", "không", "là", "làm", "lại", "lên", "mà", "mình",
    "một", "nào", "nên", "nếu", "người", "như", "những", "này", "nữa", "phải",
    "qua", "ra", "rằng", "rất", "sau", "sẽ", "tại", "thì", "theo", "thể", "thêm",
    "tôi", "trong", "trên", "trước", "từ", "và", "vào", "về", "vì", "việc", "với",
    "xin", "y", "điều", "nhiều", "ít", "hoặc", "nó", "họ", "em", "chị", "ông",
}


def clean_text(value):
    if value is None:
        return ""
    text = str(value).replace("\ufeff", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def word_count(text):
    return len(clean_text(text).split())


def stable_hash(text, length=16):
    normalized = clean_text(text).lower()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:length]


def sentence_split(text):
    text = clean_text(text)
    if not text:
        return []

    # Preserve list-like medical instructions as sentence boundaries.
    text = re.sub(r"\s*([•●▪◦])\s*", r". \1 ", text)
    text = re.sub(r"\s+(-|\d+[.)])\s+", r". \1 ", text)

    parts = re.split(r"(?<=[.!?。！？])\s+|(?<=;)\s+|\n+", text)
    sentences = [clean_text(part) for part in parts if clean_text(part)]
    return sentences or [text]


def split_long_words(text, max_words=TARGET_CHUNK_WORDS, overlap=OVERLAP_WORDS):
    words = clean_text(text).split()
    if len(words) <= max_words:
        return [clean_text(text)] if words else []

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(0, end - overlap)
    return chunks


def compact_context(text, max_words):
    words = clean_text(text).split()
    if len(words) <= max_words:
        return clean_text(text), False

    head_words = max_words * 2 // 3
    tail_words = max_words - head_words
    compacted = " ".join(words[:head_words] + ["[...]"] + words[-tail_words:])
    return compacted, True


def semantic_segments(text, max_words=TARGET_CHUNK_WORDS, overlap=OVERLAP_WORDS):
    sentences = sentence_split(text)
    segments = []
    current = []
    current_words = 0

    for sentence in sentences:
        n_words = word_count(sentence)
        if n_words > max_words:
            if current:
                segments.append(" ".join(current))
                current = []
                current_words = 0
            segments.extend(split_long_words(sentence, max_words=max_words, overlap=overlap))
            continue

        if current and current_words + n_words > max_words:
            segments.append(" ".join(current))
            overlap_tail = []
            tail_words = 0
            for old_sentence in reversed(current):
                old_words = word_count(old_sentence)
                if tail_words + old_words > overlap:
                    break
                overlap_tail.insert(0, old_sentence)
                tail_words += old_words
            current = overlap_tail[:]
            current_words = tail_words

        current.append(sentence)
        current_words += n_words

    if current:
        segments.append(" ".join(current))

    return [clean_text(segment) for segment in segments if clean_text(segment)]


def extract_keywords(text, limit=MAX_KEYWORDS):
    tokens = re.findall(r"[A-Za-zÀ-ỹ0-9][A-Za-zÀ-ỹ0-9+/.-]{2,}", clean_text(text))
    normalized = []
    for token in tokens:
        value = token.strip(".,;:!?()[]{}").lower()
        if len(value) < 3 or value in VI_STOPWORDS:
            continue
        if value.isdigit():
            continue
        normalized.append(value)

    counts = Counter(normalized)
    return [word for word, _ in counts.most_common(limit)]


def pair_user_assistant(messages):
    pairs = []
    last_user = ""
    previous_user = ""
    for index, message in enumerate(messages):
        role = message.get("role")
        content = clean_text(message.get("content"))
        if not content:
            continue
        if role == "user":
            previous_user = last_user
            last_user = content
        elif role == "assistant":
            question = last_user or previous_user
            if question:
                pairs.append(
                    {
                        "turn_index": index,
                        "previous_user_context": previous_user if previous_user != question else "",
                        "question": question,
                        "answer": content,
                    }
                )
    return pairs


def make_chunk(
    *,
    parent_id,
    source_dataset,
    record_type,
    chunk_type,
    turn_index,
    chunk_index,
    total_chunks,
    question,
    answer,
    previous_user_context="",
):
    original_question = clean_text(question)
    question, question_was_compacted = compact_context(original_question, MAX_QUESTION_CONTEXT_WORDS)
    answer = clean_text(answer)
    previous_user_context, previous_context_was_compacted = compact_context(
        previous_user_context, MAX_PREVIOUS_CONTEXT_WORDS
    )

    text_parts = []
    if previous_user_context:
        text_parts.append(f"Ngữ cảnh trước: {previous_user_context}")
    if question:
        text_parts.append(f"Câu hỏi: {question}")
    text_parts.append(f"Trả lời: {answer}")
    text = clean_text(" ".join(text_parts))

    keywords = extract_keywords(text)
    fingerprint = stable_hash(text)

    return {
        "chunk_id": f"{parent_id}__{chunk_type}_{turn_index:03d}_{chunk_index:03d}_{fingerprint}",
        "parent_id": parent_id,
        "source_dataset": source_dataset,
        "record_type": record_type,
        "domain": "general_vietnamese_medical",
        "chunk_type": chunk_type,
        "turn_index": turn_index,
        "chunk_index": chunk_index,
        "total_chunks_for_turn": total_chunks,
        "question": question,
        "answer": answer,
        "previous_user_context": previous_user_context or None,
        "question_full_word_count": word_count(original_question),
        "question_was_compacted": question_was_compacted,
        "previous_context_was_compacted": previous_context_was_compacted,
        "keywords": keywords,
        "word_count": word_count(text),
        "char_count": len(text),
        "fingerprint": fingerprint,
        "created_at": CREATED_AT,
        "text": text,
        "search_text": clean_text(" ".join([question, answer, " ".join(keywords)])),
    }


def chunks_from_qa_record(record):
    question = clean_text(record.get("question_clean") or record.get("question_raw"))
    answer = clean_text(record.get("answer_clean") or record.get("answer_raw"))
    if not question or not answer:
        return []

    answer_segments = semantic_segments(answer)
    chunks = []
    for index, segment in enumerate(answer_segments, start=1):
        if word_count(question) + word_count(segment) < MIN_CHUNK_WORDS:
            continue
        chunks.append(
            make_chunk(
                parent_id=record["id"],
                source_dataset=record.get("source_dataset"),
                record_type=record.get("record_type"),
                chunk_type="qa_answer" if len(answer_segments) == 1 else "qa_answer_part",
                turn_index=1,
                chunk_index=index,
                total_chunks=len(answer_segments),
                question=question,
                answer=segment,
            )
        )
    return chunks


def chunks_from_conversation_record(record):
    messages = record.get("messages") or []
    pairs = pair_user_assistant(messages)
    chunks = []

    for pair_number, pair in enumerate(pairs, start=1):
        answer_segments = semantic_segments(pair["answer"])
        for segment_number, segment in enumerate(answer_segments, start=1):
            if word_count(pair["question"]) + word_count(segment) < MIN_CHUNK_WORDS:
                continue
            chunks.append(
                make_chunk(
                    parent_id=record["id"],
                    source_dataset=record.get("source_dataset"),
                    record_type=record.get("record_type"),
                    chunk_type="conversation_qa" if len(answer_segments) == 1 else "conversation_qa_part",
                    turn_index=pair_number,
                    chunk_index=segment_number,
                    total_chunks=len(answer_segments),
                    previous_user_context=pair["previous_user_context"],
                    question=pair["question"],
                    answer=segment,
                )
            )

    return chunks


def chunks_from_record(record):
    if record.get("record_type") == "qa":
        return chunks_from_qa_record(record)
    if record.get("record_type") == "conversation":
        return chunks_from_conversation_record(record)
    return []


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    duplicate_fingerprints = 0
    seen_fingerprints = set()
    chunk_type_counts = Counter()
    record_type_counts = Counter()

    with INPUT_JSONL.open("r", encoding="utf-8") as source, OUTPUT_JSONL.open("w", encoding="utf-8") as output:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            stats["input_records"] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats["bad_json_lines"] += 1
                continue

            record_type_counts[record.get("record_type") or "unknown"] += 1
            chunks = chunks_from_record(record)
            if not chunks:
                stats["records_without_chunks"] += 1
                continue

            for chunk in chunks:
                if chunk["fingerprint"] in seen_fingerprints:
                    duplicate_fingerprints += 1
                    continue
                seen_fingerprints.add(chunk["fingerprint"])
                output.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                stats["output_chunks"] += 1
                chunk_type_counts[chunk["chunk_type"]] += 1

            if line_number % 10000 == 0:
                print(
                    json.dumps(
                        {
                            "processed_records": line_number,
                            "output_chunks": stats["output_chunks"],
                            "duplicates_skipped": duplicate_fingerprints,
                        },
                        ensure_ascii=False,
                    )
                )

    manifest = {
        "input": str(INPUT_JSONL),
        "output": str(OUTPUT_JSONL),
        "created_at": CREATED_AT,
        "algorithm": {
            "name": "structure_aware_vietnamese_medical_chunking",
            "steps": [
                "preserve user-assistant QA boundaries",
                "split long answers by Vietnamese sentence boundaries",
                "repeat question in each answer segment for standalone retrieval",
                "add previous user context for multi-turn disambiguation",
                "compact overly long question/context with head-tail retention",
                "deduplicate exact normalized chunks by SHA1 fingerprint",
                "extract lightweight lexical keywords for filtering/debugging",
            ],
            "max_chunk_words": MAX_CHUNK_WORDS,
            "target_chunk_words": TARGET_CHUNK_WORDS,
            "overlap_words": OVERLAP_WORDS,
            "min_chunk_words": MIN_CHUNK_WORDS,
            "max_question_context_words": MAX_QUESTION_CONTEXT_WORDS,
            "max_previous_context_words": MAX_PREVIOUS_CONTEXT_WORDS,
        },
        "stats": {
            **dict(stats),
            "duplicates_skipped": duplicate_fingerprints,
            "unique_fingerprints": len(seen_fingerprints),
            "input_record_types": dict(record_type_counts),
            "output_chunk_types": dict(chunk_type_counts),
        },
    }

    with OUTPUT_MANIFEST.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
