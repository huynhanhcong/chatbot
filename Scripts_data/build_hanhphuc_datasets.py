import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_CRAWL_DIR = ROOT / "Data_crawl"
DATA_RAG_DIR = ROOT / "Data_RAG"

DOCTORS_SOURCE = DATA_CRAWL_DIR / "hanhphuc_doctors.csv"
PACKAGES_SOURCE = DATA_CRAWL_DIR / "hanhphuc_goi_kham_merged_cleaned.csv"

HUMAN_CLEAN_DIR = DATA_RAG_DIR / "human_clean"
RETRIEVAL_DIR = DATA_RAG_DIR / "retrieval"

DOCTORS_HUMAN_CLEAN = HUMAN_CLEAN_DIR / "hanhphuc_doctors_clean.csv"
PACKAGES_HUMAN_CLEAN = HUMAN_CLEAN_DIR / "hanhphuc_packages_clean.csv"
RETRIEVAL_JSONL = RETRIEVAL_DIR / "hanhphuc_retrieval.jsonl"
RETRIEVAL_CSV = RETRIEVAL_DIR / "hanhphuc_retrieval.csv"

ZERO_WIDTH_CHARS = "\u200b\u200c\u200d\ufeff"
DOCTOR_SPECIALTY_ALIASES = {
    "bác sĩ chẩn đoán hình ảnh": "Bác sĩ Chẩn đoán Hình ảnh",
    "bác sĩ chẩn đoán hình  ảnh": "Bác sĩ Chẩn đoán Hình ảnh",
    "bác sĩ chẩn đoán hình ảnh ": "Bác sĩ Chẩn đoán Hình ảnh",
    "bác sĩ chẩn đoán hình ảnh": "Bác sĩ Chẩn đoán Hình ảnh",
    "bác sĩ chẩn đoán hình ảnh": "Bác sĩ Chẩn đoán Hình ảnh",
    "bác sĩ sản - phụ khoa": "Bác sĩ Sản – Phụ khoa",
    "trưởng khoa sản - phụ khoa": "Trưởng khoa Sản – Phụ khoa",
}
DOCTOR_MARKETING_PATTERNS = [
    r"\bphương châm\b",
    r"\bđồng hành\b",
    r"\ban tâm\b",
    r"\btận tâm\b",
    r"\bthấu hiểu\b",
    r"\bphong thái\b",
    r"\bnhẹ nhàng\b",
    r"\bđiềm tĩnh\b",
    r"\bhạnh phúc hơn\b",
    r"\bcuộc đời trọn vẹn\b",
    r"\bchất lượng cuộc sống\b",
    r"\bkhách hàng\b",
    r"\byêu thương\b",
    r"\bthoải mái\b",
    r"\blắng nghe\b",
    r"\bgiúp người bệnh cảm thấy\b",
    r"\bgiúp quá trình điều trị\b",
    r"\bgiúp mỗi\b",
    r"\bniềm đam mê\b",
    r"\bhướng đến việc mang lại\b",
    r"\bgóp phần khẳng định\b",
    r"\bđiểm đến y tế tin cậy\b",
]
DOCTOR_KEEP_PATTERNS = [
    r"\bkinh nghiệm\b",
    r"\btốt nghiệp\b",
    r"\bchuyên khoa\b",
    r"\bthạc sĩ\b",
    r"\btiến sĩ\b",
    r"\bnội trú\b",
    r"\bcông tác\b",
    r"\bđảm nhiệm\b",
    r"\btrưởng khoa\b",
    r"\bgiám đốc\b",
    r"\bbác sĩ\b",
    r"\bchuyên gia\b",
    r"\bchuyên sâu\b",
    r"\bđiều trị\b",
    r"\bkhám\b",
    r"\btư vấn\b",
    r"\bphẫu thuật\b",
    r"\bnội soi\b",
    r"\bsiêu âm\b",
    r"\bxét nghiệm\b",
    r"\btầm soát\b",
    r"\bchẩn đoán\b",
    r"\bhồi sức\b",
    r"\bnhi khoa\b",
    r"\bnội tiết\b",
    r"\btâm thần\b",
    r"\bda liễu\b",
    r"\brăng\b",
    r"\btim\b",
    r"\bbệnh lý\b",
    r"\bthực hiện\b",
    r"\bnghiên cứu\b",
    r"\bbáo cáo viên\b",
    r"\bhội nghị\b",
    r"\bcme\b",
    r"\brtms\b",
]
DOCTOR_SKIP_LINE_PATTERNS = [
    r"^đặt lịch khám\b",
    r"\bthứ 2\b",
    r"\bthứ 3\b",
    r"\bthứ 4\b",
    r"\bthứ 5\b",
    r"\bthứ 6\b",
    r"\bthứ 7\b",
]
PACKAGE_MARKETING_ONLY_PATTERNS = [
    r"^Tại Hạnh Phúc, chúng tôi tự hào\b",
    r"^Bắt đầu hành trình nuôi dạy con cái cùng chúng tôi!?$",
]
PACKAGE_TRIM_MARKETING_PATTERNS = [
    r"^Lựa chọn tối ưu cho ba mẹ\b",
]


def clean_text(value):
    if pd.isna(value):
        return pd.NA

    text = str(value)
    for char in ZERO_WIDTH_CHARS:
        text = text.replace(char, "")

    replacements = {
        "\r\n": "\n",
        "\r": "\n",
        "\u00a0": " ",
        "tếnăm": "tế năm",
        "làm việc ,": "làm việc,",
        "Tân Tạo .": "Tân Tạo.",
        "khách hàngsở hữu": "khách hàng sở hữu",
        "thông thường,mà": "thông thường, mà",
        "Đại Học": "Đại học",
        "BSCKI.": "BS.CKI.",
        "BSCKII.": "BS.CKII.",
        "Ths.BS.": "ThS.BS.",
        "Ths.BS": "ThS.BS",
        "TS.BS.": "TS.BS.",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = text.strip()

    if text == "Không rõ":
        return pd.NA
    return text or pd.NA


def normalize_whitespace_text(text):
    text = clean_text(text)
    return "" if pd.isna(text) else str(text)


def split_sentences(text):
    text = normalize_whitespace_text(text)
    if not text:
        return []

    parts = re.split(r"(?<=[.!?…])\s+", text)
    return [part.strip() for part in parts if part.strip()]


def load_json_list(value):
    text = clean_text(value)
    if pd.isna(text):
        return []

    text = str(text)
    if text.startswith("[") and text.endswith("]"):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, list):
            return [normalize_whitespace_text(item) for item in data if normalize_whitespace_text(item)]

    return [line for line in text.split("\n") if line]


def json_or_na(items):
    cleaned = []
    seen = set()
    for item in items:
        item = normalize_whitespace_text(item)
        if not item or item in seen:
            continue
        seen.add(item)
        cleaned.append(item)
    if not cleaned:
        return pd.NA
    return json.dumps(cleaned, ensure_ascii=False)


def pipe_or_na(items):
    cleaned = []
    seen = set()
    for item in items:
        item = normalize_whitespace_text(item)
        if not item or item in seen:
            continue
        seen.add(item)
        cleaned.append(item)
    if not cleaned:
        return pd.NA
    return " | ".join(cleaned)


def slug_from_url(url):
    url = normalize_whitespace_text(url)
    return url.rstrip("/").split("/")[-1]


def normalize_specialty(value):
    specialty = normalize_whitespace_text(value)
    specialty = specialty.replace(" - ", " – ")
    specialty_key = specialty.lower()
    return DOCTOR_SPECIALTY_ALIASES.get(specialty_key, specialty)


def is_doctor_skip_line(text):
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in DOCTOR_SKIP_LINE_PATTERNS)


def doctor_sentence_has_keep_signal(text):
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in DOCTOR_KEEP_PATTERNS)


def doctor_sentence_is_marketing_only(text):
    lowered = text.lower()
    has_keep = doctor_sentence_has_keep_signal(text)
    has_marketing = any(re.search(pattern, lowered) for pattern in DOCTOR_MARKETING_PATTERNS)
    if has_marketing and not has_keep:
        return True
    hard_skip_prefixes = [
        "với phong cách",
        "với phong thái",
        "đối với bác sĩ",
        "với bác sĩ",
        "không chỉ giỏi về chuyên môn",
        "với phương châm",
        "là người đồng hành",
        "từ đó đã tạo nền tảng",
        "với nền tảng chuyên môn vững chắc",
    ]
    if any(lowered.startswith(prefix) for prefix in hard_skip_prefixes):
        return True
    if lowered.startswith("với phong cách") or lowered.startswith("với phong thái"):
        return True
    if lowered.startswith("đối với bác sĩ"):
        return True
    if lowered.startswith("với bác sĩ"):
        return True
    if lowered.startswith("không chỉ giỏi về chuyên môn"):
        return True
    return False


def normalize_doctor_fact(line):
    line = normalize_whitespace_text(line)
    if not line:
        return ""
    if line[0].islower():
        line = line[0].upper() + line[1:]
    line = re.sub(r"^[-–•]\s*", "", line)
    return line


def split_doctor_content(text):
    text = normalize_whitespace_text(text)
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    paragraph_lines = []
    bullet_lines = []
    bullet_mode = False

    for line in lines:
        if is_doctor_skip_line(line):
            continue

        normalized = normalize_doctor_fact(line)
        if normalized.endswith(":"):
            bullet_mode = True
            continue

        short_item = (
            len(normalized) <= 180
            and not re.search(r"[.!?…]$", normalized)
            and (
                bullet_mode
                or doctor_sentence_has_keep_signal(normalized)
                or ":" in normalized
            )
        )

        if short_item:
            bullet_lines.append(normalized)
            continue

        bullet_mode = False
        paragraph_lines.append(normalized)

    return paragraph_lines, bullet_lines


def extract_doctor_sentences(paragraph_lines):
    sentences = []
    seen = set()

    for paragraph in paragraph_lines:
        for sentence in split_sentences(paragraph):
            sentence = normalize_doctor_fact(sentence)
            if not sentence or sentence in seen:
                continue
            if doctor_sentence_is_marketing_only(sentence):
                continue
            if not doctor_sentence_has_keep_signal(sentence):
                continue
            seen.add(sentence)
            sentences.append(sentence)

    return sentences


def build_doctor_facts(raw_text):
    paragraphs, bullets = split_doctor_content(raw_text)
    sentences = extract_doctor_sentences(paragraphs)

    cleaned_bullets = []
    seen = set(sentences)
    for bullet in bullets:
        bullet = normalize_doctor_fact(bullet)
        if not bullet or bullet in seen:
            continue
        if doctor_sentence_is_marketing_only(bullet):
            continue
        if not doctor_sentence_has_keep_signal(bullet):
            continue
        seen.add(bullet)
        cleaned_bullets.append(bullet)

    facts = []
    for item in sentences + cleaned_bullets:
        if item not in facts:
            facts.append(item)
    return facts


def pick_doctor_summary_sentences(facts):
    priority_groups = [
        [fact for fact in facts if re.search(r"\bkinh nghiệm\b|\bchuyên gia\b|\bchuyên sâu\b", fact.lower())],
        [fact for fact in facts if re.search(r"\btốt nghiệp\b|\bchuyên khoa\b|\bthạc sĩ\b|\btiến sĩ\b|\bnội trú\b", fact.lower())],
        [fact for fact in facts if re.search(r"\bcông tác\b|\bđảm nhiệm\b|\bthực hiện\b|\bđiều trị\b|\bphẫu thuật\b|\bnội soi\b|\bchẩn đoán\b", fact.lower())],
    ]

    summary = []
    for group in priority_groups:
        for item in group:
            if item not in summary:
                summary.append(item)
            if len(summary) == 3:
                return summary

    for item in facts:
        if item not in summary:
            summary.append(item)
        if len(summary) == 3:
            break
    return summary


def ensure_sentence(text):
    text = normalize_doctor_fact(text)
    if not text:
        return ""
    if re.search(r"[.!?…]$", text):
        return text
    return text + "."


def build_doctor_summary(facts):
    sentences = pick_doctor_summary_sentences(facts)
    sentences = [ensure_sentence(item) for item in sentences if item]
    if not sentences:
        return pd.NA
    return " ".join(sentences[:3])


def clean_package_summary(summary):
    summary = clean_text(summary)
    if pd.isna(summary):
        return pd.NA

    sentences = []
    for sentence in split_sentences(str(summary)):
        if any(re.search(pattern, sentence) for pattern in PACKAGE_MARKETING_ONLY_PATTERNS):
            continue
        if any(re.search(pattern, sentence) for pattern in PACKAGE_TRIM_MARKETING_PATTERNS):
            continue
        sentences.append(sentence)

    if not sentences:
        return pd.NA
    return " ".join(sentences)


def build_search_text(parts):
    values = []
    for part in parts:
        if isinstance(part, list):
            values.extend([normalize_whitespace_text(item) for item in part if normalize_whitespace_text(item)])
            continue
        text = clean_text(part)
        if pd.isna(text):
            continue
        values.append(str(text))
    return " ".join(values)


def build_doctors_human_clean():
    df = pd.read_csv(DOCTORS_SOURCE)
    records = []

    for _, row in df.iterrows():
        name = normalize_whitespace_text(row.iloc[0])
        specialty = normalize_specialty(row.iloc[1])
        facts = build_doctor_facts(row.iloc[2])
        summary = build_doctor_summary(facts)
        info = pipe_or_na(facts[:10])
        source_url = normalize_whitespace_text(row.iloc[3])

        records.append(
            {
                "id": f"doctor_{slug_from_url(source_url)}",
                "Tên bác sĩ": name,
                "Chuyên môn": specialty,
                "Tóm tắt": summary,
                "Thông tin chính": info,
                "Link": source_url,
            }
        )

    return pd.DataFrame(records)


def build_packages_human_clean():
    df = pd.read_csv(PACKAGES_SOURCE)
    records = []

    for _, row in df.iterrows():
        services = load_json_list(row.iloc[4])
        terms = load_json_list(row.iloc[5])
        preparation = load_json_list(row.iloc[6])
        source_url = normalize_whitespace_text(row.iloc[7])

        records.append(
            {
                "id": f"package_{slug_from_url(source_url)}",
                "Nhóm gói": normalize_whitespace_text(row.iloc[0]),
                "Tên gói": normalize_whitespace_text(row.iloc[1]),
                "Giá": int(row.iloc[2]) if not pd.isna(row.iloc[2]) else pd.NA,
                "Giới thiệu gói khám": clean_package_summary(row.iloc[3]),
                "Dịch vụ trong gói": pipe_or_na(services),
                "Điều kiện và điều khoản": pipe_or_na(terms),
                "Chuẩn bị trước tầm soát": pipe_or_na(preparation),
                "Link": source_url,
            }
        )

    return pd.DataFrame(records)


def build_retrieval_df(doctors_clean_df, packages_clean_df):
    records = []

    for _, row in doctors_clean_df.iterrows():
        facts = []
        info = clean_text(row["Thông tin chính"])
        if not pd.isna(info):
            facts = [item.strip() for item in str(info).split("|") if item.strip()]

        search_text = build_search_text(
            [
                row["Tên bác sĩ"],
                row["Chuyên môn"],
                row["Tóm tắt"],
                facts,
            ]
        )

        records.append(
            {
                "id": row["id"],
                "entity_type": "doctor",
                "title": row["Tên bác sĩ"],
                "category": row["Chuyên môn"],
                "summary": row["Tóm tắt"],
                "facts_json": json_or_na(facts),
                "services_json": pd.NA,
                "preparation_json": pd.NA,
                "terms_json": pd.NA,
                "price_vnd": pd.NA,
                "source_url": row["Link"],
                "search_text": search_text,
            }
        )

    packages_source_df = pd.read_csv(PACKAGES_SOURCE)
    package_lookup = {
        f"package_{slug_from_url(normalize_whitespace_text(row.iloc[7]))}": row
        for _, row in packages_source_df.iterrows()
    }

    for _, row in packages_clean_df.iterrows():
        package_source = package_lookup[row["id"]]
        services = load_json_list(package_source.iloc[4])
        terms = load_json_list(package_source.iloc[5])
        preparation = load_json_list(package_source.iloc[6])
        search_text = build_search_text(
            [
                row["Tên gói"],
                row["Nhóm gói"],
                row["Giới thiệu gói khám"],
                services,
                preparation,
                terms,
            ]
        )

        records.append(
            {
                "id": row["id"],
                "entity_type": "package",
                "title": row["Tên gói"],
                "category": row["Nhóm gói"],
                "summary": row["Giới thiệu gói khám"],
                "facts_json": pd.NA,
                "services_json": json_or_na(services),
                "preparation_json": json_or_na(preparation),
                "terms_json": json_or_na(terms),
                "price_vnd": row["Giá"],
                "source_url": row["Link"],
                "search_text": search_text,
            }
        )

    return pd.DataFrame(records)


def validate_outputs(doctors_clean_df, packages_clean_df, retrieval_df):
    assert len(doctors_clean_df) == 59, f"Unexpected doctors row count: {len(doctors_clean_df)}"
    assert len(packages_clean_df) == 43, f"Unexpected packages row count: {len(packages_clean_df)}"
    assert len(retrieval_df) == 102, f"Unexpected retrieval row count: {len(retrieval_df)}"

    for df in [doctors_clean_df, packages_clean_df, retrieval_df]:
        assert not df.astype(str).apply(lambda col: col.str.contains("Không rõ", na=False)).any().any(), "Literal 'Không rõ' remains in outputs"

    assert pd.api.types.is_integer_dtype(packages_clean_df["Giá"]), "Package price is not integer dtype"
    assert retrieval_df["search_text"].astype(str).str.strip().ne("").all(), "Empty search_text found"

    for column in ["facts_json", "services_json", "preparation_json", "terms_json"]:
        for value in retrieval_df[column].dropna():
            parsed = json.loads(value)
            assert isinstance(parsed, list), f"{column} contains non-list JSON"


def write_jsonl(df, path):
    with path.open("w", encoding="utf-8") as handle:
        for record in df.to_dict(orient="records"):
            serializable = {
                key: (None if pd.isna(value) else value)
                for key, value in record.items()
            }
            handle.write(json.dumps(serializable, ensure_ascii=False) + "\n")


def main():
    HUMAN_CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    RETRIEVAL_DIR.mkdir(parents=True, exist_ok=True)

    doctors_clean_df = build_doctors_human_clean()
    packages_clean_df = build_packages_human_clean()
    retrieval_df = build_retrieval_df(doctors_clean_df, packages_clean_df)

    validate_outputs(doctors_clean_df, packages_clean_df, retrieval_df)

    doctors_clean_df.to_csv(DOCTORS_HUMAN_CLEAN, index=False, encoding="utf-8-sig")
    packages_clean_df.to_csv(PACKAGES_HUMAN_CLEAN, index=False, encoding="utf-8-sig")
    retrieval_df.to_csv(RETRIEVAL_CSV, index=False, encoding="utf-8-sig")
    write_jsonl(retrieval_df, RETRIEVAL_JSONL)

    print(f"Wrote {DOCTORS_HUMAN_CLEAN}")
    print(f"Wrote {PACKAGES_HUMAN_CLEAN}")
    print(f"Wrote {RETRIEVAL_CSV}")
    print(f"Wrote {RETRIEVAL_JSONL}")


if __name__ == "__main__":
    main()
