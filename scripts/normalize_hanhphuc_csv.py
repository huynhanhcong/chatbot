import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DOCTORS_PATH = ROOT / "hanhphuc_doctors.csv"
PACKAGES_PATH = ROOT / "hanhphuc_goi_kham_merged_cleaned.csv"

ZERO_WIDTH_CHARS = "\u200b\u200c\u200d\ufeff"
SECTION_TITLES = [
    "Kích thích buồng trứng",
    "Chọc hút trứng",
    "Tạo phôi",
    "Chuyển phôi và thử thai",
    "Kích thích trứng",
]


def clean_text(value):
    if pd.isna(value):
        return pd.NA

    text = str(value)
    for char in ZERO_WIDTH_CHARS:
        text = text.replace(char, "")

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = text.strip()

    if text == "Không rõ":
        return pd.NA

    return text or pd.NA


def dedupe_keep_order(items):
    seen = set()
    result = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def to_json_list(items):
    items = dedupe_keep_order(items)
    if not items:
        return pd.NA
    return json.dumps(items, ensure_ascii=False)


def normalize_price(value):
    text = clean_text(value)
    if pd.isna(text):
        return pd.NA

    digits = re.sub(r"\D", "", text)
    return int(digits) if digits else pd.NA


def split_lines(text):
    text = clean_text(text)
    if pd.isna(text):
        return []

    if str(text).startswith("[") and str(text).endswith("]"):
        try:
            values = json.loads(str(text))
        except json.JSONDecodeError:
            values = None
        if isinstance(values, list):
            return [
                str(item).strip()
                for item in values
                if not pd.isna(clean_text(item))
            ]

    # Preserve section boundaries for IVF-like packages before splitting.
    text = re.sub(r"(?<!^)\s+((?:I|II|III|IV)\.\s+)", r"\n\1", text)

    lines = []
    for raw_line in text.split("\n"):
        line = clean_text(raw_line)
        if pd.isna(line):
            continue
        line = str(line).lstrip("-• ").strip()
        if line:
            lines.append(line)
    return lines


def normalize_service_list(value):
    lines = split_lines(value)
    cleaned = []

    for line in lines:
        line = re.sub(r"\s+(\d+\.\s+)", r"\n\1", line)
        for part in line.split("\n"):
            part = part.strip()
            if not part:
                continue

            match = re.match(r"^(I|II|III|IV)\.\s+(.*)$", part)
            if match:
                part = match.group(2).strip()

            match = re.match(r"^\d+\.\s+(.*)$", part)
            if match:
                part = match.group(1).strip()

            for title in SECTION_TITLES:
                prefix = f"{title} "
                if part == title:
                    cleaned.append(title)
                    part = ""
                    break
                if part.startswith(prefix):
                    remainder = part[len(prefix):].strip()
                    if remainder and re.match(r"^[A-ZÀ-Ỵ]", remainder):
                        cleaned.append(title)
                        part = remainder
                    break

            if part:
                cleaned.append(part)

    merged = []
    for item in cleaned:
        item = item.replace("Tư vấn với bác sĩ Tư vấn với bác sĩ", "Tư vấn với bác sĩ")
        if item == "Gói khám bao gồm":
            continue
        if item.startswith("(") and merged:
            merged[-1] = f"{merged[-1]} {item}"
            continue
        merged.append(item)

    return to_json_list(merged)


def normalize_term_item(line):
    mappings = [
        (
            "Trong quá trình thăm khám, ngoài các dịch vụ có trong gói",
            "Có thể phát sinh chi phí ngoài gói theo chỉ định bác sĩ.",
        ),
        (
            "Chi phí gói không bao gồm thuốc kê đơn",
            "Không bao gồm thuốc kê đơn và các mục không nêu trong gói.",
        ),
        (
            "Chi phí gói đã bao gồm thuốc kích trứng.",
            "Đã bao gồm thuốc kích trứng.",
        ),
        (
            "Trong quá trình thăm khám, ngoài các thuốc trong gói",
            "Có thể phát sinh thêm thuốc điều dưỡng ngoài gói theo chỉ định bác sĩ.",
        ),
        (
            "Gói không được chuyển nhượng",
            "Gói không được chuyển nhượng.",
        ),
        (
            "Gói khám sẽ có hiệu lực kích hoạt trong vòng 365 ngày kể từ ngày mua gói.",
            "Gói có hiệu lực kích hoạt trong 365 ngày kể từ ngày mua.",
        ),
        (
            "Các Điều Khoản & Điều Kiện này có thể thay đổi tùy theo thời điểm.",
            "Điều khoản có thể thay đổi theo thời điểm.",
        ),
        (
            "Giá gói, Điều Khoản & Điều Kiện có thể được thay đổi và cập nhật tùy theo chính sách của bệnh viện tùy từng thời điểm.",
            "Giá và điều khoản có thể thay đổi theo chính sách bệnh viện.",
        ),
    ]

    for prefix, replacement in mappings:
        if line.startswith(prefix):
            return replacement

    return line


def normalize_terms_list(value):
    lines = split_lines(value)
    return to_json_list([normalize_term_item(line) for line in lines])


def normalize_prep_item(line, index):
    if index == 0 and ":" in line:
        prefix, suffix = line.split(":", 1)
        if prefix.startswith("Để quá trình") or prefix.startswith("Quý khách vui lòng"):
            line = suffix.strip()
    elif index == 0 and (line.startswith("Để quá trình") or line.startswith("Quý khách vui lòng")):
        return None

    return line or None


def normalize_prep_list(value):
    lines = split_lines(value)
    cleaned = []
    for index, line in enumerate(lines):
        item = normalize_prep_item(line, index)
        if item:
            cleaned.append(item)
    return to_json_list(cleaned)


def normalize_doctors_csv():
    df = pd.read_csv(DOCTORS_PATH)
    for column in df.columns:
        df[column] = df[column].map(clean_text)

    df.to_csv(DOCTORS_PATH, index=False, encoding="utf-8-sig")


def normalize_packages_csv():
    df = pd.read_csv(PACKAGES_PATH)

    for column in df.columns:
        if column == "Giá":
            continue
        df[column] = df[column].map(clean_text)

    df["Giá"] = df["Giá"].map(normalize_price).astype("Int64")
    df["Dịch vụ trong gói"] = df["Dịch vụ trong gói"].map(normalize_service_list)
    df["Điều kiện và điều khoản"] = df["Điều kiện và điều khoản"].map(normalize_terms_list)
    df["Chuẩn bị trước tầm soát"] = df["Chuẩn bị trước tầm soát"].map(normalize_prep_list)

    df.to_csv(PACKAGES_PATH, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    normalize_doctors_csv()
    normalize_packages_csv()
