from __future__ import annotations

import json
import re
import unicodedata
from typing import Protocol

from .models import ProductDetail


class TextGenerator(Protocol):
    def generate(self, prompt: str) -> str:
        ...


class GeminiDrugAssistant:
    def __init__(self, generator: TextGenerator) -> None:
        self.generator = generator

    @classmethod
    def from_api_key(cls, api_key: str | None, model: str) -> GeminiDrugAssistant:
        from RAG_app.gemini_client import GeminiTextClient

        return cls(GeminiTextClient(api_key, model))

    def extract_drug_name(self, message: str) -> str | None:
        prompt = (
            "Ban la bo trich xuat ten thuoc tu tin nhan tieng Viet.\n"
            "Chi tra ve JSON hop le, khong markdown, theo schema: "
            '{"drug_name":"ten thuoc can tim"}.\n'
            "Neu khong co ten thuoc, tra ve {\"drug_name\":null}.\n\n"
            f"Tin nhan: {message}\n"
            "JSON:"
        )
        raw = self.generator.generate(prompt)
        drug_name = _parse_drug_name_json(raw)
        return drug_name or _fallback_extract_drug_name(message)

    def summarize_product(self, detail: ProductDetail) -> str:
        context = detail.to_context_text()
        prompt = (
            "Ban la tro ly thong tin thuoc cho webapp. Hay tom tat bang tieng Viet, "
            "chi dua tren DU LIEU PHARMACITY ben duoi.\n"
            "Quy tac bat buoc:\n"
            "- Khong tu them cong dung, lieu dung, canh bao hoac gia ngoai du lieu.\n"
            "- Neu truong nao khong co du lieu, bo qua truong do.\n"
            "- Neu la thuoc ke don, noi ro can tu van bac si/duoc si.\n"
            "- Luon them cau: Thong tin chi mang tinh tham khao, khong thay the tu van y te.\n"
            "- Cau truc ngan gon gom cac muc neu co: Ten thuoc, Thanh phan, Cong dung/chi dinh, "
            "Cach dung/lieu dung, Chong chi dinh/than trong/tac dung phu, Dong goi/gia, Nguon.\n\n"
            f"DU LIEU PHARMACITY:\n{context}\n\n"
            "Tom tat:"
        )
        summary = self.generator.generate(prompt).strip()
        if not summary:
            raise RuntimeError("Gemini returned an empty product summary.")
        return summary

    def answer_follow_up(
        self,
        detail: ProductDetail,
        question: str,
        previous_answer: str | None = None,
    ) -> str:
        context = detail.to_context_text()
        prompt = (
            "Ban la tro ly thong tin thuoc trong mot cuoc hoi dap lien tuc. "
            "Hay tra loi cau hoi moi bang tieng Viet, dua tren SAN PHAM DANG NOI TOI "
            "va lich su hoi thoai neu can.\n"
            "Quy tac bat buoc:\n"
            "- Khong doi sang san pham khac.\n"
            "- Khong tu them thong tin ngoai du lieu Pharmacity.\n"
            "- Dung LICH_SU_HOI_THOAI de hieu cau hoi tiep noi, nhung cau tra loi phai dua tren SAN PHAM DANG NOI TOI.\n"
            "- Neu cau hoi khong the tra loi tu du lieu, noi ro chua co thong tin trong du lieu.\n"
            "- Luon them cau: Thong tin chi mang tinh tham khao, khong thay the tu van y te.\n\n"
            f"SAN PHAM DANG NOI TOI:\n{context}\n\n"
            f"LICH_SU_HOI_THOAI:\n{previous_answer or 'Khong co'}\n\n"
            f"CAU HOI MOI: {question}\n\n"
            "Tra loi:"
        )
        answer = self.generator.generate(prompt).strip()
        if not answer:
            raise RuntimeError("Gemini returned an empty follow-up answer.")
        return answer


def _parse_drug_name_json(raw: str) -> str | None:
    text = _extract_json_object(raw.strip())
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    value = parsed.get("drug_name")
    if value is None:
        return None
    name = str(value).strip().strip("\"'")
    return name or None


def _extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return text
    return text[start : end + 1]


def _fallback_extract_drug_name(message: str) -> str | None:
    cleaned = message.strip()
    if not cleaned:
        return None

    match = re.search(
        r"(?:thuoc|thuốc|san pham|sản phẩm)\s+([A-Za-z0-9][\w .+-]{1,80})",
        cleaned,
        flags=re.IGNORECASE,
    )
    if match:
        return _strip_filler(match.group(1))

    words = [word for word in re.split(r"\s+", cleaned) if word]
    if 1 <= len(words) <= 5:
        return _strip_filler(cleaned)
    return None


def _strip_filler(value: str) -> str | None:
    value = re.split(r"[?.!,;:\n]", value, maxsplit=1)[0].strip()
    value = re.sub(r"(?i)\b(la gi|là gì|can tim|cần tìm)$", "", value).strip()
    return value or None


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value)
    normalized = "".join(char for char in normalized if unicodedata.category(char) != "Mn")
    normalized = normalized.replace("đ", "d").replace("Đ", "D")
    return normalized.lower()
