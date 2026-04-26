from __future__ import annotations

import re

from .drug_extraction import normalize_search_text


def format_user_answer(answer: str | None, *, max_chars: int = 1800) -> str:
    text = str(answer or "").replace("\r\n", "\n").strip()
    if not text:
        return ""

    lines = [_normalize_line(line) for line in text.splitlines()]
    cleaned: list[str] = []
    previous_blank = False
    for line in lines:
        if _is_noise_line(line):
            continue
        if not line:
            if cleaned and not previous_blank:
                cleaned.append("")
            previous_blank = True
            continue
        cleaned.append(line)
        previous_blank = False

    cleaned = _dedupe_nearby_lines(cleaned)
    text = "\n".join(cleaned).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    if len(text) > max_chars:
        text = text[: max_chars - 16].rstrip() + "\n...[rút gọn]"
    return text


def _normalize_line(line: str) -> str:
    line = " ".join(str(line or "").split())
    line = re.sub(r"\s+([,.;:])", r"\1", line)
    return line.strip()


def _is_noise_line(line: str) -> bool:
    if not line:
        return False
    return line in {",", ";", ".", "-", "•"}


def _dedupe_nearby_lines(lines: list[str]) -> list[str]:
    deduped: list[str] = []
    seen_nonblank: set[str] = set()
    for line in lines:
        if not line:
            deduped.append(line)
            continue
        key = normalize_search_text(line)
        # Preserve dosage rows even if active ingredient names appear elsewhere,
        # but drop exact duplicate lines produced by raw HTML/link sections.
        if key in seen_nonblank:
            continue
        seen_nonblank.add(key)
        deduped.append(line)
    while deduped and not deduped[-1]:
        deduped.pop()
    return deduped
