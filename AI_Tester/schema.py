from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


EvalMode = Literal["chat", "voice"]
ModeFilter = Literal["all", "chat", "voice"]


@dataclass(frozen=True)
class EvalTurn:
    message: str
    selected_index: int | None = None
    selected_sku: str | None = None
    expected_route: str | None = None
    expected_status: str | None = None
    required_keywords: list[str] = field(default_factory=list)
    forbidden_keywords: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalTurn":
        message = str(data.get("message") or data.get("transcript") or "").strip()
        if not message:
            raise ValueError("Each eval turn requires a non-empty message.")
        return cls(
            message=message,
            selected_index=_optional_int(data.get("selected_index")),
            selected_sku=_optional_str(data.get("selected_sku")),
            expected_route=_optional_str(data.get("expected_route")),
            expected_status=_optional_str(data.get("expected_status")),
            required_keywords=_string_list(data.get("required_keywords")),
            forbidden_keywords=_string_list(data.get("forbidden_keywords")),
        )


@dataclass(frozen=True)
class EvalCase:
    id: str
    mode: EvalMode
    turns: list[EvalTurn]
    expected_route: str | None = None
    expected_status: str | None = None
    expected_intent: list[str] = field(default_factory=list)
    required_keywords: list[str] = field(default_factory=list)
    forbidden_keywords: list[str] = field(default_factory=list)
    expected_sources: list[str] = field(default_factory=list)
    safety_expectation: str | None = None
    min_score: float = 0.75

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalCase":
        case_id = str(data.get("id") or "").strip()
        if not case_id:
            raise ValueError("Each eval case requires an id.")

        mode = str(data.get("mode") or "chat").strip().lower()
        if mode not in {"chat", "voice"}:
            raise ValueError(f"Unsupported eval mode for {case_id}: {mode}")

        turns = [EvalTurn.from_dict(turn) for turn in data.get("turns") or []]
        if not turns:
            raise ValueError(f"Eval case {case_id} requires at least one turn.")

        return cls(
            id=case_id,
            mode=mode,  # type: ignore[arg-type]
            turns=turns,
            expected_route=_optional_str(data.get("expected_route")),
            expected_status=_optional_str(data.get("expected_status")),
            expected_intent=_string_list(data.get("expected_intent")),
            required_keywords=_string_list(data.get("required_keywords")),
            forbidden_keywords=_string_list(data.get("forbidden_keywords")),
            expected_sources=_string_list(data.get("expected_sources")),
            safety_expectation=_optional_str(data.get("safety_expectation")),
            min_score=_float_or_default(data.get("min_score"), 0.75),
        )


def load_cases(
    path: str | Path,
    *,
    mode: ModeFilter = "all",
    limit: int | None = None,
) -> list[EvalCase]:
    if limit is not None and limit <= 0:
        return []

    cases: list[EvalCase] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            case = EvalCase.from_dict(raw)
            if mode != "all" and case.mode != mode:
                continue
            cases.append(case)
            if limit is not None and len(cases) >= limit:
                break
    return cases


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    return int(value)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(item).strip() for item in value if str(item).strip()]


def _float_or_default(value: Any, default: float) -> float:
    if value in {None, ""}:
        return default
    return float(value)
