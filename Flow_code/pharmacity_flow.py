from __future__ import annotations

import logging
import re
import time
from typing import Any, Protocol

from RAG_app.config import load_settings

from .cached_pharmacity_client import CachedPharmacityClient
from .drug_extraction import extract_drug_name_local
from .drug_summary import answer_product_follow_up_template, render_product_summary
from .gemini_assistant import GeminiDrugAssistant, normalize_text
from .models import ProductDetail, ProductOption
from .pharmacity_export import (
    DEFAULT_PHARMACITY_EXPORT_PATH,
    DEFAULT_PHARMACITY_EXTRACTED_PATH,
    export_extracted_product_info,
    export_product_detail,
)
from .pharmacity_client import PharmacityApiClient
from .response_formatter import format_user_answer
from .session_store import InMemorySessionStore, SearchSession, format_drug_history


class FlowValidationError(ValueError):
    pass


class FlowNotFoundError(RuntimeError):
    pass


class FlowDependencyError(RuntimeError):
    pass


logger = logging.getLogger("uvicorn.error")


class PharmacitySearchClient(Protocol):
    def search_products(self, keyword: str, max_options: int = 4) -> list[ProductOption]:
        ...

    def fetch_product_detail(self, product: ProductOption) -> ProductDetail:
        ...


class DrugAssistant(Protocol):
    def extract_drug_name(self, message: str) -> str | None:
        ...

    def summarize_product(self, detail: ProductDetail) -> str:
        ...

    def answer_follow_up(
        self,
        detail: ProductDetail,
        question: str,
        previous_answer: str | None = None,
    ) -> str:
        ...


class PharmacityFlow:
    def __init__(
        self,
        client: PharmacitySearchClient | None = None,
        assistant: DrugAssistant | None = None,
        session_store: InMemorySessionStore | None = None,
        max_options: int = 4,
        export_path: str | None = str(DEFAULT_PHARMACITY_EXPORT_PATH),
        extracted_export_path: str | None = str(DEFAULT_PHARMACITY_EXTRACTED_PATH),
    ) -> None:
        self.client = client or self._build_default_client()
        self.assistant = assistant or self._build_default_assistant()
        self.session_store = session_store or InMemorySessionStore()
        self.max_options = max_options
        self.export_path = export_path
        self.extracted_export_path = extracted_export_path

    def handle_message(
        self,
        message: str,
        conversation_id: str | None = None,
        selected_index: int | None = None,
        selected_sku: str | None = None,
    ) -> dict[str, Any]:
        message = (message or "").strip()
        if not message and selected_index is None and not selected_sku:
            raise FlowValidationError("message is required.")

        if selected_sku or selected_index is not None:
            return self._answer_selected_product(
                conversation_id=conversation_id,
                selected_index=selected_index,
                selected_sku=selected_sku,
                user_message=message,
            )

        session = self.session_store.get(conversation_id)
        text_selection = parse_product_selection(message) if session else None
        if session and text_selection is not None:
            return self._answer_selected_product(
                conversation_id=conversation_id,
                selected_index=text_selection,
                selected_sku=None,
                user_message=message,
            )
        if session and session.selected_detail is None and _should_reuse_pending_selection(session, message):
            return self._pending_selection_response(session)
        if session and session.selected_detail is not None:
            return self._answer_follow_up(session=session, question=message)

        started_at = time.perf_counter()
        drug_name = extract_drug_name_local(message)
        if not drug_name:
            drug_name = self.assistant.extract_drug_name(message)
            _log_latency("pharmacity_llm_extract", started_at)
        if not drug_name:
            raise FlowValidationError("Could not detect a drug name from message.")

        options = self.client.search_products(drug_name, max_options=self.max_options)
        _log_latency("pharmacity_search", started_at, option_count=len(options))
        session = self.session_store.save_search(
            drug_name=drug_name,
            options=options,
            question=message,
            conversation_id=conversation_id,
        )

        if not options:
            return {
                "status": "not_found",
                "conversation_id": session.conversation_id,
                "message": f"Hiện chưa tìm thấy sản phẩm phù hợp với '{drug_name}'.",
                "options": [],
                "internal_grounding": {
                    "provider": "pharmacity",
                    "drug_name": drug_name,
                },
            }

        return self._selection_response(
            session,
            message="Mình đã tìm thấy vài sản phẩm phù hợp. Bạn chọn 1 sản phẩm để mình trả lời chính xác hơn nhé.",
        )

    def start_drug_info_flow(
        self,
        message: str,
        conversation_id: str | None = None,
    ) -> dict[str, Any]:
        return self.handle_message(message=message, conversation_id=conversation_id)

    def select_product(self, conversation_id: str, choice_index: int) -> dict[str, Any]:
        return self.handle_message(
            message="",
            conversation_id=conversation_id,
            selected_index=choice_index,
        )

    def has_active_session(self, conversation_id: str | None) -> bool:
        return self.session_store.get(conversation_id) is not None

    def get_session(self, conversation_id: str | None) -> SearchSession | None:
        return self.session_store.get(conversation_id)

    def _pending_selection_response(self, session: SearchSession) -> dict[str, Any]:
        return self._selection_response(
            session,
            message="Bạn chọn giúp mình 1 sản phẩm trong danh sách ở trên để mình trả lời đúng sản phẩm nhé.",
        )

    def _selection_response(self, session: SearchSession, *, message: str) -> dict[str, Any]:
        return {
            "status": "need_selection",
            "conversation_id": session.conversation_id,
            "message": message,
            "options": [option.to_response() for option in session.options],
            "internal_grounding": {
                "provider": "pharmacity",
                "drug_name": session.drug_name,
                "option_count": len(session.options),
            },
        }

    def _answer_selected_product(
        self,
        conversation_id: str | None,
        selected_index: int | None,
        selected_sku: str | None,
        user_message: str | None = None,
    ) -> dict[str, Any]:
        session = self._require_session(conversation_id)
        option = self._resolve_option(session, selected_index, selected_sku)
        started_at = time.perf_counter()
        detail = self.client.fetch_product_detail(option)
        _log_latency("pharmacity_detail", started_at)
        self._export_product_detail(detail)
        question = _resolve_selection_question(session, user_message)

        answer = answer_product_follow_up_template(detail, question or "")
        if not answer:
            answer = self._summarize_selected_product(detail)
        answer = format_user_answer(answer)

        self.session_store.save_selected_detail(
            conversation_id=session.conversation_id,
            selected_detail=detail,
            last_answer=answer,
            question=question or f"Chọn sản phẩm {detail.name}",
        )

        return {
            "status": "answered",
            "conversation_id": session.conversation_id,
            "answer": answer,
            "selected_product": detail.selected_product_response(),
            "source_url": detail.source_url,
            "internal_grounding": {
                "provider": "pharmacity",
                "drug_name": session.drug_name,
                "selected_product": detail.selected_product_response(),
                "source_url": detail.source_url,
            },
        }

    def _answer_follow_up(self, session: SearchSession, question: str) -> dict[str, Any]:
        if session.selected_detail is None:
            raise FlowValidationError("No selected product in current conversation.")
        templated_answer = answer_product_follow_up_template(session.selected_detail, question)
        if templated_answer:
            answer = templated_answer
        else:
            previous_context = format_drug_history(session) or session.last_answer
            try:
                answer = self.assistant.answer_follow_up(
                    detail=session.selected_detail,
                    question=question,
                    previous_answer=previous_context,
                )
            except RuntimeError as exc:
                raise FlowDependencyError(str(exc)) from exc
        answer = format_user_answer(answer)
        self.session_store.save_selected_detail(
            conversation_id=session.conversation_id,
            selected_detail=session.selected_detail,
            last_answer=answer,
            question=question,
        )
        return {
            "status": "answered",
            "conversation_id": session.conversation_id,
            "answer": answer,
            "selected_product": session.selected_detail.selected_product_response(),
            "source_url": session.selected_detail.source_url,
            "internal_grounding": {
                "provider": "pharmacity",
                "drug_name": session.drug_name,
                "selected_product": session.selected_detail.selected_product_response(),
                "source_url": session.selected_detail.source_url,
            },
        }

    def _summarize_selected_product(self, detail: ProductDetail) -> str:
        try:
            return self.assistant.summarize_product(detail)
        except RuntimeError:
            return render_product_summary(detail)

    def _export_product_detail(self, detail: ProductDetail) -> None:
        if not self.export_path and not self.extracted_export_path:
            return
        try:
            if self.export_path:
                export_product_detail(detail, path=self.export_path)
            if self.extracted_export_path:
                export_extracted_product_info(detail, path=self.extracted_export_path)
        except OSError as exc:
            logger.warning("pharmacity_export_failed sku=%s error=%s", detail.sku, exc)

    def _require_session(self, conversation_id: str | None) -> SearchSession:
        session = self.session_store.get(conversation_id)
        if session is None:
            raise FlowNotFoundError("Conversation not found or expired. Please search again.")
        return session

    def _resolve_option(
        self,
        session: SearchSession,
        selected_index: int | None,
        selected_sku: str | None,
    ) -> ProductOption:
        if selected_sku:
            selected_sku = selected_sku.strip().upper()
            for option in session.options:
                if option.sku.upper() == selected_sku:
                    return option
            raise FlowValidationError("selected_sku does not match current search options.")

        if selected_index is None:
            raise FlowValidationError("selected_index or selected_sku is required.")
        if selected_index < 1 or selected_index > len(session.options):
            raise FlowValidationError(
                f"selected_index must be between 1 and {len(session.options)}."
            )
        return session.options[selected_index - 1]

    def _build_default_client(self) -> PharmacitySearchClient:
        settings = load_settings()
        return CachedPharmacityClient(
            api_client=PharmacityApiClient(timeout_seconds=6.0),
            index_path=settings.pharmacity_index_path,
            cache_ttl_seconds=settings.cache_ttl_seconds,
        )

    def _build_default_assistant(self) -> DrugAssistant:
        try:
            settings = load_settings()
            return GeminiDrugAssistant.from_api_key(
                api_key=settings.gemini_api_key,
                model=settings.gemini_generation_model,
            )
        except Exception as exc:
            raise FlowDependencyError(f"Could not initialize Gemini client: {exc}") from exc


def parse_product_selection(message: str) -> int | None:
    normalized = normalize_text(message).strip()
    normalized = re.sub(r"\s+", " ", normalized)

    ordinal_map = {
        "dau tien": 1,
        "thu nhat": 1,
        "so mot": 1,
        "mot": 1,
        "thu hai": 2,
        "so hai": 2,
        "hai": 2,
        "thu ba": 3,
        "so ba": 3,
        "ba": 3,
        "thu tu": 4,
        "so bon": 4,
        "bon": 4,
        "thu nam": 5,
        "so nam": 5,
        "nam": 5,
    }
    for phrase, value in ordinal_map.items():
        if normalized in {phrase, f"thuoc {phrase}", f"lua chon {phrase}"}:
            return value

    digit_match = re.fullmatch(
        r"(?:thuoc|lua chon|option|so|#)?\s*([1-9][0-9]?)",
        normalized,
    )
    if digit_match:
        return int(digit_match.group(1))

    explicit_match = re.fullmatch(
        r"(?:chon|toi chon|lay|xem)\s+(?:thuoc|lua chon|option|so)?\s*([1-9][0-9]?)",
        normalized,
    )
    if explicit_match:
        return int(explicit_match.group(1))

    return None


def _resolve_selection_question(
    session: SearchSession,
    user_message: str | None,
) -> str | None:
    message = (user_message or "").strip()
    if message and parse_product_selection(message) is None:
        return message
    return session.requested_question or message or None


def _should_reuse_pending_selection(session: SearchSession, message: str) -> bool:
    normalized = normalize_text(message).strip()
    if not normalized:
        return True

    if parse_product_selection(message) is not None:
        return True

    extracted_drug_name = extract_drug_name_local(message)
    if extracted_drug_name:
        if normalize_text(extracted_drug_name) != normalize_text(session.drug_name):
            return False
        return True

    return any(
        keyword in normalized
        for keyword in [
            "san pham",
            "thuoc nao",
            "loai nao",
            "chon",
            "gia",
            "bao nhieu",
            "thanh phan",
            "cong dung",
            "cach dung",
            "ke don",
            "xem them",
        ]
    )


def _log_latency(event: str, started_at: float, **fields: object) -> None:
    elapsed_ms = (time.perf_counter() - started_at) * 1000
    suffix = " ".join(f"{key}={value}" for key, value in fields.items())
    logger.info("%s latency_ms=%.2f %s", event, elapsed_ms, suffix)
