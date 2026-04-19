import json


class GeminiTextClient:
    def __init__(self, api_key: str | None, model: str):
        if not api_key:
            raise RuntimeError("Missing GEMINI_API_KEY. Set it in environment or .env.")
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError("Missing dependency: install google-genai from requirements-rag.txt.") from exc

        self.model = model
        self.client = genai.Client(api_key=api_key)

    def generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(model=self.model, contents=prompt)
        return (getattr(response, "text", None) or "").strip()

    def rewrite_query(self, query: str, intent: str) -> str:
        prompt = (
            "Viết lại câu hỏi sau thành một truy vấn retrieval tiếng Việt ngắn gọn. "
            "Giữ tên bác sĩ, tên gói khám, chuyên khoa, triệu chứng, giá, dịch vụ nếu có. "
            "Không trả lời câu hỏi.\n"
            f"Intent: {intent}\n"
            f"Câu hỏi: {query}\n"
            "Truy vấn:"
        )
        rewritten = self.generate(prompt)
        return rewritten.strip().strip('"') or query

    def rerank(self, query: str, candidates: list[dict], top_k: int = 5) -> list[str]:
        compact_candidates = [
            {
                "id": item["id"],
                "title": item["title"],
                "entity_type": item["entity_type"],
                "text": item["text"][:1500],
            }
            for item in candidates
        ]
        prompt = (
            "Bạn là bộ rerank cho RAG bệnh viện. Chọn tối đa "
            f"{top_k} context liên quan nhất đến câu hỏi. "
            "Chỉ trả JSON array các id, không giải thích.\n"
            f"Câu hỏi: {query}\n"
            f"Candidates: {json.dumps(compact_candidates, ensure_ascii=False)}"
        )
        raw = self.generate(prompt)
        try:
            parsed = json.loads(_extract_json_array(raw))
        except json.JSONDecodeError:
            return [item["id"] for item in candidates[:top_k]]
        if not isinstance(parsed, list):
            return [item["id"] for item in candidates[:top_k]]
        valid_ids = {item["id"] for item in candidates}
        return [str(item) for item in parsed if str(item) in valid_ids][:top_k]


def _extract_json_array(text: str) -> str:
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return text
    return text[start : end + 1]
