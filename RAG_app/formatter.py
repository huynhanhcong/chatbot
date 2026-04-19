from .models import RagAnswer, SearchResult


def build_sources(results: list[SearchResult]) -> list[dict[str, str | None]]:
    sources = []
    seen = set()
    for result in results:
        key = (result.id, result.source_url)
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "id": result.id,
                "title": result.title,
                "url": result.source_url,
            }
        )
    return sources


def render_answer(answer: RagAnswer) -> str:
    lines = [answer.answer.strip()]
    if answer.sources:
        lines.append("\nNguồn:")
        for index, source in enumerate(answer.sources, start=1):
            url = source.get("url") or ""
            title = source.get("title") or source.get("id") or "Nguồn"
            lines.append(f"{index}. {title}: {url}")
    lines.append(f"\nIntent: {answer.intent} | Confidence: {answer.confidence}")
    return "\n".join(lines)
