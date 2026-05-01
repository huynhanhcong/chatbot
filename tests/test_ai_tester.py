from __future__ import annotations

from collections import Counter
import json

import httpx
import pytest

from AI_Tester.judges import HeuristicJudge, RouterLLMJudge, evaluate_with_mode
from AI_Tester.runner import (
    AiEvalRunner,
    CaseResult,
    DEFAULT_CASES_PATH,
    ensure_eval_server,
    prepare_deep_eval_cases,
    run_and_write_results,
    write_results,
)
from AI_Tester.judges import JudgeResult, _extract_chat_completion_text
from AI_Tester.schema import EvalCase, EvalTurn, load_cases


def test_default_deep_case_file_has_20_cases_and_100_turns() -> None:
    cases = load_cases(DEFAULT_CASES_PATH)
    modes = Counter(case.mode for case in cases)

    assert len(cases) == 20
    assert modes == {"chat": 20}
    assert sum(len(case.turns) for case in cases) == 100
    assert all(len(case.turns) == 5 for case in cases)
    assert len({case.id for case in cases}) == 20


def test_prepare_deep_eval_cases_enforces_count_and_turns() -> None:
    cases = load_cases(DEFAULT_CASES_PATH)
    prepared = prepare_deep_eval_cases(cases, case_count=20, turns_per_case=5)

    assert len(prepared) == 20
    assert all(len(case.turns) == 5 for case in prepared)

    with pytest.raises(ValueError):
        prepare_deep_eval_cases(cases, case_count=21, turns_per_case=5)

    with pytest.raises(ValueError):
        prepare_deep_eval_cases(cases, case_count=20, turns_per_case=4)


def test_runner_preserves_conversation_id_across_chat_turns() -> None:
    seen_payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        seen_payloads.append(payload)
        if len(seen_payloads) == 1:
            return httpx.Response(
                200,
                json={
                    "status": "answered",
                    "route": "hospital_rag",
                    "conversation_id": "conv-1",
                    "answer": "Goi IVF Standard gom cac buoc chinh.",
                    "sources": [{"title": "Goi IVF Standard", "url": "https://example.test"}],
                    "intent": "package_search",
                },
            )
        return httpx.Response(
            200,
            json={
                "status": "answered",
                "route": "hospital_rag",
                "conversation_id": "conv-1",
                "answer": "Goi IVF Standard co gia 89.900.000 VND.",
                "sources": [{"title": "Goi IVF Standard", "url": "https://example.test"}],
                "intent": "price_question",
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    runner = AiEvalRunner(
        base_url="https://chat.test",
        judge_mode="heuristic",
        http_client=client,
    )
    case = EvalCase(
        id="chat-followup",
        mode="chat",
        turns=[
            EvalTurn("Goi IVF Standard gom gi?"),
            EvalTurn("Goi nay bao nhieu tien?"),
            EvalTurn("Goi nay co thuoc kich trung khong?"),
            EvalTurn("Co chi phi phat sinh nao khong?"),
            EvalTurn("Tom tat lai goi IVF Standard."),
        ],
        expected_route="hospital_rag",
        expected_status="answered",
        expected_intent=["price_question"],
        required_keywords=["89.900.000"],
        expected_sources=["IVF Standard"],
    )

    result = runner.run_case(case)

    assert result.passed
    assert seen_payloads[0] == {"message": "Goi IVF Standard gom gi?"}
    assert len(seen_payloads) == 5
    assert all(payload["conversation_id"] == "conv-1" for payload in seen_payloads[1:])
    assert result.to_dict()["turn_count"] == 5
    assert all("latency_ms" in turn for turn in result.transcript)
    assert all(turn["turn_index"] == index for index, turn in enumerate(result.transcript, start=1))
    assert all(turn["question_depth"] == index for index, turn in enumerate(result.transcript, start=1))


def test_voice_mode_uses_bridge_and_maps_spoken_selection() -> None:
    seen_payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        seen_payloads.append(payload)
        if len(seen_payloads) == 1:
            return httpx.Response(
                200,
                json={
                    "status": "need_selection",
                    "route": "pharmacity",
                    "conversation_id": "conv-1",
                    "message": "Toi tim thay cac san pham Oresol.",
                    "options": [{"index": 1, "sku": "P00219", "name": "Oresol 245"}],
                    "intent": "drug_search",
                },
            )
        return httpx.Response(
            200,
            json={
                "status": "answered",
                "route": "pharmacity",
                "conversation_id": "conv-1",
                "answer": "Thong tin thuoc Oresol 245.",
                "selected_product": {"sku": "P00219", "name": "Oresol 245"},
                "sources": [],
                "intent": "drug_followup",
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    runner = AiEvalRunner(
        base_url="https://chat.test",
        judge_mode="heuristic",
        http_client=client,
    )
    case = EvalCase(
        id="voice-selection",
        mode="voice",
        turns=[EvalTurn("Hay cho toi biet thuoc Oresol"), EvalTurn("chon so mot")],
        expected_route="pharmacity",
        expected_status="answered",
        expected_intent=["drug_followup"],
        required_keywords=["Oresol"],
    )

    result = runner.run_case(case)

    assert result.passed
    assert seen_payloads[1] == {
        "message": "1",
        "conversation_id": "conv-1",
        "selected_index": 1,
    }
    assert result.transcript[1]["assistant_text"] == "Thong tin thuoc Oresol 245."


def test_heuristic_judge_flags_route_keyword_source_and_forbidden_failures() -> None:
    case = EvalCase(
        id="bad-answer",
        mode="chat",
        turns=[EvalTurn("Goi IVF Standard gom gi?")],
        expected_route="hospital_rag",
        expected_status="answered",
        required_keywords=["IVF"],
        forbidden_keywords=["khong co"],
        expected_sources=["IVF Standard"],
    )
    transcript = [
        {
            "response": {
                "status": "answered",
                "route": "out_of_scope",
                "answer": "khong co du lieu",
                "sources": [],
            }
        }
    ]

    result = HeuristicJudge().evaluate(case, transcript)

    assert not result.passed
    assert "expected route='hospital_rag', got 'out_of_scope'" in result.issues
    assert "missing keyword/source 'IVF'" in result.issues
    assert "missing keyword/source 'IVF Standard'" in result.issues
    assert "forbidden keyword present 'khong co'" in result.issues


def test_hybrid_judge_falls_back_to_heuristic_when_llm_is_unavailable() -> None:
    case = EvalCase(
        id="fallback",
        mode="chat",
        turns=[EvalTurn("Goi IVF Standard gom gi?")],
        expected_route="hospital_rag",
        expected_status="answered",
        required_keywords=["IVF"],
    )
    transcript = [
        {
            "response": {
                "status": "answered",
                "route": "hospital_rag",
                "answer": "Goi IVF Standard gom cac buoc chinh.",
                "sources": [],
            }
        }
    ]

    result = evaluate_with_mode(
        case=case,
        transcript=transcript,
        judge_mode="hybrid",
        llm_judge=None,
    )

    assert result.passed
    assert result.judge_used == "heuristic_fallback"
    assert "Router LLM judge unavailable" in (result.rationale or "")
    for key in [
        "route",
        "context_memory",
        "retrieval_accuracy",
        "groundedness",
        "safety",
        "voice_bridge",
    ]:
        assert key in result.metrics


def test_router_llm_judge_from_env_uses_local_defaults(monkeypatch) -> None:
    monkeypatch.setenv("AI_TESTER_LLM_API_KEY", "test-key")
    monkeypatch.delenv("AI_TESTER_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("AI_TESTER_LLM_MODEL", raising=False)

    judge = RouterLLMJudge.from_env()

    assert judge is not None
    assert judge.base_url == "http://localhost:20128/v1"
    assert judge.model == "KYMA"


def test_router_llm_judge_posts_openai_compatible_request_and_parses_scores() -> None:
    seen_payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_payloads.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "relevance": 1,
                                    "groundedness": 0.8,
                                    "completeness": 0.9,
                                    "safety": 1,
                                    "conciseness": 0.7,
                                    "route": 1,
                                    "context_memory": 1,
                                    "retrieval_accuracy": 0.8,
                                    "voice_bridge": 1,
                                    "overall": 0.9,
                                    "issues": ["minor gap"],
                                    "rationale": "Looks grounded.",
                                }
                            )
                        }
                    }
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    judge = RouterLLMJudge(
        api_key="test-key",
        base_url="https://router.test/v1",
        model="KYMA",
        http_client=client,
    )
    case = EvalCase(
        id="router-smoke",
        mode="chat",
        turns=[EvalTurn("Goi IVF Standard gom gi?")],
        expected_route="hospital_rag",
        expected_status="answered",
    )
    transcript = [
        {
            "response": {
                "status": "answered",
                "route": "hospital_rag",
                "answer": "Goi IVF Standard gom cac buoc chinh.",
            }
        }
    ]

    result = judge.evaluate(case, transcript)

    assert seen_payloads[0]["model"] == "KYMA"
    assert seen_payloads[0]["stream"] is False
    assert seen_payloads[0]["temperature"] == 0
    assert seen_payloads[0]["max_tokens"] >= 2048
    assert result.judge_used == "llm"
    assert result.score == 0.9
    assert result.metrics["groundedness"] == 0.8
    assert result.issues == ["minor gap"]


def test_router_response_parser_accepts_sse_done_suffix() -> None:
    response_text = (
        'data: {"choices":[{"message":{"content":"{\\"overall\\": 1, '
        '\\"issues\\": [], \\"rationale\\": \\"ok\\"}"}}]}\n\n'
        "data: [DONE]\n"
    )

    assert _extract_chat_completion_text(response_text) == (
        '{"overall": 1, "issues": [], "rationale": "ok"}'
    )


def test_write_results_writes_jsonl_lines(tmp_path) -> None:
    results = [
        CaseResult(
            case_id=f"case-{index}",
            mode="chat",
            score=1.0,
            passed=True,
            judge=JudgeResult(judge_used="heuristic", score=1.0, passed=True),
            transcript=[{"latency_ms": 10.0}],
            latency_ms=10.0,
        )
        for index in range(20)
    ]
    output = tmp_path / "ai_eval_results.jsonl"

    write_results(output, results)

    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 20
    assert json.loads(lines[0])["turn_count"] == 1
    assert json.loads(lines[0])["avg_turn_latency_ms"] == 10.0


def test_run_and_write_results_flushes_completed_cases_before_failure(tmp_path) -> None:
    class PartialRunner:
        def __init__(self) -> None:
            self.calls = 0

        def run_case(self, case: EvalCase) -> CaseResult:
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("case failed")
            return CaseResult(
                case_id=case.id,
                mode=case.mode,
                score=1.0,
                passed=True,
                judge=JudgeResult(judge_used="heuristic", score=1.0, passed=True),
                transcript=[{"latency_ms": 5.0}],
                latency_ms=5.0,
            )

    cases = [
        EvalCase(id="case-1", mode="chat", turns=[EvalTurn("q")] * 5),
        EvalCase(id="case-2", mode="chat", turns=[EvalTurn("q")] * 5),
    ]
    output = tmp_path / "ai_eval_results.jsonl"

    with pytest.raises(RuntimeError):
        run_and_write_results(runner=PartialRunner(), cases=cases, output_path=output)

    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["case_id"] == "case-1"


def test_auto_server_reuses_existing_server_without_starting_or_stopping() -> None:
    def process_factory(command: str):
        raise AssertionError("should not start a server")

    server = ensure_eval_server(
        base_url="http://127.0.0.1:8000",
        auto_server=True,
        server_command="python -m uvicorn Flow_code.api:app",
        health_timeout_seconds=1,
        health_check=lambda url: True,
        process_factory=process_factory,
    )

    assert not server.started_by_runner
    server.stop()


def test_auto_server_stops_process_started_by_runner() -> None:
    class FakeProcess:
        def __init__(self) -> None:
            self.terminated = False

        def poll(self):
            return None

        def terminate(self) -> None:
            self.terminated = True

        def wait(self, timeout: int | float | None = None) -> int:
            return 0

    fake = FakeProcess()
    health_checks = {"count": 0}

    def health_check(url: str) -> bool:
        health_checks["count"] += 1
        return health_checks["count"] >= 2

    server = ensure_eval_server(
        base_url="http://127.0.0.1:8000",
        auto_server=True,
        server_command="python -m uvicorn Flow_code.api:app",
        health_timeout_seconds=1,
        health_check=health_check,
        process_factory=lambda command: fake,
        sleep_func=lambda seconds: None,
    )

    assert server.started_by_runner
    server.stop()
    assert fake.terminated
