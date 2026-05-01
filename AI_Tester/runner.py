from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import httpx

from Chat_Voice.agent.chat_bridge import VoiceChatBridge, assistant_text_for_voice
from Chat_Voice.agent.models import VoiceTurnRequest

from .judges import JudgeResult, LLMJudge, RouterLLMJudge, evaluate_with_mode
from .schema import EvalCase, EvalTurn, load_cases


DEFAULT_CASES_PATH = Path(__file__).resolve().parent / "cases" / "llm_deep_eval_20.jsonl"
DEFAULT_OUTPUT_PATH = Path(".runtime") / "ai_eval_results.jsonl"
DEFAULT_CASE_COUNT = 20
DEFAULT_TURNS_PER_CASE = 5
DEFAULT_SERVER_COMMAND = "python -m uvicorn Flow_code.api:app --host 127.0.0.1 --port 8000"


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    mode: str
    score: float
    passed: bool
    judge: JudgeResult
    transcript: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        turn_latencies = [
            float(turn.get("latency_ms") or 0.0)
            for turn in self.transcript
            if isinstance(turn, dict)
        ]
        avg_turn_latency = sum(turn_latencies) / len(turn_latencies) if turn_latencies else 0.0
        return {
            "case_id": self.case_id,
            "mode": self.mode,
            "turn_count": len(self.transcript),
            "score": round(self.score, 4),
            "passed": self.passed,
            "latency_ms": round(self.latency_ms, 2),
            "avg_turn_latency_ms": round(avg_turn_latency, 2),
            "max_turn_latency_ms": round(max(turn_latencies), 2) if turn_latencies else 0.0,
            "judge": self.judge.to_dict(),
            "transcript": self.transcript,
        }


@dataclass
class ManagedServer:
    process: Any | None = None
    started_by_runner: bool = False

    def stop(self) -> None:
        if not self.started_by_runner or self.process is None:
            return
        poll = getattr(self.process, "poll", None)
        if callable(poll) and poll() is not None:
            return
        terminate = getattr(self.process, "terminate", None)
        if callable(terminate):
            terminate()
        wait = getattr(self.process, "wait", None)
        if callable(wait):
            try:
                wait(timeout=10)
                return
            except Exception:
                pass
        kill = getattr(self.process, "kill", None)
        if callable(kill):
            kill()


class AiEvalRunner:
    def __init__(
        self,
        *,
        base_url: str,
        judge_mode: str = "hybrid",
        timeout_seconds: float = 60.0,
        http_client: httpx.Client | None = None,
        llm_judge: LLMJudge | None = None,
    ) -> None:
        self.chat_api_url = _chat_url(base_url)
        self.judge_mode = judge_mode
        self.timeout_seconds = timeout_seconds
        self.http_client = http_client
        self.llm_judge = llm_judge

    def run_cases(self, cases: list[EvalCase]) -> list[CaseResult]:
        return [self.run_case(case) for case in cases]

    def run_case(self, case: EvalCase) -> CaseResult:
        started_at = time.perf_counter()
        conversation_id: str | None = None
        voice_session_id = f"voice-eval-{case.id}"
        transcript: list[dict[str, Any]] = []

        for index, turn in enumerate(case.turns, start=1):
            if case.mode == "voice":
                result = self._run_voice_turn(turn, conversation_id, voice_session_id)
            else:
                result = self._run_chat_turn(turn, conversation_id)
            response = result["response"]
            conversation_id = str(response.get("conversation_id") or conversation_id or "")
            result["turn_index"] = index
            result["question_depth"] = index
            result["conversation_id"] = conversation_id
            result["route"] = response.get("route")
            result["status"] = response.get("status")
            result["intent"] = response.get("intent")
            result["confidence"] = response.get("confidence")
            transcript.append(result)

        judge = evaluate_with_mode(
            case=case,
            transcript=transcript,
            judge_mode=self.judge_mode,
            llm_judge=self.llm_judge,
        )
        latency_ms = (time.perf_counter() - started_at) * 1000
        return CaseResult(
            case_id=case.id,
            mode=case.mode,
            score=judge.score,
            passed=judge.passed,
            judge=judge,
            transcript=transcript,
            latency_ms=latency_ms,
        )

    def _run_chat_turn(self, turn: EvalTurn, conversation_id: str | None) -> dict[str, Any]:
        payload = _chat_payload(turn, conversation_id)
        started_at = time.perf_counter()
        response = self._post_chat(payload)
        latency_ms = (time.perf_counter() - started_at) * 1000
        return {
            "mode": "chat",
            "request": payload,
            "response": response,
            "assistant_text": assistant_text_for_voice(response),
            "latency_ms": round(latency_ms, 2),
            "llm_response_time_ms": round(latency_ms, 2),
        }

    def _run_voice_turn(
        self,
        turn: EvalTurn,
        conversation_id: str | None,
        voice_session_id: str,
    ) -> dict[str, Any]:
        bridge = VoiceChatBridge(
            self.chat_api_url,
            timeout_seconds=self.timeout_seconds,
            client=self.http_client,
        )
        request = VoiceTurnRequest(
            transcript=turn.message,
            conversation_id=conversation_id,
            voice_session_id=voice_session_id,
            selected_index=turn.selected_index,
            selected_sku=turn.selected_sku,
        )
        started_at = time.perf_counter()
        response = bridge.handle_turn(request)
        latency_ms = (time.perf_counter() - started_at) * 1000
        raw_response = dict(response.raw_response)
        raw_response["assistant_text"] = response.assistant_text
        return {
            "mode": "voice",
            "request": response.to_metadata(),
            "response": raw_response,
            "assistant_text": response.assistant_text,
            "latency_ms": round(latency_ms, 2),
            "llm_response_time_ms": round(latency_ms, 2),
        }

    def _post_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.http_client is not None:
            response = self.http_client.post(
                self.chat_api_url,
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            return response.json()

        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(self.chat_api_url, json=payload)
            response.raise_for_status()
            return response.json()


def write_results(path: str | Path, results: list[CaseResult]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")


def run_and_write_results(
    *,
    runner: AiEvalRunner,
    cases: list[EvalCase],
    output_path: str | Path,
) -> list[CaseResult]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    results: list[CaseResult] = []
    with output.open("w", encoding="utf-8") as handle:
        for case in cases:
            result = runner.run_case(case)
            results.append(result)
            handle.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
            handle.flush()
    return results


def summarize_results(results: list[CaseResult]) -> dict[str, Any]:
    total = len(results)
    passed = sum(1 for result in results if result.passed)
    average_score = sum(result.score for result in results) / total if total else 0.0
    turn_latencies = [
        float(turn.get("latency_ms") or 0.0)
        for result in results
        for turn in result.transcript
        if isinstance(turn, dict)
    ]
    average_turn_latency = (
        sum(turn_latencies) / len(turn_latencies)
        if turn_latencies
        else 0.0
    )
    return {
        "total": total,
        "turns": sum(len(result.transcript) for result in results),
        "passed": passed,
        "failed": total - passed,
        "average_score": round(average_score, 4),
        "average_turn_latency_ms": round(average_turn_latency, 2),
        "max_turn_latency_ms": round(max(turn_latencies), 2) if turn_latencies else 0.0,
        "failed_cases": [result.case_id for result in results if not result.passed],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI evaluator for chatbot and voice flows.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--judge", choices=["heuristic", "llm", "hybrid"], default="hybrid")
    parser.add_argument("--cases", default=str(DEFAULT_CASES_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--fail-under", type=float, default=0.75)
    parser.add_argument("--case-count", type=int, default=DEFAULT_CASE_COUNT)
    parser.add_argument("--turns-per-case", type=int, default=DEFAULT_TURNS_PER_CASE)
    parser.add_argument("--limit", type=int, default=None, help="Deprecated alias for --case-count.")
    parser.add_argument("--mode", choices=["chat", "voice", "all"], default="all")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--auto-server", action="store_true")
    parser.add_argument("--server-command", default=DEFAULT_SERVER_COMMAND)
    parser.add_argument("--server-health-timeout", type=float, default=60.0)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    case_count = args.limit if args.limit is not None else args.case_count
    cases = load_cases(args.cases, mode=args.mode)
    cases = prepare_deep_eval_cases(
        cases,
        case_count=case_count,
        turns_per_case=args.turns_per_case,
    )
    llm_judge = RouterLLMJudge.from_env() if args.judge in {"llm", "hybrid"} else None
    server = ensure_eval_server(
        base_url=args.base_url,
        auto_server=args.auto_server,
        server_command=args.server_command,
        health_timeout_seconds=args.server_health_timeout,
    )
    try:
        runner = AiEvalRunner(
            base_url=args.base_url,
            judge_mode=args.judge,
            timeout_seconds=args.timeout,
            llm_judge=llm_judge,
        )
        results = run_and_write_results(
            runner=runner,
            cases=cases,
            output_path=args.output,
        )
        summary = summarize_results(results)
    finally:
        server.stop()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.verbose:
        for result in results:
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    if summary["average_score"] < args.fail_under or summary["failed"]:
        return 1
    return 0


def _chat_payload(turn: EvalTurn, conversation_id: str | None) -> dict[str, Any]:
    payload: dict[str, Any] = {"message": turn.message}
    if conversation_id:
        payload["conversation_id"] = conversation_id
    if turn.selected_index is not None:
        payload["selected_index"] = turn.selected_index
    if turn.selected_sku:
        payload["selected_sku"] = turn.selected_sku
    return payload


def prepare_deep_eval_cases(
    cases: list[EvalCase],
    *,
    case_count: int,
    turns_per_case: int,
) -> list[EvalCase]:
    if case_count <= 0:
        raise ValueError("--case-count must be greater than 0.")
    if turns_per_case <= 0:
        raise ValueError("--turns-per-case must be greater than 0.")
    if len(cases) < case_count:
        raise ValueError(f"Expected at least {case_count} cases, got {len(cases)}.")

    prepared: list[EvalCase] = []
    for case in cases[:case_count]:
        if len(case.turns) != turns_per_case:
            raise ValueError(
                f"Eval case {case.id!r} must have exactly {turns_per_case} turns; "
                f"got {len(case.turns)}."
            )
        prepared.append(case)
    return prepared


def ensure_eval_server(
    *,
    base_url: str,
    auto_server: bool,
    server_command: str,
    health_timeout_seconds: float,
    health_check: Callable[[str], bool] | None = None,
    process_factory: Callable[[str], Any] | None = None,
    sleep_func: Callable[[float], None] = time.sleep,
) -> ManagedServer:
    if not auto_server:
        return ManagedServer()

    health_url = _health_url(base_url)
    checker = health_check or _is_healthy
    if checker(health_url):
        return ManagedServer(started_by_runner=False)

    factory = process_factory or _start_server_process
    process = factory(server_command)
    deadline = time.perf_counter() + health_timeout_seconds
    while time.perf_counter() < deadline:
        if checker(health_url):
            return ManagedServer(process=process, started_by_runner=True)
        poll = getattr(process, "poll", None)
        if callable(poll) and poll() is not None:
            raise RuntimeError("Auto-server process exited before /health became ready.")
        sleep_func(0.5)

    ManagedServer(process=process, started_by_runner=True).stop()
    raise TimeoutError(f"Timed out waiting for server health at {health_url}.")


def _is_healthy(health_url: str) -> bool:
    try:
        response = httpx.get(health_url, timeout=2.0)
    except httpx.HTTPError:
        return False
    return response.status_code == 200


def _start_server_process(server_command: str) -> subprocess.Popen[Any]:
    return subprocess.Popen(  # noqa: S603 - command is an explicit CLI option for local eval.
        shlex.split(server_command),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _health_url(base_url: str) -> str:
    trimmed = base_url.rstrip("/")
    if trimmed.endswith("/chat"):
        trimmed = trimmed[: -len("/chat")]
    return f"{trimmed}/health"


def _chat_url(base_url: str) -> str:
    trimmed = base_url.rstrip("/")
    return trimmed if trimmed.endswith("/chat") else f"{trimmed}/chat"


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
