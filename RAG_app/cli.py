import argparse
import json
import sys
from pathlib import Path

from rich.console import Console

from .config import load_settings
from .evaluator import evaluate
from .formatter import render_answer
from .ingest import ingest_hanhphuc
from .pipeline import RagPipeline


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

console = Console(force_terminal=False, legacy_windows=False)


def _cmd_ingest(args: argparse.Namespace) -> None:
    if args.source != "hanhphuc":
        raise SystemExit("V1 only supports --source hanhphuc")
    settings = load_settings()
    stats = ingest_hanhphuc(settings, recreate=args.recreate, batch_size=args.batch_size)
    console.print(json.dumps(stats, ensure_ascii=False, indent=2))


def _cmd_ask(args: argparse.Namespace) -> None:
    settings = load_settings()
    answer = RagPipeline(settings).answer(args.question)
    if args.json:
        console.print(answer.model_dump_json(indent=2))
    else:
        console.print(render_answer(answer))


def _cmd_eval(args: argparse.Namespace) -> None:
    settings = load_settings()
    qa_path = Path(args.qa) if args.qa else settings.qa_path
    stats = evaluate(settings, qa_path=qa_path, limit=args.limit)
    console.print(json.dumps(stats, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local RAG CLI for Hanh Phuc hospital data.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Embed and upload Hanh Phuc entities to Qdrant.")
    ingest.add_argument("--source", default="hanhphuc", help="Only 'hanhphuc' is supported in v1.")
    ingest.add_argument("--recreate", action="store_true", help="Recreate the Qdrant collection before upload.")
    ingest.add_argument("--batch-size", type=int, default=16, help="Gemini embedding batch size.")
    ingest.set_defaults(func=_cmd_ingest)

    ask = subparsers.add_parser("ask", help="Ask a question through the local RAG pipeline.")
    ask.add_argument("question", help="User question.")
    ask.add_argument("--json", action="store_true", help="Print the structured JSON response.")
    ask.set_defaults(func=_cmd_ask)

    eval_cmd = subparsers.add_parser("eval", help="Evaluate retrieval/generation on QA JSONL.")
    eval_cmd.add_argument("--qa", default=None, help="Path to eval QA JSONL.")
    eval_cmd.add_argument("--limit", type=int, default=None, help="Optional max questions for smoke eval.")
    eval_cmd.set_defaults(func=_cmd_eval)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except RuntimeError as exc:
        raise SystemExit(f"Error: {exc}") from exc
    except FileNotFoundError as exc:
        raise SystemExit(f"Missing file: {exc}") from exc


if __name__ == "__main__":
    main()
