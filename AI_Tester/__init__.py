"""AI evaluation harness for the chatbot and voice adapter."""

__all__ = [
    "AiEvalRunner",
    "EvalCase",
    "EvalTurn",
    "HeuristicJudge",
    "JudgeResult",
    "RouterLLMJudge",
    "load_cases",
]


def __getattr__(name: str):
    if name == "AiEvalRunner":
        from .runner import AiEvalRunner

        return AiEvalRunner
    if name in {"HeuristicJudge", "JudgeResult", "RouterLLMJudge"}:
        from .judges import HeuristicJudge, JudgeResult, RouterLLMJudge

        return {
            "HeuristicJudge": HeuristicJudge,
            "JudgeResult": JudgeResult,
            "RouterLLMJudge": RouterLLMJudge,
        }[name]
    if name in {"EvalCase", "EvalTurn", "load_cases"}:
        from .schema import EvalCase, EvalTurn, load_cases

        return {"EvalCase": EvalCase, "EvalTurn": EvalTurn, "load_cases": load_cases}[name]
    raise AttributeError(name)
