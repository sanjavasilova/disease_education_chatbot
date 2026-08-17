"""Shared helpers for low-cost LLM-as-judge sampling."""

from __future__ import annotations

AVAILABLE_JUDGE_METRICS = (
    "faithfulness",
    "answer_relevance",
    "context_relevance",
    "correctness",
)


def resolve_judge_metrics(raw: str | None) -> list[str]:
    if not raw:
        return list(AVAILABLE_JUDGE_METRICS)
    requested = [m.strip() for m in raw.split(",") if m.strip()]
    if not requested:
        raise ValueError("--judge-metrics was empty")
    unknown = [m for m in requested if m not in AVAILABLE_JUDGE_METRICS]
    if unknown:
        raise ValueError(
            f"Unknown judge metric(s): {unknown}. "
            f"Choose from: {', '.join(AVAILABLE_JUDGE_METRICS)}"
        )
    return requested


def evenly_spaced_indices(n_total: int, n_sample: int) -> list[int]:
    """Pick n_sample indices spread across [0, n_total)."""
    if n_total <= 0:
        return []
    if n_sample is None or n_sample >= n_total:
        return list(range(n_total))
    if n_sample <= 0:
        return []
    if n_sample == 1:
        return [n_total // 2]
    # Inclusive endpoints for better coverage across the dataset
    return sorted({int(round(i * (n_total - 1) / (n_sample - 1))) for i in range(n_sample)})


def estimate_judge_api_calls(n_questions: int, n_metrics: int, include_answer: bool = True) -> int:
    """Rough LLM call count (answer generation + one call per judge metric)."""
    per_q = (1 if include_answer else 0) + n_metrics
    return n_questions * per_q
