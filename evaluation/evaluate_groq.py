"""
RAG Evaluation for MediChat

Metrics:
  RETRIEVAL (no LLM calls needed — free):
    1. Hit Rate       — Did retrieval return at least one chunk from the expected source?
    2. MRR            — Mean Reciprocal Rank of the first relevant chunk
    3. Source Precision — What fraction of retrieved chunks are from expected sources?

  LLM-AS-JUDGE (uses Gemini API calls):
    4. Faithfulness    — Is the answer grounded in the retrieved context?
    5. Answer Relevance — Does the answer address the question?
    6. Context Relevance — Are the retrieved chunks relevant to the question?
    7. Correctness      — Is the answer factually consistent with the ground truth?

Usage:
    python3 -m evaluation.evaluate_groq --retrieval-only
    python3 -m evaluation.evaluate_groq --augment-results evaluation/results_groq.json \\
        --judge-sample 5 --judge-metrics correctness
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from dotenv import load_dotenv
load_dotenv(_root / ".env", override=True)

from groq import Groq
from groq import APIStatusError
from app.rag_groq import retrieve_with_metadata
from app.prompts import SYSTEM_PROMPT
from evaluation.test_dataset import TEST_QUESTIONS
from evaluation.eval_common import (
    estimate_judge_api_calls,
    evenly_spaced_indices,
    resolve_judge_metrics,
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
JUDGE_MODEL = "llama-3.3-70b-versatile"

# Free-tier limit: 5 requests/min. We pace calls and retry on 429.
REQUEST_DELAY = 4  # seconds between API calls to stay under limit
MAX_RETRIES = 6


def _call_with_retry(messages: list, temperature: float = 1.0) -> str:
    for attempt in range(MAX_RETRIES):
        try:
            time.sleep(REQUEST_DELAY)
            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except APIStatusError as e:
            if e.status_code == 429:
                wait = 65 * (attempt + 1)
                print(f"    Rate limited. Waiting {wait}s before retry ({attempt+1}/{MAX_RETRIES})...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Max retries exceeded for Groq API call")

# ---------------------------------------------------------------------------
# 1. Retrieval metrics (FREE — no API calls)
# ---------------------------------------------------------------------------
def compute_hit_rate(retrieved_sources: list[str], expected_sources: list[str]) -> float:
    """1.0 if any expected source appears in retrieved sources, else 0.0."""
    for expected in expected_sources:
        if expected in retrieved_sources:
            return 1.0
    return 0.0


def compute_mrr(retrieved_sources: list[str], expected_sources: list[str]) -> float:
    """Reciprocal rank of the first relevant chunk (1/rank). 0 if not found."""
    for rank, source in enumerate(retrieved_sources, start=1):
        if source in expected_sources:
            return 1.0 / rank
    return 0.0


def compute_source_precision(retrieved_sources: list[str], expected_sources: list[str]) -> float:
    """Fraction of retrieved chunks that come from expected sources."""
    if not retrieved_sources:
        return 0.0
    relevant = sum(1 for s in retrieved_sources if s in expected_sources)
    return relevant / len(retrieved_sources)


# ---------------------------------------------------------------------------
# 2. Generate answer (same logic as main.py /ask)
# ---------------------------------------------------------------------------
def generate_answer(question: str, context_chunks: list[str]) -> str:
    context = "\n".join(context_chunks)
    user_content = f"Context:\n{context}\n\nQuestion:\n{question}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return _call_with_retry(messages)


# ---------------------------------------------------------------------------
# 3. LLM-as-Judge scoring
# ---------------------------------------------------------------------------
SCORING_PROMPTS = {
    "faithfulness": (
        "You are an evaluation judge. Given a CONTEXT and an ANSWER, score how "
        "faithful the answer is to the context. A faithful answer only contains "
        "information that can be derived from the context, with no hallucinations.\n\n"
        "CONTEXT:\n{context}\n\n"
        "ANSWER:\n{answer}\n\n"
        "Respond with ONLY a JSON object: {{\"score\": <float 0-1>, \"reason\": \"<brief explanation>\"}}"
    ),
    "answer_relevance": (
        "You are an evaluation judge. Given a QUESTION and an ANSWER, score how "
        "relevant and helpful the answer is in addressing the question.\n\n"
        "QUESTION:\n{question}\n\n"
        "ANSWER:\n{answer}\n\n"
        "Respond with ONLY a JSON object: {{\"score\": <float 0-1>, \"reason\": \"<brief explanation>\"}}"
    ),
    "context_relevance": (
        "You are an evaluation judge. Given a QUESTION and retrieved CONTEXT chunks, "
        "score how relevant the retrieved context is to answering the question. "
        "A score of 1 means all chunks are highly relevant; 0 means none are relevant.\n\n"
        "QUESTION:\n{question}\n\n"
        "CONTEXT:\n{context}\n\n"
        "Respond with ONLY a JSON object: {{\"score\": <float 0-1>, \"reason\": \"<brief explanation>\"}}"
    ),
    "correctness": (
        "You are an evaluation judge. Given a GROUND TRUTH reference answer and the "
        "GENERATED ANSWER, score how factually correct and consistent the generated "
        "answer is compared to the ground truth. Both may be correct but phrased "
        "differently — focus on factual agreement, not wording.\n\n"
        "GROUND TRUTH:\n{ground_truth}\n\n"
        "GENERATED ANSWER:\n{answer}\n\n"
        "Respond with ONLY a JSON object: {{\"score\": <float 0-1>, \"reason\": \"<brief explanation>\"}}"
    ),
}


def judge_score(metric: str, **kwargs) -> dict:
    prompt = SCORING_PROMPTS[metric].format(**kwargs)
    messages = [{"role": "user", "content": prompt}]
    text = _call_with_retry(messages, temperature=0.0)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"score": 0.0, "reason": f"Failed to parse judge response: {text[:200]}"}

def _judge_entry(entry: dict, metrics: list[str]) -> dict:
    """Generate answer (if needed) and LLM-judge scores for one result entry."""
    question = entry["question"]
    ground_truth = entry["ground_truth"]
    expected_sources = entry.get("expected_sources", [])

    if entry.get("context_chunks"):
        context_chunks = entry["context_chunks"]
    else:
        # Re-retrieve and keep retrieval_scores consistent with the context used for judging.
        context_chunks, source_ids, _ = retrieve_with_metadata(question)
        entry["retrieved_sources"] = source_ids
        entry["context_chunks"] = context_chunks
        entry["retrieval_scores"] = {
            "hit_rate": compute_hit_rate(source_ids, expected_sources),
            "mrr": compute_mrr(source_ids, expected_sources),
            "source_precision": compute_source_precision(source_ids, expected_sources),
        }

    context = "\n---\n".join(context_chunks)
    needs_answer = any(
        m in ("faithfulness", "answer_relevance", "correctness") for m in metrics
    )
    if needs_answer:
        answer = entry.get("answer") or generate_answer(question, context_chunks)
        entry["answer"] = answer
        print(f"  Answer: {answer[:120]}...")
    else:
        answer = entry.get("answer") or ""

    judge_scores = dict(entry.get("judge_scores") or {})
    for metric in metrics:
        result = judge_score(
            metric,
            question=question,
            answer=answer,
            context=context,
            ground_truth=ground_truth,
        )
        judge_scores[metric] = result
        reason = result.get("reason", "")
        print(f"  {metric:20s}: {result['score']:.2f}  ({str(reason)[:80]})")
    entry["judge_scores"] = judge_scores
    return entry


def augment_results_with_judge(
    results_path: Path,
    judge_sample: int = 5,
    judge_metrics: list[str] | None = None,
    output_path: Path | None = None,
):
    """Add LLM-as-judge scores to a subset of an existing results JSON."""
    metrics = judge_metrics or resolve_judge_metrics(None)
    results = json.loads(Path(results_path).read_text(encoding="utf-8"))
    indices = evenly_spaced_indices(len(results), judge_sample)
    needs_answer = any(
        m in ("faithfulness", "answer_relevance", "correctness") for m in metrics
    )
    est = estimate_judge_api_calls(
        len(indices), len(metrics), include_answer=needs_answer
    )

    print(f"\n{'='*70}")
    print(f"  MediChat LLM-as-judge sample — {len(indices)}/{len(results)} questions")
    print(f"  metrics: {', '.join(metrics)}")
    print(f"  estimated LLM calls (answer+judge): ~{est}")
    print(f"{'='*70}\n")

    out = Path(output_path) if output_path else Path(results_path)

    def _save():
        out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    for rank, idx in enumerate(indices):
        entry = results[idx]
        already = entry.get("judge_scores") or {}
        missing = [m for m in metrics if m not in already]
        if not missing:
            print(f"[{rank+1}/{len(indices)}] (row {idx}) SKIP (already judged) {entry['question']}")
            continue
        print(f"[{rank+1}/{len(indices)}] (row {idx}) {entry['question']}")
        try:
            _judge_entry(entry, missing)
        finally:
            _save()  # keep answer/partial scores even if a later judge call fails
        print()

    _save()
    print(f"Updated results saved to {out}")
    print(
        "Statistical analysis:\n"
        f"  python -m evaluation.statistical_analysis auto-vs-judge {out} "
        f"--report-mk evaluation/statistical_report_mk.md "
        f"--json-out evaluation/statistical_report.json"
    )
    return results


def run_evaluation(
    limit: int | None = None,
    retrieval_only: bool = False,
    judge_metrics: list[str] | None = None,
    judge_sample: int | None = None,
    output_path: Path | None = None,
):
    questions = TEST_QUESTIONS[:limit] if limit else TEST_QUESTIONS
    results = []
    metrics = judge_metrics or resolve_judge_metrics(None)

    retrieval_metrics = {"hit_rate": 0.0, "mrr": 0.0, "source_precision": 0.0}
    judge_metric_totals = {m: 0.0 for m in metrics}
    judge_count = 0

    judge_indices = set()
    if not retrieval_only:
        n_judge = judge_sample if judge_sample is not None else len(questions)
        judge_indices = set(evenly_spaced_indices(len(questions), n_judge))
        est = estimate_judge_api_calls(len(judge_indices), len(metrics))
        mode = f"retrieval + LLM judge on {len(judge_indices)}/{len(questions)} (~{est} calls)"
    else:
        mode = "retrieval-only"

    print(f"\n{'='*70}")
    print(f"  MediChat RAG Evaluation — {len(questions)} questions [{mode}]")
    print(f"{'='*70}\n")

    for i, item in enumerate(questions):
        question = item["question"]
        ground_truth = item["ground_truth"]
        expected_sources = item.get("expected_sources", [])

        print(f"[{i+1}/{len(questions)}] {question}")

        context_chunks, source_ids, distances = retrieve_with_metadata(question)

        hit = compute_hit_rate(source_ids, expected_sources)
        mrr = compute_mrr(source_ids, expected_sources)
        precision = compute_source_precision(source_ids, expected_sources)

        retrieval_metrics["hit_rate"] += hit
        retrieval_metrics["mrr"] += mrr
        retrieval_metrics["source_precision"] += precision

        print(f"  Sources retrieved : {source_ids}")
        print(f"  Expected sources  : {expected_sources}")
        print(f"  hit_rate          : {hit:.2f}")
        print(f"  mrr               : {mrr:.2f}")
        print(f"  source_precision  : {precision:.2f}")

        entry = {
            "question": question,
            "ground_truth": ground_truth,
            "expected_sources": expected_sources,
            "retrieved_sources": source_ids,
            "retrieval_scores": {"hit_rate": hit, "mrr": mrr, "source_precision": precision},
            "context_chunks": context_chunks,
        }

        if i in judge_indices:
            _judge_entry(entry, metrics)
            judge_count += 1
            for metric in metrics:
                judge_metric_totals[metric] += entry["judge_scores"][metric]["score"]

        results.append(entry)
        print()

    n = len(questions)
    print(f"{'='*70}")
    print(f"  RETRIEVAL SCORES (averaged over {n} questions)")
    print(f"{'='*70}")
    for metric, total in retrieval_metrics.items():
        print(f"  {metric:20s}: {total / n:.2f}")

    if judge_count:
        print(f"\n{'='*70}")
        print(f"  LLM JUDGE SCORES (averaged over {judge_count} judged questions)")
        print(f"{'='*70}")
        for metric, total in judge_metric_totals.items():
            print(f"  {metric:20s}: {total / judge_count:.2f}")

        all_count = len(retrieval_metrics) + len(judge_metric_totals)
        overall = (
            sum(t / n for t in retrieval_metrics.values())
            + sum(t / judge_count for t in judge_metric_totals.values())
        ) / all_count
    else:
        overall = sum(retrieval_metrics.values()) / (n * len(retrieval_metrics))

    print(f"\n{'='*70}")
    print(f"  OVERALL SCORE     : {overall:.2f}")
    print(f"{'='*70}\n")

    out = Path(output_path) if output_path else (_root / "evaluation" / "results_groq.json")
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Detailed results saved to {out}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MediChat RAG pipeline (Groq)")
    parser.add_argument("--limit", type=int, default=None, help="Number of questions to evaluate")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Only evaluate retrieval metrics (no LLM judge, saves API quota)",
    )
    parser.add_argument(
        "--judge-metrics",
        default=None,
        help="Comma-separated judge metrics (default: all). Example: correctness",
    )
    parser.add_argument(
        "--judge-sample",
        type=int,
        default=None,
        help="Only LLM-judge this many evenly spaced questions",
    )
    parser.add_argument(
        "--augment-results",
        type=Path,
        default=None,
        help="Add judge scores onto an existing results JSON",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional output path")
    args = parser.parse_args()

    if args.augment_results:
        metrics = resolve_judge_metrics(args.judge_metrics or "correctness")
        sample = args.judge_sample if args.judge_sample is not None else 5
        augment_results_with_judge(
            args.augment_results,
            judge_sample=sample,
            judge_metrics=metrics,
            output_path=args.output,
        )
    else:
        metrics = resolve_judge_metrics(args.judge_metrics)
        run_evaluation(
            limit=args.limit,
            retrieval_only=args.retrieval_only,
            judge_metrics=metrics,
            judge_sample=args.judge_sample,
            output_path=args.output,
        )
