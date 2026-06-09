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
    python3 -m evaluation.evaluate                  # full eval (all 10 questions)
    python3 -m evaluation.evaluate --limit 3        # first 3 only
    python3 -m evaluation.evaluate --retrieval-only  # skip LLM judge (no API cost)
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

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
JUDGE_MODEL = "llama-3.3-70b-versatile"

# Free-tier limit: 5 requests/min. We pace calls and retry on 429.
REQUEST_DELAY = 2  # seconds between API calls to stay under limit
MAX_RETRIES = 3


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

# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
def run_evaluation(limit: int | None = None, retrieval_only: bool = False):
    questions = TEST_QUESTIONS[:limit] if limit else TEST_QUESTIONS
    results = []

    # Retrieval metric accumulators
    retrieval_metrics = {"hit_rate": 0.0, "mrr": 0.0, "source_precision": 0.0}
    # LLM judge accumulators
    judge_metrics = {m: 0.0 for m in SCORING_PROMPTS}

    mode = "retrieval-only" if retrieval_only else "full (retrieval + LLM judge)"
    print(f"\n{'='*70}")
    print(f"  MediChat RAG Evaluation — {len(questions)} questions [{mode}]")
    print(f"{'='*70}\n")

    for i, item in enumerate(questions):
        question = item["question"]
        ground_truth = item["ground_truth"]
        expected_sources = item.get("expected_sources", [])

        print(f"[{i+1}/{len(questions)}] {question}")

        # Step 1: Retrieve context with metadata (1 API call for embedding)
        context_chunks, source_ids, distances = retrieve_with_metadata(question)
        context = "\n---\n".join(context_chunks)

        # Step 2: Retrieval metrics (free)
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
        }

        # Step 3: LLM judge (optional — costs API calls)
        if not retrieval_only:
            answer = generate_answer(question, context_chunks)
            print(f"  Answer: {answer[:120]}...")

            judge_scores = {}
            for metric in SCORING_PROMPTS:
                result = judge_score(
                    metric,
                    question=question,
                    answer=answer,
                    context=context,
                    ground_truth=ground_truth,
                )
                judge_scores[metric] = result
                judge_metrics[metric] += result["score"]
                print(f"  {metric:20s}: {result['score']:.2f}  ({result['reason'][:80]})")

            entry["answer"] = answer
            entry["context_chunks"] = context_chunks
            entry["judge_scores"] = judge_scores

        results.append(entry)
        print()

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    n = len(questions)
    print(f"{'='*70}")
    print(f"  RETRIEVAL SCORES (averaged over {n} questions)")
    print(f"{'='*70}")
    for metric, total in retrieval_metrics.items():
        avg = total / n
        print(f"  {metric:20s}: {avg:.2f}")

    if not retrieval_only:
        print(f"\n{'='*70}")
        print(f"  LLM JUDGE SCORES (averaged over {n} questions)")
        print(f"{'='*70}")
        for metric, total in judge_metrics.items():
            avg = total / n
            print(f"  {metric:20s}: {avg:.2f}")

        all_totals = list(retrieval_metrics.values()) + list(judge_metrics.values())
        all_count = len(retrieval_metrics) + len(judge_metrics)
    else:
        all_totals = list(retrieval_metrics.values())
        all_count = len(retrieval_metrics)

    overall = sum(all_totals) / (n * all_count)
    print(f"\n{'='*70}")
    print(f"  OVERALL SCORE     : {overall:.2f}")
    print(f"{'='*70}\n")

    # Save detailed results
    output_path = _root / "evaluation" / "results_groq.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Detailed results saved to {output_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MediChat RAG pipeline")
    parser.add_argument("--limit", type=int, default=None, help="Number of questions to evaluate")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="Only evaluate retrieval metrics (no LLM judge, saves API quota)")
    args = parser.parse_args()
    run_evaluation(limit=args.limit, retrieval_only=args.retrieval_only)
