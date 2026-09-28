"""
No-RAG baseline evaluation for MediChat.

Compares Gemini 2.5 Flash without retrieval against the existing
Gemini + RAG configuration.

Applicable LLM-as-a-Judge metrics:
    - Answer Relevance
    - Correctness

Not applicable:
    - Hit Rate
    - MRR
    - Source Precision
    - Context Relevance
    - Faithfulness (as currently defined against retrieved context)

Usage:
    python3 -m evaluation.evaluate_no_rag --limit 3
    python3 -m evaluation.evaluate_no_rag
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

_root = Path(__file__).resolve().parent.parent

if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from dotenv import load_dotenv

load_dotenv(_root / ".env")

from google import genai
from google.genai import types

from app.prompts import NO_RAG_SYSTEM_PROMPT
from evaluation.test_dataset import TEST_QUESTIONS


client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL = "gemini-2.5-flash"
JUDGE_MODEL = "gemini-2.5-flash"

REQUEST_DELAY = 2
MAX_RETRIES = 8

def _call_with_retry(fn, *args, **kwargs):
    from google.genai.errors import ClientError

    for attempt in range(MAX_RETRIES):
        try:
            time.sleep(REQUEST_DELAY)
            return fn(*args, **kwargs)

        except ClientError as e:
            if e.code == 429:
                wait = 90 * (attempt + 1)

                print(
                    f"    Rate limited. Waiting {wait}s "
                    f"before retry ({attempt + 1}/{MAX_RETRIES})..."
                )

                time.sleep(wait)

            else:
                raise

    raise RuntimeError("Max retries exceeded for Gemini API call")


def generate_no_rag_answer(question: str) -> str:

    contents = [
        types.Content(
            role="user",
            parts=[types.Part(text=question)]
        )
    ]

    response = _call_with_retry(
        client.models.generate_content,
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=NO_RAG_SYSTEM_PROMPT
        ),
    )

    return response.text


SCORING_PROMPTS = {

    "answer_relevance": (
        "You are an evaluation judge. Given a QUESTION and an ANSWER, "
        "score how relevant and helpful the answer is in addressing "
        "the question.\n\n"
        "QUESTION:\n{question}\n\n"
        "ANSWER:\n{answer}\n\n"
        "Respond with ONLY a JSON object: "
        "{{\"score\": <float 0-1>, "
        "\"reason\": \"<brief explanation>\"}}"
    ),

    "correctness": (
        "You are an evaluation judge. Given a GROUND TRUTH reference "
        "answer and the GENERATED ANSWER, score how factually correct "
        "and consistent the generated answer is compared to the ground "
        "truth. Both may be correct but phrased differently — focus on "
        "factual agreement, not wording.\n\n"
        "GROUND TRUTH:\n{ground_truth}\n\n"
        "GENERATED ANSWER:\n{answer}\n\n"
        "Respond with ONLY a JSON object: "
        "{{\"score\": <float 0-1>, "
        "\"reason\": \"<brief explanation>\"}}"
    ),
}

def judge_score(metric: str, **kwargs) -> dict:

    prompt = SCORING_PROMPTS[metric].format(**kwargs)

    response = _call_with_retry(
        client.models.generate_content,
        model=JUDGE_MODEL,
        contents=[
            types.Content(
                role="user",
                parts=[types.Part(text=prompt)]
            )
        ],
        config=types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
        ),
    )

    try:
        return json.loads(response.text)

    except json.JSONDecodeError:

        return {
            "score": 0.0,
            "reason": (
                "Failed to parse judge response: "
                + response.text[:200]
            ),
        }


def run_evaluation(limit=None, output_path=None):

    questions = TEST_QUESTIONS[:limit] if limit else TEST_QUESTIONS

    results = []

    totals = {
        "answer_relevance": 0.0,
        "correctness": 0.0,
    }

    print()
    print("=" * 70)
    print(
        f"  MediChat NO-RAG Baseline Evaluation "
        f"— {len(questions)} questions"
    )
    print("=" * 70)
    print()

    for i, item in enumerate(questions):

        question = item["question"]
        ground_truth = item["ground_truth"]

        print(
            f"[{i + 1}/{len(questions)}] "
            f"{question}"
        )

        answer = generate_no_rag_answer(question)

        print(
            f"  Answer: {answer[:150]}..."
        )

        relevance = judge_score(
            "answer_relevance",
            question=question,
            answer=answer,
        )

        correctness = judge_score(
            "correctness",
            ground_truth=ground_truth,
            answer=answer,
        )

        totals["answer_relevance"] += relevance["score"]
        totals["correctness"] += correctness["score"]

        print(
            f"  answer_relevance : "
            f"{relevance['score']:.2f}"
        )

        print(
            f"  correctness      : "
            f"{correctness['score']:.2f}"
        )

        entry = {
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,

            "judge_scores": {

                "answer_relevance": relevance,

                "correctness": correctness,
            },
        }

        results.append(entry)

        out = (
            Path(output_path)
            if output_path
            else _root / "evaluation" / "results_no_rag.json"
        )

        out.write_text(
            json.dumps(
                results,
                indent=2,
                ensure_ascii=False
            ),
            encoding="utf-8",
        )

        print()

    n = len(results)

    avg_relevance = totals["answer_relevance"] / n
    avg_correctness = totals["correctness"] / n

    print("=" * 70)
    print(
        f"  NO-RAG LLM JUDGE SCORES "
        f"(averaged over {n} questions)"
    )
    print("=" * 70)

    print(
        f"  answer_relevance : "
        f"{avg_relevance:.2f}"
    )

    print(
        f"  correctness      : "
        f"{avg_correctness:.2f}"
    )

    print("=" * 70)

    print(
        f"\nDetailed results saved to {out}"
    )

    return results

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Evaluate Gemini without RAG"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of questions to evaluate",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path",
    )

    args = parser.parse_args()

    run_evaluation(
        limit=args.limit,
        output_path=args.output,
    )