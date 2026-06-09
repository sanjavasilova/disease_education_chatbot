import sys
import os
import time
import random
import logging
import argparse
from collections import defaultdict, Counter
from pathlib import Path
from dotenv import load_dotenv

_root = Path(__file__).resolve().parent.parent
if _root not in (Path(p).resolve() for p in sys.path):
    sys.path.insert(0, str(_root))

from ingest.extract import extract_html, extract_pdf, extract_csv
from ingest.chunk import chunk_text
from data.sources import SOURCES
from vectorstore.chroma_client_groq import get_chroma_client
from groq import Groq
from sentence_transformers import SentenceTransformer

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("build_db_groq.log"),
    ],
)
log = logging.getLogger(__name__)

BATCH_SIZE = 100
DELAY_BETWEEN_BATCHES = 7.0

SYMPTOM_FREQUENCY_THRESHOLD = 0.2

def _process_disease_symptoms(rows: list[dict], threshold: float) -> list[str]:
    disease_case_counts: Counter = Counter()
    disease_symptom_counts: dict[str, Counter] = defaultdict(Counter)

    for row in rows:
        disease = row.get("diseases", "Unknown").strip()
        if not disease:
            continue
        disease_case_counts[disease] += 1
        for symptom, value in row.items():
            if symptom != "diseases" and value == "1":
                disease_symptom_counts[disease][symptom] += 1

    chunks = []
    for disease, total_cases in disease_case_counts.items():
        frequent_symptoms = sorted(
            symptom
            for symptom, count in disease_symptom_counts[disease].items()
            if count / total_cases >= threshold
        )

        if not frequent_symptoms:
            frequent_symptoms = [
                s for s, _ in disease_symptom_counts[disease].most_common(5)
            ]

        if frequent_symptoms:
            chunks.append(
                f"Disease: {disease}. "
                f"Common symptoms (present in \u2265{int(threshold * 100)}% of cases): "
                f"{', '.join(frequent_symptoms)}."
            )
        log.debug(
            "  %s: %d cases, %d symptoms above threshold",
            disease, total_cases, len(frequent_symptoms),
        )

    log.info(
        "  csv_special: %d diseases processed with %.0f%% frequency threshold",
        len(chunks), threshold * 100,
    )
    return chunks


def _embed(model: SentenceTransformer, texts: list[str]) -> list:
    return model.encode(texts, normalize_embeddings=True).tolist()

def build_db(
    sample_size: int | None = None,
    shuffle: bool = True,
    symptom_threshold: float = SYMPTOM_FREQUENCY_THRESHOLD,
) -> None:
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    chroma = get_chroma_client()
    collection = chroma.get_or_create_collection(name="medical_docs")

    failed_sources: list[str] = []

    for src_idx, src in enumerate(SOURCES):
        src_id = src["id"]

        try:
            log.info("Processing %s ...", src_id)

            if src["type"] == "html":
                text   = extract_html(src["url"])
                chunks = chunk_text(text)

            elif src["type"] == "pdf":
                path = Path(src["path"])
                if not path.is_absolute():
                    path = _root / path
                text   = extract_pdf(str(path))
                chunks = chunk_text(text)

            elif src["type"] == "csv":
                path = Path(src["path"])
                if not path.is_absolute():
                    path = _root / path
                rows   = extract_csv(str(path))
                chunks = [
                    ", ".join(f"{k}: {v}" for k, v in row.items())
                    for row in rows
                ]

            elif src["type"] == "csv_special":
                path = Path(src["path"])
                if not path.is_absolute():
                    path = _root / path
                rows   = extract_csv(str(path))
                chunks = _process_disease_symptoms(rows, symptom_threshold)

            else:
                log.warning("Unknown source type '%s' for %s — skipping.", src["type"], src_id)
                continue

            existing_result = collection.get(where={"source": src_id}, include=[])
            existing_ids    = set(existing_result["ids"])
            to_embed_count  = sum(
                1 for i in range(len(chunks))
                if f"{src_id}_{i}" not in existing_ids
            )
            log.info(
                "%s: %d chunks produced, %d already stored, %d to embed",
                src_id, len(chunks), len(existing_ids), to_embed_count,
            )

            for batch_start in range(0, len(chunks), BATCH_SIZE):
                batch     = chunks[batch_start : batch_start + BATCH_SIZE]
                batch_ids = [
                    f"{src_id}_{i}"
                    for i in range(batch_start, batch_start + len(batch))
                ]

                new_indices = [i for i, bid in enumerate(batch_ids) if bid not in existing_ids]
                if not new_indices:
                    continue

                texts_to_embed = [batch[i]     for i in new_indices]
                ids_to_embed   = [batch_ids[i] for i in new_indices]

                if sample_size and len(texts_to_embed) > sample_size:
                    combined = list(zip(texts_to_embed, ids_to_embed))
                    if shuffle:
                        random.shuffle(combined)
                    combined       = combined[:sample_size]
                    texts_to_embed = [c[0] for c in combined]
                    ids_to_embed   = [c[1] for c in combined]

                embeddings = _embed(embed_model, texts_to_embed)

                collection.add(
                    documents=texts_to_embed,
                    embeddings=embeddings,
                    metadatas=[{"source": src_id}] * len(texts_to_embed),
                    ids=ids_to_embed,
                )

            log.info("%s: done.", src_id)

        except Exception:
            log.exception("ERROR processing %s — continuing with remaining sources.", src_id)
            failed_sources.append(src_id)

    if failed_sources:
        log.error(
            "Pipeline completed with %d failed source(s): %s",
            len(failed_sources),
            ", ".join(failed_sources),
        )
        sys.exit(1)
    else:
        log.info("Pipeline completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the medical vector database.")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Randomly sample this many chunks per batch (useful for testing).",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable shuffling before sampling.",
    )
    parser.add_argument(
        "--symptom-threshold",
        type=float,
        default=SYMPTOM_FREQUENCY_THRESHOLD,
        help=(
            "Minimum fraction of cases a symptom must appear in to be included "
            "in the disease chunk (default: 0.2 = 20%%)."
        ),
    )
    args = parser.parse_args()

    build_db(
        sample_size=args.sample_size,
        shuffle=not args.no_shuffle,
        symptom_threshold=args.symptom_threshold,
    )