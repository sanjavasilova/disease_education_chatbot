import sys
import os
import time
from pathlib import Path
from dotenv import load_dotenv

_root = Path(__file__).resolve().parent.parent
if _root not in (Path(p).resolve() for p in sys.path):
    sys.path.insert(0, str(_root))

from ingest.extract import extract_html, extract_pdf, extract_csv
from ingest.chunk import chunk_text
from data.sources import SOURCES
from vectorstore.chroma_client import get_chroma_client
from google import genai
from google.genai.errors import ClientError

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

chroma = get_chroma_client()
collection = chroma.get_or_create_collection(name="medical_docs")

BATCH_SIZE = 10
DELAY_BETWEEN_BATCHES = 7.0 
def build_db():
    for src_idx, src in enumerate(SOURCES):
        src_id = src["id"]
        if src_idx > 0:
            time.sleep(DELAY_BETWEEN_BATCHES)
        try:
            print(f"Processing {src_id}...", flush=True)
            if src["type"] == "html":
                text = extract_html(src["url"])
                chunks = chunk_text(text)
            elif src["type"] == "pdf":
                path = src["path"]
                if not Path(path).is_absolute():
                    path = _root / path
                text = extract_pdf(str(path))
                chunks = chunk_text(text)
            elif src["type"] == "csv":
                path = src["path"]
                if not Path(path).is_absolute():
                    path = _root / path
                rows = extract_csv(str(path))
                # Convert rows to descriptive text
                chunks = [", ".join([f"{k}: {v}" for k, v in row.items()]) for row in rows]
                # Limit to first 200 for demo purposes
                chunks = chunks[:200]
            elif src["type"] == "csv_special":
                # Specialized handling for Mendeley disease-symptom dataset
                path = src["path"]
                if not Path(path).is_absolute():
                    path = _root / path
                rows = extract_csv(str(path))
                chunks = []
                for row in rows:
                    disease = row.get("diseases", "Unknown")
                    symptoms = [k for k, v in row.items() if v == '1' and k != "diseases"]
                    if symptoms:
                        chunks.append(f"Disease: {disease}. Symptoms: {', '.join(symptoms)}.")
                
                # Limit to first 200 for demo purposes to avoid hitting quota/time limits
                chunks = chunks[:200] 

            # chunks already prepared for CSV types
            if src["type"] in ["html", "pdf"]:
                 pass # already handled inside if/else if they needed further chunking
            print(f"  {src_id}: {len(chunks)} chunks", flush=True)

            for batch_start in range(0, len(chunks), BATCH_SIZE):
                batch = chunks[batch_start : batch_start + BATCH_SIZE]
                batch_ids = range(batch_start, batch_start + len(batch))

                try:
                    result = client.models.embed_content(
                        model="gemini-embedding-001",
                        contents=batch,
                    )
                except ClientError as e:
                    if e.code == 429:
                        print(f"  Rate limit (429) on {src_id}. Wait ~1 min and re-run.", flush=True)
                    raise

                collection.add(
                    documents=batch,
                    embeddings=[e.values for e in result.embeddings],
                    metadatas=[{"source": src_id}] * len(batch),
                    ids=[f"{src_id}_{i}" for i in batch_ids],
                )

                if batch_start + BATCH_SIZE < len(chunks):
                    time.sleep(DELAY_BETWEEN_BATCHES)

            print(f"  {src_id}: done.", flush=True)
        except Exception as e:
            print(f"  ERROR on {src_id}: {e}", flush=True)
            raise

if __name__ == "__main__":
    build_db()
