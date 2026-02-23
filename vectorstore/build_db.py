import sys
import os
import time
import random
from collections import defaultdict
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

BATCH_SIZE = 100
DELAY_BETWEEN_BATCHES = 7.0 

SAMPLE_SIZE = None 
SHUFFLE_BEFORE_SAMPLE = True

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
                if src_id == "mimic_diagnoses":
                    print(f"  Enriching {src_id} with observations from other MIMIC files...", flush=True)
                    from collections import Counter
                    import csv
                    
                    base_dir = Path(path).parent
                    
                    hadm_to_icd = defaultdict(list)
                    diag_path = base_dir / "DIAGNOSES_ICD.csv"
                    if diag_path.exists():
                        with open(diag_path, 'r', encoding='utf-8') as f:
                            for r in csv.DictReader(f):
                                hadm_to_icd[r['hadm_id']].append(r['icd9_code'])
                    
                    item_map = {}
                    relevant_item_ids = set()
                    relevant_categories = {'Abdominal', 'Neurological', 'Respiratory', 'Skin - Impairment', 'Pain/Sedation'}
                    symptom_keywords = {'pain', 'tremor', 'nausea', 'vomiting', 'cough', 'shortness', 'dizziness', 'fever', 'chills', 'weakness', 'swelling', 'edema'}
                    
                    items_path = base_dir / "D_ITEMS.csv"
                    if items_path.exists():
                        with open(items_path, 'r', encoding='utf-8') as f:
                            for r in csv.DictReader(f):
                                item_id, label = r['itemid'], r['label']
                                item_map[item_id] = label
                                if (r['category'] in relevant_categories) or \
                                   any(kw in label.lower() for kw in symptom_keywords):
                                    relevant_item_ids.add(item_id)
                    
                    icd_to_items = defaultdict(Counter)
                    chartevents_path = base_dir / "CHARTEVENTS.csv"
                    if chartevents_path.exists():
                        with open(chartevents_path, 'r', encoding='utf-8') as f:
                            reader = csv.DictReader(f)
                            for i, r in enumerate(reader):
                                item_id, hadm_id = r['itemid'], r['hadm_id']
                                if item_id in relevant_item_ids and hadm_id in hadm_to_icd:
                                    for icd in hadm_to_icd[hadm_id]:
                                        icd_to_items[icd][item_id] += 1
                                if i > 1000000: 
                                    break
                    
                    chunks = []
                    for row in rows:
                        code = row.get("icd9_code", "unknown")
                        title = row.get("long_title", "Unknown Diagnosis")
                        
                        top_items = [item_map[i] for i, _ in icd_to_items[code].most_common(8)]
                        if top_items:
                            chunks.append(f"ICD-9 Code {code}: {title}. Commonly associated clinical observations: {', '.join(top_items)}.")
                        else:
                            chunks.append(f"ICD-9 Code {code}: {title} ({row.get('short_title', '')})")
                else:
                    chunks = [", ".join([f"{k}: {v}" for k, v in row.items()]) for row in rows]
            elif src["type"] == "csv_special":
                path = src["path"]
                if not Path(path).is_absolute():
                    path = _root / path
                rows = extract_csv(str(path))
                disease_symptoms = defaultdict(set)
                for row in rows:
                    disease = row.get("diseases", "Unknown")
                    symptoms = [k for k, v in row.items() if v == '1' and k != "diseases"]
                    for sym in symptoms:
                        disease_symptoms[disease].add(sym)
                
                chunks = []
                for disease, symptoms in disease_symptoms.items():
                    if symptoms:
                        chunks.append(f"Disease: {disease}. Symptoms: {', '.join(sorted(list(symptoms)))}.")
                
            if src["type"] in ["html", "pdf"]:
                 pass 

            existing_ids = set(collection.get(include=[])["ids"])

            print(f"  {src_id}: {len(chunks)} chunks", flush=True)

            for batch_start in range(0, len(chunks), BATCH_SIZE):
                batch = chunks[batch_start : batch_start + BATCH_SIZE]
                batch_ids = [f"{src_id}_{i}" for i in range(batch_start, batch_start + len(batch))]

                new_batch_indices = [i for i, bid in enumerate(batch_ids) if bid not in existing_ids]
                
                if not new_batch_indices:
                    continue  
                
                batch_to_process = [batch[i] for i in new_batch_indices]
                ids_to_process = [batch_ids[i] for i in new_batch_indices]

                if SAMPLE_SIZE and len(batch_to_process) > SAMPLE_SIZE:
                    print(f"  Sampling {SAMPLE_SIZE} new chunks from {len(batch_to_process)} available...", flush=True)
                    combined = list(zip(batch_to_process, ids_to_process))
                    if SHUFFLE_BEFORE_SAMPLE:
                        random.shuffle(combined)
                    sampled_combined = combined[:SAMPLE_SIZE]
                    batch_to_process, ids_to_process = zip(*sampled_combined)

                retries = 0
                max_retries = 5
                while retries < max_retries:
                    try:
                        result = client.models.embed_content(
                            model="gemini-embedding-001",
                            contents=batch_to_process,
                        )
                        break 
                    except ClientError as e:
                        if e.code == 429:
                            wait_time = 2 ** retries + 1 
                            print(f"  Rate limit (429) on {src_id}. Retrying in {wait_time}s...", flush=True)
                            time.sleep(wait_time)
                            retries += 1
                        else:
                            raise e
                else:
                    raise Exception(f"Failed to embed batch for {src_id} after {max_retries} retries due to rate limiting.")

                collection.add(
                    documents=batch_to_process,
                    embeddings=[e.values for e in result.embeddings],
                    metadatas=[{"source": src_id}] * len(batch_to_process),
                    ids=ids_to_process,
                )

                if batch_start + BATCH_SIZE < len(chunks):
                    time.sleep(DELAY_BETWEEN_BATCHES)

            print(f"  {src_id}: done.", flush=True)
        except Exception as e:
            print(f"  ERROR on {src_id}: {e}", flush=True)
            pass

if __name__ == "__main__":
    build_db()
