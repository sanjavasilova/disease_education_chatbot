import sys
import os
from dotenv import load_dotenv
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from sentence_transformers import SentenceTransformer
from vectorstore.chroma_client_open_router import get_chroma_client

load_dotenv()

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

chroma = get_chroma_client()
collection = chroma.get_collection(name="medical_docs")

def retrieve(query, k=10):
    emb = embed_model.encode(query, normalize_embeddings=True).tolist()

    results = collection.query(
        query_embeddings=[emb],
        n_results=k
    )

    sources = results.get("metadatas", [[]])[0]
    source_ids = [s.get("source", "unknown") for s in sources]
    print(f"Retrieved from: {source_ids}")

    return results["documents"][0]


def retrieve_with_metadata(query, k=10):
    emb = embed_model.encode(query, normalize_embeddings=True).tolist()

    results = collection.query(
        query_embeddings=[emb],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    documents = results["documents"][0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    source_ids = [m.get("source", "unknown") for m in metadatas]

    return documents, source_ids, distances