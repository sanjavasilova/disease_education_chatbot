import sys
import os
from dotenv import load_dotenv
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from google import genai
from vectorstore.chroma_client import get_chroma_client

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

chroma = get_chroma_client()
collection = chroma.get_collection(name="medical_docs")

def retrieve(query, k=10):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=query,
    )
    emb = result.embeddings[0].values

    results = collection.query(
        query_embeddings=[emb],
        n_results=k
    )
    
    sources = results.get("metadatas", [[]])[0]
    source_ids = [s.get("source", "unknown") for s in sources]
    print(f"Retrieved from: {source_ids}")

    return results["documents"][0]


def retrieve_with_metadata(query, k=10):
    """Same as retrieve but also returns source IDs and distances."""
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=query,
    )
    emb = result.embeddings[0].values

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
