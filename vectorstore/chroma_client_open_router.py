import chromadb
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_PATH = str(_PROJECT_ROOT / "chroma_db_open_router")

def get_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_PATH)
