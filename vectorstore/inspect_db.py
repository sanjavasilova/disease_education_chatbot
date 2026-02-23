import sys
import pandas as pd
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from vectorstore.chroma_client import get_chroma_client

def inspect_db():
    client = get_chroma_client()
    try:
        collection = client.get_collection(name="medical_docs")
    except Exception as e:
        print(f"Error accessing collection: {e}")
        return

    results = collection.get()
    
    if not results['ids']:
        print("The database is currently empty.")
        return

    data = {
        'ID': results['ids'],
        'Source': [m.get('source', 'Unknown') for m in results['metadatas']],
        'Content': [d[:100] + "..." if len(d) > 100 else d for d in results['documents']]
    }
    
    df = pd.DataFrame(data)
    
    print("\n=== Database Overview ===")
    print(f"Total entries: {len(df)}")
    print("\n--- Summary by Source ---")
    print(df['Source'].value_counts().to_string())
    
    print("\n--- Detailed View (Top 50 entries) ---")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 50)
    
    print(df.head(50).to_string(index=False))
    # df.to_csv(r'/home/s/Desktop/disease_chatbot/database.csv', index=False)
    print("\nTip: Use `df.to_csv()` or similar if you want to export the whole database.")

if __name__ == "__main__":
    inspect_db()