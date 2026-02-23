# MediChat: Smart Disease Education Chatbot

```mermaid
graph TD
    User([User]) <--> Frontend[React Frontend]
    Frontend <--> Backend[FastAPI Server]
    Backend -- Search Query --> VectorDB[(ChromaDB)]
    VectorDB -- Relevant Context --> Backend
    Backend -- Augment Prompt --> LLM[Gemini 2.5 Flash]
    LLM -- Generated Response --> Backend
    
    Data[WHO, PDF, CSV] --> Ingest[build_db.py]
    Ingest -- Embeddings --> VectorDB
```

MediChat is a Retrieval-Augmented Generation (RAG) chatbot designed to provide warm, calm, and accurate educational information about diseases. It combines a modern React frontend with a powerful FastAPI backend, utilizing ChromaDB for vector storage and Google's Gemini AI for embeddings and generation.

## 🚀 Key Features

-   **RAG Architecture**: Uses real-time retrieval from a curated medical knowledge base to ground AI responses.
-   **Multi-Source Ingestion**: Automatically processes WHO fact sheets, clinical PDFs, and large-scale clinical datasets (MIMIC-III).
-   **Smart Formatting**: Frontend supports full Markdown rendering for clear medical lists and bold highlights.
-   **API Quota Management**: Built-in optional chunk sampling to handle free-tier API limits during large data ingestion.
-   **Empathetic UI**: Designed with a "glassmorphism" aesthetic, micro-animations, and a supportive tone.

## 🛠️ Tech Stack

-   **Frontend**: React (Vite), Tailwind CSS, Framer Motion, Lucide Icons.
-   **Backend**: FastAPI, Uvicorn, Python.
-   **AI/ML**: Google Gemini API (`gemini-2.5-flash`, `gemini-embedding-001`).
-   **Vector Store**: ChromaDB.
-   **Data Processing**: BeautifulSoup4, PyPDF, Pandas.

## 📋 Prerequisites

-   Python 3.10+
-   Node.js & NPM
-   A Google Gemini API Key

## ⚙️ Setup & Installation

### 1. Environment Setup
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_api_key_here
```

### 2. Backend Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Frontend Installation
```bash
cd frontend
npm install
cd ..
```

## 🏃 Running the Application

### Phase 1: Build the Knowledge Base
Ingest the medical data into the vector database:
```bash
python3 vectorstore/build_db.py
```
*Note: You can adjust `SAMPLE_SIZE` in `build_db.py` to limit the number of new chunks ingested per run.*

### Phase 2: Start the Backend
```bash
python3 -m uvicorn main:app --reload
```

### Phase 3: Start the Frontend
```bash
cd frontend
npm run dev
```

## 🧠 How it Works (RAG Flow)

1.  **Ingestion**: `build_db.py` extracts text from HTML, PDF, and CSV files, chunks it, and generates embeddings.
2.  **Storage**: Embeddings and metadata are stored in a local ChromaDB instance (`chroma_db/`).
3.  **Retrieval**: When a user asks a question, the server embeds the query and searches ChromaDB for the most relevant context chunks.
4.  **Augmentation**: The context is injected into a specialized `SYSTEM_PROMPT` that enforces medical safety and calm reassurance.
5.  **Generation**: Gemini 2.0 Flash generates a response grounded strictly in the provided context.
6.  **Formatting**: The frontend renders the response as Markdown for a clean, professional look.

## 🛡️ Medical Disclaimer
This application is for educational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment.
