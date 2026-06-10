# MediChat: Smart Disease Education Chatbot

```mermaid
graph TD
    User([User]) <--> Frontend[React Frontend]
    Frontend <--> Backend[FastAPI Server]
    Backend -- Search Query --> VectorDB[(ChromaDB)]
    VectorDB -- Relevant Context --> Backend
    Backend -- Augment Prompt --> LLM["Gemini 2.5 Flash / Groq (Llama 3.3 70B) / OpenRouter (Gemma 4 31B)"]
    LLM -- Generated Response --> Backend
    
    Data[WHO, PDF, CSV] --> Ingest[build_db.py]
    Ingest -- Embeddings --> VectorDB
```

MediChat is a Retrieval-Augmented Generation (RAG) chatbot designed to provide warm, calm, and accurate educational information about diseases. It combines a modern React frontend with a powerful FastAPI backend, utilizing ChromaDB for vector storage and a choice of AI providers for generation.

## 🚀 Key Features

-   **RAG Architecture**: Uses real-time retrieval from a curated medical knowledge base to ground AI responses.
-   **Multi-Source Ingestion**: Automatically processes WHO fact sheets, clinical PDFs, and large-scale clinical datasets (MIMIC-III).
-   **Multi-Provider Support**: Supports Google Gemini, Groq, and OpenRouter as interchangeable LLM backends.
-   **Smart Formatting**: Frontend supports full Markdown rendering for clear medical lists and bold highlights.
-   **API Quota Management**: Built-in optional chunk sampling to handle free-tier API limits during large data ingestion.
-   **Empathetic UI**: Designed with a "glassmorphism" aesthetic, micro-animations, and a supportive tone.

## 🛠️ Tech Stack

-   **Frontend**: React (Vite), Tailwind CSS, Framer Motion, Lucide Icons.
-   **Backend**: FastAPI, Uvicorn, Python.
-   **AI/ML**: Google Gemini API (`gemini-2.5-flash`); Groq API (`llama-3.3-70b-versatile`); OpenRouter API (`google/gemma-4-31b-it:free` or any free model); all use `sentence_transformers` for local embeddings.
-   **Vector Store**: ChromaDB.
-   **Data Processing**: BeautifulSoup4, PyPDF, Pandas.

## 📋 Prerequisites

-   Python 3.10+
-   Node.js & NPM
-   A Google Gemini API Key, Groq API Key, or OpenRouter API Key (depending on which backend you use)

## ⚙️ Setup & Installation

### 1. Environment Setup
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
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
# Gemini
python3 vectorstore/build_db.py

# Groq
python3 vectorstore/build_db_groq.py

# OpenRouter (uses the same local sentence-transformers embeddings as Groq)
python3 vectorstore/build_db_open_router.py
```
*Note: You can adjust `SAMPLE_SIZE` in `build_db.py` to limit the number of new chunks ingested per run.*

### Phase 2: Start the Backend
```bash
# Gemini
python3 -m uvicorn main:app --reload

# Groq
python3 -m uvicorn main_groq:app --reload

# OpenRouter
python3 -m uvicorn main_open_router:app --reload
```

### Phase 3: Start the Frontend
```bash
cd frontend
npm run dev
```

## 🧠 How it Works (RAG Flow)

1.  **Ingestion**: `build_db.py` extracts text from HTML, PDF, and CSV files, chunks it, and generates embeddings using `all-MiniLM-L6-v2` (local, no API cost).
2.  **Storage**: Embeddings and metadata are stored in a local ChromaDB instance (`chroma_db/`, `chroma_db_groq/`, `chroma_db_open_router/`).
3.  **Retrieval**: When a user asks a question, the server embeds the query and searches ChromaDB for the most relevant context chunks.
4.  **Augmentation**: The context is injected into a specialized `SYSTEM_PROMPT` that enforces medical safety and calm reassurance.
5.  **Generation**: The configured LLM generates a response grounded strictly in the provided context.
6.  **Formatting**: The frontend renders the response as Markdown for a clean, professional look.

## 📊 Evaluation

The RAG pipeline is evaluated using retrieval quality metrics over a test dataset of 50 medical questions covering diseases, mental health, injuries, and more.

### Running the Evaluation
```bash
# Gemini — retrieval only
python -m evaluation.evaluate --retrieval-only
# Gemini — full LLM-as-judge
python -m evaluation.evaluate --limit 5

# Groq — retrieval only
python -m evaluation.evaluate_groq --retrieval-only
# Groq — full LLM-as-judge
python -m evaluation.evaluate_groq --limit 5

# OpenRouter — retrieval only
python -m evaluation.evaluate_open_router --retrieval-only
# OpenRouter — full LLM-as-judge
python -m evaluation.evaluate_open_router --limit 5
```

### Retrieval Metrics (Gemini)

| Metric | Score | Description |
|---|---|---|
| **Hit Rate** | 1.00 | Retrieval always returns at least one chunk from the correct source |
| **MRR** | 0.94 | Correct source is ranked 1st almost every time |
| **Source Precision** | 0.68 | 68% of retrieved chunks come from the expected source |
| **Overall** | 0.87 | Simple average of retrieval metrics |

### Retrieval Metrics (Groq — Llama 3.3 70B)

| Metric | Score | Description |
|---|---|---|
| **Hit Rate** | 1.00 | Retrieval always returns at least one chunk from the correct source |
| **MRR** | 0.89 | Correct source is ranked 1st almost every time |
| **Source Precision** | 0.58 | 58% of retrieved chunks come from the expected source |
| **Overall** | 0.82 | Simple average of retrieval metrics |

### Retrieval Metrics (OpenRouter — Gemma 4 31B)

| Metric | Score | Description |
|---|---|---|
| **Hit Rate** | 1.00 | Retrieval always returns at least one chunk from the correct source |
| **MRR** | 0.88 | Correct source is ranked 1st almost every time |
| **Source Precision** | 0.59 | 59% of retrieved chunks come from the expected source |
| **Overall** | 0.82 | Simple average of retrieval metrics |

### LLM-as-Judge Metrics (optional, requires API calls)

| Metric | Description |
|---|---|
| **Faithfulness** | Is the answer grounded in the retrieved context (no hallucination)? |
| **Answer Relevance** | Does the answer address the user's question? |
| **Context Relevance** | Are the retrieved chunks relevant to the question? |
| **Correctness** | Is the answer factually consistent with the ground truth? |

## 🛡️ Medical Disclaimer
This application is for educational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment.

## 📸 Screenshots

![MediChat UI](docs/example_question_1.png)

![MediChat UI](docs/example_question_2.png)

![MediChat UI](docs/example_question_3.png)
