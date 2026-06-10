import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from fastapi.middleware.cors import CORSMiddleware

from app.rag_open_router import retrieve
from app.prompts import SYSTEM_PROMPT
from openai import OpenAI

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

MODEL = "google/gemma-4-31b-it:free"

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    question: str
    history: Optional[List[ChatMessage]] = []

@app.post("/ask")
def ask(request: ChatRequest):
    question = request.question
    history = request.history
    context_chunks = retrieve(question)
    context = "\n".join(context_chunks)
    user_content = f"Context:\n{context}\n\nQuestion:\n{question}"

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for msg in history:
        role = "assistant" if msg.role == "model" else msg.role
        messages.append({"role": role, "content": msg.content})

    messages.append({"role": "user", "content": user_content})

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
        )
    except Exception as e:
        status = getattr(e, "status_code", None)
        if status == 429:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. The free-tier quota may be used up. Try again in a few minutes.",
            )
        print(f"error {status} {str(e)}")
        raise HTTPException(status_code=502, detail=str(e))
    print(f"{response.choices[0]}")
    return {"answer": response.choices[0].message.content}