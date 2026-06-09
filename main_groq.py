import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from fastapi.middleware.cors import CORSMiddleware

from app.rag import retrieve
from app.prompts import SYSTEM_PROMPT
from groq import Groq
from groq import APIStatusError

load_dotenv()

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

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
            model="llama-3.3-70b-versatile",
            messages=messages,
        )
    except APIStatusError as e:
        if e.status_code == 429:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Your free-tier quota may be used up. Try again in a few minutes or tomorrow (quotas reset daily).",
            )
        print(f"Groq error {e.status_code}: {e.message}")  # add this
        raise HTTPException(status_code=502, detail=str(e))

    return {"answer": response.choices[0].message.content}
