import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from fastapi.middleware.cors import CORSMiddleware

from app.rag import retrieve
from app.prompts import SYSTEM_PROMPT
from google import genai
from google.genai import types
from google.genai.errors import ClientError

load_dotenv()

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))



@app.post("/ask")
def ask(question: str):
    context_chunks = retrieve(question)
    context = "\n".join(context_chunks)
    user_content = f"Context:\n{context}\n\nQuestion:\n{question}"

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
            ),
        )
    except ClientError as e:
        if e.code == 429:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Your free-tier quota may be used up. Try again in a few minutes or tomorrow (quotas reset daily).",
            )
        raise HTTPException(status_code=502, detail=str(e))

    return {"answer": response.text}
