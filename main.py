import os
import uuid

from fastapi import FastAPI
from fastapi import UploadFile
from fastapi import File
from fastapi import HTTPException

from openai import OpenAI

from rag import (
    extract_text,
    chunk_text,
    create_embeddings,
    embed_query,
)

from vectorstore import VectorStore
from models import QuestionRequest
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

vector_store = None

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1"
)


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global vector_store

    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files allowed"
        )

    pdf_path = os.path.join(
        UPLOAD_DIR,
        f"{uuid.uuid4()}.pdf"
    )

    with open(pdf_path, "wb") as f:
        f.write(await file.read())

    text = extract_text(pdf_path)

    chunks = chunk_text(text)

    embeddings = create_embeddings(chunks)

    dimension = len(embeddings[0])

    vector_store = VectorStore(dimension)
    vector_store.add(embeddings, chunks)

    return {
        "status": "success",
        "chunks": len(chunks)
    }


@app.post("/ask")
async def ask_question(req: QuestionRequest):
    global vector_store

    if vector_store is None:
        raise HTTPException(
            status_code=400,
            detail="Upload PDF first"
        )

    query_embedding = embed_query(req.question)

    contexts = vector_store.search(
        query_embedding,
        k=5
    )

    context_text = "\n\n".join(contexts)

    prompt = f"""
You are a helpful document assistant.

Use ONLY the provided context.

Context:
{context_text}

Question:
{req.question}

Answer:
"""

    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-8B-Instruct",
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "sources": contexts
    }