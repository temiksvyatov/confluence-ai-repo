from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import tempfile
import os
from .rag_chain import create_chain
from .confluence_indexer import reindex_confluence
from .session_manager import add_documents_to_session, get_session_history
from .config import settings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredImageLoader
import asyncio
import logging
import httpx

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"]
)
_reindex_lock = asyncio.Lock()

class ReindexRequest(BaseModel):
    password: str = "admin123"

@app.post("/reindex")
async def trigger_reindex(req: ReindexRequest):
    if req.password != "admin123":
        return {"error": "Неверный пароль"}, 403
    if _reindex_lock.locked():
        return {"error": "Переиндексация уже запущена"}, 409
    asyncio.create_task(_safe_reindex())
    return {"status": "reindex started"}

async def _safe_reindex():
    async with _reindex_lock:
        await reindex_confluence()

@app.post("/chat")
async def chat(message: str = Form(...), session_id: str = Form(...), files: list[UploadFile] = File(default=[])):
    temp_docs = []
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        try:
            if file.filename.lower().endswith(".pdf"):
                loader = PyPDFLoader(tmp_path)
            elif file.filename.lower().endswith((".docx", ".doc")):
                loader = Docx2txtLoader(tmp_path)
            elif file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
                loader = UnstructuredImageLoader(tmp_path)
            else:
                continue
            docs = loader.load()
            temp_docs.extend(docs)
        finally:
            os.unlink(tmp_path)
    if temp_docs:
        add_documents_to_session(session_id, temp_docs)

    async def generate():
        chain = create_chain(session_id)
        full_response = ""
        async for chunk in chain.astream(message):
            full_response += chunk
            yield f"data: {chunk}\n\n"

        get_session_history(session_id).save_context(
            {"input": message},
            {"output": full_response}
        )

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/health")
async def health():
    return {"status": "ok"}
