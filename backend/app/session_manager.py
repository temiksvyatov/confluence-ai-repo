from collections import defaultdict
from langchain.schema import Document
from datetime import datetime, timedelta
from langchain.memory import ConversationBufferMemory
from langchain_core.vectorstores import InMemoryVectorStore
from .vector_store import embeddings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

session_vectors = defaultdict(lambda: None)
session_memory = defaultdict(lambda: ConversationBufferMemory(return_messages=True))
session_last_access = {}
SESSION_TTL = timedelta(hours=2)

def cleanup_old_sessions():
    now = datetime.now()
    to_remove = [
        sid for sid, last_time in session_last_access.items()
        if now - last_time > SESSION_TTL
    ]
    for sid in to_remove:
        session_vectors.pop(sid, None)
        session_memory.pop(sid, None)
        session_last_access.pop(sid, None)

def get_session_vectorstore(session_id: str):
    cleanup_old_sessions()
    session_last_access[session_id] = datetime.now()
    if session_id not in session_vectors:
        session_vectors[session_id] = InMemoryVectorStore(embedding=embeddings)
    return session_vectors[session_id]

def add_documents_to_session(session_id: str, docs: list[Document]):
    vs = get_session_vectorstore(session_id)
    vs.add_documents(docs)

def get_session_history(session_id: str):
    return session_memory[session_id]
