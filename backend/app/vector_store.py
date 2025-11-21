from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.vectorstores import InMemoryVectorStore
from .config import settings
import os
import gc

embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
CHROMA_PATH = "/app/data/chroma"

_vectorstore = None

def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        if settings.vector_db_mode == "memory":
            _vectorstore = InMemoryVectorStore(embedding=embeddings)
        else:
            os.makedirs(CHROMA_PATH, exist_ok=True)
            _vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    return _vectorstore

def clear_persistent_store():
    global _vectorstore
    if settings.vector_db_mode == "persistent" and _vectorstore is not None:
        try:
            client = _vectorstore._client
            for coll in client.list_collections():
                client.delete_collection(coll.name)
        except:
            pass
        _vectorstore = None
        gc.collect()
        for item in os.listdir(CHROMA_PATH):
            path = os.path.join(CHROMA_PATH, item)
            try:
                if os.path.isfile(path):
                    os.unlink(path)
                else:
                    import shutil
                    shutil.rmtree(path, ignore_errors=True)
            except:
                pass
