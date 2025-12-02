# backend/retrievers/chroma_store.py
from langchain_huggingface import HuggingFaceEmbeddings  # CHANGED
from langchain_community.vectorstores import Chroma
from backend.config import CHROMA_DIR, EMBEDDING_MODEL

def get_chroma():
    # UPDATED: Force CPU usage
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}  # ADDED
    )
    vs = Chroma(
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    return vs

def retrieve_chroma(query: str, k: int = 5):
    vs = get_chroma()
    docs = vs.similarity_search(query, k=k)
    return docs