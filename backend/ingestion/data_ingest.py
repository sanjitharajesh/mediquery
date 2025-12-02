# backend/ingestion/data_ingest.py
import json
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # CHANGED
from langchain_community.vectorstores import Chroma
from backend.config import DATA_DIR, CHROMA_DIR, CHUNKS_PATH, EMBEDDING_MODEL

def load_fda_pdfs():
    pdf_paths = sorted(Path(DATA_DIR).glob("*.pdf"))
    docs = []
    for pdf in pdf_paths:
        loader = PyPDFLoader(str(pdf))
        docs.extend(loader.load())
    return docs

def build_chunks(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    return splitter.split_documents(docs)

def save_chunks_jsonl(chunks):
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            record = {
                "id": i,
                "text": chunk.page_content,
                "metadata": chunk.metadata,
            }
            f.write(json.dumps(record) + "\n")
    print(f"Saved {len(chunks)} chunks to {CHUNKS_PATH}")

def build_chroma_store(chunks):
    # UPDATED: Force CPU usage
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}  # ADDED
    )
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    print(f"Chroma DB persisted at {CHROMA_DIR}")
    return vs

def run_ingestion():
    print(f"Loading PDFs from {DATA_DIR} ...")
    docs = load_fda_pdfs()
    print(f"Loaded {len(docs)} document pages")
    
    print("Chunking PDFs...")
    chunks = build_chunks(docs)
    print(f"Created {len(chunks)} chunks")
    
    print("Saving chunks to JSONL (for BM25 etc.)...")
    save_chunks_jsonl(chunks)
    
    print("Building Chroma vector store...")
    build_chroma_store(chunks)
    
    print("Ingestion complete.")

if __name__ == "__main__":
    run_ingestion()