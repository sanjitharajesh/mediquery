# backend/config.py
from pathlib import Path

# Project root (mediquery/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Where the raw PDFs live
DATA_DIR = BASE_DIR / "data" / "fda_pdfs"

# Where Chroma will persist its vector store
CHROMA_DIR = BASE_DIR / "chroma_db"

# Where we'll store preprocessed text chunks for BM25, etc.
CHUNKS_PATH = BASE_DIR / "data" / "chunks.jsonl"

# Embedding model for Chroma
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# (You can later swap to Instructor if you want)
