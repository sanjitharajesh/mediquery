A local, privacy-preserving medical assistant that answers drug-related questions using RAG with Mistral-7B, Instructor embeddings, and ChromaDB.
Built with FastAPI, SQLite, and Docker, MediQuery retrieves FDA medication guides, processes them locally, and returns accurate, explainable responses with no external API calls.
Includes full logging (queries, processing time, source count) and a clean pipeline for ingestion, retrieval, and generation.
Run locally using uvicorn backend.main:app after installing dependencies and downloading the GGUF model.
Lightweight, fast, and fully offline.
