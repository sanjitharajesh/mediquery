# Mediquery — LLM-Powered Drug Query System

AI-powered FDA drug information assistant that answers medication 
questions using only official FDA drug labels, eliminating hallucination 
by grounding every response in evidence-based sources.

## Features
- Responses sourced exclusively from official FDA drug labels
- Hybrid BM25 + Pinecone retrieval with round-robin merge
- Optional user context (age, conditions, medications) for tailored answers
- Full source attribution on every response

## Tech Stack

**Backend:** FastAPI, LangChain, Groq (llama-3.3-70b-versatile), 
HuggingFace Embeddings (all-MiniLM-L6-v2), Pinecone (384-dim cosine index)

**Frontend:** HTML, CSS, JavaScript

**Observability:** Langfuse (per-query latency, token usage, context tracing), 
RAGAS (faithfulness, answer relevancy, context precision), 
rolling faithfulness drift detection

**Deployment:** Render

## Evaluation
RAGAS scores posted back to Langfuse 
per trace for per-query quality tracking.

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env  
uvicorn main:app --reload
```
