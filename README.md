# MediQuery — RAG-Powered FDA Drug Query System

AI-powered drug information assistant that answers medication questions 
using only official FDA drug labels — eliminating hallucination by 
constraining every response to evidence-based, authoritative sources.

## The Problem
FDA drug labels are the legal ground truth for drug safety information 
but are effectively unqueryable. General-purpose LLMs answer drug 
questions using training weights that go stale, mix up drugs, and 
hallucinate dosages. In a medical context, that's a liability.

## The Architecture
MediQuery grounds every response in retrieved FDA label context via a 
hybrid retrieval pipeline, with a hard fallback gate that refuses to 
answer rather than guess.

## Tech Stack
**Backend:** FastAPI, LangChain, Groq (Llama-3.3-70B), HuggingFace 
Embeddings (all-MiniLM-L6-v2), Pinecone (384-dim cosine index)  
**Retrieval:** Hybrid BM25 + Pinecone with Reciprocal Rank Fusion (k=60)  
**Frontend:** HTML, CSS, JavaScript  
**Observability:** Langfuse (per-query latency, token cost, retrieval metadata)  
**Evaluation:** Custom keyword-coverage + RAGAS framework across 15 queries 
in 7 clinical categories: 80% accuracy, 100% on factual retrieval queries

## Key Design Decisions
- **Hybrid retrieval** — BM25 catches exact medical terminology; dense 
  embeddings handle semantic variants. RRF merges both without score normalization.
- **Failing-closed fallback** — queries targeting out-of-index drugs or 
  below-threshold retrievals bypass the LLM entirely
- **Prompt-level constraint** — generation is explicitly instructed to use 
  only retrieved context, never training weights
- **Query expansion** — patient language is mapped to FDA label terminology 
  before retrieval

## Evaluation
80% accuracy across 15 structured queries spanning side effects, 
contraindications, dosage, drug interactions, pregnancy safety, and 
clinical monitoring. Failures are constrained to cross-chunk reasoning 
queries, a known RAG architecture limitation documented as future work.

## Roadmap
- Layout-aware PDF ingestion (Marker/Unstructured) for multi-column FDA tables
- Cross-encoder re-ranking layer between retrieval and generation
- Full FDA label index (currently 17 drugs, scales to thousands)
- Streaming responses to eliminate perceived latency

## Setup
```bash
pip install -r backend/requirements.txt
cp .env.example .env
PYTHONPATH=backend uvicorn backend.api:app --reload --port 8000
```
