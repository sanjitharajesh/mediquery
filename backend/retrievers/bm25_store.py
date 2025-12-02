# backend/retrievers/bm25_store.py

import json
from pathlib import Path
from typing import List, Dict

from rank_bm25 import BM25Okapi

from backend.config import CHUNKS_PATH

_BM25 = None
_CHUNKS: List[Dict] = []


def _load_chunks():
    global _CHUNKS
    if _CHUNKS:
        return _CHUNKS

    if not Path(CHUNKS_PATH).exists():
        raise FileNotFoundError(f"Chunks file not found at {CHUNKS_PATH}. "
                                f"Run ingestion first (ingest_pdfs.py).")

    chunks = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            chunks.append(record)
    _CHUNKS = chunks
    return _CHUNKS


def _build_bm25():
    global _BM25
    if _BM25 is not None:
        return _BM25

    chunks = _load_chunks()
    corpus_tokens = [chunk["text"].lower().split() for chunk in chunks]
    _BM25 = BM25Okapi(corpus_tokens)
    return _BM25


def retrieve_bm25(query: str, k: int = 5) -> List[Dict]:
    """
    Returns a list of dicts:
      { "text": ..., "score": ..., "metadata": {...} }
    """
    bm25 = _build_bm25()
    chunks = _load_chunks()

    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)

    # top-k indices by score
    scored_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:k]

    results = []
    for idx in scored_indices:
        chunk = chunks[idx]
        results.append(
            {
                "text": chunk["text"],
                "score": float(scores[idx]),
                "metadata": chunk.get("metadata", {}),
            }
        )
    return results
