# backend/retrievers/hybrid.py

import re
from typing import Dict, List

from langchain_core.documents import Document

from retrievers.bm25_store import retrieve_bm25
from retrievers.pinecone_store import get_pinecone_retriever


RRF_K = 60


def hybrid_retrieve(
    query: str,
    k_vector: int = 5,
    k_bm25: int = 10,
    k_final: int = 7,
) -> List[Document]:
    """
    Hybrid retrieval with Reciprocal Rank Fusion (RRF).

    Pinecone vector results and local rank-bm25 results are fused by rank:
        score += 1 / (RRF_K + rank)

    Chunks are deduplicated by normalized text content and the top k_final
    fused documents are returned.
    """
    fused: Dict[str, Document] = {}

    def text_key(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip().lower()

    def add_result(doc: Document, source: str, rank: int, raw_score: float | None = None) -> None:
        key = text_key(doc.page_content)
        if not key:
            return

        rrf_score = 1.0 / (RRF_K + rank)

        if key not in fused:
            metadata = doc.metadata.copy()
            metadata["rrf_score"] = 0.0
            metadata["retrieval_sources"] = []
            fused[key] = Document(page_content=doc.page_content, metadata=metadata)

        existing = fused[key]
        existing.metadata["rrf_score"] += rrf_score
        existing.metadata["retrieval_sources"].append(source)
        existing.metadata[f"{source}_rank"] = rank
        if raw_score is not None:
            existing.metadata[f"{source}_score"] = raw_score

    vector_docs = get_pinecone_retriever().invoke(query)[:k_vector]
    for rank, doc in enumerate(vector_docs, start=1):
        add_result(doc, source="pinecone", rank=rank)

    for rank, result in enumerate(retrieve_bm25(query, k=k_bm25), start=1):
        metadata = result.get("metadata", {}) or {}
        doc = Document(page_content=result["text"], metadata=metadata)
        add_result(doc, source="bm25", rank=rank, raw_score=result.get("score"))

    ranked_docs = sorted(
        fused.values(),
        key=lambda doc: doc.metadata.get("rrf_score", 0.0),
        reverse=True,
    )

    return ranked_docs[:k_final]
