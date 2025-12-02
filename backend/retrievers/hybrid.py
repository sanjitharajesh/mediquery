# backend/retrievers/hybrid.py

from typing import List, Set, Tuple

from langchain_core.documents import Document

from backend.retrievers.chroma_store import retrieve_chroma
from backend.retrievers.bm25_store import retrieve_bm25


def hybrid_retrieve(
    query: str,
    k_chroma: int = 2,
    k_bm25: int = 2,
    k_final: int = 4,
) -> List[Document]:
    """
    Simple hybrid retrieval:
    - Get top-k from Chroma (semantic)
    - Get top-k from BM25 (lexical)
    - Merge & dedupe by (source, page) if available
    - Return up to k_final Documents
    """

    # 1) Get Chroma Documents (already LangChain Document objects)
    chroma_docs: List[Document] = retrieve_chroma(query, k=k_chroma)

    # 2) Get BM25 results (plain dicts) and wrap as Documents
    bm25_results = retrieve_bm25(query, k=k_bm25)
    bm25_docs: List[Document] = []
    for r in bm25_results:
        metadata = r.get("metadata", {}) or {}
        bm25_docs.append(
            Document(
                page_content=r["text"],
                metadata=metadata,
            )
        )

    # 3) Merge with de-duplication
    seen: Set[Tuple] = set()
    merged: List[Document] = []

    def doc_key(doc: Document) -> Tuple:
        # Try to identify a chunk by (source, page); fall back to content prefix
        src = doc.metadata.get("source", "")
        page = doc.metadata.get("page", "")
        prefix = doc.page_content[:100]
        return (src, page, prefix)

    for doc in chroma_docs + bm25_docs:
        key = doc_key(doc)
        if key in seen:
            continue
        seen.add(key)
        merged.append(doc)

    # 4) Truncate to k_final
    return merged[:k_final]
