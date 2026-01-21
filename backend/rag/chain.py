# backend/rag/chain.py
import re
from typing import List
from langchain_core.documents import Document
from backend.retrievers.hybrid import hybrid_retrieve
from backend.rag.prompts import RAG_PROMPT
from backend.llm import generate_answer

def _clean_text(text: str) -> str:
    """Remove problematic characters that break LLM processing"""
    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # Remove control characters except newlines
    text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
    # Collapse multiple whitespaces
    text = re.sub(r'\s+', ' ', text)
    # Remove excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def _retriever_fn(question: str) -> str:
    """
    Optimized retrieval for comprehensive answers.
    """
    docs: List[Document] = hybrid_retrieve(
        question,
        k_chroma=3,    # More from vector search
        k_bm25=3,      # More from keyword search
        k_final=4,     # Get top 4 docs
    )
    
    if not docs:
        return "No relevant information found."
    
    # Use top 2 documents with good content
    context_parts = []
    for i, doc in enumerate(docs[:2], 1):
        content = _clean_text(doc.page_content)
        content = content[:900]  # More content per doc
        
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        
        context_parts.append(f"[Source {i}: {src}, p.{page}]\n{content}")
    
    context = "\n\n".join(context_parts)
    
    # Allow more context (up to 1800 chars)
    if len(context) > 1800:
        context = context[:1800]
    
    return context

class SimpleRAGChain:
    def invoke(self, question: str, verbose: bool = False) -> str:
        # Get context
        context = _retriever_fn(question)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Context length: {len(context)} chars")
            print(f"Context preview: {context[:200]}...")
            print(f"{'='*60}\n")
        
        # Build prompt
        prompt = RAG_PROMPT.format(context=context, question=question)
        
        if verbose:
            print(f"Full prompt length: {len(prompt)} chars\n")
        
        # Generate answer (no automatic disclaimer)
        answer = generate_answer(prompt, verbose=verbose)
        
        return answer

def get_rag_chain() -> SimpleRAGChain:
    return SimpleRAGChain()