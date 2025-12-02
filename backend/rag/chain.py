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
    Ultra-minimal context retrieval with aggressive cleaning.
    """
    docs: List[Document] = hybrid_retrieve(
        question,
        k_chroma=1,
        k_bm25=1,
        k_final=1,
    )
    
    if not docs:
        return "No relevant information found."
    
    # Take first doc
    doc = docs[0]
    
    # Clean and truncate aggressively
    content = _clean_text(doc.page_content)
    content = content[:300]  # VERY SHORT for now
    
    src = doc.metadata.get("source", "unknown")
    page = doc.metadata.get("page", "?")
    
    context = f"[{src}, p.{page}]\n{content}"
    
    # Final length check
    if len(context) > 500:
        context = context[:500]
    
    return context

class SimpleRAGChain:
    def invoke(self, question: str, verbose: bool = False) -> str:
        # Get minimal, clean context
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
        
        # Generate answer
        answer = generate_answer(prompt, verbose=verbose)
        
        # Add disclaimer
        if answer and "Error:" not in answer:
            answer += "\n\nDisclaimer: Not medical advice. Consult a healthcare professional."
        
        return answer

def get_rag_chain() -> SimpleRAGChain:
    return SimpleRAGChain()