# backend/rag/chain.py
import re
from typing import List
from langchain_core.documents import Document
from retrievers.pinecone_store import get_pinecone_retriever
from rag.prompts import RAG_PROMPT
from llm import generate_answer


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

def _expand_query(question: str) -> str:
    """Add synonyms and related terms to improve retrieval with enhanced medical terminology"""
    question_lower = question.lower()
    
    # Drug name expansions with comprehensive medical terms
    drug_map = {
        "adderall": " amphetamine dextroamphetamine adhd stimulant adverse reactions cardiovascular psychiatric appetite insomnia tachycardia hypertension",
        "ritalin": " methylphenidate adhd stimulant adverse reactions cardiovascular psychiatric appetite",
        "accutane": " isotretinoin acne contraindications warnings teratogenic pregnancy birth-defects",
        "lipitor": " atorvastatin statin cholesterol ldl hdl hyperlipidemia dyslipidemia cardiovascular coronary heart-disease myocardial-infarction stroke atherosclerosis",
        "prozac": " fluoxetine ssri depression dosage milligrams mg daily administration",
        "metformin": " glucophage diabetes type-2-diabetes lactic-acidosis warnings renal kidney impairment contraindications",
        "lisinopril": " ace-inhibitor angiotensin-converting-enzyme antihypertensive blood-pressure hypertension contraindications angioedema pregnancy fetal-harm hypersensitivity",
        "tretinoin": " retin-a retinoid acne skin photosensitivity pregnancy teratogenic topical",
        "ibuprofen": " nsaid nonsteroidal anti-inflammatory pain analgesic bleeding anticoagulant warfarin aspirin interactions",
    }
    
    for drug, expansion in drug_map.items():
        if drug in question_lower:
            question += expansion
            break
    
    # Category expansions with enhanced medical terminology
    if "side effect" in question_lower or "adverse" in question_lower:
        question += " adverse-reactions side-effects adverse-events toxicity safety-profile common serious"
    elif "contraindication" in question_lower or "should not" in question_lower or "avoid" in question_lower or "not be used" in question_lower:
        question += " contraindications warnings precautions boxed-warning contraindicated do-not-use avoid-use"
    elif "interaction" in question_lower:
        question += " drug-interactions concomitant-use drug-combinations pharmacokinetic pharmacodynamic"
    elif "dosage" in question_lower or "dose" in question_lower:
        question += " dosage-administration recommended-dose milligrams mg daily administration dosing-schedule"
    elif "pregnancy" in question_lower or "pregnant" in question_lower:
        question += " pregnancy contraindications teratogenic fetal-harm lactation nursing-mothers use-in-pregnancy pregnancy-category"
    elif "warning" in question_lower:
        question += " warnings-and-precautions boxed-warning contraindications serious-adverse-events"
    elif "use" in question_lower and "for" in question_lower or "used for" in question_lower:
        question += " indications-and-usage therapeutic-indication clinical-indications approved-uses treatment"
    
    return question

def _retriever_fn(question: str) -> str:
    """
    Enhanced retrieval using Pinecone vector store.
    """
    # Expand query for better matching
    expanded_question = _expand_query(question)
    
    # Get Pinecone retriever
    retriever = get_pinecone_retriever()
    
    # Retrieve documents
    docs: List[Document] = retriever.get_relevant_documents(expanded_question)
    
    if not docs:
        return "No relevant information found."
    
    # Use top 5 documents
    context_parts = []
    for i, doc in enumerate(docs[:5], 1):
        content = _clean_text(doc.page_content)
        content = content[:1000]
        
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        
        context_parts.append(f"[Source {i}: {src}, p.{page}]\n{content}")
    
    context = "\n\n".join(context_parts)
    
    # Allow more total context for comprehensive coverage
    if len(context) > 4000:
        context = context[:4000]
    
    return context

class SimpleRAGChain:
    def invoke(self, question: str, verbose: bool = False) -> str:
        # Get context with query expansion
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
        
        return answer

def get_rag_chain() -> SimpleRAGChain:
    return SimpleRAGChain()