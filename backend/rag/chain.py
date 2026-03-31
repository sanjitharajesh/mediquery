# backend/rag/chain.py
import re
import time
from typing import List, Tuple
from langchain_core.documents import Document
from langfuse import observe
from observability.langfuse_tracing import langfuse_client  # init registers global OTEL tracer
from retrievers.pinecone_store import get_pinecone_retriever
from rag.prompts import RAG_PROMPT
from llm import generate_answer, _last_token_usage


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

@observe(name="retrieval")
def _retriever_fn(question: str) -> Tuple[str, List[str]]:
    """
    Enhanced retrieval using Pinecone vector store.
    Returns (formatted_context_str, raw_context_texts) for tracing and RAGAS.
    """
    # Expand query for better matching
    expanded_question = _expand_query(question)

    # Get Pinecone retriever
    retriever = get_pinecone_retriever()

    # Retrieve documents
    docs: List[Document] = retriever.invoke(expanded_question)

    if not docs:
        langfuse_client.update_current_span(output="No relevant information found.")
        return "No relevant information found.", []

    # Use top 5 documents
    context_parts = []
    raw_contexts: List[str] = []
    for i, doc in enumerate(docs[:5], 1):
        content = _clean_text(doc.page_content)
        content = content[:1000]
        raw_contexts.append(content)

        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")

        context_parts.append(f"[Source {i}: {src}, p.{page}]\n{content}")

    context = "\n\n".join(context_parts)

    # Allow more total context for comprehensive coverage
    if len(context) > 4000:
        context = context[:4000]

    langfuse_client.update_current_span(
        input=question,
        output=raw_contexts,
        metadata={"num_docs": len(raw_contexts), "context_length": len(context)},
    )

    return context, raw_contexts

class SimpleRAGChain:
    def __init__(self):
        self._last_trace_id: str | None = None
        self._last_contexts: List[str] = []

    @observe(name="rag_query")
    def invoke(self, question: str, verbose: bool = False) -> str:
        start = time.time()

        # Log the incoming question on the trace
        langfuse_client.update_current_span(input=question)

        # Get context with query expansion (sub-span logged inside _retriever_fn)
        context, raw_contexts = _retriever_fn(question)

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

        latency_ms = round((time.time() - start) * 1000, 2)

        # Log output, latency, token counts, and retrieved contexts to Langfuse
        langfuse_client.update_current_span(
            output=answer,
            metadata={
                "latency_ms": latency_ms,
                "context_length": len(context),
                "retrieved_contexts": raw_contexts,
                **_last_token_usage,
            },
        )

        self._last_trace_id = langfuse_client.get_current_trace_id()
        self._last_contexts = raw_contexts

        return answer

def get_rag_chain() -> SimpleRAGChain:
    return SimpleRAGChain()