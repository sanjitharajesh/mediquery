# backend/rag/chain.py
import re
import time
from typing import List, Tuple
from langchain_core.documents import Document
from langfuse.decorators import observe
from observability.langfuse_tracing import langfuse_client  # init registers global OTEL tracer
from retrievers.hybrid import hybrid_retrieve
from rag.prompts import RAG_PROMPT
import llm
from llm import generate_answer


FINAL_CONTEXT_K = 5
FALLBACK_MESSAGE = "The requested medication is outside the current FDA drug label index."
MIN_FUSED_RRF_SCORE = 1.0 / (60 + 10)

INDEXED_MEDICATION_TERMS = {
    "accutane",
    "adderall",
    "amphetamine",
    "atorvastatin",
    "dextroamphetamine",
    "fluoxetine",
    "glucophage",
    "ibuprofen",
    "isotretinoin",
    "lipitor",
    "lisinopril",
    "metformin",
    "methylphenidate",
    "prozac",
    "retin-a",
    "ritalin",
    "tretinoin",
}
KNOWN_UNSUPPORTED_MEDICATIONS = {
    "eliquis",
    "humira",
    "jardiance",
    "keytruda",
    "ozempic",
    "warfarin",
}
QUESTION_STOPWORDS = {
    "a",
    "about",
    "and",
    "are",
    "can",
    "does",
    "do",
    "for",
    "has",
    "have",
    "how",
    "i",
    "is",
    "it",
    "me",
    "my",
    "of",
    "should",
    "take",
    "the",
    "there",
    "to",
    "what",
    "when",
    "which",
    "with",
}


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


def _mentions_indexed_medication(question: str) -> bool:
    question_lower = question.lower()
    return any(re.search(rf"\b{re.escape(term)}\b", question_lower) for term in INDEXED_MEDICATION_TERMS)


def _mentions_known_unsupported_medication(question: str) -> bool:
    question_lower = question.lower()
    return any(re.search(rf"\b{re.escape(term)}\b", question_lower) for term in KNOWN_UNSUPPORTED_MEDICATIONS)


def _candidate_medication_terms(question: str) -> List[str]:
    candidates = []
    for token in re.findall(r"\b[A-Za-z][A-Za-z0-9-]*\b", question):
        token_lower = token.lower()
        if token_lower in QUESTION_STOPWORDS:
            continue
        if token_lower in INDEXED_MEDICATION_TERMS:
            continue
        if token_lower in KNOWN_UNSUPPORTED_MEDICATIONS or token[:1].isupper():
            candidates.append(token_lower)
    return candidates


def _is_outside_index_request(question: str) -> bool:
    if _mentions_indexed_medication(question):
        return False
    if _mentions_known_unsupported_medication(question):
        return True
    return bool(_candidate_medication_terms(question))

def _expand_query(question: str) -> str:
    """Add synonyms and related terms to improve retrieval with enhanced medical terminology"""
    question_lower = question.lower()
    
    # Drug name expansions with comprehensive medical terms
    drug_map = {
        "adderall": " amphetamine dextroamphetamine adhd stimulant adverse reactions cardiovascular psychiatric appetite insomnia tachycardia hypertension",
        "ritalin": " methylphenidate adhd stimulant adverse reactions cardiovascular psychiatric appetite",
        "accutane": " isotretinoin acne contraindications warnings teratogenic pregnancy birth-defects iPLEDGE precautions monitoring liver function lipids contraception blood donation",
        "isotretinoin": " accutane acne contraindications warnings teratogenic pregnancy birth-defects iPLEDGE precautions monitoring liver function lipids contraception blood donation",
        "lipitor": " atorvastatin statin cholesterol ldl hdl hyperlipidemia dyslipidemia cardiovascular coronary heart-disease myocardial-infarction stroke atherosclerosis",
        "prozac": " fluoxetine ssri depression dosage milligrams mg daily administration taper discontinuation syndrome gradual dose reduction withdrawal",
        "fluoxetine": " prozac ssri depression dosage milligrams mg daily administration taper discontinuation syndrome gradual dose reduction withdrawal",
        "metformin": " glucophage diabetes type-2-diabetes lactic-acidosis warnings renal kidney impairment contraindications black box warning lactic acidosis renal function monitoring serum creatinine hepatic impairment",
        "lisinopril": " ace-inhibitor angiotensin-converting-enzyme antihypertensive blood-pressure hypertension contraindications angioedema pregnancy fetal-harm hypersensitivity",
        "tretinoin": " retin-a retinoid acne skin photosensitivity pregnancy teratogenic topical avoid waxing abrasives sunscreen photosensitivity medicated cosmetics concomitant",
        "retin-a": " tretinoin retinoid acne skin photosensitivity pregnancy teratogenic topical avoid waxing abrasives sunscreen photosensitivity medicated cosmetics concomitant",
        "ibuprofen": " nsaid nonsteroidal anti-inflammatory pain analgesic bleeding anticoagulant warfarin aspirin interactions",
    }
    
    for drug, expansion in drug_map.items():
        if drug in question_lower:
            question += expansion
            break
    
    # Category expansions with enhanced medical terminology
    if "side effect" in question_lower or "adverse" in question_lower:
        question += " adverse-reactions side-effects adverse-events toxicity safety-profile common serious"
    if "contraindication" in question_lower or "should not" in question_lower or "avoid" in question_lower or "not be used" in question_lower:
        question += " contraindications warnings precautions boxed-warning contraindicated do-not-use avoid-use"
    if "interaction" in question_lower:
        question += " drug-interactions concomitant-use drug-combinations pharmacokinetic pharmacodynamic"
    if "dosage" in question_lower or "dose" in question_lower:
        question += " dosage-administration recommended-dose milligrams mg daily administration dosing-schedule"
    if "pregnancy" in question_lower or "pregnant" in question_lower:
        question += " pregnancy contraindications teratogenic fetal-harm lactation nursing-mothers use-in-pregnancy pregnancy-category"
    if "warning" in question_lower:
        question += " warnings-and-precautions boxed-warning contraindications serious-adverse-events"
    if "black box" in question_lower or "boxed warning" in question_lower:
        question += " boxed warning serious risk FDA black box contraindicated fatal"
    if "monitor" in question_lower or "monitoring" in question_lower:
        question += " monitoring parameters lab tests serum levels renal hepatic function"
    if "taper" in question_lower or "discontinu" in question_lower or "stop" in question_lower:
        question += " tapering discontinuation gradual reduction withdrawal syndrome"
    if ("use" in question_lower and "for" in question_lower) or "used for" in question_lower:
        question += " indications-and-usage therapeutic-indication clinical-indications approved-uses treatment"
    
    return question

@observe(name="retrieval")
def _retriever_fn(question: str) -> Tuple[str, List[str]]:
    """
    Enhanced retrieval using RRF over Pinecone vector search and local BM25.
    Returns (formatted_context_str, raw_context_texts) for tracing and RAGAS.
    """
    if _is_outside_index_request(question):
        langfuse_client.update_current_span(
            input=question,
            output=FALLBACK_MESSAGE,
            metadata={"fallback": True, "reason": "medication_outside_index"},
        )
        return FALLBACK_MESSAGE, []

    # Expand query for better matching
    expanded_question = _expand_query(question)

    # Retrieve exactly the top 5 fused documents.
    docs: List[Document] = hybrid_retrieve(expanded_question, k_final=FINAL_CONTEXT_K)

    top_score = docs[0].metadata.get("rrf_score", 0.0) if docs else 0.0
    if not docs or top_score < MIN_FUSED_RRF_SCORE:
        langfuse_client.update_current_span(
            input=question,
            output=FALLBACK_MESSAGE,
            metadata={
                "fallback": True,
                "reason": "retrieval_below_threshold",
                "top_rrf_score": top_score,
            },
        )
        return FALLBACK_MESSAGE, []

    # Use top 5 documents
    context_parts = []
    raw_contexts: List[str] = []
    for i, doc in enumerate(docs[:FINAL_CONTEXT_K], 1):
        content = _clean_text(doc.page_content)
        content = content[:1000]
        raw_contexts.append(content)

        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        score = doc.metadata.get("rrf_score", 0.0)

        context_parts.append(f"[Source {i}: {src}, p.{page}, rrf={score:.4f}]\n{content}")

    context = "\n\n".join(context_parts)

    # Allow more total context for comprehensive coverage
    if len(context) > 4000:
        context = context[:4000]

    langfuse_client.update_current_span(
        input=question,
        output=raw_contexts,
        metadata={
            "num_docs": len(raw_contexts),
            "context_length": len(context),
            "top_rrf_score": top_score,
            "retrieval": "rrf_bm25_pinecone",
        },
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

        if context == FALLBACK_MESSAGE and not raw_contexts:
            latency_ms = round((time.time() - start) * 1000, 2)
            langfuse_client.update_current_span(
                output=context,
                metadata={
                    "latency_ms": latency_ms,
                    "fallback": True,
                    "context_length": 0,
                    "retrieved_contexts": [],
                },
            )
            self._last_trace_id = langfuse_client.get_current_trace_id()
            self._last_contexts = []
            return context

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
                **llm._last_token_usage,
            },
        )

        self._last_trace_id = langfuse_client.get_current_trace_id()
        self._last_contexts = raw_contexts

        return answer

def get_rag_chain() -> SimpleRAGChain:
    return SimpleRAGChain()
