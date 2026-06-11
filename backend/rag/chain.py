# backend/rag/chain.py
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langfuse.decorators import observe

from observability.langfuse_tracing import langfuse_client  # init registers global OTEL tracer
from rag import drug_aliases
from rag.prompts import RAG_PROMPT
from retrievers.hybrid import hybrid_retrieve
import llm
from llm import generate_answer


FINAL_CONTEXT_K = 5
FALLBACK_MESSAGE = "The requested medication is outside the current FDA drug label index."
MIN_FUSED_RRF_SCORE = 1.0 / (60 + 10)

# Kept for backward-compatibility; _CORPUS_DRUG_SET supersedes it at runtime.
INDEXED_MEDICATION_TERMS = {
    "accutane", "adderall", "amphetamine", "atorvastatin", "dextroamphetamine",
    "fluoxetine", "glucophage", "ibuprofen", "isotretinoin", "lipitor",
    "lisinopril", "metformin", "methylphenidate", "prozac", "retin-a",
    "ritalin", "tretinoin",
}
KNOWN_UNSUPPORTED_MEDICATIONS = {
    "eliquis", "humira", "jardiance", "keytruda", "ozempic", "warfarin",
}
QUESTION_STOPWORDS = {
    "a", "about", "and", "are", "can", "does", "do", "for", "has", "have",
    "how", "i", "is", "it", "me", "my", "of", "should", "take", "the",
    "there", "to", "what", "when", "which", "with",
}


# ============================================================================
# Corpus drug set — built once at import time
# ============================================================================

def _load_corpus_drug_set() -> frozenset:
    """
    Build the set of all drug names (brands + generics) known to be in the
    corpus.  Sources merged in priority order:
      1. INDEXED_MEDICATION_TERMS (hardcoded baseline)
      2. All keys in drug_aliases.BRAND_TO_GENERIC (brand names)
      3. All values in drug_aliases.BRAND_TO_GENERIC (generic names)
      4. data/corpus_drugs.json written by ingestion scripts
    """
    base = set(INDEXED_MEDICATION_TERMS)
    base.update(k.lower() for k in drug_aliases.BRAND_TO_GENERIC.keys())
    base.update(v.lower() for v in drug_aliases.BRAND_TO_GENERIC.values())

    # corpus_drugs.json lives next to chunks.jsonl (data/)
    try:
        from config import CHUNKS_PATH
        corpus_json = Path(CHUNKS_PATH).parent / "corpus_drugs.json"
        if corpus_json.exists():
            drugs = json.loads(corpus_json.read_text())
            base.update(d.lower() for d in drugs)
    except Exception:
        pass

    return frozenset(base)


_CORPUS_DRUG_SET: frozenset = _load_corpus_drug_set()


# ============================================================================
# Pre-retrieval: drug extraction + coverage analysis
# ============================================================================

def _extract_drugs_groq(question: str) -> List[str]:
    """
    Use Groq (llama-3.1-8b-instant) to extract all drug/medication names from
    the query, normalise brand→generic via drug_aliases, and return a deduplicated
    list of lowercase generic names.

    Returns [] on any error or when USE_GROQ is False.
    """
    if not os.getenv("USE_GROQ", "false").lower() == "true":
        return []

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return []

    try:
        from groq import Groq  # already a project dependency

        client = Groq(api_key=api_key)
        prompt = (
            "Extract all medication or drug names mentioned in the query below. "
            "Include both brand names and generic names as separate entries. "
            "Return ONLY valid JSON with this exact structure: "
            '{"drugs": ["name1", "name2"]}. '
            "Use lowercase for every name. "
            'Return {"drugs": []} if no drugs are mentioned.\n\n'
            f"Query: {question}"
        )
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0,
            max_tokens=120,
        )
        raw = resp.choices[0].message.content.strip()

        # Strip markdown code fences if the model wraps the JSON
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip("`").strip()

        try:
            data = json.loads(raw)
            names = [str(d).lower().strip() for d in data.get("drugs", []) if d]
        except json.JSONDecodeError:
            # Fallback: pull quoted words from the response
            names = re.findall(r'"([a-z][a-z0-9 /-]*)"', raw)

        # Normalise brand→generic and deduplicate
        seen: set = set()
        result: List[str] = []
        for n in names:
            generic = drug_aliases.normalize_drug_name(n)
            if generic and generic not in seen:
                seen.add(generic)
                result.append(generic)

        return result

    except Exception:
        return []


def _is_in_corpus(drug_name: str) -> bool:
    """Check whether a (normalised) drug name is in the known corpus set."""
    return drug_name.lower() in _CORPUS_DRUG_SET


def _analyze_query_drugs(
    question: str,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Extract drug names, normalise, and check corpus coverage.

    Returns:
        all_drugs — every normalised generic name extracted from the query
        covered   — subset present in the corpus
        missing   — subset absent from the corpus
    """
    all_drugs = _extract_drugs_groq(question)
    if not all_drugs:
        return [], [], []

    covered = [d for d in all_drugs if _is_in_corpus(d)]
    missing = [d for d in all_drugs if not _is_in_corpus(d)]
    return all_drugs, covered, missing


def _no_coverage_response(drug_names: List[str]) -> str:
    names_str = ", ".join(drug_aliases.display_name(d) for d in drug_names)
    examples = ", ".join(drug_aliases.EXAMPLE_CORPUS_DRUGS)
    return (
        f"None of the medications you mentioned ({names_str}) are in my current "
        f"FDA label database and could not be answered. "
        f"I can answer questions about: {examples}, and more."
    )


def _partial_coverage_note(covered: List[str], missing: List[str]) -> str:
    covered_str = ", ".join(drug_aliases.display_name(d) for d in covered)
    missing_str = ", ".join(drug_aliases.display_name(d) for d in missing)
    verb = "is" if len(missing) == 1 else "are"
    return (
        f"\n\n---\n\n**Note:** This response only covers {covered_str}. "
        f"{missing_str} {verb} not currently in my FDA label database "
        f"and could not be included in this answer."
    )


# ============================================================================
# Label version date note (Feature 2)
# ============================================================================

def _build_version_note(doc_metadatas: List[Dict]) -> str:
    """
    Build a label attribution line from retrieved doc metadata.
    Only fires when at least one chunk carries a label_version_date field.

    For a single drug:   "Source: FDA label for ibuprofen, last updated 2023-06-15."
    For multiple drugs:  "Source: FDA labels (ibuprofen, sertraline). Oldest label:
                          ibuprofen, last updated 2022-01-01 (least recently updated
                          label in this response)."
    """
    # Collect per-drug oldest version date
    drug_oldest: Dict[str, str] = {}
    for meta in doc_metadatas:
        drug = meta.get("drug_name", "").strip().lower()
        date = meta.get("label_version_date", "").strip()
        if not drug or not date:
            continue
        if not re.match(r"\d{4}-\d{2}-\d{2}", date):
            continue
        # Keep the oldest (earliest) date seen for this drug
        if drug not in drug_oldest or date < drug_oldest[drug]:
            drug_oldest[drug] = date

    if not drug_oldest:
        return ""

    if len(drug_oldest) == 1:
        drug, date = next(iter(drug_oldest.items()))
        display = drug_aliases.display_name(drug)
        return f"\n\n---\n\n*Source: FDA label for {display}, last updated {date}.*"

    # Multiple drugs — find the globally oldest date
    oldest_drug = min(drug_oldest, key=lambda d: drug_oldest[d])
    oldest_date = drug_oldest[oldest_drug]
    all_display = ", ".join(
        drug_aliases.display_name(d) for d in sorted(drug_oldest)
    )
    oldest_display = drug_aliases.display_name(oldest_drug)
    return (
        f"\n\n---\n\n*Source: FDA labels for {all_display}. "
        f"Oldest label: {oldest_display}, last updated {oldest_date} "
        f"(reflects the least recently updated label in this response).*"
    )


# ============================================================================
# Text utilities
# ============================================================================

def _clean_text(text: str) -> str:
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ============================================================================
# Existing fallback helpers (retained for Ollama / non-Groq mode)
# ============================================================================

def _mentions_indexed_medication(question: str) -> bool:
    """Check whether the question mentions any known corpus drug (brand or generic)."""
    q = question.lower()
    return any(re.search(rf"\b{re.escape(term)}\b", q) for term in _CORPUS_DRUG_SET)


def _mentions_known_unsupported_medication(question: str) -> bool:
    """Only flag as unsupported if the drug is not already in the corpus set."""
    truly_unsupported = KNOWN_UNSUPPORTED_MEDICATIONS - _CORPUS_DRUG_SET
    q = question.lower()
    return any(re.search(rf"\b{re.escape(term)}\b", q) for term in truly_unsupported)


def _candidate_medication_terms(question: str) -> List[str]:
    candidates = []
    for token in re.findall(r"\b[A-Za-z][A-Za-z0-9-]*\b", question):
        token_lower = token.lower()
        if token_lower in QUESTION_STOPWORDS:
            continue
        if token_lower in _CORPUS_DRUG_SET:
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


# ============================================================================
# Query expansion (unchanged from original)
# ============================================================================

def _expand_query(question: str) -> str:
    question_lower = question.lower()

    drug_map = {
        "adderall":    " amphetamine dextroamphetamine adhd stimulant adverse reactions cardiovascular psychiatric appetite insomnia tachycardia hypertension",
        "ritalin":     " methylphenidate adhd stimulant adverse reactions cardiovascular psychiatric appetite",
        "accutane":    " isotretinoin acne contraindications warnings teratogenic pregnancy birth-defects iPLEDGE precautions monitoring liver function lipids contraception blood donation",
        "isotretinoin":" accutane acne contraindications warnings teratogenic pregnancy birth-defects iPLEDGE precautions monitoring liver function lipids contraception blood donation",
        "lipitor":     " atorvastatin statin cholesterol ldl hdl hyperlipidemia dyslipidemia cardiovascular coronary heart-disease myocardial-infarction stroke atherosclerosis",
        "prozac":      " fluoxetine ssri depression dosage milligrams mg daily administration taper discontinuation syndrome gradual dose reduction withdrawal",
        "fluoxetine":  " prozac ssri depression dosage milligrams mg daily administration taper discontinuation syndrome gradual dose reduction withdrawal",
        "metformin":   " glucophage diabetes type-2-diabetes lactic-acidosis warnings renal kidney impairment contraindications black box warning lactic acidosis renal function monitoring serum creatinine hepatic impairment",
        "lisinopril":  " ace-inhibitor angiotensin-converting-enzyme antihypertensive blood-pressure hypertension contraindications angioedema pregnancy fetal-harm hypersensitivity",
        "tretinoin":   " retin-a retinoid acne skin photosensitivity pregnancy teratogenic topical avoid waxing abrasives sunscreen photosensitivity medicated cosmetics concomitant",
        "retin-a":     " tretinoin retinoid acne skin photosensitivity pregnancy teratogenic topical avoid waxing abrasives sunscreen photosensitivity medicated cosmetics concomitant",
        "ibuprofen":   " nsaid nonsteroidal anti-inflammatory pain analgesic bleeding anticoagulant warfarin aspirin interactions",
    }

    for drug, expansion in drug_map.items():
        if drug in question_lower:
            question += expansion
            break

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


# ============================================================================
# Retrieval
# ============================================================================

@observe(name="retrieval")
def _retriever_fn(
    question: str,
) -> Tuple[str, List[str], List[Dict]]:
    """
    Enhanced retrieval using RRF over Pinecone vector search and local BM25.

    Returns:
        context       — formatted context string for the prompt
        raw_contexts  — raw text chunks (used by RAGAS / eval)
        doc_metadatas — metadata dicts from each retrieved document
    """
    if _is_outside_index_request(question):
        pass  # langfuse span update skipped for compatibility
        return FALLBACK_MESSAGE, [], []

    expanded_question = _expand_query(question)
    docs: List[Document] = hybrid_retrieve(expanded_question, k_final=FINAL_CONTEXT_K)

    top_score = docs[0].metadata.get("rrf_score", 0.0) if docs else 0.0
    if not docs or top_score < MIN_FUSED_RRF_SCORE:
        pass  # langfuse span update skipped for compatibility
        return FALLBACK_MESSAGE, [], []

    context_parts: List[str] = []
    raw_contexts: List[str] = []
    doc_metadatas: List[Dict] = []

    for i, doc in enumerate(docs[:FINAL_CONTEXT_K], 1):
        content = _clean_text(doc.page_content)
        content = content[:1000]
        raw_contexts.append(content)
        doc_metadatas.append(dict(doc.metadata))

        src   = doc.metadata.get("source", "unknown")
        page  = doc.metadata.get("page", "?")
        score = doc.metadata.get("rrf_score", 0.0)
        context_parts.append(f"[Source {i}: {src}, p.{page}, rrf={score:.4f}]\n{content}")

    context = "\n\n".join(context_parts)
    if len(context) > 4000:
        context = context[:4000]

    pass  # langfuse span update skipped for compatibility

    return context, raw_contexts, doc_metadatas


# ============================================================================
# RAG chain
# ============================================================================

class SimpleRAGChain:
    def __init__(self) -> None:
        self._last_trace_id: Optional[str] = None
        self._last_contexts: List[str] = []
        self._last_doc_metadata: List[Dict] = []

    @observe(name="rag_query")
    def invoke(self, question: str, verbose: bool = False) -> str:
        start = time.time()
        pass  # langfuse span update skipped for compatibility

        # ── Feature 1: pre-retrieval drug analysis ────────────────────────
        all_drugs, covered, missing = _analyze_query_drugs(question)

        if all_drugs and not covered:
            # No mentioned drug is in the corpus — skip retrieval entirely.
            self._last_trace_id = None
            self._last_contexts = []
            self._last_doc_metadata = []
            return _no_coverage_response(all_drugs)

        # ── Retrieval ─────────────────────────────────────────────────────
        context, raw_contexts, doc_metadatas = _retriever_fn(question)

        if context == FALLBACK_MESSAGE and not raw_contexts:
            pass  # langfuse span update skipped for compatibility
            self._last_trace_id = None
            self._last_contexts = []
            self._last_doc_metadata = []
            return context

        if verbose:
            print(f"\n{'='*60}")
            print(f"Context length: {len(context)} chars")
            print(f"Context preview: {context[:200]}...")
            print(f"{'='*60}\n")

        # ── Generation ────────────────────────────────────────────────────
        prompt = RAG_PROMPT.format(context=context, question=question)

        if verbose:
            print(f"Full prompt length: {len(prompt)} chars\n")

        answer = generate_answer(prompt, verbose=verbose)

        latency_ms = round((time.time() - start) * 1000, 2)
        pass  # langfuse span update skipped for compatibility

        self._last_trace_id = None
        self._last_contexts = raw_contexts
        self._last_doc_metadata = doc_metadatas

        # ── Feature 2: append label version note ─────────────────────────
        version_note = _build_version_note(doc_metadatas)
        if version_note:
            answer += version_note

        # ── Feature 1: partial coverage note ─────────────────────────────
        if covered and missing:
            answer += _partial_coverage_note(covered, missing)

        return answer


def get_rag_chain() -> SimpleRAGChain:
    return SimpleRAGChain()
