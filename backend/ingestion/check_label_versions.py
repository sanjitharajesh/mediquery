"""
backend/ingestion/check_label_versions.py

Label staleness checker for the MediQuery Pinecone corpus.

Steps:
  1. For each drug in the known corpus, fetch one representative Pinecone
     chunk and read its stored label_version_date metadata field.
  2. Query the DailyMed API for the drug's current published_date.
  3. Compare. If DailyMed has a newer version, mark the drug stale.
  4. For each stale drug: delete all Pinecone vectors that carry that
     drug_name, then re-fetch and re-ingest the updated label via the
     same pipeline as expand_corpus.py.
  5. Print a clear summary.

Usage (run from project root):
    PYTHONPATH=backend python backend/ingestion/check_label_versions.py
    PYTHONPATH=backend python backend/ingestion/check_label_versions.py --dry-run
    PYTHONPATH=backend python backend/ingestion/check_label_versions.py --drug Gabapentin
"""
import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CHUNKS_PATH, EMBEDDING_MODEL

# Import shared pipeline functions from expand_corpus so we don't duplicate logic.
from ingestion.expand_corpus import (
    TARGET_DRUGS,
    DAILYMED_SPLS_URL,
    PINECONE_INDEX_NAME,
    RATE_LIMIT_DELAY,
    REQUEST_TIMEOUT,
    PINECONE_BATCH_SIZE,
    _norm,
    _dailymed_search,
    _init_pinecone,
    _build_splitter,
    fetch_spl_record,
    fetch_spl_xml,
    parse_spl_xml,
    build_chunks,
    upsert_chunks,
    _update_corpus_drugs_json,
)

load_dotenv()

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


# ============================================================================
# Step 1 — get stored version dates from Pinecone
# ============================================================================

def get_stored_version_date(
    drug_name: str,
    index,
    probe_vec: List[float],
) -> Optional[str]:
    """
    Query Pinecone for one chunk belonging to drug_name and return its
    label_version_date, or None if not found / not set.
    """
    try:
        results = index.query(
            vector=probe_vec,
            top_k=1,
            filter={"drug_name": {"$eq": drug_name.lower()}},
            include_metadata=True,
        )
        if not results.matches:
            return None
        meta = results.matches[0].metadata or {}
        date = meta.get("label_version_date", "")
        return date if re.match(r"\d{4}-\d{2}-\d{2}", date) else None
    except Exception as exc:
        log.debug("Pinecone query failed for %s: %s", drug_name, exc)
        return None


# ============================================================================
# Step 2 — get current DailyMed published date
# ============================================================================

def get_current_dailymed_date(drug_name: str) -> Optional[str]:
    """
    Query DailyMed for the drug's current published_date.
    Tries the primary name first, then brand fallbacks from expand_corpus.
    Returns ISO date string YYYY-MM-DD, or None if not found.
    """
    from ingestion.expand_corpus import SEARCH_FALLBACKS

    search_terms = [drug_name] + SEARCH_FALLBACKS.get(drug_name, [])

    for term in search_terms:
        time.sleep(RATE_LIMIT_DELAY)
        try:
            results = _dailymed_search(term, pagesize=3)
        except Exception as exc:
            log.debug("DailyMed search failed for '%s': %s", term, exc)
            continue

        if not results:
            continue

        # Prefer a result whose title contains the term
        norm_term = _norm(term)
        ranked = sorted(
            results,
            key=lambda r: (0 if norm_term in _norm(r.get("title", "")) else 1),
        )
        date_raw = ranked[0].get("published_date", "")
        if not date_raw:
            continue

        # Normalise to YYYY-MM-DD
        if re.match(r"\d{4}-\d{2}-\d{2}", date_raw):
            return date_raw[:10]
        if re.match(r"\d{8}", date_raw):
            return f"{date_raw[:4]}-{date_raw[4:6]}-{date_raw[6:]}"

    return None


# ============================================================================
# Step 4 — delete + re-ingest
# ============================================================================

def delete_drug_vectors(index, drug_name: str) -> None:
    """Delete all Pinecone vectors whose drug_name metadata matches."""
    index.delete(filter={"drug_name": {"$eq": drug_name.lower()}})
    log.info("  Deleted existing vectors for '%s'", drug_name)


def reingest_drug(
    drug_name: str,
    index,
    embeddings: HuggingFaceEmbeddings,
) -> Tuple[bool, str]:
    """
    Re-fetch and re-ingest a drug label from DailyMed.
    Returns (success, reason_string).
    """
    splitter = _build_splitter()

    record = fetch_spl_record(drug_name)
    if record is None:
        return False, "No SPL record found on DailyMed"

    setid, label_title, source_url = record

    try:
        xml_str = fetch_spl_xml(setid)
    except Exception as exc:
        return False, f"XML download failed: {exc}"

    sections, extracted_name, brand_names, label_version_date = parse_spl_xml(xml_str)
    if not sections:
        return False, "No keepable sections in parsed XML"

    chunks = build_chunks(
        sections, drug_name, brand_names, source_url, splitter, label_version_date
    )
    if not chunks:
        return False, "Chunking produced no output"

    try:
        n = upsert_chunks(chunks, index, embeddings)
        log.info("  Re-ingested %d vectors  [version: %s]", n, label_version_date or "unknown")
        _update_corpus_drugs_json(drug_name)
        return True, label_version_date
    except Exception as exc:
        return False, f"Pinecone upsert failed: {exc}"


# ============================================================================
# Main check loop
# ============================================================================

def _get_drugs_to_check(single_drug: Optional[str]) -> List[str]:
    """Build the list of drugs to check. Single-drug mode or full corpus."""
    if single_drug:
        return [single_drug]

    # Start with TARGET_DRUGS (DailyMed-ingested), then add any extras from
    # corpus_drugs.json that aren't already in TARGET_DRUGS.
    drugs = list(TARGET_DRUGS)
    corpus_json = Path(CHUNKS_PATH).parent / "corpus_drugs.json"
    if corpus_json.exists():
        try:
            extra = json.loads(corpus_json.read_text())
            known_norm = {_norm(d) for d in drugs}
            for e in extra:
                if _norm(e) not in known_norm:
                    drugs.append(e)
        except Exception:
            pass

    return drugs


def run(single_drug: Optional[str] = None, dry_run: bool = False) -> None:
    log.info("=" * 65)
    log.info("  MediQuery Label Version Checker")
    log.info("=" * 65)

    log.info("Loading embedding model (%s)…", EMBEDDING_MODEL)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )

    log.info("Connecting to Pinecone index '%s'…", PINECONE_INDEX_NAME)
    _, index = _init_pinecone()

    # Use a shared probe vector (the model is deterministic, so we can reuse it)
    probe_vec = embeddings.embed_query("medication prescribing information FDA label")

    drugs = _get_drugs_to_check(single_drug)
    log.info("Checking %d drugs…\n", len(drugs))

    checked = 0
    up_to_date: List[str] = []
    stale: List[Tuple[str, str, str]] = []       # (drug, stored_date, current_date)
    no_stored_date: List[str] = []
    no_current_date: List[str] = []
    check_failed: List[Tuple[str, str]] = []

    for drug in drugs:
        log.info("Checking  %s", drug)

        stored = get_stored_version_date(drug, index, probe_vec)
        if stored is None:
            log.info("  ⚠️   No label_version_date in Pinecone (PDF-based or pre-versioning ingest)")
            no_stored_date.append(drug)
            checked += 1
            continue

        current = get_current_dailymed_date(drug)
        if current is None:
            log.info("  ⚠️   Could not fetch current date from DailyMed")
            no_current_date.append(drug)
            checked += 1
            continue

        if current <= stored:
            log.info("  ✅  Up to date  (stored: %s  DailyMed: %s)", stored, current)
            up_to_date.append(drug)
        else:
            log.info("  🔄  STALE  (stored: %s  →  DailyMed: %s)", stored, current)
            stale.append((drug, stored, current))

        checked += 1

    # ── Refresh stale drugs ───────────────────────────────────────────────
    refreshed: List[str] = []
    refresh_failed: List[Tuple[str, str]] = []

    if stale:
        log.info("")
        log.info("━" * 65)
        log.info("Refreshing %d stale drug(s)%s", len(stale),
                 "  [DRY RUN — skipping]" if dry_run else "…")
        log.info("━" * 65)

    for drug, stored_date, current_date in stale:
        log.info("")
        log.info("Refreshing  %s  (%s → %s)", drug, stored_date, current_date)

        if dry_run:
            log.info("  [dry-run] would delete + re-ingest")
            continue

        delete_drug_vectors(index, drug)
        ok, detail = reingest_drug(drug, index, embeddings)
        if ok:
            log.info("  ✅  Refreshed successfully  (new version: %s)", detail)
            refreshed.append(drug)
        else:
            log.error("  ❌  Refresh failed: %s", detail)
            refresh_failed.append((drug, detail))

    # ── Summary ───────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 65)
    log.info("SUMMARY")
    log.info("=" * 65)
    log.info("Drugs checked        : %d", checked)
    log.info("Up to date           : %d", len(up_to_date))
    log.info("Stale (found)        : %d", len(stale))
    log.info("Refreshed            : %d", len(refreshed))
    log.info("Refresh failed       : %d", len(refresh_failed))
    log.info("No stored date       : %d  (not version-tracked)", len(no_stored_date))
    log.info("No DailyMed date     : %d  (DailyMed lookup failed)", len(no_current_date))

    if stale and dry_run:
        log.info("")
        log.info("Stale drugs that would be refreshed:")
        for drug, stored, current in stale:
            log.info("  🔄  %-35s  %s → %s", drug, stored, current)

    if refresh_failed:
        log.info("")
        log.info("Refresh failures:")
        for drug, reason in refresh_failed:
            log.info("  ❌  %-35s  %s", drug, reason)

    log.info("=" * 65)


# ============================================================================
# Entry point
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check MediQuery corpus label versions against DailyMed"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Identify stale drugs but do not delete or re-ingest",
    )
    parser.add_argument(
        "--drug",
        metavar="NAME",
        help="Check a single drug instead of the full corpus",
    )
    args = parser.parse_args()
    run(single_drug=args.drug, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
