"""
backend/ingestion/expand_corpus.py

Corpus expansion pipeline for MediQuery.

Steps:
  1. Audit Pinecone index for existing coverage of the target drug list.
  2. Fetch missing FDA labels from the DailyMed API as SPL XML.
  3. Parse XML into named clinical sections (discards patient info / references).
  4. Chunk each section with a 512-token / 50-token-overlap splitter.
  5. Embed with all-MiniLM-L6-v2 and upsert to Pinecone with rich metadata.

Usage (run from project root):
    PYTHONPATH=backend python backend/ingestion/expand_corpus.py
    PYTHONPATH=backend python backend/ingestion/expand_corpus.py --dry-run
    PYTHONPATH=backend python backend/ingestion/expand_corpus.py --drug Gabapentin
"""
import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CHUNKS_PATH, EMBEDDING_MODEL

load_dotenv()

# ============================================================================
# Logging
# ============================================================================

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

PINECONE_API_KEY    = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "mediquery"

DAILYMED_SPLS_URL = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json"
DAILYMED_XML_URL  = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{setid}.xml"

RATE_LIMIT_DELAY   = 0.5    # seconds between DailyMed calls
REQUEST_TIMEOUT    = 30     # seconds
MAX_CHUNK_TOKENS   = 512
CHUNK_OVERLAP_TOKENS = 50
PINECONE_BATCH_SIZE  = 100

# SPL LOINC codes for sections to keep → canonical key stored in metadata
KEEP_SECTIONS: Dict[str, str] = {
    "42229-5": "boxed_warning",
    "34067-9": "indications_and_usage",
    "34068-7": "dosage_and_administration",
    "34070-3": "contraindications",
    "43685-7": "warnings_and_precautions",
    "34084-4": "adverse_reactions",
    "34073-7": "drug_interactions",
    "43684-0": "use_in_specific_populations",
    "34089-3": "description",
    "34090-1": "clinical_pharmacology",
}

SPL_NS = "urn:hl7-org:v3"


def _t(tag: str) -> str:
    return f"{{{SPL_NS}}}{tag}"


# ============================================================================
# Target drug list
# ============================================================================

TARGET_DRUGS: List[str] = [
    # Pain / Anti-inflammatory
    "Ibuprofen", "Acetaminophen", "Aspirin", "Naproxen", "Tramadol",
    "Oxycodone", "Hydrocodone", "Gabapentin", "Pregabalin", "Celecoxib",
    # Antibiotics
    "Amoxicillin", "Azithromycin", "Doxycycline", "Ciprofloxacin",
    "Metronidazole", "Clindamycin", "Augmentin", "Trimethoprim-Sulfamethoxazole",
    "Levofloxacin",
    # Antivirals / Antifungals
    "Acyclovir", "Valacyclovir", "Oseltamivir", "Fluconazole",
    # Psychiatric / Neurological
    "Sertraline", "Escitalopram", "Fluoxetine", "Bupropion", "Venlafaxine",
    "Duloxetine", "Alprazolam", "Lorazepam", "Clonazepam",
    "Quetiapine", "Aripiprazole", "Lithium", "Lamotrigine",
    "Zolpidem", "Trazodone",
    # ADHD
    "Adderall", "Methylphenidate", "Atomoxetine",
    # Cardiovascular
    "Lisinopril", "Metoprolol", "Atorvastatin", "Amlodipine", "Losartan",
    "Warfarin", "Clopidogrel", "Furosemide", "Spironolactone", "Carvedilol",
    "Digoxin", "Apixaban", "Rivaroxaban",
    # Diabetes
    "Metformin", "Insulin Glargine", "Insulin Lispro", "Semaglutide",
    "Tirzepatide", "Jardiance", "Farxiga", "Januvia", "Glipizide",
    # Respiratory / Allergy
    "Albuterol", "Montelukast", "Fluticasone", "Budesonide", "Formoterol",
    "Tiotropium", "Prednisone", "Methylprednisolone",
    "Cetirizine", "Loratadine", "Fexofenadine", "Diphenhydramine", "Hydroxyzine",
    # GI
    "Omeprazole", "Pantoprazole", "Esomeprazole", "Famotidine", "Ondansetron",
    "Loperamide", "Polyethylene Glycol", "Simethicone", "Metoclopramide",
    "Dicyclomine",
    # Thyroid / Hormones
    "Levothyroxine", "Liothyronine", "Estradiol", "Progesterone",
    "Medroxyprogesterone", "Clomiphene", "Letrozole", "Norethindrone",
    # Men's health
    "Sildenafil", "Tadalafil", "Finasteride", "Dutasteride",
    "Testosterone Cypionate", "Testosterone Gel", "Tamsulosin",
    # Dermatology
    "Tretinoin", "Adapalene", "Clindamycin topical", "Isotretinoin",
    "Hydroquinone", "Azelaic Acid", "Tacrolimus", "Clobetasol",
    "Triamcinolone", "Desonide", "Spironolactone topical",
    "Minoxidil topical", "Minoxidil oral", "Ketoconazole shampoo",
    # Ophthalmic
    "Latanoprost", "Ciprofloxacin eye drops", "Prednisolone eye drops",
    # Bone / Joint / Rheumatology
    "Alendronate", "Colchicine", "Allopurinol", "Methotrexate",
    "Hydroxychloroquine",
    # Neurology
    "Donepezil", "Memantine", "Carbidopa-Levodopa", "Ropinirole",
    # Urology
    "Oxybutynin", "Solifenacin",
    # Oncology / Hormone therapy
    "Tamoxifen", "Anastrozole",
    # Addiction / SUD
    "Naloxone", "Naltrexone", "Buprenorphine", "Methadone", "Varenicline",
    # HIV
    "Dolutegravir", "Tenofovir",
    # OTC / Supplements
    "Folic Acid", "Vitamin D3", "Vitamin B12", "Iron supplements",
    "Magnesium Oxide", "Potassium Chloride",
]

# Tried in order when the primary name returns no DailyMed results
SEARCH_FALLBACKS: Dict[str, List[str]] = {
    "Adderall":                     ["amphetamine mixed salts", "amphetamine dextroamphetamine"],
    "Augmentin":                    ["amoxicillin clavulanate", "amoxicillin-clavulanate"],
    "Trimethoprim-Sulfamethoxazole":["sulfamethoxazole trimethoprim", "Bactrim", "Septra"],
    "Jardiance":                    ["empagliflozin"],
    "Farxiga":                      ["dapagliflozin"],
    "Januvia":                      ["sitagliptin"],
    "Insulin Lispro":               ["Humalog", "insulin lispro"],
    "Semaglutide":                  ["Ozempic", "Wegovy", "Rybelsus"],
    "Tirzepatide":                  ["Mounjaro", "Zepbound"],
    "Polyethylene Glycol":          ["polyethylene glycol 3350", "MiraLax", "PEG 3350"],
    "Carbidopa-Levodopa":           ["carbidopa levodopa", "Sinemet"],
    "Testosterone Cypionate":       ["testosterone cypionate injection"],
    "Testosterone Gel":             ["AndroGel", "testosterone gel"],
    "Clindamycin topical":          ["clindamycin phosphate topical", "clindamycin topical solution"],
    "Spironolactone topical":       ["spironolactone topical"],
    "Minoxidil topical":            ["minoxidil topical solution", "minoxidil 5%"],
    "Minoxidil oral":               ["minoxidil tablet"],
    "Ketoconazole shampoo":         ["ketoconazole shampoo", "Nizoral"],
    "Ciprofloxacin eye drops":      ["ciprofloxacin ophthalmic"],
    "Prednisolone eye drops":       ["prednisolone acetate ophthalmic"],
    "Iron supplements":             ["ferrous sulfate", "ferric sulfate"],
    "Vitamin D3":                   ["cholecalciferol"],
    "Vitamin B12":                  ["cyanocobalamin"],
    "Magnesium Oxide":              ["magnesium oxide tablet"],
    "Potassium Chloride":           ["potassium chloride tablet"],
    "Dolutegravir":                 ["Tivicay"],
    "Tenofovir":                    ["tenofovir disoproxil fumarate", "Viread"],
    "Buprenorphine":                ["buprenorphine naloxone", "Suboxone", "Subutex"],
    "Varenicline":                  ["Chantix"],
    "Oseltamivir":                  ["Tamiflu"],
    "Valacyclovir":                 ["Valtrex"],
    "Acyclovir":                    ["Zovirax"],
    "Azelaic Acid":                 ["azelaic acid cream", "Finacea"],
    "Hydroquinone":                 ["hydroquinone cream"],
    "Latanoprost":                  ["Xalatan"],
    "Desonide":                     ["DesOwen", "desonide cream"],
    "Allopurinol":                  ["Zyloprim"],
    "Colchicine":                   ["Colcrys"],
    "Donepezil":                    ["Aricept"],
    "Memantine":                    ["Namenda"],
    "Ropinirole":                   ["Requip"],
    "Solifenacin":                  ["VESIcare"],
    "Tamsulosin":                   ["Flomax"],
    "Anastrozole":                  ["Arimidex"],
    "Tamoxifen":                    ["Nolvadex"],
    "Naloxone":                     ["Narcan"],
    "Naltrexone":                   ["Vivitrol", "ReVia"],
    "Apixaban":                     ["Eliquis"],
    "Rivaroxaban":                  ["Xarelto"],
    "Warfarin":                     ["Coumadin"],
    "Clopidogrel":                  ["Plavix"],
    "Losartan":                     ["Cozaar"],
    "Amlodipine":                   ["Norvasc"],
    "Carvedilol":                   ["Coreg"],
    "Furosemide":                   ["Lasix"],
    "Digoxin":                      ["Lanoxin"],
    "Glipizide":                    ["Glucotrol"],
    "Albuterol":                    ["Proventil", "Ventolin", "ProAir"],
    "Montelukast":                  ["Singulair"],
    "Tiotropium":                   ["Spiriva"],
    "Formoterol":                   ["Foradil"],
    "Fexofenadine":                 ["Allegra"],
    "Cetirizine":                   ["Zyrtec"],
    "Loratadine":                   ["Claritin"],
    "Hydroxyzine":                  ["Vistaril", "Atarax"],
    "Ondansetron":                  ["Zofran"],
    "Metoclopramide":               ["Reglan"],
    "Esomeprazole":                 ["Nexium"],
    "Pantoprazole":                 ["Protonix"],
    "Omeprazole":                   ["Prilosec"],
    "Famotidine":                   ["Pepcid"],
    "Pregabalin":                   ["Lyrica"],
    "Gabapentin":                   ["Neurontin"],
    "Tramadol":                     ["Ultram"],
    "Celecoxib":                    ["Celebrex"],
    "Oxycodone":                    ["OxyContin", "Roxicodone"],
    "Hydrocodone":                  ["Vicodin", "Norco"],
    "Venlafaxine":                  ["Effexor"],
    "Quetiapine":                   ["Seroquel"],
    "Aripiprazole":                 ["Abilify"],
    "Lamotrigine":                  ["Lamictal"],
    "Trazodone":                    ["Desyrel"],
    "Alprazolam":                   ["Xanax"],
    "Lorazepam":                    ["Ativan"],
    "Clonazepam":                   ["Klonopin"],
    "Zolpidem":                     ["Ambien"],
    "Lithium":                      ["lithium carbonate", "Eskalith"],
    "Liothyronine":                 ["Cytomel"],
    "Progesterone":                 ["Prometrium"],
    "Norethindrone":                ["Aygestin"],
    "Medroxyprogesterone":          ["Depo-Provera", "Provera"],
    "Clomiphene":                   ["Clomid", "clomiphene citrate"],
    "Letrozole":                    ["Femara"],
    "Sildenafil":                   ["Viagra", "Revatio"],
    "Tadalafil":                    ["Cialis"],
    "Finasteride":                  ["Proscar", "Propecia"],
    "Dutasteride":                  ["Avodart"],
    "Oxybutynin":                   ["Ditropan"],
    "Adapalene":                    ["Differin"],
    "Tacrolimus":                   ["Protopic"],
    "Clobetasol":                   ["Temovate"],
    "Triamcinolone":                ["Kenalog"],
    "Methotrexate":                 ["methotrexate tablet", "Trexall"],
    "Hydroxychloroquine":           ["Plaquenil"],
    "Alendronate":                  ["Fosamax"],
}

# Maps filename prefix in chunks.jsonl → target drug name already covered by PDFs
_PDF_COVERAGE: Dict[str, str] = {
    "prozac":     "Fluoxetine",
    "zoloft":     "Sertraline",
    "lexapro":    "Escitalopram",
    "wellbutrin": "Bupropion",
    "cymbalta":   "Duloxetine",
    "ritalin":    "Methylphenidate",
    "concerta":   "Methylphenidate",
    "strattera":  "Atomoxetine",
    "lipitor":    "Atorvastatin",
    "atorvastatin": "Atorvastatin",
    "glumetza":   "Metformin",
    "metformin":  "Metformin",
    "ozempic":    "Semaglutide",
    "accutane":   "Isotretinoin",
    "adderall":   "Adderall",
    "insulin":    "Insulin Glargine",
    "lisinopril": "Lisinopril",
    "metoprolol": "Metoprolol",
    "naproxen":   "Naproxen",
    "ibuprofen":  "Ibuprofen",
    "tretinoin":  "Tretinoin",
}


# ============================================================================
# Coverage audit
# ============================================================================

def _norm(name: str) -> str:
    """Lowercase, strip non-alphanumeric — for loose name matching."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _covered_by_pdfs(target_drugs: List[str]) -> List[str]:
    """
    Check chunks.jsonl source filenames for PDF-based coverage.
    Returns the subset of target_drugs already present.
    """
    if not Path(CHUNKS_PATH).exists():
        return []

    # Collect all unique filename stems from the chunks corpus
    stems: set = set()
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            src = rec.get("metadata", {}).get("source", "")
            if src:
                stems.add(Path(src).stem.lower())

    present: List[str] = []
    for drug in target_drugs:
        drug_norm = _norm(drug)
        # Direct: does any stem contain the normalized drug name?
        if any(drug_norm in _norm(s) for s in stems):
            present.append(drug)
            continue
        # Via _PDF_COVERAGE brand→generic table
        for prefix, mapped_drug in _PDF_COVERAGE.items():
            if _norm(mapped_drug) == drug_norm and any(prefix in s for s in stems):
                present.append(drug)
                break

    return present


def _covered_by_pinecone_metadata(
    drugs: List[str],
    index,
    embeddings: HuggingFaceEmbeddings,
) -> List[str]:
    """
    For drugs that may have been ingested via a previous DailyMed run,
    check for the drug_name metadata field in Pinecone.
    Uses a semantic probe so we don't need to enumerate all vectors.
    """
    present: List[str] = []
    for drug in drugs:
        try:
            vec = embeddings.embed_query(f"{drug} medication prescribing information")
            results = index.query(
                vector=vec,
                top_k=1,
                filter={"drug_name": {"$eq": drug.lower()}},
                include_metadata=False,
            )
            if results.matches:
                present.append(drug)
            time.sleep(0.1)
        except Exception as exc:
            log.debug("Pinecone metadata check failed for %s: %s", drug, exc)
    return present


def audit_coverage(
    target_drugs: List[str],
    index,
    embeddings: HuggingFaceEmbeddings,
) -> Tuple[List[str], List[str]]:
    """
    Returns (present, missing) for the target drug list.

    Phase 1 — fast, in-process: scan chunks.jsonl source filenames.
    Phase 2 — Pinecone metadata filter: catches drugs from prior DailyMed runs.
    """
    log.info("Auditing corpus coverage for %d target drugs…", len(target_drugs))

    pdf_present = _covered_by_pdfs(target_drugs)
    log.info("  PDF corpus covers %d / %d drugs", len(pdf_present), len(target_drugs))

    not_in_pdf = [d for d in target_drugs if d not in pdf_present]
    if not_in_pdf:
        log.info("  Checking Pinecone metadata for %d remaining drugs…", len(not_in_pdf))
        pinecone_present = _covered_by_pinecone_metadata(not_in_pdf, index, embeddings)
        log.info("  Pinecone metadata covers %d additional drugs", len(pinecone_present))
    else:
        pinecone_present = []

    covered = set(pdf_present) | set(pinecone_present)
    present = [d for d in target_drugs if d in covered]
    missing = [d for d in target_drugs if d not in covered]
    return present, missing


# ============================================================================
# DailyMed API
# ============================================================================

def _dailymed_search(drug_name: str, pagesize: int = 5) -> List[dict]:
    """Query /spls.json for a drug name. Returns the data list."""
    params = {"drug_name": drug_name, "name_type": "both", "pagesize": pagesize, "page": 1}
    resp = requests.get(DAILYMED_SPLS_URL, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json().get("data", [])


def fetch_spl_record(drug_name: str) -> Optional[Tuple[str, str, str]]:
    """
    Try to find a DailyMed SPL record for a drug.
    Tries the primary name first, then any fallback search terms.

    Returns (setid, title, source_url) or None if not found.
    """
    search_terms = [drug_name] + SEARCH_FALLBACKS.get(drug_name, [])

    for term in search_terms:
        time.sleep(RATE_LIMIT_DELAY)
        try:
            results = _dailymed_search(term)
        except Exception as exc:
            log.warning("  DailyMed search error for '%s': %s", term, exc)
            continue

        if not results:
            continue

        # Prefer results whose title contains the search term
        norm_term = _norm(term)
        ranked = sorted(
            results,
            key=lambda r: (0 if norm_term in _norm(r.get("title", "")) else 1),
        )
        rec = ranked[0]
        setid = rec.get("setid", "")
        title = rec.get("title", drug_name)
        source_url = f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={setid}"
        if setid:
            log.info(
                "  Found via '%s': %s  [setid=%s]",
                term, title[:70], setid,
            )
            return setid, title, source_url

    return None


def fetch_spl_xml(setid: str) -> str:
    """Download the full SPL XML for a given setid."""
    time.sleep(RATE_LIMIT_DELAY)
    url = DAILYMED_XML_URL.format(setid=setid)
    resp = requests.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.text


# ============================================================================
# SPL XML parsing
# ============================================================================

def _extract_text(elem) -> str:
    """
    Recursively extract plain text from an SPL XML element.
    Handles paragraph, list/item, content, table rows. Skips media/footnotes.
    """
    if elem is None:
        return ""

    skip_tags = {_t("renderMultiMedia"), _t("footnote"), _t("footnoteRef")}
    parts: List[str] = []

    def _walk(node, depth: int = 0) -> None:
        tag = node.tag
        local = tag.split("}")[-1] if "}" in tag else tag

        if tag in skip_tags:
            return

        text = (node.text or "").strip()
        if text:
            parts.append(text)

        for child in node:
            child_local = (child.tag.split("}")[-1] if "}" in child.tag else child.tag)
            if child_local == "list":
                for item in child:
                    item_text = _extract_text(item).strip()
                    if item_text:
                        parts.append(f"• {item_text}")
            elif child_local in ("tr",):
                # Table row — join cells with tab
                cell_texts = []
                for cell in child:
                    ct = _extract_text(cell).strip()
                    if ct:
                        cell_texts.append(ct)
                if cell_texts:
                    parts.append("\t".join(cell_texts))
            elif child_local in ("thead", "tbody", "tfoot", "table"):
                _walk(child, depth + 1)
            else:
                _walk(child, depth + 1)

            tail = (child.tail or "").strip()
            if tail:
                parts.append(tail)

    _walk(elem)
    # Collapse runs of whitespace / blank lines
    text = " ".join(parts)
    text = re.sub(r"  +", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _collect_sections(root_elem, parent_code: Optional[str] = None) -> List[Dict]:
    """
    Walk the structuredBody and collect all sections whose LOINC code is in
    KEEP_SECTIONS, recursing into subsections. Returns a list of dicts:
      {section_key, section_title, text}
    """
    collected: List[Dict] = []

    for component in root_elem.iter(_t("section")):
        code_elem = component.find(_t("code"))
        if code_elem is None:
            continue

        code = code_elem.get("code", "")
        if code not in KEEP_SECTIONS:
            continue

        section_key = KEEP_SECTIONS[code]
        title_elem = component.find(_t("title"))
        section_title = (title_elem.text or section_key).strip() if title_elem is not None else section_key

        text_elem = component.find(_t("text"))
        body_text = _extract_text(text_elem) if text_elem is not None else ""

        # Also pull text from subsections (nested components under this section)
        for sub in component.findall(f".//{_t('section')}"):
            sub_code_elem = sub.find(_t("code"))
            if sub_code_elem is None or sub_code_elem.get("code", "") in KEEP_SECTIONS:
                continue  # keep only sub-sections without their own top-level code
            sub_text_elem = sub.find(_t("text"))
            if sub_text_elem is not None:
                sub_text = _extract_text(sub_text_elem)
                if sub_text:
                    body_text = f"{body_text}\n\n{sub_text}".strip()

        if body_text and len(body_text) > 30:
            collected.append({
                "section_key": section_key,
                "section_title": section_title,
                "text": body_text,
            })

    return collected


def parse_spl_xml(xml_str: str) -> Tuple[List[Dict], str, List[str], str]:
    """
    Parse a DailyMed SPL XML string.

    Returns:
        sections           — list of {section_key, section_title, text}
        drug_name          — extracted generic name (best-effort)
        brand_names        — list of brand name strings found in the label
        label_version_date — ISO date string YYYY-MM-DD from <effectiveTime>, or ""
    """
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError as exc:
        log.error("  XML parse error: %s", exc)
        return [], "", [], ""

    # Extract label version date from top-level <effectiveTime value="YYYYMMDD"/>
    label_version_date = ""
    for eff in root.iter(_t("effectiveTime")):
        val = eff.get("value", "")
        if re.match(r"^\d{8}$", val):
            label_version_date = f"{val[:4]}-{val[4:6]}-{val[6:]}"
            break
        if re.match(r"^\d{4}-\d{2}-\d{2}$", val):
            label_version_date = val
            break

    # Extract drug / brand names from the SPL header
    generic_name = ""
    brand_names: List[str] = []

    for name_elem in root.iter(_t("name")):
        val = (name_elem.text or "").strip()
        if val:
            brand_names.append(val)

    # Prefer the first <name> under manufacturedMedicinalProduct as the primary name
    for mmp in root.iter(_t("manufacturedMedicinalProduct")):
        name_elem = mmp.find(_t("name"))
        if name_elem is not None and name_elem.text:
            generic_name = name_elem.text.strip()
            break

    brand_names = list(dict.fromkeys(b for b in brand_names if b))  # dedupe, keep order

    sections = _collect_sections(root)
    return sections, generic_name, brand_names, label_version_date


# ============================================================================
# Chunking
# ============================================================================

def _build_splitter() -> RecursiveCharacterTextSplitter:
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=MAX_CHUNK_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
    )


def build_chunks(
    sections: List[Dict],
    drug_name: str,
    brand_names: List[str],
    source_url: str,
    splitter: RecursiveCharacterTextSplitter,
    label_version_date: str = "",
) -> List[Dict]:
    """
    Chunk each section and return a list of vector-ready dicts:
      {id, text, drug_name, brand_names, section, section_title, chunk_index,
       total_chunks_in_section, source_url, label_version_date}
    """
    chunks: List[Dict] = []
    drug_norm = re.sub(r"[^a-z0-9]", "", drug_name.lower())

    for sec in sections:
        raw_chunks = splitter.split_text(sec["text"])
        total = len(raw_chunks)

        for idx, chunk_text in enumerate(raw_chunks):
            if not chunk_text.strip():
                continue

            # Deterministic, stable ID: drug + section + position
            chunk_id = hashlib.sha1(
                f"{drug_norm}|{sec['section_key']}|{idx}".encode()
            ).hexdigest()[:16]
            vec_id = f"{drug_norm}_{sec['section_key']}_{idx}_{chunk_id}"

            chunks.append({
                "id": vec_id,
                "text": chunk_text.strip(),
                "metadata": {
                    "drug_name":          drug_name.lower(),
                    "brand_names":        ", ".join(brand_names[:5]),
                    "section":            sec["section_key"],
                    "section_title":      sec["section_title"][:200],
                    "chunk_index":        idx,
                    "total_chunks":       total,
                    "source_url":         source_url,
                    "label_version_date": label_version_date,
                    "text":               chunk_text.strip()[:1000],
                },
            })

    return chunks


def _update_corpus_drugs_json(drug_name: str) -> None:
    """Append drug_name (lowercase) to data/corpus_drugs.json, creating it if needed."""
    corpus_path = Path(CHUNKS_PATH).parent / "corpus_drugs.json"
    try:
        existing: List[str] = json.loads(corpus_path.read_text()) if corpus_path.exists() else []
        entry = drug_name.lower()
        if entry not in existing:
            existing.append(entry)
            corpus_path.write_text(json.dumps(sorted(existing), indent=2))
    except Exception as exc:
        log.warning("  Could not update corpus_drugs.json: %s", exc)


# ============================================================================
# Pinecone upsert
# ============================================================================

def upsert_chunks(
    chunks: List[Dict],
    index,
    embeddings: HuggingFaceEmbeddings,
) -> int:
    """Embed and upsert chunks in batches. Returns the number of vectors upserted."""
    if not chunks:
        return 0

    texts = [c["text"] for c in chunks]
    vectors = embeddings.embed_documents(texts)

    records = [
        {"id": c["id"], "values": vec, "metadata": c["metadata"]}
        for c, vec in zip(chunks, vectors)
    ]

    upserted = 0
    for i in range(0, len(records), PINECONE_BATCH_SIZE):
        batch = records[i : i + PINECONE_BATCH_SIZE]
        index.upsert(vectors=batch)
        upserted += len(batch)

    return upserted


# ============================================================================
# Pinecone / embedding setup
# ============================================================================

def _init_pinecone() -> tuple:
    """Return (Pinecone client, index object)."""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        log.info("Creating Pinecone index '%s'…", PINECONE_INDEX_NAME)
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    index = pc.Index(PINECONE_INDEX_NAME)
    return pc, index


# ============================================================================
# Main pipeline
# ============================================================================

def run(target_drugs: List[str], dry_run: bool = False) -> None:
    log.info("=" * 65)
    log.info("  MediQuery Corpus Expansion")
    log.info("=" * 65)

    # ── Setup ──────────────────────────────────────────────────────────────
    log.info("Loading embedding model (%s)…", EMBEDDING_MODEL)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )
    splitter = _build_splitter()

    log.info("Connecting to Pinecone index '%s'…", PINECONE_INDEX_NAME)
    _, index = _init_pinecone()

    # ── Audit ──────────────────────────────────────────────────────────────
    present, missing = audit_coverage(target_drugs, index, embeddings)

    log.info("")
    log.info("━" * 65)
    log.info("COVERAGE AUDIT  (%d present / %d missing)", len(present), len(missing))
    log.info("━" * 65)
    log.info("Already ingested (%d):", len(present))
    for drug in present:
        log.info("  ✅  %s", drug)
    log.info("")
    log.info("Missing — will fetch from DailyMed (%d):", len(missing))
    for drug in missing:
        log.info("  ❌  %s", drug)
    log.info("━" * 65)

    if dry_run:
        log.info("Dry-run mode — stopping before ingestion.")
        return

    if not missing:
        log.info("Nothing to do — all target drugs are already ingested.")
        return

    # ── Ingestion loop ─────────────────────────────────────────────────────
    succeeded: List[str] = []
    failed: List[Tuple[str, str]] = []

    for i, drug in enumerate(missing, 1):
        log.info("")
        log.info("[%d/%d]  %s", i, len(missing), drug)

        # 1. Fetch SPL record
        record = fetch_spl_record(drug)
        if record is None:
            reason = "No SPL record found on DailyMed (tried primary + all fallbacks)"
            log.warning("  ❌  %s — %s", drug, reason)
            failed.append((drug, reason))
            continue

        setid, label_title, source_url = record

        # 2. Fetch SPL XML
        try:
            xml_str = fetch_spl_xml(setid)
        except Exception as exc:
            reason = f"XML download failed: {exc}"
            log.warning("  ❌  %s — %s", drug, reason)
            failed.append((drug, reason))
            continue

        # 3. Parse XML
        sections, extracted_name, brand_names, label_version_date = parse_spl_xml(xml_str)
        if not sections:
            reason = "XML parsed but no keepable sections found"
            log.warning("  ❌  %s — %s", drug, reason)
            failed.append((drug, reason))
            continue

        section_summary = ", ".join(s["section_key"] for s in sections)
        log.info(
            "  Parsed %d sections: %s  [version: %s]",
            len(sections), section_summary[:100],
            label_version_date or "unknown",
        )

        # 4. Chunk
        chunks = build_chunks(
            sections, drug, brand_names, source_url, splitter, label_version_date
        )
        log.info("  %d chunks from %d sections", len(chunks), len(sections))

        if not chunks:
            reason = "Chunking produced no output"
            log.warning("  ❌  %s — %s", drug, reason)
            failed.append((drug, reason))
            continue

        # 5. Embed + upsert
        try:
            n = upsert_chunks(chunks, index, embeddings)
            log.info("  ✅  Upserted %d vectors to Pinecone", n)
            succeeded.append(drug)
            _update_corpus_drugs_json(drug)
        except Exception as exc:
            reason = f"Pinecone upsert failed: {exc}"
            log.error("  ❌  %s — %s", drug, reason)
            failed.append((drug, reason))

    # ── Summary ────────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 65)
    log.info("INGESTION SUMMARY")
    log.info("=" * 65)
    log.info("Already present  : %d", len(present))
    log.info("Newly ingested   : %d", len(succeeded))
    log.info("Failed           : %d", len(failed))

    if succeeded:
        log.info("")
        log.info("Succeeded:")
        for drug in succeeded:
            log.info("  ✅  %s", drug)

    if failed:
        log.info("")
        log.info("Failed:")
        for drug, reason in failed:
            log.info("  ❌  %-35s  %s", drug, reason)

    log.info("=" * 65)


# ============================================================================
# Entry point
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Expand MediQuery Pinecone corpus")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Audit coverage and print missing drugs without fetching or upserting",
    )
    parser.add_argument(
        "--drug",
        metavar="NAME",
        help="Ingest a single drug by name instead of the full target list",
    )
    args = parser.parse_args()

    if args.drug:
        drugs = [args.drug]
    else:
        drugs = TARGET_DRUGS

    run(drugs, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
