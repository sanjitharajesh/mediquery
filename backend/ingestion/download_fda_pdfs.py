import re
from pathlib import Path
from typing import List, Dict

import requests

from backend.config import DATA_DIR

# DailyMed API endpoints
SPLS_JSON_URL = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json"
PDF_DOWNLOAD_URL = "https://dailymed.nlm.nih.gov/dailymed/downloadpdffile.cfm"


DRUG_CATEGORIES: Dict[str, List[str]] = {
    "ADHD": [
        "Adderall",
        "Ritalin",
        "Vyvanse",
        "Concerta",
        "Strattera",
    ],
    "Antidepressants": [
        "Prozac",
        "Zoloft",
        "Lexapro",
        "Wellbutrin",
        "Cymbalta",
    ],
    "Dermatology": [
        "Accutane",
        "Tretinoin",
        "Benzoyl peroxide",
    ],
    "Cardiovascular": [
        "Lipitor",
        "Atorvastatin",
        "Lisinopril",
        "Metoprolol",
    ],
    "Diabetes": [
        "Metformin",
        "Insulin glargine",
        "Ozempic",
    ],
    "Pain": [
        "Ibuprofen",
        "Naproxen",
        "Acetaminophen",
    ],
}


# ---- 2. Helpers ----

def safe_filename(title: str) -> str:
    """
    Convert a drug title into a safe filename.
    e.g. 'ZOCOR (SIMVASTATIN) TABLET [MERCK]' -> 'zocor_simvastatin_tablet_merck'
    """
    title = title.lower()
    title = re.sub(r"[\[\]\(\)]", " ", title)         # remove brackets/parens
    title = re.sub(r"[^a-z0-9]+", "_", title)         # non-alphanum -> _
    title = re.sub(r"_+", "_", title).strip("_")      # collapse multiple _
    return title or "drug_label"


def fetch_spls_for_drug(drug_name: str, pagesize: int = 3) -> List[dict]:
    """
    Query DailyMed /spls.json for a specific drug name.
    We use the drug_name filter with name_type=both to match brand/generic.
    """
    params = {
        "drug_name": drug_name,
        "name_type": "both",  # match both brand + generic names
        "pagesize": pagesize,
        "page": 1,
    }
    print(f"  -> Querying SPLs for drug_name='{drug_name}'")
    resp = requests.get(SPLS_JSON_URL, params=params, timeout=30)
    resp.raise_for_status()
    json_data = resp.json()
    return json_data.get("data", [])


def download_pdf(setid: str, title: str, out_dir: Path) -> Path | None:
    """
    Given a setid + title, download the PDF from DailyMed and save it
    into out_dir. Returns the file path, or None on failure.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    base_name = safe_filename(title) if title else setid
    out_path = out_dir / f"{base_name}_{setid}.pdf"

    if out_path.exists():
        print(f"      [skip] Already exists: {out_path.name}")
        return out_path

    params = {"setId": setid}
    print(f"      [download] setid={setid}")
    try:
        with requests.get(PDF_DOWNLOAD_URL, params=params, timeout=60, stream=True) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        print(f"      [saved] {out_path}")
        return out_path
    except Exception as e:
        print(f"      [error] Failed to download PDF for setid={setid}: {e}")
        return None


# ---- 3. Main function ----

def download_selected_drug_pdfs(max_per_drug: int = 2) -> None:
    """
    For each drug in DRUG_CATEGORIES:
    - query DailyMed /spls.json filtered by drug_name
    - take up to `max_per_drug` SPLs
    - download their PDFs into DATA_DIR (data/fda_pdfs)

    This should give you ~20–30 PDFs total across all categories.
    """
    out_dir = Path(DATA_DIR)
    print(f"Output directory for PDFs: {out_dir}")
    total_downloaded = 0

    for category, drug_list in DRUG_CATEGORIES.items():
        print(f"\n=== Category: {category} ===")
        for drug_name in drug_list:
            try:
                spls = fetch_spls_for_drug(drug_name)
            except Exception as e:
                print(f"  [error] Failed to fetch SPLs for '{drug_name}': {e}")
                continue

            if not spls:
                print(f"  [info] No SPLs found for '{drug_name}'")
                continue

            # Take up to max_per_drug SPLs
            for spl in spls[:max_per_drug]:
                setid = spl.get("setid")
                title = spl.get("title", drug_name)
                if not setid:
                    continue
                result = download_pdf(setid=setid, title=title, out_dir=out_dir)
                if result is not None:
                    total_downloaded += 1

    print(f"\nDone. Total PDFs downloaded: {total_downloaded}")


if __name__ == "__main__":
    download_selected_drug_pdfs(max_per_drug=2)
