# backend/ingestion/data_ingest.py
import json
import re
from pathlib import Path
from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from config import DATA_DIR, CHROMA_DIR, CHUNKS_PATH, EMBEDDING_MODEL

# ============================================================================
# SMART CHUNKING (Section-Aware)
# ============================================================================

# Common FDA label section headers
SECTION_PATTERNS = [
    r'\b(INDICATIONS AND USAGE|INDICATION)\b',
    r'\b(DOSAGE AND ADMINISTRATION|DOSING)\b',
    r'\b(CONTRAINDICATIONS?)\b',
    r'\b(WARNINGS? AND PRECAUTIONS?|WARNINGS?|PRECAUTIONS?)\b',
    r'\b(ADVERSE REACTIONS?|SIDE EFFECTS?)\b',
    r'\b(DRUG INTERACTIONS?)\b',
    r'\b(USE IN SPECIFIC POPULATIONS?)\b',
    r'\b(PREGNANCY|LACTATION|NURSING MOTHERS?)\b',
    r'\b(PEDIATRIC USE)\b',
    r'\b(GERIATRIC USE)\b',
    r'\b(CLINICAL PHARMACOLOGY)\b',
    r'\b(OVERDOSAGE)\b',
    r'\b(DESCRIPTION)\b',
    r'\b(HOW SUPPLIED)\b',
    r'\b(BOXED WARNING|BLACK BOX WARNING)\b',
]


def detect_sections(text: str) -> List[Dict]:
    """Find section boundaries in FDA label text."""
    combined_pattern = '|'.join(SECTION_PATTERNS)
    
    matches = []
    for match in re.finditer(combined_pattern, text, re.IGNORECASE):
        matches.append({
            'start': match.start(),
            'end': match.end(),
            'name': match.group().strip()
        })
    
    if not matches:
        return [{
            'start': 0,
            'end': len(text),
            'section_name': 'GENERAL',
            'content': text
        }]
    
    sections = []
    for i, match in enumerate(matches):
        section_start = match['end']
        section_end = matches[i + 1]['start'] if i + 1 < len(matches) else len(text)
        content = text[section_start:section_end].strip()
        
        if content and len(content) > 50:
            sections.append({
                'start': match['start'],
                'end': section_end,
                'section_name': match['name'],
                'content': content
            })
    
    return sections


def categorize_section(section_name: str) -> str:
    """Map section names to query-relevant categories."""
    name_lower = section_name.lower()
    
    if any(x in name_lower for x in ['adverse', 'side effect']):
        return 'side_effects'
    elif 'contraindication' in name_lower:
        return 'contraindications'
    elif any(x in name_lower for x in ['warning', 'precaution', 'boxed', 'black box']):
        return 'warnings'
    elif 'interaction' in name_lower:
        return 'interactions'
    elif any(x in name_lower for x in ['dosage', 'administration', 'dosing']):
        return 'dosage'
    elif any(x in name_lower for x in ['pregnancy', 'lactation', 'nursing']):
        return 'pregnancy'
    elif any(x in name_lower for x in ['indication', 'usage']):
        return 'indications'
    else:
        return 'general'


def chunk_section(content: str, max_size: int = 1200) -> List[str]:
    """Split large sections while preserving paragraphs."""
    if len(content) <= max_size:
        return [content]
    
    paragraphs = content.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        if len(para) > max_size:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sent in sentences:
                if len(current_chunk) + len(sent) <= max_size:
                    current_chunk += sent + " "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sent + " "
        else:
            if len(current_chunk) + len(para) <= max_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks if chunks else [content[:max_size]]


def create_smart_chunks(documents: List[Document], max_chunk_size: int = 1200) -> List[Document]:
    """Convert documents to section-aware chunks."""
    smart_chunks = []
    
    for doc in documents:
        sections = detect_sections(doc.page_content)
        
        for section_data in sections:
            section_name = section_data['section_name']
            section_type = categorize_section(section_name)
            content = section_data['content']
            
            sub_chunks = chunk_section(content, max_chunk_size)
            
            for i, chunk_text in enumerate(sub_chunks):
                metadata = doc.metadata.copy()
                metadata['section'] = section_name
                metadata['section_type'] = section_type
                metadata['section_part'] = i + 1
                metadata['total_parts'] = len(sub_chunks)
                
                smart_chunks.append(
                    Document(
                        page_content=chunk_text,
                        metadata=metadata
                    )
                )
    
    return smart_chunks


# ============================================================================
# INGESTION FUNCTIONS
# ============================================================================

def load_fda_pdfs():
    """Load all PDF files from data directory."""
    pdf_paths = sorted(Path(DATA_DIR).glob("*.pdf"))
    docs = []
    for pdf in pdf_paths:
        loader = PyPDFLoader(str(pdf))
        docs.extend(loader.load())
    return docs


def build_chunks(docs, use_smart_chunking: bool = True):
    """Build chunks using smart or basic chunking."""
    if use_smart_chunking:
        print("✅ Using SMART section-aware chunking...")
        return create_smart_chunks(docs, max_chunk_size=1200)
    else:
        print("⚠️  Using basic chunking...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        return splitter.split_documents(docs)


def save_chunks_jsonl(chunks):
    """Save chunks to JSONL file for BM25."""
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            record = {
                "id": i,
                "text": chunk.page_content,
                "metadata": chunk.metadata,
            }
            f.write(json.dumps(record) + "\n")
    print(f"✅ Saved {len(chunks)} chunks to {CHUNKS_PATH}")


def build_chroma_store(chunks):
    """Build ChromaDB vector store with embeddings."""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}
    )
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    print(f"✅ Chroma DB persisted at {CHROMA_DIR}")
    return vs


def run_ingestion(use_smart_chunking: bool = True):
    """
    Main ingestion pipeline.
    
    Args:
        use_smart_chunking: If True, uses section-aware chunking (recommended)
    """
    print("="*70)
    print("  📚 MediQuery Data Ingestion")
    print("="*70)
    
    print(f"\n1️⃣  Loading PDFs from {DATA_DIR} ...")
    docs = load_fda_pdfs()
    print(f"   ✅ Loaded {len(docs)} document pages from {len(set(d.metadata.get('source') for d in docs))} PDFs")
    
    print(f"\n2️⃣  Chunking PDFs...")
    chunks = build_chunks(docs, use_smart_chunking=use_smart_chunking)
    print(f"   ✅ Created {len(chunks)} chunks")
    
    # Show section distribution if smart chunking
    if use_smart_chunking:
        from collections import Counter
        section_types = Counter(c.metadata.get('section_type', 'unknown') for c in chunks)
        print(f"\n   📊 Section distribution:")
        for stype, count in section_types.most_common():
            print(f"      • {stype}: {count} chunks")
    
    print(f"\n3️⃣  Saving chunks to JSONL (for BM25)...")
    save_chunks_jsonl(chunks)
    
    print(f"\n4️⃣  Building Chroma vector store...")
    build_chroma_store(chunks)
    
    print("\n" + "="*70)
    print("  ✅ Ingestion complete!")
    print("="*70)


if __name__ == "__main__":
    # Run with smart chunking (recommended)
    run_ingestion(use_smart_chunking=True)
    
    # To use basic chunking instead:
    # run_ingestion(use_smart_chunking=False)