"""
preprocess.py
=============
Indian Legal Documents ka full preprocessing pipeline.

Kya karta hai:
  1. Raw judgment text clean karta hai
  2. Metadata extract karta hai (court, year, IPC sections)
  3. Legal entities dhundhta hai (NER)
  4. Documents ko chunks mein todta hai
  5. Model-ready CSV files save karta hai

Run: python preprocess.py
"""

import re
import os
import json
import pandas as pd
from tqdm import tqdm

print("=" * 55)
print("  INDIAN LEGAL AI — Preprocessing Pipeline")
print("=" * 55)

# ── Output folders banao ───────────────────────────────────
os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/chunks",    exist_ok=True)
os.makedirs("data/stats",     exist_ok=True)

# ══════════════════════════════════════════════════════════
# STEP 1 — DATA LOAD
# ══════════════════════════════════════════════════════════

def load_data():
    """
    data/raw/train.json se data load karta hai.
    Agar file nahi mili toh synthetic data use karta hai
    taaki preprocessing demo ho sake.
    """
    if os.path.exists("data/raw/train.json"):
        print("\n[STEP 1] Loading downloaded ILDC data...")
        with open("data/raw/train.json", "r", encoding="utf-8") as f:
            records = json.load(f)
        print(f"         {len(records)} cases loaded ✅")
        return records
    else:
        print("\n[STEP 1] data/raw/train.json nahi mila.")
        print("         Synthetic demo data use ho raha hai...")
        print("         (Pehle download_data.py run karo real data ke liye)\n")
        return _make_synthetic_data(500)

def _make_synthetic_data(n=500):
    """Demo ke liye synthetic Indian legal cases banata hai."""
    import random
    random.seed(42)

    ipc_sections = [
        "302 IPC", "304B IPC", "376 IPC", "420 IPC",
        "498A IPC", "307 IPC", "392 IPC", "395 IPC",
        "506 IPC", "354 IPC"
    ]
    courts = [
        "Supreme Court of India",
        "Delhi High Court",
        "Bombay High Court",
        "Madras High Court",
        "Allahabad High Court",
        "Punjab and Haryana High Court",
        "Karnataka High Court",
        "Calcutta High Court",
    ]
    crime_types = [
        "Murder", "Dowry Death", "Rape", "Fraud",
        "Domestic Violence", "Attempt to Murder",
        "Robbery", "Dacoity", "Criminal Intimidation",
        "Outraging Modesty"
    ]
    judges = [
        "Justice D.Y. Chandrachud",
        "Justice N.V. Ramana",
        "Justice S.K. Kaul",
        "Justice Indira Banerjee",
        "Justice A.M. Khanwilkar",
    ]

    records = []
    for i in range(n):
        sec    = random.choice(ipc_sections)
        court  = random.choice(courts)
        crime  = random.choice(crime_types)
        judge  = random.choice(judges)
        year   = random.randint(2010, 2023)
        label  = random.choice([0, 1])
        w_count = random.randint(3, 8)
        outcome_word = "allow" if label == 1 else "dismiss"

        text = (
            f"IN THE {court.upper()}\n"
            f"Criminal Appeal No. {1000+i} of {year}\n\n"
            f"BEFORE: {judge}\n\n"
            f"FACTS:\n"
            f"The appellant was convicted under Section {sec} for the offence of {crime}. "
            f"The learned Sessions Judge convicted the accused and sentenced him to "
            f"rigorous imprisonment. The matter was taken up before the {court}.\n\n"
            f"SUBMISSIONS:\n"
            f"The learned counsel for the appellant submitted that the trial court had "
            f"erred in appreciating the evidence on record. The prosecution examined "
            f"{w_count} witnesses to prove its case beyond reasonable doubt.\n\n"
            f"ANALYSIS:\n"
            f"After careful consideration of the facts and circumstances of the case "
            f"and the law laid down by this Court in various judgments, we find that "
            f"the impugned order {'requires interference' if label==1 else 'does not require interference'}.\n\n"
            f"ORDER:\n"
            f"For the reasons stated above, we {outcome_word} this appeal. "
            f"({year}) {random.randint(1,15)} SCC {random.randint(1,500)}"
        )
        records.append({"text": text, "label": label})

    print(f"         {n} synthetic cases created ✅")
    return records


# ══════════════════════════════════════════════════════════
# STEP 2 — TEXT CLEANING
# ══════════════════════════════════════════════════════════

# Patterns jo Indian legal text mein common hain
_PAGE_NUM   = re.compile(r"Page\s+\d+\s+of\s+\d+", re.I)
_WHITESPACE = re.compile(r"\s{2,}")
_NON_ASCII  = re.compile(r"[^\x00-\x7F]+")
_IPC_REF    = re.compile(
    r"[Ss]ections?\s+([\d,\s/]+[A-Z]?)\s+"
    r"(?:of\s+the\s+)?(?:Indian Penal Code|IPC|CrPC|CPC)"
)
_CITATION   = re.compile(
    r"\((\d{4})\)\s+(\d+)\s+(SCC|SCR|AIR|CrLJ|MLJ)\s+(\d+)"
)

# Common abbreviations jo Indian judgments mein milti hain
ABBREVS = {
    r"\bHC\b"   : "High Court",
    r"\bSC\b"   : "Supreme Court",
    r"\bFIR\b"  : "First Information Report",
    r"\bSHO\b"  : "Station House Officer",
    r"\bCBI\b"  : "Central Bureau of Investigation",
    r"\bNDPS\b" : "Narcotic Drugs and Psychotropic Substances Act",
    r"\bMVA\b"  : "Motor Vehicles Act",
    r"\bPOCO\b" : "Protection of Children from Sexual Offences Act",
}

def clean_text(raw_text):
    """
    Ek judgment ka text clean karta hai.

    Steps:
      - Line breaks normalize karo
      - Page numbers hatao
      - Non-ASCII characters hatao (OCR artifacts)
      - Legal abbreviations expand karo
      - Extra whitespace hatao
    """
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    text = _PAGE_NUM.sub("", text)
    text = _NON_ASCII.sub(" ", text)

    # Abbreviations expand karo
    for pattern, replacement in ABBREVS.items():
        text = re.sub(pattern, replacement, text)

    # Extra spaces hatao
    text = _WHITESPACE.sub(" ", text).strip()

    return text


# ══════════════════════════════════════════════════════════
# STEP 3 — METADATA EXTRACTION
# ══════════════════════════════════════════════════════════

COURT_NAMES = {
    "supreme court"       : "Supreme Court of India",
    "delhi high"          : "Delhi High Court",
    "bombay high"         : "Bombay High Court",
    "madras high"         : "Madras High Court",
    "allahabad high"      : "Allahabad High Court",
    "calcutta high"       : "Calcutta High Court",
    "karnataka high"      : "Karnataka High Court",
    "kerala high"         : "Kerala High Court",
    "gujarat high"        : "Gujarat High Court",
    "punjab and haryana"  : "Punjab and Haryana High Court",
    "rajasthan high"      : "Rajasthan High Court",
    "gauhati high"        : "Gauhati High Court",
}

def extract_metadata(text, label):
    """
    Judgment text se structured information nikalata hai.
    Returns: dict with doc_id, year, court, ipc_sections,
             citations, outcome, word_count
    """
    meta = {
        "label"        : label,
        "outcome"      : "Accepted" if label == 1 else "Dismissed",
        "year"         : None,
        "court"        : "Unknown",
        "ipc_sections" : [],
        "citations"    : [],
        "word_count"   : len(text.split()),
    }

    # Year nikalo
    year_match = re.search(r"\b(20\d{2}|19\d{2})\b", text)
    if year_match:
        meta["year"] = int(year_match.group())

    # Court nikalo
    text_lower = text[:500].lower()
    for key, val in COURT_NAMES.items():
        if key in text_lower:
            meta["court"] = val
            break

    # IPC Sections nikalo
    ipc_hits = _IPC_REF.findall(text)
    sections = []
    for hit in ipc_hits:
        for s in re.split(r"[,/\s]+", hit):
            s = s.strip()
            if re.match(r"^\d+[A-Z]?$", s):
                sections.append(s)
    meta["ipc_sections"] = list(set(sections))

    # Legal citations nikalo e.g. (2019) 3 SCC 45
    cites = _CITATION.findall(text)
    meta["citations"] = [f"({y}) {v} {r} {p}" for y, v, r, p in cites]

    return meta


# ══════════════════════════════════════════════════════════
# STEP 4 — NAMED ENTITY RECOGNITION (Rule-based)
# ══════════════════════════════════════════════════════════

NER_PATTERNS = {
    "JUDGE"       : re.compile(
        r"(?:Justice|HON'BLE\s+(?:MR\.|MS\.)\s+JUSTICE)\s+"
        r"[A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+"
    ),
    "CASE_NUMBER" : re.compile(
        r"(?:Criminal Appeal|Civil Appeal|Writ Petition|SLP)\s+"
        r"No\.?\s*\d+\s+(?:of|/)\s*\d{4}"
    ),
    "IPC_SECTION" : re.compile(
        r"[Ss]ection\s+\d+[A-Z]?\s+(?:IPC|Indian Penal Code)"
    ),
    "ACT"         : re.compile(
        r"(?:the\s+)?[A-Z][A-Za-z\s]+Act,?\s+\d{4}"
    ),
    "DATE"        : re.compile(
        r"\b\d{1,2}(?:st|nd|rd|th)?\s+"
        r"(?:January|February|March|April|May|June|"
        r"July|August|September|October|November|December)"
        r"\s*,?\s*\d{4}\b"
    ),
}

def extract_entities(text):
    """
    Rule-based NER — Indian legal documents ke liye
    specific entities dhundhta hai.
    """
    entities = {}
    for ent_type, pattern in NER_PATTERNS.items():
        found = pattern.findall(text)
        entities[ent_type] = list({e.strip() for e in found})
    return entities


# ══════════════════════════════════════════════════════════
# STEP 5 — CHUNKING
# ══════════════════════════════════════════════════════════

# Section headers jo Indian judgments mein commonly milte hain
SECTION_HEADERS = re.compile(
    r"\b(FACTS|BACKGROUND|SUBMISSIONS?|ARGUMENTS?|"
    r"ANALYSIS|FINDINGS?|REASONING|JUDGMENT|ORDER|"
    r"HELD|CONCLUSION|DECISION)\b",
    re.IGNORECASE
)

def chunk_document(doc_id, text, label, max_words=400, overlap=50):
    """
    Ek document ko model-ready chunks mein todhta hai.

    Strategy:
      1. Pehle natural section headings pe split karo
         (FACTS, SUBMISSIONS, ANALYSIS, ORDER, etc.)
      2. Agar headings nahi hain toh sliding window use karo
         (400 words, 50 word overlap)

    Args:
        doc_id   : unique document ID
        text     : cleaned text
        label    : 0 (dismissed) / 1 (accepted)
        max_words: ek chunk mein maximum words
        overlap  : consecutive chunks mein shared words

    Returns:
        List of chunk dicts
    """
    chunks = []

    # Strategy 1: Section-based split
    parts   = SECTION_HEADERS.split(text)
    headers = SECTION_HEADERS.findall(text)

    if len(headers) >= 2:
        sections = []
        section_names = ["PREAMBLE"] + headers
        for header, body in zip(section_names, parts):
            section_text = f"{header}: {body.strip()}"
            words = section_text.split()

            if len(words) <= max_words:
                sections.append({
                    "doc_id"           : doc_id,
                    "chunk_id"         : len(chunks) + len(sections),
                    "text"             : section_text,
                    "word_count"       : len(words),
                    "section"          : header,
                    "is_section_start" : True,
                    "label"            : label,
                })
            else:
                # Badi section ko aur tod do
                sub_chunks = _sliding_window(words, max_words, overlap)
                for sc in sub_chunks:
                    sections.append({
                        "doc_id"           : doc_id,
                        "chunk_id"         : len(chunks) + len(sections),
                        "text"             : sc,
                        "word_count"       : len(sc.split()),
                        "section"          : header,
                        "is_section_start" : False,
                        "label"            : label,
                    })
        return sections

    # Strategy 2: Sliding window (jab sections nahi milte)
    words = text.split()
    for i, chunk_text in enumerate(_sliding_window(words, max_words, overlap)):
        chunks.append({
            "doc_id"           : doc_id,
            "chunk_id"         : i,
            "text"             : chunk_text,
            "word_count"       : len(chunk_text.split()),
            "section"          : "FULL_TEXT",
            "is_section_start" : (i == 0),
            "label"            : label,
        })
    return chunks

def _sliding_window(words, window, overlap):
    """Words ki list se overlapping text windows banata hai."""
    results = []
    step = max(1, window - overlap)
    for start in range(0, len(words), step):
        chunk = words[start: start + window]
        if chunk:
            results.append(" ".join(chunk))
        if start + window >= len(words):
            break
    return results


# ══════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════

def run():
    # ── Load ──────────────────────────────────────────────
    records = load_data()

    # ── Process each document ─────────────────────────────
    print("\n[STEP 2] Cleaning + Metadata + NER + Chunking...")

    all_meta   = []
    all_chunks = []
    all_ner    = []

    for idx, record in enumerate(tqdm(records, desc="         Processing")):
        raw_text = record.get("text", "")
        label    = record.get("label", 0)
        doc_id   = f"doc_{idx:05d}"

        if not raw_text or len(raw_text.split()) < 30:
            continue

        # Clean
        cleaned = clean_text(raw_text)

        # Metadata
        meta          = extract_metadata(cleaned, label)
        meta["doc_id"] = doc_id
        meta["original_words"] = len(raw_text.split())
        meta["cleaned_words"]  = len(cleaned.split())
        meta["ipc_sections"]   = ", ".join(meta["ipc_sections"])
        meta["citations"]      = ", ".join(meta["citations"])
        all_meta.append(meta)

        # NER
        ents          = extract_entities(cleaned)
        ents["doc_id"] = doc_id
        for k in ents:
            if isinstance(ents[k], list):
                ents[k] = " | ".join(ents[k])
        all_ner.append(ents)

        # Chunks
        chunks = chunk_document(doc_id, cleaned, label)
        all_chunks.extend(chunks)

    # ── Save outputs ───────────────────────────────────────
    print("\n[STEP 3] Saving processed files...")

    # 1. Metadata CSV — model training ke liye main file
    meta_df = pd.DataFrame(all_meta)
    meta_df.to_csv("data/processed/metadata.csv", index=False)
    print(f"         ✅ data/processed/metadata.csv  ({len(meta_df)} rows)")

    # 2. Chunks CSV — BERT input ke liye
    chunks_df = pd.DataFrame(all_chunks)
    chunks_df.to_csv("data/chunks/chunks.csv", index=False)
    print(f"         ✅ data/chunks/chunks.csv       ({len(chunks_df)} rows)")

    # 3. NER CSV
    ner_df = pd.DataFrame(all_ner).fillna("")
    ner_df.to_csv("data/processed/ner_entities.csv", index=False)
    print(f"         ✅ data/processed/ner_entities.csv")

    # ── Stats ─────────────────────────────────────────────
    print("\n[STEP 4] Dataset Statistics:")
    print(f"         Total documents   : {len(meta_df)}")
    print(f"         Total chunks      : {len(chunks_df)}")
    print(f"         Avg words/doc     : {meta_df['cleaned_words'].mean():.0f}")
    print(f"         Avg chunks/doc    : {len(chunks_df)/max(len(meta_df),1):.1f}")
    print(f"\n         Label Distribution:")
    label_counts = meta_df['label'].value_counts()
    for lbl, cnt in label_counts.items():
        name = "Accepted" if lbl == 1 else "Dismissed"
        pct  = cnt / len(meta_df) * 100
        print(f"           {name:10} : {cnt} ({pct:.1f}%)")

    if "court" in meta_df.columns:
        print(f"\n         Top Courts:")
        for court, cnt in meta_df['court'].value_counts().head(5).items():
            print(f"           {court[:35]:35} : {cnt}")

    # Save stats
    stats = {
        "total_documents"  : len(meta_df),
        "total_chunks"     : len(chunks_df),
        "avg_words_per_doc": round(meta_df['cleaned_words'].mean(), 1),
        "label_0_dismissed": int((meta_df['label'] == 0).sum()),
        "label_1_accepted" : int((meta_df['label'] == 1).sum()),
    }
    with open("data/stats/summary.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*55}")
    print(f"  ✅ PREPROCESSING COMPLETE!")
    print(f"  Ab run karo: python train.py")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    run()