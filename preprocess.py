

import os, re, csv, json
from pathlib import Path





for folder in ["data/processed", "data/chunks", "data/stats"]:
    os.makedirs(folder, exist_ok=True)

RAW = Path("data/raw")



def read_txt(filepath):
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            return Path(filepath).read_text(encoding=enc)
        except:
            continue
    return ""

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip()

def chunk_text(text, doc_id, doc_type, chunk_size=512, overlap=50):
    words = text.split()
    chunks, start, idx = [], 0, 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append({
            "chunk_id"  : f"{doc_id}_c{idx}",
            "doc_id"    : doc_id,
            "doc_type"  : doc_type,
            "chunk_idx" : idx,
            "text"      : " ".join(words[start:end]),
            "word_count": end - start
        })
        idx += 1
        start += chunk_size - overlap
    return chunks

def save_csv(filepath, rows):
    if not rows:
        print(f"      WARNING: No data to save for {filepath}")
        return
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

def load_relevance(filepath):
  
    entries = []
    raw = read_txt(filepath)
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 2:
            query_id = parts[0]
            for doc_id in parts[1:]:
                doc_id = doc_id.strip()
                if doc_id:
                    entries.append({
                        "query_id": query_id,
                        "doc_id"  : doc_id,
                        "relevant": 1
                    })
    return entries


print("\n[1/4] Case documents load ho rahe hain...")

cases = []
case_folder = RAW / "Object_casedocs"

for filepath in sorted(case_folder.glob("*.txt")):
    raw = read_txt(filepath)
    if not raw.strip():
        continue
    cleaned = clean_text(raw)
    cases.append({
        "case_id"   : filepath.stem,
        "filename"  : filepath.name,
        "text"      : cleaned,
        "word_count": len(cleaned.split()),
        "char_count": len(cleaned)
    })

print(f"      {len(cases)} cases loaded")
save_csv("data/processed/cases.csv", cases)
print("      Saved: data/processed/cases.csv")


print("\n[2/4] Queries load ho rahi hain...")

queries = []
query_file = RAW / "Query_doc.txt"
raw_q = read_txt(query_file)

for line in raw_q.splitlines():
    line = line.strip()
    if not line or '||' not in line:
        continue
    parts = line.split('||', 1)
    if len(parts) != 2:
        continue
    query_id = parts[0].strip()
    text     = parts[1].strip()
    if len(text) > 30:
        cleaned = clean_text(text)
        queries.append({
            "query_id"  : query_id,
            "text"      : cleaned,
            "word_count": len(cleaned.split())
        })

print(f"      {len(queries)} queries loaded")
save_csv("data/processed/queries.csv", queries)
print("      Saved: data/processed/queries.csv")


print("\n[3/4] Statutes load ho rahe hain...")

statutes = []
statute_folder = RAW / "Object_statutes"

for filepath in sorted(statute_folder.glob("*.txt")):
    raw = read_txt(filepath)
    if not raw.strip():
        continue
    cleaned = clean_text(raw)
    statutes.append({
        "statute_id" : filepath.stem,
        "filename"   : filepath.name,
        "text"       : cleaned,
        "word_count" : len(cleaned.split()),
        "char_count" : len(cleaned)
    })

print(f"      {len(statutes)} statutes loaded")
save_csv("data/processed/statutes.csv", statutes)
print("      Saved: data/processed/statutes.csv")


print("\n[4a/4] Relevance judgments load ho rahe hain...")

# Prior cases relevance
prior_relevance = load_relevance(RAW / "relevance_judgments_priorcases.txt")
# Rename doc_id to case_id
prior_relevance = [{"query_id": e["query_id"],
                    "case_id" : e["doc_id"],
                    "relevant": e["relevant"]} for e in prior_relevance]

print(f"      {len(prior_relevance)} prior case relevance entries")
save_csv("data/processed/relevance_cases.csv", prior_relevance)
print("      Saved: data/processed/relevance_cases.csv")

# Statute relevance
statute_relevance = load_relevance(RAW / "relevance_judgments_statutes.txt")
# Rename doc_id to statute_id
statute_relevance = [{"query_id"  : e["query_id"],
                      "statute_id": e["doc_id"],
                      "relevant"  : e["relevant"]} for e in statute_relevance]

print(f"      {len(statute_relevance)} statute relevance entries")
save_csv("data/processed/relevance_statutes.csv", statute_relevance)
print("      Saved: data/processed/relevance_statutes.csv")


print("\n[4b/4] Chunks ban rahe hain...")

all_chunks = []

for case in cases:
    all_chunks.extend(
        chunk_text(case["text"], case["case_id"], "case")
    )
for statute in statutes:
    all_chunks.extend(
        chunk_text(statute["text"], statute["statute_id"], "statute", chunk_size=256)
    )
for query in queries:
    all_chunks.extend(
        chunk_text(query["text"], query["query_id"], "query", chunk_size=256)
    )

save_csv("data/chunks/chunks.csv", all_chunks)
print(f"      {len(all_chunks)} total chunks ")
print("      Saved: data/chunks/chunks.csv")


stats = {
    "data_source"             : "AILA FIRE 2019",
    "total_cases"             : len(cases),
    "total_queries"           : len(queries),
    "total_statutes"          : len(statutes),
    "total_chunks"            : len(all_chunks),
    "prior_relevance_pairs"   : len(prior_relevance),
    "statute_relevance_pairs" : len(statute_relevance),
    "avg_case_words"          : round(sum(c["word_count"] for c in cases) / max(len(cases), 1), 1),
    "avg_query_words"         : round(sum(q["word_count"] for q in queries) / max(len(queries), 1), 1),
    "chunk_size"              : 512,
    "chunk_overlap"           : 50,
}

with open("data/stats/summary.json", "w") as f:
    json.dump(stats, f, indent=2)


print("  PREPROCESSING COMPLETE! ")

print(json.dumps(stats, indent=2))
