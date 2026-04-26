import os
import json
import random
import pandas as pd
from tqdm import tqdm

# ── Config ───────────────────────────────────────────────────────────────────
QUERY_FILE      = r"C:\Users\ASUS\Downloads\aila_data\Query_doc.txt"
RELEVANCE_FILE  = r"C:\Users\ASUS\Downloads\aila_data\relevance_judgments_priorcases.txt"
CASES_DIR       = r"C:\Users\ASUS\Downloads\aila_data\Object_casedocs"
OUTPUT_PATH     = "data/training/training_pairs.json"
MAX_TEXT_LEN    = 300  # words per chunk for training

os.makedirs("data/training", exist_ok=True)

# ── Step 1: Load queries ──────────────────────────────────────────────────────
print("Loading queries...")
queries = {}
with open(QUERY_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if "||" in line:
            parts = line.split("||", 1)
            qid   = parts[0].strip()
            text  = parts[1].strip()
            # Use first 300 words as query representation
            words = text.split()[:MAX_TEXT_LEN]
            queries[qid] = " ".join(words)

print(f"Loaded {len(queries)} queries")

# ── Step 2: Load relevance judgments ─────────────────────────────────────────
print("Loading relevance judgments...")
relevant   = {}  # qid -> list of relevant case ids
irrelevant = {}  # qid -> list of irrelevant case ids

with open(RELEVANCE_FILE, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        qid     = parts[0]
        case_id = parts[2]
        score   = int(parts[3])

        if score == 1:
            relevant.setdefault(qid, []).append(case_id)
        else:
            irrelevant.setdefault(qid, []).append(case_id)

print(f"Queries with relevant cases: {len(relevant)}")
total_relevant = sum(len(v) for v in relevant.values())
print(f"Total relevant pairs: {total_relevant}")

# ── Step 3: Load case documents ───────────────────────────────────────────────
print("\nLoading case documents...")
case_texts = {}
case_files = os.listdir(CASES_DIR)

for fname in tqdm(case_files):
    if not fname.endswith(".txt"):
        continue
    case_id = fname.replace(".txt", "")
    fpath   = os.path.join(CASES_DIR, fname)
    try:
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            text  = f.read().strip()
            words = text.split()[:MAX_TEXT_LEN]
            case_texts[case_id] = " ".join(words)
    except:
        continue

print(f"Loaded {len(case_texts)} case documents")

# ── Step 4: Build training pairs ──────────────────────────────────────────────
print("\nBuilding training pairs...")

# Format for MultipleNegativesRankingLoss:
# Each sample = {"query": ..., "positive": ..., "negative": ...}
# OR just {"query": ..., "positive": ...} — negatives sampled automatically

positive_pairs = []  # (query_text, positive_case_text)
triplets       = []  # (query_text, positive_case_text, negative_case_text)

skipped = 0
for qid, pos_cases in relevant.items():
    if qid not in queries:
        skipped += 1
        continue

    query_text = queries[qid]
    neg_cases  = irrelevant.get(qid, [])

    for pos_case_id in pos_cases:
        if pos_case_id not in case_texts:
            continue

        pos_text = case_texts[pos_case_id]

        # Add positive pair
        positive_pairs.append({
            "query"   : query_text,
            "positive": pos_text
        })

        # Add triplet with random negative
        if neg_cases:
            neg_case_id = random.choice(neg_cases)
            if neg_case_id in case_texts:
                triplets.append({
                    "query"   : query_text,
                    "positive": pos_text,
                    "negative": case_texts[neg_case_id]
                })

print(f"\nSkipped queries (not in query file): {skipped}")
print(f"Positive pairs created: {len(positive_pairs)}")
print(f"Triplets created      : {len(triplets)}")

# ── Step 5: Save ──────────────────────────────────────────────────────────────
output = {
    "positive_pairs": positive_pairs,
    "triplets"      : triplets,
    "stats": {
        "total_queries"       : len(queries),
        "queries_with_positives": len(relevant),
        "total_positives"     : len(positive_pairs),
        "total_triplets"      : len(triplets)
    }
}

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

print(f"\n✅ Saved training data to: {OUTPUT_PATH}")
print(f"\n══════════════════════════════")
print(f"       TRAINING DATA SUMMARY  ")
print(f"══════════════════════════════")
print(f"  Queries          : {len(queries)}")
print(f"  Positive pairs   : {len(positive_pairs)}")
print(f"  Triplets         : {len(triplets)}")
print(f"══════════════════════════════")

# ── Step 6: Show a sample ─────────────────────────────────────────────────────
print("\nSample triplet:")
if triplets:
    s = triplets[0]
    print(f"  QUERY    : {s['query'][:150]}...")
    print(f"  POSITIVE : {s['positive'][:150]}...")
    print(f"  NEGATIVE : {s['negative'][:150]}...")