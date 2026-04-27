import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

# ── Config ───────────────────────────────────────────────────────────────────
INDEX_DIR  = "faiss_indexes"
MODEL_NAME = "models/finetuned_legal_bert"

# ── Load models ───────────────────────────────────────────────────────────────
print("Loading models...")
bi_encoder    = SentenceTransformer(MODEL_NAME, device="cuda")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", device="cuda")
print("Models loaded ✅")

# ── Load indexes ──────────────────────────────────────────────────────────────
ROLES = ["RPC", "RATIO", "FAC", "STA", "ISSUE",
         "PRE_RELIED", "RLC", "PREAMBLE", "ARG_PETITIONER", "ALL"]

indexes = {}
meta    = {}

print("Loading indexes...")
for role in ROLES:
    idx_path  = os.path.join(INDEX_DIR, f"index_{role}.bin")
    meta_path = os.path.join(INDEX_DIR, f"meta_{role}.csv")

    if os.path.exists(idx_path):
        indexes[role] = faiss.read_index(idx_path)
        meta[role]    = pd.read_csv(meta_path)

print(f"Loaded {len(indexes)} indexes ✅\n")


# ── Intent Detection ──────────────────────────────────────────────────────────
def detect_intent(query):
    q = query.lower()

    if any(w in q for w in ["bail", "custody", "arrest"]):
        return ["RPC", "RATIO"]

    if any(w in q for w in ["murder", "homicide", "death"]):
        return ["RPC", "RATIO", "FAC"]

    if any(w in q for w in ["rape", "sexual"]):
        return ["RPC", "RATIO", "FAC"]

    if any(w in q for w in ["fraud", "cheating", "420"]):
        return ["RPC", "RATIO", "STA", "ALL"]

    if any(w in q for w in ["section", "law", "act"]):
        return ["STA", "ISSUE"]

    return ["ALL"]


# ── Role-Aware Search (WITH CHUNK MERGING) ────────────────────────────────────
def search_and_rerank(query, top_k=5, initial_k=20):

    query_embedding = bi_encoder.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    roles = detect_intent(query)

    doc_chunks = {}

    for role in roles:
        if role not in indexes:
            continue

        idx     = indexes[role]
        role_df = meta[role]

        k = min(initial_k, idx.ntotal)
        _, idxs = idx.search(query_embedding, k)

        for i in idxs[0]:
            if i < 0 or i >= len(role_df):
                continue

            row = role_df.iloc[i]
            doc_id = row["doc_id"]

            if doc_id not in doc_chunks:
                doc_chunks[doc_id] = []

            doc_chunks[doc_id].append((row["text"], role))

    # Merge chunks per case
    merged_docs = []

    for doc_id, chunks in doc_chunks.items():
        combined_text = " ".join([c[0] for c in chunks[:3]])  # top 3 chunks
        roles_used = list(set([c[1] for c in chunks]))

        merged_docs.append((doc_id, combined_text, roles_used))

    # Cross encoder reranking
    pairs  = [[query, doc[1]] for doc in merged_docs]
    scores = cross_encoder.predict(pairs)

    ranked = sorted(
        [(merged_docs[i], scores[i]) for i in range(len(scores))],
        key=lambda x: x[1],
        reverse=True
    )

    return ranked[:top_k]


# ── Better Reasoning Extraction ───────────────────────────────────────────────
def extract_reasoning(query, text):

    sentences = text.split(".")
    query_words = query.lower().split()

    important = []

    for s in sentences:
        s = s.strip().lower()
        if len(s) < 40:
            continue

        if any(qw in s for qw in query_words):
            important.append(s)

        if any(w in s for w in ["held", "observed", "court", "evidence", "intention"]):
            important.append(s)

        if len(important) >= 3:
            break

    return list(set(important))[:3]


# ── NEW Answer Generation (CORE FIX) ──────────────────────────────────────────
def generate_answer(query, results):

    answer = "\n🧠 Answer:\n\n"

    # ── Case-Based Insights ──
    answer += "📚 Relevant Case Insights:\n\n"

    reasoning_all = []

    for i, ((doc_id, text, roles), score) in enumerate(results, 1):

        short_text = text[:200].replace("\n", " ")

        answer += f"{i}. Case {doc_id}\n"
        answer += f"   → {short_text}...\n"

        reasoning = extract_reasoning(query, text)

        for r in reasoning:
            reasoning_all.append(r)

        answer += "\n"

    # ── Pattern Extraction ──
    answer += "🔍 Legal Patterns Observed:\n"

    reasoning_all = list(set(reasoning_all))[:5]

    for r in reasoning_all:
        answer += f"- {r.capitalize()}.\n"

    # ── Final Answer ──
    answer += "\n⚖️ Final Answer:\n"

    answer += (
        "Based on the above cases, the outcome depends on specific facts, "
        "intent, and strength of evidence. Courts rely heavily on reasoning "
        "from similar precedents rather than a fixed rule."
    )

    return answer


# ── Decision Support ──────────────────────────────────────────────────────────
def decision_support(results):

    positive = 0
    negative = 0

    for (doc, score) in results:
        text = doc[1].lower()

        if "granted" in text or "allowed" in text:
            positive += 1
        if "denied" in text or "dismissed" in text:
            negative += 1

    if positive > negative:
        return "⚖️ Trend: Relief is often granted in similar cases"
    elif negative > positive:
        return "⚖️ Trend: Relief is often denied in similar cases"
    else:
        return "⚖️ Trend: Mixed outcomes (case-specific)"


# ── Main ──────────────────────────────────────────────────────────────────────
def main():

    while True:
        print("\n" + "="*50)

        query = input("Enter your legal query (or 'quit'): ").strip()

        if query.lower() == "quit":
            break

        if not query:
            continue

        print("\n🔍 Searching relevant cases...\n")

        results = search_and_rerank(query)

        if not results:
            print("No results found.")
            continue

        print("="*50)
        print("        🧠 LEGAL ASSISTANT RESPONSE")
        print("="*50)

        print(generate_answer(query, results))
        print("\n" + decision_support(results))


if __name__ == "__main__":
    main()