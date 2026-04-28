# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# import pandas as pd
# import faiss
# import numpy as np
# from sentence_transformers import SentenceTransformer, CrossEncoder

# # ── Config ───────────────────────────────────────────────────────────────────
# INDEX_DIR  = "faiss_indexes"
# MODEL_NAME = "Sayyam9/legal-bert-aila-finetuned"

# # ── Load models ───────────────────────────────────────────────────────────────
# print("Loading models...")
# bi_encoder    = SentenceTransformer(MODEL_NAME, device="cpu")
# cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", device="cpu")
# print("Models loaded ✅")

# # ── Load indexes ──────────────────────────────────────────────────────────────
# ROLES = ["RPC", "RATIO", "FAC", "STA", "ISSUE",
#          "PRE_RELIED", "RLC", "PREAMBLE", "ARG_PETITIONER", "ALL"]

# indexes = {}
# meta    = {}

# print("Loading indexes...")
# for role in ROLES:
#     idx_path  = os.path.join(INDEX_DIR, f"index_{role}.bin")
#     meta_path = os.path.join(INDEX_DIR, f"meta_{role}.csv")

#     if os.path.exists(idx_path):
#         indexes[role] = faiss.read_index(idx_path)
#         meta[role]    = pd.read_csv(meta_path)

# print(f"Loaded {len(indexes)} indexes ✅\n")


# # ── Intent Detection ──────────────────────────────────────────────────────────
# def detect_intent(query):
#     q = query.lower()

#     if any(w in q for w in ["bail", "custody", "arrest"]):
#         return ["RPC", "RATIO"], False

#     if any(w in q for w in ["murder", "homicide", "death"]):
#         return ["RPC", "RATIO", "FAC"], False

#     if any(w in q for w in ["rape", "sexual"]):
#         return ["RPC", "RATIO", "FAC"], False

#     if any(w in q for w in ["fraud", "cheating", "420"]):
#         return ["RPC", "RATIO", "STA", "ALL"], False

#     if any(w in q for w in ["section", "law", "act"]):
#         return ["STA", "ISSUE"], False

#     # ── NEW: generalised fallback — search ALL roles broadly ──
#     return ["RPC", "RATIO", "FAC", "STA", "ISSUE", "PRE_RELIED", "ALL"], True


# # ── Confidence Score Computation (NEW) ───────────────────────────────────────
# def compute_confidence_scores(scores):
#     """
#     Normalise raw cross-encoder scores to [0, 1] using min-max scaling.
#     Returns a list of floats in the same order as input scores.
#     """
#     arr = np.array(scores, dtype=np.float32)
#     min_s, max_s = arr.min(), arr.max()
#     if max_s - min_s < 1e-6:
#         return [1.0] * len(scores)
#     return ((arr - min_s) / (max_s - min_s)).tolist()


# # ── Role-Aware Search ─────────────────────────────────────────────────────────
# def search_and_rerank(query, top_k=5, initial_k=20):

#     query_embedding = bi_encoder.encode([query])
#     query_embedding = np.array(query_embedding).astype("float32")

#     # ── CHANGED: unpack the new is_general flag ──
#     roles, is_general = detect_intent(query)

#     # For generalised queries, cast a wider net
#     if is_general:
#         initial_k = min(initial_k * 2, 40)

#     doc_chunks = {}

#     for role in roles:
#         if role not in indexes:
#             continue

#         idx     = indexes[role]
#         role_df = meta[role]

#         k = min(initial_k, idx.ntotal)
#         _, idxs = idx.search(query_embedding, k)

#         for i in idxs[0]:
#             if i < 0 or i >= len(role_df):
#                 continue

#             row    = role_df.iloc[i]
#             doc_id = row["doc_id"]

#             if doc_id not in doc_chunks:
#                 doc_chunks[doc_id] = []

#             doc_chunks[doc_id].append((row["text"], role))

#     # Merge chunks
#     merged_docs = []

#     for doc_id, chunks in doc_chunks.items():
#         combined_text = " ".join([c[0] for c in chunks[:3]])
#         roles_used    = list(set([c[1] for c in chunks]))
#         merged_docs.append((doc_id, combined_text, roles_used))

#     # Rerank
#     pairs  = [[query, doc[1]] for doc in merged_docs]
#     scores = cross_encoder.predict(pairs)

#     # ── NEW: compute confidence scores before sorting ──
#     confidence_scores = compute_confidence_scores(scores)

#     ranked = sorted(
#         [
#             (merged_docs[i], scores[i], confidence_scores[i])
#             for i in range(len(scores))
#         ],
#         key=lambda x: x[1],
#         reverse=True
#     )

#     return ranked[:top_k]


# # ── Reasoning Extraction ──────────────────────────────────────────────────────
# def extract_reasoning(query, text):

#     sentences   = text.split(".")
#     query_words = query.lower().split()
#     important   = []

#     for s in sentences:
#         s = s.strip().lower()
#         if len(s) < 40:
#             continue

#         if any(qw in s for qw in query_words):
#             important.append(s)

#         if any(w in s for w in ["held", "observed", "court", "evidence", "intention"]):
#             important.append(s)

#         if len(important) >= 3:
#             break

#     return list(set(important))[:3]


# # ── Answer Generation ─────────────────────────────────────────────────────────
# def generate_answer(query, results):

#     answer  = "\n🧠 Answer:\n\n"
#     answer += "📚 Relevant Case Insights:\n\n"

#     reasoning_all = []

#     # ── CHANGED: unpack the extra confidence value from results ──
#     for i, ((doc_id, text, roles), score, confidence) in enumerate(results, 1):

#         short_text = text[:600].replace("\n", " ")
#         conf_pct   = f"{confidence * 100:.1f}%"   # NEW: format as percentage

#         answer += f"{i}. Case {doc_id}  |  Confidence: {conf_pct}\n"  # NEW: shown here
#         answer += f"   → {short_text}...\n"

#         reasoning = extract_reasoning(query, text)
#         reasoning_all.extend(reasoning)

#         answer += "\n"

#     answer += "🔍 Legal Patterns Observed:\n"

#     reasoning_all = list(set(reasoning_all))[:5]

#     for r in reasoning_all:
#         answer += f"- {r.capitalize()}.\n"

#     answer += "\n⚖️ Final Answer:\n"
#     answer += (
#         "Based on the above cases, the outcome depends on specific facts, "
#         "intent, and strength of evidence. Courts rely heavily on reasoning "
#         "from similar precedents rather than a fixed rule."
#     )

#     return answer


# # ── Decision Support ──────────────────────────────────────────────────────────
# def decision_support(results):

#     positive = 0
#     negative = 0

#     # ── CHANGED: unpack the extra confidence value ──
#     for (doc, score, confidence) in results:
#         text = doc[1].lower()

#         if "granted" in text or "allowed" in text:
#             positive += 1
#         if "denied" in text or "dismissed" in text:
#             negative += 1

#     if positive > negative:
#         return "⚖️ Trend: Relief is often granted in similar cases"
#     elif negative > positive:
#         return "⚖️ Trend: Relief is often denied in similar cases"
#     else:
#         return "⚖️ Trend: Mixed outcomes (case-specific)"


# # ── Main ──────────────────────────────────────────────────────────────────────
# def main():

#     while True:
#         print("\n" + "="*50)

#         query = input("Enter your legal query (or 'quit'): ").strip()

#         if query.lower() == "quit":
#             break

#         if not query:
#             continue

#         print("\n🔍 Searching relevant cases...\n")

#         results = search_and_rerank(query)

#         if not results:
#             print("No results found.")
#             continue

#         print("="*50)
#         print("        🧠 LEGAL ASSISTANT RESPONSE")
#         print("="*50)

#         print(generate_answer(query, results))
#         print("\n" + decision_support(results))


# if __name__ == "__main__":
#     main()


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

# ── Config ───────────────────────────────────────────────────────────────────
INDEX_DIR  = "faiss_indexes"
MODEL_NAME = "Sayyam9/legal-bert-aila-finetuned"

# ── Load models ───────────────────────────────────────────────────────────────
print("Loading models...")
bi_encoder    = SentenceTransformer(MODEL_NAME, device="cpu")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", device="cpu")
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
        return ["RPC", "RATIO"], False

    if any(w in q for w in ["murder", "homicide", "death"]):
        return ["RPC", "RATIO", "FAC"], False

    if any(w in q for w in ["rape", "sexual"]):
        return ["RPC", "RATIO", "FAC"], False

    if any(w in q for w in ["fraud", "cheating", "420"]):
        return ["RPC", "RATIO", "STA", "ALL"], False

    if any(w in q for w in ["section", "law", "act"]):
        return ["STA", "ISSUE"], False

    # ── NEW: generalised fallback — search ALL roles broadly ──
    return ["RPC", "RATIO", "FAC", "STA", "ISSUE", "PRE_RELIED", "ALL"], True


# ── Confidence Score Computation (NEW) ───────────────────────────────────────
def compute_confidence_scores(scores):
    """
    Normalise raw cross-encoder scores to [0, 1] using min-max scaling.
    Returns a list of floats in the same order as input scores.
    """
    arr = np.array(scores, dtype=np.float32)
    min_s, max_s = arr.min(), arr.max()
    if max_s - min_s < 1e-6:
        return [1.0] * len(scores)
    return ((arr - min_s) / (max_s - min_s)).tolist()


# ── Role-Aware Search ─────────────────────────────────────────────────────────
def search_and_rerank(query, top_k=5, initial_k=20):

    query_embedding = bi_encoder.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    # ── CHANGED: unpack the new is_general flag ──
    roles, is_general = detect_intent(query)

    # For generalised queries, cast a wider net
    if is_general:
        initial_k = min(initial_k * 2, 40)

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

            row    = role_df.iloc[i]
            doc_id = row["doc_id"]

            if doc_id not in doc_chunks:
                doc_chunks[doc_id] = []

            doc_chunks[doc_id].append((row["text"], role))

    # Merge chunks
    merged_docs = []

    for doc_id, chunks in doc_chunks.items():
        combined_text = " ".join([c[0] for c in chunks[:3]])
        roles_used    = list(set([c[1] for c in chunks]))
        merged_docs.append((doc_id, combined_text, roles_used))

    # Rerank
    pairs  = [[query, doc[1]] for doc in merged_docs]
    scores = cross_encoder.predict(pairs)

    # ── NEW: compute confidence scores before sorting ──
    confidence_scores = compute_confidence_scores(scores)

    ranked = sorted(
        [
            (merged_docs[i], scores[i], confidence_scores[i])
            for i in range(len(scores))
        ],
        key=lambda x: x[1],
        reverse=True
    )

    return ranked[:top_k]


# ── Reasoning Extraction (query-aware) ───────────────────────────────────────
def extract_reasoning(query, text):
    """
    Return sentences that are directly relevant to the query.
    Priority: query-word match > legal signal words.
    Returns up to 5 sentences (increased from 3) for richer output.
    """
    sentences   = text.split(".")
    query_words = [w for w in query.lower().split() if len(w) > 3]  # skip stopwords
    important   = []
    seen        = set()

    for s in sentences:
        s_clean = s.strip()
        if len(s_clean) < 40:
            continue
        s_lower = s_clean.lower()
        if s_lower in seen:
            continue

        # Higher priority: sentence directly mentions query terms
        query_hits = sum(1 for qw in query_words if qw in s_lower)
        legal_hit  = any(w in s_lower for w in [
            "held", "observed", "court", "evidence", "intention",
            "principle", "liability", "convicted", "acquitted",
            "section", "ipc", "accused", "judgment", "appeal"
        ])

        if query_hits >= 2 or (query_hits >= 1 and legal_hit):
            important.insert(0, s_clean)   # query-matched sentences go first
            seen.add(s_lower)
        elif legal_hit and len(important) < 5:
            important.append(s_clean)
            seen.add(s_lower)

        if len(important) >= 5:
            break

    return important[:5]


# ── Query-Aware Final Answer Builder ─────────────────────────────────────────
def build_final_answer(query, reasoning_all):
    """
    Construct a specific answer paragraph using extracted sentences
    rather than a fixed generic string.
    """
    if not reasoning_all:
        return (
            "Based on the retrieved cases, the outcome depends on specific facts, "
            "intent, and strength of evidence."
        )

    # Deduplicate while preserving order
    seen, unique = set(), []
    for s in reasoning_all:
        key = s.lower()[:60]
        if key not in seen:
            seen.add(key)
            unique.append(s)

    answer  = f'Regarding "{query}", the Supreme Court has observed:\n\n'
    for s in unique[:5]:
        answer += f"  • {s.strip().rstrip('.')}.\n"
    return answer


# ── Answer Generation ─────────────────────────────────────────────────────────
def generate_answer(query, results):

    answer  = "\n🧠 Answer:\n\n"
    answer += "📚 Relevant Case Insights:\n\n"

    reasoning_all = []

    for i, ((doc_id, text, roles), score, confidence) in enumerate(results, 1):

        # ── CHANGED: show up to 600 chars, no trailing ellipsis mid-sentence ──
        excerpt    = text[:600].replace("\n", " ").rsplit(" ", 1)[0]
        conf_pct   = f"{confidence * 100:.1f}%"
        roles_str  = ", ".join(roles)

        answer += f"{i}. Case {doc_id}  |  Confidence: {conf_pct}  |  Roles: {roles_str}\n"
        answer += f"   {excerpt}\n\n"

        reasoning = extract_reasoning(query, text)
        reasoning_all.extend(reasoning)

    answer += "🔍 Legal Patterns Observed:\n"
    patterns = list(dict.fromkeys(                        # deduplicate, keep order
        s for s in reasoning_all if len(s) > 40
    ))[:5]
    for r in patterns:
        answer += f"  • {r.rstrip('.')}.\n"

    answer += "\n⚖️ Final Answer:\n"
    answer += build_final_answer(query, reasoning_all)    # ── CHANGED: query-aware

    return answer


# ── Decision Support ──────────────────────────────────────────────────────────
def decision_support(results):

    positive = 0
    negative = 0

    # ── CHANGED: unpack the extra confidence value ──
    for (doc, score, confidence) in results:
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