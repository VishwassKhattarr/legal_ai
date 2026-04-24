import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

# ── Config ───────────────────────────────────────────────────────────────────
INDEX_DIR  = "faiss_indexes"
MODEL_NAME = "all-MiniLM-L6-v2"

# ── Load models ───────────────────────────────────────────────────────────────
print("Loading models...")
bi_encoder    = SentenceTransformer(MODEL_NAME, device="cuda")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", device="cuda")
print("Models loaded ✅")

# ── Load all indexes and metadata ─────────────────────────────────────────────
ROLES = ["RPC", "RATIO", "FAC", "STA", "ISSUE",
         "PRE_RELIED", "RLC", "PREAMBLE", "ARG_PETITIONER", "ALL"]

print("Loading indexes...")
indexes = {}
meta    = {}
for role in ROLES:
    idx_path  = os.path.join(INDEX_DIR, f"index_{role}.bin")
    meta_path = os.path.join(INDEX_DIR, f"meta_{role}.csv")
    if os.path.exists(idx_path):
        indexes[role] = faiss.read_index(idx_path)
        meta[role]    = pd.read_csv(meta_path)
print(f"Loaded {len(indexes)} indexes ✅\n")


# ── Intent Detection → Role Mapping ──────────────────────────────────────────
def detect_intent(query):
    q = query.lower()

    # Bail / arrest / custody → look at final rulings + reasoning
    if any(w in q for w in ["bail", "anticipatory bail", "custody",
                              "arrest", "remand", "detention"]):
        return ["RPC", "RATIO"]

    # Murder / death / homicide → facts + ruling + reasoning
    if any(w in q for w in ["murder", "homicide", "culpable", "killing",
                              "death", "accidental", "manslaughter"]):
        return ["RPC", "RATIO", "FAC"]

    # Rape / sexual assault
    if any(w in q for w in ["rape", "sexual assault", "molestation",
                              "sexual offence", "pocso"]):
        return ["RPC", "RATIO", "FAC"]

    # Fraud / cheating / financial crime
    if any(w in q for w in ["fraud", "cheating", "forgery",
                              "embezzlement", "misappropriation"]):
        return ["RPC", "RATIO", "STA"]

    # What law applies / which section
    if any(w in q for w in ["which section", "what law", "which act",
                              "applicable law", "legal provision",
                              "what does ipc", "under which"]):
        return ["STA", "ISSUE"]

    # Divorce / matrimonial
    if any(w in q for w in ["divorce", "maintenance", "alimony",
                              "matrimonial", "husband", "wife", "dowry"]):
        return ["RPC", "RATIO", "FAC"]

    # Property / land
    if any(w in q for w in ["property", "land", "possession",
                              "title", "ownership", "eviction"]):
        return ["RPC", "RATIO", "FAC"]

    # What happened in the case / facts
    if any(w in q for w in ["what happened", "facts of", "background",
                              "story", "incident", "events"]):
        return ["FAC"]

    # Precedent / prior case
    if any(w in q for w in ["precedent", "prior case", "previous judgment",
                              "earlier ruling", "similar case"]):
        return ["PRE_RELIED"]

    # Lower court / appeal
    if any(w in q for w in ["lower court", "high court", "sessions court",
                              "appeal from", "trial court"]):
        return ["RLC"]

    # Default → full index
    return ["ALL"]


def extract_ipc_section(query):
    for word in query.lower().split():
        if word.isdigit() and 100 <= int(word) <= 600:
            return word
    return None


# ── Role-Aware Search ─────────────────────────────────────────────────────────
def search_and_rerank(query, top_k=5, initial_k=20):
    query_embedding = bi_encoder.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    ipc      = extract_ipc_section(query)
    roles    = detect_intent(query)
    seen_docs = set()
    candidates = []
    candidate_meta = []

    # Search role-specific indexes first
    for role in roles:
        if role not in indexes:
            continue
        idx      = indexes[role]
        role_df  = meta[role]
        k        = min(initial_k, idx.ntotal)
        _, idxs  = idx.search(query_embedding, k)

        for i in idxs[0]:
            if i < 0 or i >= len(role_df):
                continue
            row    = role_df.iloc[i]
            doc_id = row["doc_id"]
            if doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)
            text = row["text"]

            # IPC filter if section detected
            if ipc:
                if str(ipc) not in text:
                    continue

            candidates.append(text)
            candidate_meta.append((doc_id, row["doc_type"], role))

    # Fallback to full index if not enough candidates
    if len(candidates) < top_k:
        idx     = indexes["ALL"]
        all_df  = meta["ALL"]
        k       = min(initial_k * 2, idx.ntotal)
        _, idxs = idx.search(query_embedding, k)

        for i in idxs[0]:
            if i < 0 or i >= len(all_df):
                continue
            row    = all_df.iloc[i]
            doc_id = row["doc_id"]
            if doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)
            candidates.append(row["text"])
            candidate_meta.append((doc_id, row["doc_type"], "ALL"))

    if not candidates:
        return []

    # Cross encoder reranking
    pairs  = [[query, doc] for doc in candidates]
    scores = cross_encoder.predict(pairs)

    ranked = sorted(
        zip(candidates, scores, candidate_meta),
        key=lambda x: x[1],
        reverse=True
    )
    return ranked[:top_k]


# ── Answer Generation ─────────────────────────────────────────────────────────
def generate_answer(query, results):
    texts = [t.lower() for t, _, _ in results]

    positive = sum("granted" in t or "allowed" in t for t in texts)
    negative = sum("rejected" in t or "denied" in t or "dismissed" in t for t in texts)

    if positive > negative:
        trend = "courts have granted relief in several similar cases"
    elif negative > positive:
        trend = "courts have denied relief in many similar cases"
    else:
        trend = "courts show mixed decisions depending on circumstances"

    combined = " ".join(texts[:3])
    keywords_map = {
        "delay"        : "delay in proceedings",
        "evidence"     : "strength of evidence",
        "condition"    : "legal conditions",
        "serious"      : "seriousness of offence",
        "custody"      : "custodial requirements",
        "investigation": "stage of investigation",
        "charge"       : "nature of charges",
        "appeal"       : "procedural history",
        "accident"     : "accidental nature of offence",
        "intention"    : "criminal intention",
        "negligence"   : "negligence of the accused"
    }
    signals = [v for k, v in keywords_map.items() if k in combined]
    signal_text = ", ".join(signals) if signals else "case-specific factors"

    return f"""
🧠 Answer:
Based on the retrieved legal cases, {trend}.
Courts typically consider factors such as {signal_text} when making decisions.
The outcome depends on judicial discretion and the specific facts of each case.
"""


# ── Decision Support ──────────────────────────────────────────────────────────
def decision_support(results):
    positive = negative = 0
    for text, _, _ in results:
        t = text.lower()
        if "granted" in t or "allowed" in t:
            positive += 1
        if "rejected" in t or "denied" in t or "dismissed" in t:
            negative += 1

    if positive > negative:
        return "⚖️  Decision Insight: The trend suggests relief is possible under certain conditions."
    elif negative > positive:
        return "⚖️  Decision Insight: The trend suggests relief is less likely based on precedents."
    else:
        return "⚖️  Decision Insight: The outcome is highly case-dependent."


# ── Explainability ────────────────────────────────────────────────────────────
ROLE_EXPLANATIONS = {
    "RPC"           : "Final ruling by the court",
    "RATIO"         : "Court's reasoning / ratio decidendi",
    "FAC"           : "Facts of the case",
    "STA"           : "Relevant statute or legal provision",
    "ISSUE"         : "Legal issue framed by the court",
    "PRE_RELIED"    : "Precedent relied upon by the court",
    "RLC"           : "Ruling by the lower court",
    "PREAMBLE"      : "Case header / preamble",
    "ARG_PETITIONER": "Argument by the petitioner",
    "ALL"           : "General legal content",
}

def explain(text, role):
    t = text.lower()
    reasons = []

    role_desc = ROLE_EXPLANATIONS.get(role, "Legal content")
    reasons.append(f"[{role}] {role_desc}")

    if "bail"         in t: reasons.append("Discusses bail")
    if "ipc"          in t: reasons.append("Refers to IPC section")
    if "murder"       in t or "homicide" in t: reasons.append("Discusses homicide/murder")
    if "accident"     in t or "negligence" in t: reasons.append("Mentions accidental death")
    if "granted"      in t or "allowed"   in t: reasons.append("Relief was granted")
    if "dismissed"    in t or "rejected"  in t: reasons.append("Relief was denied")
    if "conviction"   in t: reasons.append("Discusses conviction")
    if "acquittal"    in t: reasons.append("Discusses acquittal")
    if "section 302"  in t: reasons.append("IPC 302 — Murder")
    if "section 304"  in t: reasons.append("IPC 304 — Culpable homicide")
    if "section 304a" in t: reasons.append("IPC 304A — Death by negligence")
    if "section 376"  in t: reasons.append("IPC 376 — Rape")
    if "section 420"  in t: reasons.append("IPC 420 — Cheating")

    return " | ".join(reasons)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    while True:
        print("\n" + "="*50)
        query = input("Enter your legal query (or 'quit' to exit): ").strip()
        if query.lower() == "quit":
            break
        if not query:
            continue

        # Show detected intent
        roles = detect_intent(query)
        print(f"\n🎯 Detected intent → searching indexes: {roles}")

        expanded = query + " " + " ".join(roles)
        results  = search_and_rerank(query)

        if not results:
            print("No relevant cases found. Try rephrasing your query.")
            continue

        print("\n" + "="*50)
        print("        🧠 LEGAL ASSISTANT RESPONSE")
        print("="*50)
        print(generate_answer(query, results))
        print(decision_support(results))
        print("\n🔍 Supporting Cases:\n")

        for i, (text, score, (doc_id, doc_type, role)) in enumerate(results):
            confidence = max(0, min(100, int((score + 10) * 5)))
            print(f"{i+1}. [{doc_type}] {doc_id} | Relevance: {confidence}% | Role: {role}")
            print(f"   {text[:200]}...")
            print(f"   Reason: {explain(text, role)}")
            print("-" * 70)


if __name__ == "__main__":
    main()