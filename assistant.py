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


# ── Intent Detection ──────────────────────────────────────────────────────────
def detect_intent(query):
    q = query.lower()

    if any(w in q for w in ["bail", "anticipatory bail", "custody",
                              "arrest", "remand", "detention"]):
        return ["RPC", "RATIO"]

    if any(w in q for w in ["murder", "homicide", "culpable", "killing",
                              "death", "accidental", "manslaughter"]):
        return ["RPC", "RATIO", "FAC"]

    if any(w in q for w in ["rape", "sexual assault", "molestation",
                              "sexual offence", "pocso"]):
        return ["RPC", "RATIO", "FAC"]

    if any(w in q for w in ["fraud", "cheating", "forgery",
                              "embezzlement", "misappropriation"]):
        return ["RPC", "RATIO", "STA", "ALL"]

    if any(w in q for w in ["which section", "what law", "which act",
                              "applicable law", "legal provision",
                              "what does ipc", "under which"]):
        return ["STA", "ISSUE"]

    if any(w in q for w in ["divorce", "maintenance", "alimony",
                              "matrimonial", "husband", "wife", "dowry"]):
        return ["RPC", "RATIO", "FAC"]

    if any(w in q for w in ["property", "land", "possession",
                              "title", "ownership", "eviction"]):
        return ["RPC", "RATIO", "FAC"]

    if any(w in q for w in ["what happened", "facts of", "background",
                              "story", "incident", "events"]):
        return ["FAC"]

    if any(w in q for w in ["precedent", "prior case", "previous judgment",
                              "earlier ruling", "similar case"]):
        return ["PRE_RELIED"]

    if any(w in q for w in ["lower court", "high court", "sessions court",
                              "appeal from", "trial court"]):
        return ["RLC"]

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

    ipc           = extract_ipc_section(query)
    roles         = detect_intent(query)
    seen_docs     = set()
    candidates    = []
    candidate_meta = []

    for role in roles:
        if role not in indexes:
            continue
        idx     = indexes[role]
        role_df = meta[role]
        k       = min(initial_k, idx.ntotal)
        _, idxs = idx.search(query_embedding, k)

        for i in idxs[0]:
            if i < 0 or i >= len(role_df):
                continue
            row    = role_df.iloc[i]
            doc_id = row["doc_id"]
            if doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)
            text = row["text"]

            if ipc:
                if str(ipc) not in text:
                    continue

            candidates.append(text)
            candidate_meta.append((doc_id, row["doc_type"], role))

    # Fallback
    if len(candidates) < top_k:
        idx    = indexes["ALL"]
        all_df = meta["ALL"]
        k      = min(initial_k * 2, idx.ntotal)
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

    # Keyword boosting
    q = query.lower()
    boost_keywords = []
    if any(w in q for w in ["cheating", "fraud", "420"]):
        boost_keywords = ["420", "cheating", "fraud", "deceiv", "dishonest"]
    elif any(w in q for w in ["bail", "anticipatory"]):
        boost_keywords = ["bail", "anticipatory", "granted", "custody"]
    elif any(w in q for w in ["murder", "homicide", "accidental"]):
        boost_keywords = ["302", "304", "304a", "murder", "homicide", "accidental"]
    elif any(w in q for w in ["rape", "sexual"]):
        boost_keywords = ["376", "rape", "sexual", "penetration", "victim"]

    adjusted = []
    for text, score, m in zip(candidates, scores, candidate_meta):
        boost = 0
        if boost_keywords:
            text_lower = text.lower()
            matches    = sum(1 for kw in boost_keywords if kw in text_lower)
            boost      = matches * 0.3
        adjusted.append((text, score + boost, m))

    ranked = sorted(adjusted, key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


# ── Answer Generation ─────────────────────────────────────────────────────────
def generate_answer(query, results):
    texts = [t.lower() for t, _, _ in results]

    positive = sum("granted" in t or "allowed" in t for t in texts)
    negative = sum("rejected" in t or "denied" in t or "dismissed" in t for t in texts)

    if positive > negative:
        trend = "courts have generally granted relief in similar cases"
    elif negative > positive:
        trend = "courts have generally denied relief in similar cases"
    else:
        trend = "courts show mixed decisions in similar cases"

    # Extract specific principles from RATIO/RPC chunks
    principles = []
    for text, score, (doc_id, doc_type, role) in results:
        if role in ["RATIO", "RPC"]:
            sentences = text.lower().replace("\n", " ").split(".")
            for sent in sentences:
                sent = sent.strip()
                if len(sent) < 30:
                    continue
                if any(w in sent for w in [
                    "court held", "it was held", "we hold",
                    "it is settled", "the principle",
                    "bail", "conviction", "acquitt",
                    "section 302", "section 304", "section 376", "section 420",
                    "granted", "denied", "dismissed", "allowed",
                    "cheating", "murder", "rape", "homicide",
                    "intention", "negligence", "evidence"
                ]):
                    principles.append(sent[:200])
                if len(principles) >= 3:
                    break
        if len(principles) >= 3:
            break

    q = query.lower()

    if any(w in q for w in ["bail", "anticipatory", "custody", "arrest"]):
        intro   = "Regarding bail in this type of case"
        factors = "courts examine: (1) prima facie case against the accused, (2) nature and gravity of the offence, (3) likelihood of fleeing justice, (4) tampering with evidence, and (5) criminal antecedents of the accused."

    elif any(w in q for w in ["murder", "homicide", "culpable", "accidental", "killing"]):
        intro   = "In cases involving murder or culpable homicide"
        factors = "courts distinguish between IPC Section 302 (murder with intention), Section 304 (culpable homicide not amounting to murder), and Section 304A (death by negligence). The key factor is criminal intention (mens rea)."

    elif any(w in q for w in ["rape", "sexual", "assault", "pocso"]):
        intro   = "In rape and sexual assault cases"
        factors = "courts apply IPC Section 376. Conviction typically rests on the sole testimony of the victim if found credible. Corroboration is not mandatory but desirable."

    elif any(w in q for w in ["cheating", "fraud", "420", "forgery"]):
        intro   = "In cheating and fraud cases"
        factors = "courts apply IPC Section 420. The essential ingredients are: (1) deception of a person, (2) fraudulent or dishonest inducement, and (3) delivery of property or alteration of a valuable document."

    elif any(w in q for w in ["dowry", "matrimonial", "divorce", "maintenance"]):
        intro   = "In matrimonial and dowry cases"
        factors = "courts apply IPC Section 498A (cruelty) and Section 304B (dowry death). Both physical and mental cruelty are recognized."

    elif any(w in q for w in ["property", "land", "possession", "title"]):
        intro   = "In property dispute cases"
        factors = "courts examine title documents, possession records, and chain of ownership. Limitation periods are strictly applied."

    else:
        intro   = "Based on the retrieved legal cases"
        factors = "courts examine the specific facts, evidence, applicable statutes, and relevant precedents before arriving at a decision."

    answer  = f"\n🧠 Answer:\n"
    answer += f"{intro}, {trend}.\n\n"
    answer += f"⚖️  Legal Position: {factors}\n"

    if principles:
        answer += f"\n📋 Key Legal Principles from Retrieved Cases:\n"
        for i, p in enumerate(principles, 1):
            answer += f"   {i}. {p.strip().capitalize()}.\n"

    return answer


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
    t       = text.lower()
    reasons = []

    role_desc = ROLE_EXPLANATIONS.get(role, "Legal content")
    reasons.append(f"[{role}] {role_desc}")

    if "bail"          in t: reasons.append("Discusses bail")
    if "ipc"           in t: reasons.append("Refers to IPC section")
    if "murder"        in t or "homicide" in t: reasons.append("Discusses homicide/murder")
    if "accident"      in t or "negligence" in t: reasons.append("Mentions accidental death")
    if "granted"       in t or "allowed"   in t: reasons.append("Relief was granted")
    if "dismissed"     in t or "rejected"  in t: reasons.append("Relief was denied")
    if "conviction"    in t: reasons.append("Discusses conviction")
    if "acquittal"     in t: reasons.append("Discusses acquittal")
    if "cheating"      in t or "420"       in t: reasons.append("IPC 420 — Cheating")
    if "section 302"   in t: reasons.append("IPC 302 — Murder")
    if "section 304"   in t: reasons.append("IPC 304 — Culpable homicide")
    if "section 304a"  in t: reasons.append("IPC 304A — Death by negligence")
    if "section 376"   in t: reasons.append("IPC 376 — Rape")

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

        roles = detect_intent(query)
        print(f"\n🎯 Detected intent → searching indexes: {roles}")

        results = search_and_rerank(query)

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