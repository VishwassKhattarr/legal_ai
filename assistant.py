import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

# Load data
df = pd.read_csv("data/chunks/chunks.csv")

# Models
bi_encoder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", device="cpu")

# Load index
index = faiss.read_index("faiss_index.bin")


# ---------------- HELPER FUNCTIONS ---------------- #

def expand_query(query):
    return query + " criminal law IPC bail anticipatory bail legal conditions court decision judgment"


def is_ipc_relevant(text, ipc):
    text_lower = text.lower()

    # Must contain IPC section in meaningful context
    if ipc:
        patterns = [
            f"section {ipc}",
            f"{ipc} ipc",
            f"under {ipc}",
            f"u/s {ipc}"
        ]
        return any(p in text_lower for p in patterns)

    return True


def extract_ipc_section(query):
    for word in query.lower().split():
        if word.isdigit():
            return word
    return None


def is_relevant(text, query):
    text_lower = text.lower()
    query_tokens = [token for token in query.lower().split() if len(token) > 2]
    if not query_tokens:
        return True
    return any(token in text_lower for token in query_tokens)


# ---------------- CORE PIPELINE ---------------- #

def search_and_rerank(query, top_k=5, initial_k=40):
    query_embedding = bi_encoder.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, initial_k)

    ipc = extract_ipc_section(query)

    candidates = []
    meta = []

    for idx in indices[0]:
        row = df.iloc[idx]
        text = row["text"]
        text_lower = text.lower()

        if is_relevant(text, query):
            # Soft IPC filtering (NOT strict, NOT hardcoded)
            if ipc:
                if is_ipc_relevant(text, ipc):
                    candidates.append(text)
                    meta.append((row["doc_id"], row["doc_type"]))
            else:
                candidates.append(text)
                meta.append((row["doc_id"], row["doc_type"]))

    # fallback if nothing found
    if len(candidates) == 0:
        for idx in indices[0]:
            row = df.iloc[idx]
            candidates.append(row["text"])
            meta.append((row["doc_id"], row["doc_type"]))

    # Cross encoder scoring
    pairs = [[query, doc] for doc in candidates]
    scores = cross_encoder.predict(pairs)

    # Score adjustment (general, not hardcoded)
    adjusted = []
    for text, score, m in zip(candidates, scores, meta):
        text_lower = text.lower()
        penalty = 0

        # Penalize if query terms missing
        for word in query.lower().split():
            if word not in text_lower:
                penalty += 0.05

        adjusted.append((text, score - penalty, m))

    ranked = sorted(adjusted, key=lambda x: x[1], reverse=True)

    return ranked[:top_k]


# ---------------- ANSWER GENERATION ---------------- #

def generate_answer(query, results):
    texts = [text.lower() for text, _, _ in results]

    positive = sum("granted" in t or "allowed" in t for t in texts)
    negative = sum("rejected" in t or "denied" in t or "cancelled" in t for t in texts)

    # Determine trend purely from data
    if positive > negative:
        trend = "courts have granted relief in several similar cases"
    elif negative > positive:
        trend = "courts have denied relief in many similar cases"
    else:
        trend = "courts show mixed decisions depending on circumstances"

    # Extract reasoning signals
    combined = " ".join(texts[:3])

    keywords_map = {
        "delay": "delay in proceedings",
        "evidence": "strength of evidence",
        "condition": "legal conditions",
        "serious": "seriousness of offence",
        "custody": "custodial requirements",
        "investigation": "stage of investigation",
        "charge": "nature of charges",
        "appeal": "procedural history"
    }

    signals = []
    for k, v in keywords_map.items():
        if k in combined:
            signals.append(v)

    signal_text = ", ".join(signals) if signals else "case-specific factors"

    answer = f"""
🧠 Answer:
Based on the retrieved legal cases, {trend}.

Courts typically consider factors such as {signal_text} when making decisions. The outcome depends on judicial discretion and the specific facts of each case.
"""
    return answer


# ---------------- DECISION SUPPORT ---------------- #

def decision_support(results):
    positive = 0
    negative = 0

    for text, _, _ in results:
        t = text.lower()
        if "granted" in t or "allowed" in t:
            positive += 1
        if "rejected" in t or "denied" in t or "cancelled" in t:
            negative += 1

    if positive > negative:
        return "⚖️ Decision Insight: The trend suggests relief is possible under certain conditions."
    elif negative > positive:
        return "⚖️ Decision Insight: The trend suggests relief is less likely based on precedents."
    else:
        return "⚖️ Decision Insight: The outcome is highly case-dependent."


# ---------------- EXPLAINABILITY ---------------- #

def explain(text):
    t = text.lower()
    explanation = []

    if "bail" in t:
        explanation.append("Mentions bail")
    if "anticipatory bail" in t:
        explanation.append("Discusses anticipatory bail")
    if "ipc" in t:
        explanation.append("Refers IPC section")
    if "condition" in t or "criteria" in t:
        explanation.append("Explains legal conditions")

    return ", ".join(explanation)


# ---------------- MAIN ---------------- #

query = input("Enter your legal query: ")
query = expand_query(query)

results = search_and_rerank(query)

print("\n==============================")
print("🧠 LEGAL ASSISTANT RESPONSE")
print("==============================\n")

print(generate_answer(query, results))
print(decision_support(results))
print("\n🔍 Supporting Cases:\n")

for i, (text, score, (doc_id, doc_type)) in enumerate(results):
    print(f"{i+1}. [{doc_type}] {doc_id} | Score: {score:.4f}")
    print(text[:200])
    print("Reason:", explain(text))
    print("-" * 80)