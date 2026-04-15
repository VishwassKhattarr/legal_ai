"""
train_biencoder.py
==================
Stage 1: Bi-Encoder Training
Model  : InLegalBERT (Indian legal domain)
Task   : Given a query, retrieve relevant cases + statutes

Run: python train_biencoder.py
"""

import os
import json
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    evaluation,
)
from torch.utils.data import DataLoader

print("=" * 55)
print("  INDIAN LEGAL AI — Bi-Encoder Training")
print("=" * 55)

# ── Config ────────────────────────────────────────────────
CONFIG = {
    "model_name"    : "law-ai/InLegalBERT",   # Indian legal BERT
    "fallback_model": "distilbert-base-uncased",  # agar InLegalBERT na chale
    "max_length"    : 256,
    "batch_size"    : 8,
    "epochs"        : 3,
    "warmup_steps"  : 100,
    "learning_rate" : 2e-5,
    "save_path"     : "model/biencoder/",
    "neg_per_query" : 5,   # har query ke liye kitne negative samples
}

os.makedirs(CONFIG["save_path"], exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n  Device      : {device.upper()}")
print(f"  Model       : {CONFIG['model_name']}")
print(f"  Epochs      : {CONFIG['epochs']}")
print(f"  Batch size  : {CONFIG['batch_size']}")


# ══════════════════════════════════════════════════════════
# STEP 1 — DATA LOAD
# ══════════════════════════════════════════════════════════

print("\n[STEP 1] Loading preprocessed data...")

queries_df  = pd.read_csv("data/processed/queries.csv")
cases_df    = pd.read_csv("data/processed/cases.csv")
statutes_df = pd.read_csv("data/processed/statutes.csv")
rel_cases   = pd.read_csv("data/processed/relevance_cases.csv")
rel_stat    = pd.read_csv("data/processed/relevance_statutes.csv")

print(f"         Queries  : {len(queries_df)}")
print(f"         Cases    : {len(cases_df)}")
print(f"         Statutes : {len(statutes_df)}")
print(f"         Rel pairs (cases)   : {len(rel_cases)}")
print(f"         Rel pairs (statutes): {len(rel_stat)}")

# Dictionaries banao — fast lookup ke liye
query_dict   = dict(zip(queries_df["query_id"],   queries_df["text"]))
case_dict    = dict(zip(cases_df["case_id"],       cases_df["text"]))
statute_dict = dict(zip(statutes_df["statute_id"], statutes_df["text"]))

# Text truncate karo — memory ke liye
def truncate(text, max_words=200):
    words = str(text).split()
    return " ".join(words[:max_words])


# ══════════════════════════════════════════════════════════
# STEP 2 — TRAINING PAIRS BANANA
# ══════════════════════════════════════════════════════════

print("\n[STEP 2] Training pairs ban rahe hain...")
print("         (Positive + Negative samples)")

train_examples = []

# ── Case retrieval pairs ──────────────────────────────────
# Positive pairs: query → relevant case (label=1)
# Negative pairs: query → random irrelevant case (label=0)

all_case_ids = list(case_dict.keys())

for _, row in tqdm(queries_df.iterrows(), total=len(queries_df),
                   desc="         Case pairs"):
    qid        = row["query_id"]
    query_text = truncate(row["text"])

    # Positive cases — jo relevant hain
    pos_cases = rel_cases[rel_cases["query_id"] == qid]["case_id"].tolist()

    if not pos_cases:
        continue

    for pos_id in pos_cases[:10]:   # max 10 positive per query
        if pos_id not in case_dict:
            continue
        pos_text = truncate(case_dict[pos_id])

        # Positive pair
        train_examples.append(InputExample(
            texts  = [query_text, pos_text],
            label  = 1.0
        ))

        # Negative pairs — random cases jo relevant nahi hain
    neg_pool = [c for c in all_case_ids if c not in pos_cases]

if len(neg_pool) > 0:
    neg_ids = random.sample(
        neg_pool,
        min(CONFIG["neg_per_query"], len(neg_pool))
    )

    for neg_id in neg_ids:
        neg_text = truncate(case_dict[neg_id])
        train_examples.append(InputExample(
            texts  = [query_text, neg_text],
            label  = 0.0
        ))

# ── Statute retrieval pairs ───────────────────────────────
all_statute_ids = list(statute_dict.keys())

for _, row in tqdm(queries_df.iterrows(), total=len(queries_df),
                   desc="         Statute pairs"):
    qid        = row["query_id"]
    query_text = truncate(row["text"])

    pos_statutes = rel_stat[rel_stat["query_id"] == qid]["statute_id"].tolist()

    if not pos_statutes:
        continue

    for pos_id in pos_statutes[:5]:
        if pos_id not in statute_dict:
            continue
        pos_text = truncate(statute_dict[pos_id], max_words=100)

        train_examples.append(InputExample(
            texts = [query_text, pos_text],
            label = 1.0
        ))

    neg_pool = [s for s in all_statute_ids if s not in pos_statutes]

if len(neg_pool) > 0:
    neg_ids = random.sample(
        neg_pool,
        min(3, len(neg_pool))
    )

    for neg_id in neg_ids:
        neg_text = truncate(statute_dict[neg_id], max_words=100)
        train_examples.append(InputExample(
            texts = [query_text, neg_text],
            label = 0.0
        ))

# Shuffle
random.shuffle(train_examples)
print(f"\n         Total training pairs : {len(train_examples)}")
print(f"         Positive pairs       : {sum(1 for e in train_examples if e.label == 1.0)}")
print(f"         Negative pairs       : {sum(1 for e in train_examples if e.label == 0.0)}")


# ══════════════════════════════════════════════════════════
# STEP 3 — MODEL LOAD
# ══════════════════════════════════════════════════════════

print(f"\n[STEP 3] Loading model: {CONFIG['model_name']}...")
print(f"         (Pehli baar ~400MB download hoga)\n")

try:
    model = SentenceTransformer(CONFIG["model_name"])
    print(f"         InLegalBERT loaded ✅")
except Exception as e:
    print(f"         InLegalBERT failed: {e}")
    print(f"         Fallback: {CONFIG['fallback_model']}")
    model = SentenceTransformer(CONFIG["fallback_model"])
    print(f"         {CONFIG['fallback_model']} loaded ✅")

model.max_seq_length = CONFIG["max_length"]


# ══════════════════════════════════════════════════════════
# STEP 4 — TRAINING
# ══════════════════════════════════════════════════════════

print(f"\n[STEP 4] Training shuru ho raha hai...")
print(f"         Yeh 1-2 ghante le sakta hai CPU pe.")
print(f"         Band mat karna! ☕\n")

# DataLoader
train_loader = DataLoader(
    train_examples,
    shuffle    = True,
    batch_size = CONFIG["batch_size"],
)

# Loss — CosineSimilarityLoss
# Query aur relevant document ke vectors close aane chahiye
# Query aur irrelevant document ke vectors door rehne chahiye
loss_fn = losses.CosineSimilarityLoss(model)

# Train
model.fit(
    train_objectives  = [(train_loader, loss_fn)],
    epochs            = CONFIG["epochs"],
    warmup_steps      = CONFIG["warmup_steps"],
    output_path       = CONFIG["save_path"],
    show_progress_bar = True,
    optimizer_params  = {"lr": CONFIG["learning_rate"]},
)

print(f"\n  ✅ Bi-Encoder training complete!")
print(f"  Model saved: {CONFIG['save_path']}")


# ══════════════════════════════════════════════════════════
# STEP 5 — EMBEDDINGS GENERATE KARO
# ══════════════════════════════════════════════════════════

print(f"\n[STEP 5] Case + Statute embeddings generate ho rahe hain...")
print(f"         (Yeh retrieve.py ke liye save honge)\n")

os.makedirs("model/embeddings", exist_ok=True)

# Case embeddings
case_texts = [truncate(t) for t in cases_df["text"].tolist()]
case_ids   = cases_df["case_id"].tolist()

print("         Cases encode ho rahe hain...")
case_embeddings = model.encode(
    case_texts,
    batch_size        = 32,
    show_progress_bar = True,
    convert_to_numpy  = True,
)
np.save("model/embeddings/case_embeddings.npy", case_embeddings)
with open("model/embeddings/case_ids.json", "w") as f:
    json.dump(case_ids, f)
print(f"         ✅ Case embeddings saved: {case_embeddings.shape}")

# Statute embeddings
statute_texts = [truncate(t, max_words=100) for t in statutes_df["text"].tolist()]
statute_ids   = statutes_df["statute_id"].tolist()

print("\n         Statutes encode ho rahe hain...")
statute_embeddings = model.encode(
    statute_texts,
    batch_size        = 32,
    show_progress_bar = True,
    convert_to_numpy  = True,
)
np.save("model/embeddings/statute_embeddings.npy", statute_embeddings)
with open("model/embeddings/statute_ids.json", "w") as f:
    json.dump(statute_ids, f)
print(f"         ✅ Statute embeddings saved: {statute_embeddings.shape}")

# Query embeddings bhi save karo
query_texts = [truncate(t) for t in queries_df["text"].tolist()]
query_ids   = queries_df["query_id"].tolist()

print("\n         Queries encode ho rahe hain...")
query_embeddings = model.encode(
    query_texts,
    batch_size        = 32,
    show_progress_bar = True,
    convert_to_numpy  = True,
)
np.save("model/embeddings/query_embeddings.npy", query_embeddings)
with open("model/embeddings/query_ids.json", "w") as f:
    json.dump(query_ids, f)
print(f"         ✅ Query embeddings saved: {query_embeddings.shape}")


# ══════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════

summary = {
    "model"              : CONFIG["model_name"],
    "epochs"             : CONFIG["epochs"],
    "training_pairs"     : len(train_examples),
    "case_embeddings"    : list(case_embeddings.shape),
    "statute_embeddings" : list(statute_embeddings.shape),
    "query_embeddings"   : list(query_embeddings.shape),
}

with open("model/biencoder_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 55)
print("  BIENCODER COMPLETE! ✅")
print("=" * 55)
print(json.dumps(summary, indent=2))
print("\nAb run karo: python train_crossencoder.py")