import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import math
import random
import torch
import numpy as np
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from tqdm import tqdm

# ── Config ───────────────────────────────────────────────────────────────────
TRAINING_DATA_PATH = "data/training/training_pairs.json"
BASE_MODEL         = "law-ai/InLegalBERT"
OUTPUT_MODEL_DIR   = "models/finetuned_legal_bert"
EPOCHS             = 10
BATCH_SIZE         = 8   # safe for 8GB VRAM
WARMUP_RATIO       = 0.1
EVAL_SPLIT         = 0.15  # 15% for evaluation

os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

# ── Load training data ────────────────────────────────────────────────────────
print("Loading training data...")
with open(TRAINING_DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

triplets = data["triplets"]
print(f"Total triplets: {len(triplets)}")

# ── Shuffle and split ─────────────────────────────────────────────────────────
random.shuffle(triplets)
split_idx  = int(len(triplets) * (1 - EVAL_SPLIT))
train_data = triplets[:split_idx]
eval_data  = triplets[split_idx:]
print(f"Train: {len(train_data)} | Eval: {len(eval_data)}")

# ── Build InputExamples ───────────────────────────────────────────────────────
print("\nBuilding training examples...")
train_examples = []
for t in train_data:
    train_examples.append(InputExample(
        texts=[t["query"], t["positive"], t["negative"]]
    ))

eval_examples = []
for t in eval_data:
    eval_examples.append(InputExample(
        texts=[t["query"], t["positive"], t["negative"]]
    ))

# ── Load model ────────────────────────────────────────────────────────────────
print(f"\nLoading base model: {BASE_MODEL}")
model = SentenceTransformer(BASE_MODEL, device="cuda")
print("Model loaded on GPU ✅")

# ── DataLoader ────────────────────────────────────────────────────────────────
train_dataloader = DataLoader(
    train_examples,
    shuffle=True,
    batch_size=BATCH_SIZE
)

# ── Loss function ─────────────────────────────────────────────────────────────
# TripletLoss: pushes positive closer, negative further
train_loss = losses.TripletLoss(model=model)

# ── Evaluator ─────────────────────────────────────────────────────────────────
# Use TripletEvaluator to track improvement during training
from sentence_transformers.evaluation import TripletEvaluator

anchors   = [e.texts[0] for e in eval_examples]
positives = [e.texts[1] for e in eval_examples]
negatives = [e.texts[2] for e in eval_examples]

evaluator = TripletEvaluator(
    anchors=anchors,
    positives=positives,
    negatives=negatives,
    name="legal_eval"
)

# ── Warmup steps ──────────────────────────────────────────────────────────────
total_steps  = len(train_dataloader) * EPOCHS
warmup_steps = math.ceil(total_steps * WARMUP_RATIO)
print(f"\nTotal training steps : {total_steps}")
print(f"Warmup steps         : {warmup_steps}")
print(f"Epochs               : {EPOCHS}")
print(f"Batch size           : {BATCH_SIZE}")

# ── Train ─────────────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("  STARTING FINE-TUNING")
print("="*50 + "\n")

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator,
    epochs=EPOCHS,
    warmup_steps=warmup_steps,
    output_path=OUTPUT_MODEL_DIR,
    save_best_model=True,
    show_progress_bar=True,
    evaluation_steps=len(train_dataloader),  # evaluate every epoch
)

print("\n" + "="*50)
print("  FINE-TUNING COMPLETE ✅")
print("="*50)
print(f"\nBest model saved to: {OUTPUT_MODEL_DIR}")

# ── Quick evaluation ──────────────────────────────────────────────────────────
print("\nRunning final evaluation...")
final_score = evaluator(model)
print(f"Final triplet accuracy: {final_score:.4f}")
print("\n✅ Model is ready for use!")
print(f"Load it with: SentenceTransformer('{OUTPUT_MODEL_DIR}')")