# 🧠 AI Legal Assistant — Indian Case Law

An end-to-end AI-based legal assistant for analyzing, retrieving, and interpreting
Indian Supreme Court judgments using transformer-based models and semantic search.

---

## 📌 Problem Statement

Legal documents are long, complex, and context-heavy. Traditional keyword search
fails because the same legal concept can be expressed in many ways, and relevance
depends on legal reasoning — not just text matching.

This system addresses that by combining semantic retrieval, domain-aware filtering,
transformer reranking, and decision-support reasoning.

---

## ✅ Features

- 🔍 **Rhetorical Role-Aware Retrieval** — chunks tagged with 13 legal roles (FAC, RATIO, RPC, STA etc.) and searched selectively based on query intent
- 🎯 **Intent Detection** — automatically identifies query type (bail, murder, rape, cheating, property etc.) and routes to relevant indexes
- 🤖 **Fine-tuned InLegalBERT** — bi-encoder fine-tuned on AILA 2019 relevance judgments (96.67% triplet accuracy)
- ⚡ **FAISS Role-Specific Indexes** — 10 separate indexes for fast, targeted retrieval
- 🏆 **Cross-Encoder Reranking** — reranks candidates for higher precision
- 📋 **Structured Legal Answers** — specific legal position per query type with key principles
- ⚖️ **Decision Support** — trend analysis from retrieved precedents (relief likely / unlikely / mixed)
- 💡 **Explainability** — every result tagged with rhetorical role and legal reasoning

---

## 🧱 Tech Stack

| Component | Technology |
|---|---|
| Bi-Encoder (Retrieval) | `law-ai/InLegalBERT` (fine-tuned) |
| Cross-Encoder (Reranking) | `cross-encoder/ms-marco-MiniLM-L-12-v2` |
| Vector Search | FAISS |
| Rhetorical Role Tagging | Rule-based (13 roles from LREC-2022) |
| Fine-tuning Loss | Triplet Loss (Multiple Negatives Ranking) |
| Data | AILA FIRE 2019 (~2900 cases, 197 statutes) |
| Language | Python |

---

## 🔄 System Pipeline
User Query
│
▼
Intent Detection (bail / murder / cheating / rape / property...)
│
▼
Role-Specific FAISS Index Selection (RPC, RATIO, FAC, STA...)
│
▼
Bi-Encoder Retrieval (Fine-tuned InLegalBERT)
│
▼
Keyword Boosting (IPC section matching)
│
▼
Cross-Encoder Reranking (ms-marco-MiniLM)
│
▼
Answer Generation (intent-specific legal position)
│
▼
Decision Support (granted / denied / mixed trend)
│
▼
Explainable Results (rhetorical role + legal reasoning)

---

## 📂 Project Structure
legal_ai/
├── assistant.py              # Main assistant — full pipeline
├── preprocess.py             # Data preprocessing
├── tag_roles.py              # Rhetorical role tagging (13 roles)
├── build_role_index.py       # Build per-role FAISS indexes
├── prepare_training_data.py  # Create fine-tuning triplets from AILA
├── fintune.py                # Fine-tune InLegalBERT (bi-encoder)
├── search.py                 # Search utilities
├── rerank.py                 # Reranking utilities
├── app.py                    # App entry point
├── download_data.py          # Dataset download script
├── requirements.txt          # Dependencies
├── data/                     # Processed chunks
├── faiss_indexes/            # Role-specific FAISS indexes (gitignored)
└── models/                   # Fine-tuned models (gitignored)

---

## ⚙️ Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/VishwassKhattarr/legal_ai.git
cd legal_ai
```

### 2. Create environment
```bash
conda create -n legal_ai python=3.10 -y
conda activate legal_ai
pip install -r requirements.txt
```

### 3. Download AILA dataset
Download from [Kaggle](https://www.kaggle.com/datasets/ananyapam7/legalai) and place in:
C:\Users<you>\Downloads\aila_data\

### 4. Preprocess & build indexes
```bash
python preprocess.py
python tag_roles.py
python build_role_index.py
```

### 5. Fine-tune the model (optional but recommended)
```bash
python prepare_training_data.py
python fintune.py
```

### 6. Run the assistant
```bash
python assistant.py
```

---

## 📊 Model Performance

| Component | Metric | Score |
|---|---|---|
| Fine-tuned InLegalBERT | Triplet Accuracy (eval) | **96.67%** |
| Rhetorical Role Tagging | Coverage | 56.9% (43.1% NONE) |
| Role-Specific Retrieval | Unique cases per query | 5 diverse cases |

---

## 🔬 Novel Contributions

1. **Rhetorical Role-Aware Retrieval** — first system to combine LREC-2022 rhetorical roles with FAISS retrieval for Indian legal cases
2. **Intent-Driven Index Routing** — query intent mapped to specific role indexes
3. **Domain Fine-Tuning on AILA** — InLegalBERT fine-tuned on real AILA relevance judgments
4. **Hybrid Ranking** — semantic retrieval + keyword boosting + cross-encoder reranking
5. **Decision Support Layer** — precedent trend analysis for legal decision-making
6. **Explainable Outputs** — rhetorical role labels as structured explanations

---

## 📚 References

- AILA FIRE 2019 Dataset
- Kalamkar et al. (2022) — Corpus for Automatic Structuring of Legal Documents (LREC)
- Furniturewala et al. (2021) — Legal Text Classification using Transformers (FIRE)
- Gupta et al. (2023) — Rhetorical Role Labeling using GCN (SemEval)
- Chalkidis et al. (2020) — LEGAL-BERT

---

## 👥 Contributors

- Vishwass Khattarr
- Sayyam Wad