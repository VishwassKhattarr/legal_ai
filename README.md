# AI Legal Assistant for Indian Case Law

An end to end AI based legal assistant for analyzing, retrieving, and interpreting Indian Supreme Court judgments using transformer based models and semantic search.

---

## Problem Statement

Legal documents are long, complex, and context heavy. Traditional keyword search fails because the same legal concept can be expressed in many ways, and relevance depends on legal reasoning rather than simple text matching.

This system addresses that by combining semantic retrieval, domain aware filtering, transformer reranking, and decision support reasoning.

---

## Features

Rhetorical role aware retrieval  
Chunks are tagged with 13 legal roles such as FAC, RATIO, RPC, STA and searched selectively based on query intent  

Intent detection  
Automatically identifies query type such as bail, murder, rape, cheating, property and routes to relevant indexes  

Fine tuned legal encoder  
Bi encoder based on InLegalBERT fine tuned on AILA 2019 relevance judgments  

FAISS role specific indexes  
Multiple indexes for fast and targeted retrieval  

Cross encoder reranking  
Improves precision by reranking retrieved results  

Structured legal answers  
Outputs concise legal insights from retrieved cases  

Decision support  
Provides trend analysis such as relief likely, unlikely or mixed  

Explainability  
Each result is linked with reasoning extracted from case text  

---

## Tech Stack

Bi encoder retrieval  
law ai InLegalBERT fine tuned or Hugging Face hosted model  

Cross encoder reranking  
cross encoder ms marco MiniLM L 12 v2  

Vector search  
FAISS  

Rhetorical role tagging  
Rule based tagging system with 13 roles  

Fine tuning loss  
Triplet loss  

Dataset  
AILA FIRE 2019  

Language  
Python  

---

## System Pipeline

User Query  
Intent Detection  
Role specific index selection  
Bi encoder retrieval  
Cross encoder reranking  
Answer generation  
Decision support  
Explainable results  

---

## Project Structure

legal_ai/  
assistant.py              main assistant pipeline  
preprocess.py             data preprocessing  
tag_roles.py              rhetorical role tagging  
build_role_index.py       FAISS index creation  
prepare_training_data.py  triplet generation  
fintune.py                model fine tuning  
search.py                 search utilities  
rerank.py                 reranking utilities  
app.py                    application entry  
download_data.py          dataset download helper  
requirements.txt          dependencies  
data/                     processed data and chunks  
faiss_indexes/            generated indexes  
models/                   local models if used  

---

## Setup and Run

### 1 Clone the repository
```bash
git clone https://github.com/VishwassKhattarr/legal_ai.git
cd legal_ai

### 2 Create environment
conda create -n legal_ai python=3.11 -y
conda activate legal_ai
pip install -r requirements.txt

### 3 Download dataset
Download the dataset from Kaggle
https://www.kaggle.com/datasets/ananyapam7/legalai

Place the extracted files inside:
data/raw/

### 4 Preprocess and build indexes
python preprocess.py
python tag_roles.py
python build_role_index.py
This step is mandatory. Without FAISS indexes, the system will not return any results.

### 5 Optional fine tuning
python prepare_training_data.py
python fintune.py

### 6 Run the assistant
python assistant.py

## Pretrained Model
A pretrained model is available on Hugging Face and can be used directly without training.

https://huggingface.co/Sayyam9/legal-bert-aila-finetuned

Set in code:
MODEL_NAME = "Sayyam9/legal-bert-aila-finetuned"

## Important Notes

The system will not work unless the dataset is downloaded and FAISS indexes are built.

faiss_indexes folder is not included in the repository and must be generated locally.

No results found usually indicates missing indexes or missing data.



## Model Performance

### Fine tuned InLegalBERT
Triplet accuracy approximately 96.67 percent

### Rhetorical role tagging
Coverage approximately 56.9 percent

### Retrieval
Returns diverse relevant cases per query


### Contributions
Rhetorical role aware retrieval integrated with semantic search
Intent driven index routing
Fine tuning on legal relevance dataset
Hybrid ranking combining retrieval and reranking
Decision support layer based on precedent trends
Explainable outputs using structured reasoning

### References
AILA FIRE 2019 dataset
Legal BERT research
LREC rhetorical role corpus
SemEval rhetorical role labeling

### Contributors
Vishwass Khattarr
Sayyam Wad







