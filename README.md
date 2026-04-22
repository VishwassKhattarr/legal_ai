#  AI Legal Assistant (Indian Case Law)

An AI-based legal assistant system for analyzing and retrieving relevant Indian legal case laws using NLP and transformer-based models.

---

##  Features

- Semantic search using Sentence Transformers
- Fast retrieval with FAISS vector index
- Cross-encoder reranking for improved relevance
- Query-aware filtering (IPC sections)
- Explainable results with reasoning
- Decision-support insights based on precedents

---

##  Tech Stack

- Python
- Sentence Transformers (BERT-based models)
- FAISS (vector similarity search)
- Pandas / NumPy

---

##  Pipeline

Query  
→ Query Expansion  
→ FAISS Retrieval  
→ Filtering  
→ Cross-Encoder Reranking  
→ Answer Generation  
→ Decision Insight  

---

##  How to Run

```bash
pip install -r requirements.txt
python assistant.py
