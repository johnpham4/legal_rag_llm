# Vietnamese Legal RAG System

End-to-end Retrieval-Augmented Generation (RAG) system for Vietnamese legal documents, implementing patterns from [LLM Engineer's Handbook](https://github.com/PacktPublishing/LLM-Engineers-Handbook).

## Overview

Production-grade RAG pipeline featuring:
- **Automated Data Ingestion**: Web crawling + parsing Vietnamese legal documents
- **Advanced Retrieval**: Hybrid search (Dense embeddings + Sparse encoders: BM25/TF-IDF)
- **Smart Query Processing**: Self-query metadata extraction, query expansion, cross-encoder reranking
- **Evaluation Framework**: Rigorous retrieval metrics (MRR, Hit Rate@K, NDCG@K, Recall@K) + LLM-as-a-judge

**Performance**: BM25 hybrid achieved **52% MRR** (+13% vs dense-only), **30% top-1 accuracy**, **90% Hit@5** on 10 Vietnamese legal queries.

## Project Structure

```
legal-llm/
├── llm_engineering/
│   ├── application/          # Business logic
│   │   ├── crawlers/         # Web scraping (BeautifulSoup)
│   │   ├── preprocessing/    # Chunking, cleaning, embedding
│   │   ├── networks/         # Embedding models, sparse encoders
│   │   ├── rag/              # Retrieval, reranking, query expansion
│   │   └── evaluation/       # Metrics, LLM judge
│   ├── domain/               # Domain models (DDD)
│   │   └── orm/              # Qdrant, MongoDB integrations
│   └── infrastructure/       # LLM clients, DB connections
├── pipelines/                # ZenML pipelines
│   ├── legal_data_etl.py     # Crawl → Parse → Store
│   ├── feature_engineering.py # Clean → Chunk → Embed → Vector DB
│   └── train_sparse_embedding.py # Train BM25/TF-IDF
├── notebooks/                # Evaluation notebooks
├── data/                     # Legal URLs, test queries
└── configs/                  # Pipeline configurations
```

## Quick Start

### 1. Setup Environment

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Configure environment
cp .env.example .env
# Edit .env with your API keys (Cohere, OpenAI, MongoDB, Qdrant)
```

### 2. Run Pipelines

#### **Pipeline 1: Legal Data ETL**
Crawl Vietnamese legal documents and store to MongoDB.

```bash
uv run python -m pipelines.legal_data_etl
```

**Config**: `configs/legal_data_etl.yaml`
**Input**: `data/labor_law_urls.json` (list of URLs to crawl)
**Output**: MongoDB collection `cleaned_documents`

---

#### **Pipeline 2: Feature Engineering**
Clean, chunk, embed documents → Load to Qdrant vector DB.

```bash
uv run python -m pipelines.feature_engineering
```

**Config**: `configs/feature_engineering.yaml`
**Steps**:
1. Clean documents (remove noise, normalize)
2. Chunk by legal structure (điều khoản-aware)
3. Generate dense embeddings (Sentence-Transformers)
4. Load to Qdrant collection `embedded_chunks`

**Output**: Qdrant vector DB ready for search

---

#### **Pipeline 3: Train Sparse Encoders**
Train BM25 and TF-IDF sparse encoders on legal corpus.

```bash
uv run python -m pipelines.train_sparse_embedding
```

**Config**: `configs/train_sparse_embedding.yaml`
**Output**: Saved models in `models/sparse_encoder_{bm25|tfidf}.pkl`

---

### 3. Evaluate Retrieval

Run evaluation comparing Dense vs Hybrid (BM25/TF-IDF):

```bash
# Evaluate with BM25
uv run python notebooks/001-evaluate_search.py --sparse-encoder bm25

# Evaluate with TF-IDF
uv run python notebooks/001-evaluate_search.py --sparse-encoder tfidf
```

**Output**: Notebook with metrics, charts → `notebooks/outputs/`

## Tech Stack

- **Framework**: Python 3.11, FastAPI, LangChain
- **Vector DB**: Qdrant (hybrid search)
- **Document DB**: MongoDB (metadata)
- **Embeddings**: Sentence-Transformers, Cohere
- **LLMs**: Cohere Command-R, OpenAI GPT-4
- **Orchestration**: ZenML
- **Evaluation**: ranx (IR metrics)

## Key Features

### Retrieval Pipeline
```python
query → Self-Query (metadata extraction)
      → Query Expansion (generate variations)
      → Hybrid Search (Dense + BM25/TF-IDF)
      → Cross-Encoder Reranking
      → Top-K results
```

### Evaluation Metrics
- **MRR (Primary)**: Mean Reciprocal Rank - measures first relevant doc position
- **Hit Rate@K**: % queries with ≥1 relevant doc in top K
- **Recall@K**: % relevant docs found in top K
- **NDCG@K**: Ranking quality score

## Results

| Method | MRR | Top-1 Accuracy | Hit@5 |
|--------|-----|----------------|-------|
| Dense-only | 0.46 | 20% | 90% |
| **BM25 Hybrid** | **0.52** | **30%** | **90%** |
| TF-IDF Hybrid | 0.48 | 30% | 70% |

→ **BM25 Hybrid recommended**: +13% MRR improvement, best balance of precision and recall

## License

MIT

## References

- [LLM Engineer's Handbook](https://github.com/PacktPublishing/LLM-Engineers-Handbook)
- [ranx: Fast Evaluation Library](https://github.com/AmenRa/ranx)
