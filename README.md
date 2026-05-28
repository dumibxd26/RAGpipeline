# RAG Pipeline - MDAD Assignment 2

Hybrid retrieval-augmented generation pipeline over three BEIR datasets:
NFCorpus, SciFact, and FiQA. The project uses Weaviate as the vector database
in BYO-vectors mode, `all-MiniLM-L6-v2` for embeddings, and a local Ollama LLM
for generation.

## System Requirements

- Python 3.10+
- Docker + Docker Compose
- Ollama for the RAG generation experiments

## Setup

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
docker compose up -d            # starts Weaviate on :8080 / :50051

# Ollama, one-time setup
winget install Ollama.Ollama
ollama serve                    # keep the daemon running in the background
ollama pull gemma2:2b
```

## Full Reproduction Flow

```powershell
# Requirement 1 - index all datasets in Weaviate
python -m src.indexing

# Requirement 2 - evaluate BM25 / dense / hybrid retrieval
python -m src.evaluate

# Requirement 3 - RAG on FiQA with 10 fixed queries
python -m src.rag --dataset fiqa --retriever dense --top-k 5 --num-queries 10 --seed 42 `
    --out results/rag_fiqa_dense_k5_baseline.json

# Requirement 4 - calibration sweeps
python -m src.sweep alpha
python -m src.sweep chunk --chunk-sizes 256 512 1024
python -m src.sweep topk --topks 3 5 10
```

Generated experiment artifacts are written to `results/` as CSV, JSON and PNG
files.

## Chunking Strategy

Documents are split into fixed-size chunks of **256 tokens** with 10% overlap,
using the embedding model tokenizer rather than whitespace splitting. This
matches the `max_seq_length=256` limit of MiniLM. Requirement 4 also tests
512-token and 1024-token chunks to measure the effect of larger passages.

## Project Structure

```text
docker-compose.yml      # Weaviate 1.37.4
requirements.txt
src/
  config.py             # global constants and collection naming
  data.py               # BEIR download/load helpers
  chunking.py           # token-level chunking with overlap
  weaviate_client.py    # Weaviate connection and collection setup
  indexing.py           # Requirement 1 indexing pipeline
  retrievers.py         # BM25 / dense / hybrid retrieval
  metrics.py            # Recall@k, MRR@k, nDCG@k
  evaluate.py           # Requirement 2 evaluation
  rag.py                # Requirement 3 RAG pipeline via Ollama
  sweep.py              # Requirement 4 alpha / chunk_size / top-k sweeps
results/                # generated experiment artifacts
datasets/               # local BEIR cache
```

## Design Notes

- **BYO-vectors**: Weaviate does not vectorize documents itself. Embeddings are
  produced locally with MiniLM and sent explicitly during indexing.
- **Chunk-to-document aggregation**: Retrieval happens at chunk level, while
  BEIR qrels are document-level. Evaluation collapses chunks to `doc_id` using
  the best chunk score per document.
- **Hybrid retrieval**: Weaviate's built-in hybrid query combines BM25 and dense
  vector search. The default is alpha=0.5; the FiQA sweep finds alpha=0.75 to be
  stronger.
- **HNSW**: Weaviate's default HNSW settings are used, as requested by the
  assignment.

## Quick Custom Query Demo

```powershell
python -m src.rag `
  --dataset fiqa `
  --retriever hybrid `
  --alpha 0.75 `
  --top-k 5 `
  --question "How should I rebalance my retirement portfolio?" `
  --out results/demo_fiqa_rebalancing.json
```

The output shows the retrieved chunks and the final cited answer produced by
the local Ollama model.