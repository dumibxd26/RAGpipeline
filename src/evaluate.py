"""Compare BM25 / Dense / Hybrid on all 3 BEIR datasets.

For each (dataset, retriever) pair we compute Recall@10, MRR@10 and nDCG@10
averaged over the test queries, and dump the results to CSV plus a printed
table.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .config import DATASETS, EMBED_MODEL, collection_name
from .data import load
from .metrics import mrr_at_k, ndcg_at_k, recall_at_k
from .retrievers import bm25_search, dense_search, hybrid_search
from .weaviate_client import connect

K = 10
RETRIEVERS = ("bm25", "dense", "hybrid")
METRICS = ("recall@10", "mrr@10", "ndcg@10")


def evaluate_dataset(coll, model: SentenceTransformer, queries: dict, qrels: dict):
    qids = [qid for qid in queries if qid in qrels and qrels[qid]]
    q_texts = [queries[qid] for qid in qids]
    q_vecs = model.encode(
        q_texts, batch_size=64, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=True,
    )

    sums = {r: defaultdict(float) for r in RETRIEVERS}
    n = 0
    for qid, qtext, qvec in tqdm(list(zip(qids, q_texts, q_vecs)), desc="queries"):
        rel = qrels[qid]
        runs = {
            "bm25":   [d for d, _, _ in bm25_search(coll, qtext, k=K)],
            "dense":  [d for d, _, _ in dense_search(coll, qvec.tolist(), k=K)],
            "hybrid": [d for d, _, _ in hybrid_search(coll, qtext, qvec.tolist(), k=K)],
        }
        for r, retrieved in runs.items():
            sums[r]["recall@10"] += recall_at_k(retrieved, rel, K)
            sums[r]["mrr@10"]    += mrr_at_k(retrieved, rel, K)
            sums[r]["ndcg@10"]   += ndcg_at_k(retrieved, rel, K)
        n += 1

    return {
        r: {m: sums[r][m] / max(n, 1) for m in METRICS} for r in RETRIEVERS
    }, n


def print_table(results: dict):
    print("\n" + "=" * 100)
    print("Retriever comparison (rows = retriever, columns = dataset / metric)")
    print("=" * 100)
    header = f"{'retriever':<10}"
    for ds in DATASETS:
        for m in ("R@10", "MRR", "nDCG"):
            header += f" {ds[:8]+'/'+m:>16}"
    print(header)
    print("-" * len(header))
    for r in RETRIEVERS:
        row = f"{r:<10}"
        for ds in DATASETS:
            for m in METRICS:
                row += f" {results[ds][r][m]:>16.4f}"
        print(row)


def save_csv(results: dict, n_queries: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["retriever"]
        for ds in DATASETS:
            for m in METRICS:
                header.append(f"{ds}_{m}")
        w.writerow(header)
        for r in RETRIEVERS:
            row = [r]
            for ds in DATASETS:
                for m in METRICS:
                    row.append(f"{results[ds][r][m]:.4f}")
            w.writerow(row)
        w.writerow([])
        w.writerow(["#queries"] + [str(n_queries[ds]) for ds in DATASETS for _ in METRICS])


def main():
    model = SentenceTransformer(EMBED_MODEL)
    client = connect()
    results: dict = {}
    n_queries: dict = {}
    try:
        for ds in DATASETS:
            print(f"\n=== {ds} ===")
            _, queries, qrels = load(ds, split="test")
            coll = client.collections.get(collection_name(ds))
            res, n = evaluate_dataset(coll, model, queries, qrels)
            results[ds] = res
            n_queries[ds] = n
            for r, m in res.items():
                print(f"  {r:<6}  R@10={m['recall@10']:.4f}  MRR={m['mrr@10']:.4f}  nDCG={m['ndcg@10']:.4f}")
    finally:
        client.close()

    print_table(results)

    out_dir = Path("results")
    save_csv(results, n_queries, out_dir / "retriever_comparison.csv")
    (out_dir / "retriever_comparison.json").write_text(
        json.dumps({"results": results, "n_queries": n_queries}, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved: {out_dir/'retriever_comparison.csv'}")


if __name__ == "__main__":
    main()
