"""Requirement 4 - Hyperparameter sweep on the chosen dataset (FiQA).

We sweep three knobs:
  * alpha (hybrid fusion weight): {0.0, 0.25, 0.5, 0.75, 1.0}    quantitative, no re-indexing required
  * chunk_size: {256, 512, 1024}                                  quantitative, requires re-indexing
  * top_k (chunks fed to the LLM): {3, 5, 10}                     qualitative, evaluated via RAG

The quantitative sweeps produce nDCG@10 (plus Recall@10 and MRR for context)
saved as CSV and a matplotlib PNG. The top_k sweep just dumps the RAG outputs
so we can compare answers manually.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .config import CHUNK_OVERLAP_RATIO, EMBED_MODEL, collection_name
from .data import load
from .metrics import mrr_at_k, ndcg_at_k, recall_at_k
from .retrievers import bm25_search, dense_search, hybrid_search
from .weaviate_client import connect

K_EVAL = 10
DATASET = "fiqa"


def _eval_runs(coll, model, queries, qrels, alphas):
    """Evaluate BM25, dense, and hybrid (for each alpha) on a single collection."""
    qids = [q for q in queries if qrels.get(q)]
    q_texts = [queries[q] for q in qids]
    q_vecs = model.encode(
        q_texts, batch_size=64, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=True,
    )

    metric_names = ("recall@10", "mrr@10", "ndcg@10")
    configs = ["bm25", "dense"] + [f"hybrid_a{a}" for a in alphas]
    sums = {c: defaultdict(float) for c in configs}

    for qid, qtext, qvec in tqdm(list(zip(qids, q_texts, q_vecs)), desc="queries"):
        rel = qrels[qid]
        runs = {
            "bm25":  [d for d, _, _ in bm25_search(coll, qtext, k=K_EVAL)],
            "dense": [d for d, _, _ in dense_search(coll, qvec.tolist(), k=K_EVAL)],
        }
        for a in alphas:
            runs[f"hybrid_a{a}"] = [
                d for d, _, _ in hybrid_search(coll, qtext, qvec.tolist(), k=K_EVAL, alpha=a)
            ]
        for cfg, retrieved in runs.items():
            sums[cfg]["recall@10"] += recall_at_k(retrieved, rel, K_EVAL)
            sums[cfg]["mrr@10"]    += mrr_at_k(retrieved, rel, K_EVAL)
            sums[cfg]["ndcg@10"]   += ndcg_at_k(retrieved, rel, K_EVAL)

    n = len(qids)
    return {
        cfg: {m: sums[cfg][m] / max(n, 1) for m in metric_names} for cfg in configs
    }, n


def sweep_alpha(args):
    """Sweep hybrid alpha at fixed chunk_size (uses existing Beir_Fiqa collection)."""
    print("\n### Sweep alpha (chunk_size = current index) ###")
    model = SentenceTransformer(EMBED_MODEL)
    client = connect()
    try:
        _, queries, qrels = load(DATASET, split="test")
        coll = client.collections.get(collection_name(DATASET))
        alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
        results, n = _eval_runs(coll, model, queries, qrels, alphas)
    finally:
        client.close()

    out_dir = Path("results"); out_dir.mkdir(exist_ok=True)
    out_json = out_dir / "sweep_alpha.json"
    out_json.write_text(json.dumps({"results": results, "n_queries": n}, indent=2), encoding="utf-8")

    # Plot nDCG@10 vs alpha (including BM25 and dense as reference lines)
    xs = alphas
    ys = [results[f"hybrid_a{a}"]["ndcg@10"] for a in alphas]
    bm25_ndcg = results["bm25"]["ndcg@10"]
    dense_ndcg = results["dense"]["ndcg@10"]

    plt.figure(figsize=(7, 4.2))
    plt.plot(xs, ys, marker="o", linewidth=2, label="hybrid(alpha)")
    plt.axhline(bm25_ndcg, color="tab:red", linestyle="--", label=f"BM25 ({bm25_ndcg:.3f})")
    plt.axhline(dense_ndcg, color="tab:green", linestyle="--", label=f"dense ({dense_ndcg:.3f})")
    plt.xlabel("alpha (0 = BM25, 1 = dense)")
    plt.ylabel("nDCG@10")
    plt.title(f"FiQA: hybrid alpha sweep (n={n} queries)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "sweep_alpha.png", dpi=140)
    plt.close()

    print(f"  bm25  nDCG@10 = {bm25_ndcg:.4f}")
    for a, y in zip(xs, ys):
        print(f"  hybrid alpha={a}  nDCG@10 = {y:.4f}")
    print(f"  dense nDCG@10 = {dense_ndcg:.4f}")
    print(f"  -> saved {out_json}, results/sweep_alpha.png")


def _collection_for(chunk_size: int) -> str:
    return f"{collection_name(DATASET)}_cs{chunk_size}"


def sweep_chunk_size_clean(args):
    """Re-index FiQA at each chunk size into a dedicated collection, then evaluate."""
    from .indexing import build_chunks
    from .weaviate_client import ensure_collection
    import time

    print("\n### Sweep chunk_size ###")
    sizes = args.chunk_sizes
    model = SentenceTransformer(EMBED_MODEL)
    client = connect()
    overlap_ratio = CHUNK_OVERLAP_RATIO

    per_size_results = {}
    n_queries_total = None
    try:
        _, queries, qrels = load(DATASET, split="test")

        # Pre-encode queries once (same model = same vectors)
        qids = [q for q in queries if qrels.get(q)]
        q_texts = [queries[q] for q in qids]
        q_vecs = model.encode(
            q_texts, batch_size=64, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True,
        )
        n_queries_total = len(qids)

        corpus, _, _ = load(DATASET, split="test")

        for cs in sizes:
            target = _collection_for(cs)
            overlap = max(1, int(round(cs * overlap_ratio)))
            print(f"\n-- re-indexing {target} (chunk_size={cs}, overlap={overlap}) --")

            records = build_chunks(corpus, model.tokenizer, cs, overlap)
            print(f"   chunks: {len(records)}")

            t0 = time.time()
            texts = [r[3] for r in records]
            vectors = model.encode(
                texts, batch_size=64, show_progress_bar=True,
                convert_to_numpy=True, normalize_embeddings=True,
            )
            print(f"   encoded {len(vectors)} in {time.time()-t0:.0f}s")

            coll = ensure_collection(client, target, recreate=True)
            t0 = time.time()
            with coll.batch.dynamic() as batch:
                for (doc_id, idx, title, text), vec in zip(records, vectors):
                    batch.add_object(
                        properties={"doc_id": doc_id, "chunk_index": int(idx),
                                    "title": title, "text": text},
                        vector=vec.tolist(),
                    )
            failed = len(coll.batch.failed_objects)
            total = coll.aggregate.over_all(total_count=True).total_count
            print(f"   inserted {total} (failed={failed}) in {time.time()-t0:.0f}s")

            # Evaluate hybrid at default alpha=0.5, plus BM25 and dense, on this collection
            print("   evaluating...")
            sums = defaultdict(lambda: defaultdict(float))
            for qid, qtext, qvec in tqdm(list(zip(qids, q_texts, q_vecs)), desc=f"eval cs={cs}"):
                rel = qrels[qid]
                runs = {
                    "bm25":   [d for d, _, _ in bm25_search(coll, qtext, k=K_EVAL)],
                    "dense":  [d for d, _, _ in dense_search(coll, qvec.tolist(), k=K_EVAL)],
                    "hybrid": [d for d, _, _ in hybrid_search(coll, qtext, qvec.tolist(), k=K_EVAL, alpha=0.5)],
                }
                for c, retrieved in runs.items():
                    sums[c]["recall@10"] += recall_at_k(retrieved, rel, K_EVAL)
                    sums[c]["mrr@10"]    += mrr_at_k(retrieved, rel, K_EVAL)
                    sums[c]["ndcg@10"]   += ndcg_at_k(retrieved, rel, K_EVAL)
            n = len(qids)
            per_size_results[cs] = {
                c: {m: sums[c][m] / n for m in ("recall@10", "mrr@10", "ndcg@10")}
                for c in ("bm25", "dense", "hybrid")
            }
            for c, m in per_size_results[cs].items():
                print(f"   cs={cs} {c:<6}  R@10={m['recall@10']:.4f}  MRR={m['mrr@10']:.4f}  nDCG={m['ndcg@10']:.4f}")
    finally:
        client.close()

    out_dir = Path("results"); out_dir.mkdir(exist_ok=True)
    (out_dir / "sweep_chunk_size.json").write_text(
        json.dumps({"results": per_size_results, "n_queries": n_queries_total}, indent=2),
        encoding="utf-8",
    )

    plt.figure(figsize=(7, 4.2))
    for ret in ("bm25", "dense", "hybrid"):
        ys = [per_size_results[cs][ret]["ndcg@10"] for cs in sizes]
        plt.plot(sizes, ys, marker="o", linewidth=2, label=ret)
    plt.xlabel("chunk_size (tokens)")
    plt.ylabel("nDCG@10")
    plt.title(f"FiQA: chunk_size sweep (n={n_queries_total} queries)")
    plt.xticks(sizes)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "sweep_chunk_size.png", dpi=140)
    plt.close()
    print(f"\n  -> saved results/sweep_chunk_size.json + .png")


def sweep_topk_rag(args):
    """Re-run the RAG pipeline at several top-k values on the same 10 queries."""
    import subprocess
    for k in args.topks:
        out = f"results/rag_fiqa_dense_k{k}.json"
        print(f"\n-- RAG run with top_k={k} -> {out} --")
        cmd = [
            ".\\venv\\Scripts\\python.exe", "-m", "src.rag",
            "--dataset", "fiqa", "--retriever", "dense",
            "--top-k", str(k),
            "--llm", args.llm, "--num-queries", "10", "--seed", "42",
            "--out", out,
        ]
        subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_alpha = sub.add_parser("alpha")
    p_alpha.set_defaults(func=sweep_alpha)

    p_cs = sub.add_parser("chunk")
    p_cs.add_argument("--chunk-sizes", nargs="+", type=int, default=[256, 512, 1024],
                      dest="chunk_sizes")
    p_cs.set_defaults(func=sweep_chunk_size_clean)

    p_k = sub.add_parser("topk")
    p_k.add_argument("--topks", nargs="+", type=int, default=[3, 5, 10])
    p_k.add_argument("--llm", default="gemma2:2b")
    p_k.set_defaults(func=sweep_topk_rag)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
