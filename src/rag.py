"""End-to-end RAG on a single dataset using a local Ollama LLM.

Pipeline per query:
  1. encode the question with the same MiniLM model used at indexing time,
  2. retrieve top-k chunks with the chosen retriever (default: dense on fiqa),
  3. build a prompt that includes the chunks tagged with [doc_id#chunk_idx],
  4. ask the LLM to answer and cite the IDs.

Results (question, retrieved chunks, generated answer, qrels) are dumped as
JSON for manual qualitative analysis.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import ollama
from sentence_transformers import SentenceTransformer

from .config import EMBED_MODEL, collection_name
from .data import load
from .retrievers import fetch_chunks_for_rag
from .weaviate_client import connect

PROMPT_TEMPLATE = """You are a careful assistant. Answer the user's question using ONLY the information in the context chunks below. Every factual statement you make MUST cite the chunk it came from using square brackets like [doc_id#chunk_index]. If the context does not contain enough information to answer, reply exactly: "I don't know based on the provided context."

Context:
{context}

Question: {question}

Answer (with citations):"""


def build_prompt(question: str, chunks) -> str:
    ctx = "\n\n".join(f"[{d}#{i}] {t}" for d, i, t in chunks)
    return PROMPT_TEMPLATE.format(context=ctx, question=question)


def ask_llm(model_name: str, prompt: str) -> str:
    resp = ollama.generate(
        model=model_name,
        prompt=prompt,
        options={"temperature": 0.0, "num_ctx": 4096},
    )
    return resp["response"].strip()


def main():
    parser = argparse.ArgumentParser(description="RAG over a BEIR dataset via Ollama.")
    parser.add_argument("--dataset", default="fiqa")
    parser.add_argument(
        "--retriever", default="dense", choices=["bm25", "dense", "hybrid"]
    )
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="alpha for hybrid retriever (ignored otherwise)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="number of chunks injected into the prompt")
    parser.add_argument("--llm", default="gemma2:2b",
                        help="Ollama model tag (must be pulled locally first)")
    parser.add_argument("--num-queries", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--question", type=str, default=None,
                        help="Ask one custom question instead of sampling BEIR queries")
    args = parser.parse_args()

    if args.question:
        sampled_items = [("custom", args.question, [])]
    else:
        _, queries, qrels = load(args.dataset, split="test")
        eligible_qids = sorted([q for q in queries if qrels.get(q)])
        random.seed(args.seed)
        sampled_qids = random.sample(eligible_qids, min(args.num_queries, len(eligible_qids)))
        sampled_items = [
            (qid, queries[qid], sorted([d for d, s in qrels[qid].items() if s > 0]))
            for qid in sampled_qids
        ]

    print(f"Loading embedding model: {EMBED_MODEL}")
    encoder = SentenceTransformer(EMBED_MODEL)

    print(f"Connecting to Weaviate ({collection_name(args.dataset)})")
    client = connect()

    outputs = []
    try:
        coll = client.collections.get(collection_name(args.dataset))
        for n, (qid, question, relevant_ids) in enumerate(sampled_items, 1):
            qvec = encoder.encode(
                [question], normalize_embeddings=True, convert_to_numpy=True,
            )[0].tolist()
            chunks = fetch_chunks_for_rag(
                coll, question, qvec,
                top_k_chunks=args.top_k,
                retriever=args.retriever,
                alpha=args.alpha,
            )
            prompt = build_prompt(question, chunks)
            try:
                answer = ask_llm(args.llm, prompt)
            except Exception as exc:
                print(f"[!] LLM call failed for {qid}: {exc}", file=sys.stderr)
                answer = f"<LLM error: {exc}>"

            retrieved_ids = [d for d, _, _ in chunks]
            hit = any(d in relevant_ids for d in retrieved_ids) if relevant_ids else None

            print(f"\n--- [{n}/{len(sampled_items)}] qid={qid}  retrieval_hit={hit} ---")
            print(f"Q: {question}")
            if relevant_ids:
                print(f"Relevant doc_ids: {relevant_ids}")
            print(f"Retrieved (top-{args.top_k}): {[(d,i) for d,i,_ in chunks]}")
            print(f"A: {answer}")

            outputs.append({
                "query_id": qid,
                "question": question,
                "relevant_doc_ids": relevant_ids,
                "retrieved": [
                    {"doc_id": d, "chunk_index": i, "text": t}
                    for d, i, t in chunks
                ],
                "retrieval_hit": hit,
                "answer": answer,
            })
    finally:
        client.close()

    out_path = Path(args.out) if args.out else Path(
        f"results/rag_{args.dataset}_{args.retriever}_k{args.top_k}_{args.llm.replace(':','-')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(outputs, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {len(outputs)} RAG runs to {out_path}")


if __name__ == "__main__":
    main()
