"""Indexing into Weaviate.

For each BEIR dataset:
  1. load the corpus,
  2. split documents into fixed-size chunks (with overlap),
  3. encode chunks with sentence-transformers (BYO vectors),
  4. insert (text + vector) into a dedicated Weaviate collection.
BM25 over the chunk text is indexed automatically by Weaviate.
"""
from __future__ import annotations

import argparse
import time
from typing import List, Tuple

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .chunking import chunk_text
from .config import (
    CHUNK_OVERLAP_RATIO,
    CHUNK_SIZE,
    DATASETS,
    EMBED_MODEL,
    collection_name,
)
from .data import load
from .weaviate_client import connect, ensure_collection


def build_chunks(corpus: dict, tokenizer, chunk_size: int, overlap: int):
    """Return a list of (doc_id, chunk_index, title, text) tuples."""
    records: List[Tuple[str, int, str, str]] = []
    for doc_id, doc in corpus.items():
        title = (doc.get("title") or "").strip()
        body = (doc.get("text") or "").strip()
        full = (title + ". " + body) if title else body
        for i, ch in enumerate(chunk_text(tokenizer, full, chunk_size, overlap)):
            records.append((doc_id, i, title, ch))
    return records


def index_dataset(
    client,
    model: SentenceTransformer,
    dataset: str,
    chunk_size: int = CHUNK_SIZE,
    overlap_ratio: float = CHUNK_OVERLAP_RATIO,
    batch_encode: int = 64,
) -> None:
    overlap = max(1, int(round(chunk_size * overlap_ratio)))
    print(f"\n=== {dataset} | chunk_size={chunk_size} overlap={overlap} ===")

    t0 = time.time()
    corpus, _, _ = load(dataset, split="test")
    print(f"  corpus: {len(corpus)} documents (load {time.time()-t0:.1f}s)")

    t0 = time.time()
    records = build_chunks(corpus, model.tokenizer, chunk_size, overlap)
    print(f"  chunks: {len(records)} (build {time.time()-t0:.1f}s)")

    t0 = time.time()
    texts = [r[3] for r in records]
    vectors = model.encode(
        texts,
        batch_size=batch_encode,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    print(f"  embed: {len(vectors)} vectors (encode {time.time()-t0:.1f}s)")

    name = collection_name(dataset)
    coll = ensure_collection(client, name, recreate=True)

    t0 = time.time()
    with coll.batch.dynamic() as batch:
        for (doc_id, idx, title, text), vec in zip(records, vectors):
            batch.add_object(
                properties={
                    "doc_id": doc_id,
                    "chunk_index": int(idx),
                    "title": title,
                    "text": text,
                },
                vector=vec.tolist(),
            )
    failed = coll.batch.failed_objects
    total = coll.aggregate.over_all(total_count=True).total_count
    print(
        f"  insert: {total} objects in Weaviate "
        f"(failed={len(failed)}, took {time.time()-t0:.1f}s)"
    )


def main():
    parser = argparse.ArgumentParser(description="Index BEIR datasets into Weaviate.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DATASETS,
        help=f"Datasets to index (default: {DATASETS})",
    )
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--overlap-ratio", type=float, default=CHUNK_OVERLAP_RATIO)
    args = parser.parse_args()

    print(f"Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    client = connect()
    try:
        for ds in args.datasets:
            index_dataset(
                client,
                model,
                ds,
                chunk_size=args.chunk_size,
                overlap_ratio=args.overlap_ratio,
            )
    finally:
        client.close()
    print("\nDone. All datasets have been indexed.")


if __name__ == "__main__":
    main()
