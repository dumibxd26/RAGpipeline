"""Three retrieval strategies over a Weaviate collection:
BM25-only, dense-only (vector search) and hybrid (Weaviate built-in fusion).

Each function returns the top-k *documents* (not chunks): we fetch a larger
window of chunks and collapse to doc_id keeping the best chunk score per doc.
"""
from typing import List, Tuple

from weaviate.classes.query import MetadataQuery

CHUNK_FETCH = 100  # how many chunks to fetch before collapsing to doc_id


def _aggregate(objs) -> List[Tuple[str, float, int]]:
    """Collapse chunk-level results to (doc_id, best_score, best_chunk_index)."""
    best: dict[str, Tuple[float, int]] = {}
    for o in objs:
        doc_id = o.properties["doc_id"]
        chunk_idx = int(o.properties["chunk_index"])
        meta = o.metadata
        # BM25 + hybrid expose .score; near_vector exposes .distance
        if meta.score is not None:
            score = float(meta.score)
        elif meta.distance is not None:
            score = -float(meta.distance)  # smaller distance = higher rank
        else:
            score = 0.0
        prev = best.get(doc_id)
        if prev is None or score > prev[0]:
            best[doc_id] = (score, chunk_idx)
    return sorted(
        ((d, s, ci) for d, (s, ci) in best.items()),
        key=lambda x: -x[1],
    )


def bm25_search(coll, query: str, k: int = 10, chunk_fetch: int = CHUNK_FETCH):
    res = coll.query.bm25(
        query=query,
        limit=chunk_fetch,
        return_metadata=MetadataQuery(score=True),
    )
    return _aggregate(res.objects)[:k]


def dense_search(coll, query_vec, k: int = 10, chunk_fetch: int = CHUNK_FETCH):
    res = coll.query.near_vector(
        near_vector=list(query_vec),
        limit=chunk_fetch,
        return_metadata=MetadataQuery(distance=True),
    )
    return _aggregate(res.objects)[:k]


def hybrid_search(
    coll,
    query: str,
    query_vec,
    k: int = 10,
    alpha: float = 0.5,
    chunk_fetch: int = CHUNK_FETCH,
):
    res = coll.query.hybrid(
        query=query,
        vector=list(query_vec),
        alpha=alpha,
        limit=chunk_fetch,
        return_metadata=MetadataQuery(score=True),
    )
    return _aggregate(res.objects)[:k]


def fetch_chunks_for_rag(
    coll,
    query: str,
    query_vec,
    top_k_chunks: int,
    retriever: str = "hybrid",
    alpha: float = 0.5,
):
    """Return raw top-k *chunks* (not collapsed) for RAG context building.
    Each item is (doc_id, chunk_index, text).
    """
    if retriever == "bm25":
        res = coll.query.bm25(
            query=query, limit=top_k_chunks,
            return_metadata=MetadataQuery(score=True),
        )
    elif retriever == "dense":
        res = coll.query.near_vector(
            near_vector=list(query_vec), limit=top_k_chunks,
            return_metadata=MetadataQuery(distance=True),
        )
    elif retriever == "hybrid":
        res = coll.query.hybrid(
            query=query, vector=list(query_vec), alpha=alpha,
            limit=top_k_chunks,
            return_metadata=MetadataQuery(score=True),
        )
    else:
        raise ValueError(f"unknown retriever: {retriever}")
    return [
        (o.properties["doc_id"], int(o.properties["chunk_index"]), o.properties["text"])
        for o in res.objects
    ]
