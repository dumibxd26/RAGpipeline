"""IR evaluation metrics: Recall@k, MRR@k, nDCG@k."""
import math
from typing import Dict, List


def recall_at_k(retrieved: List[str], qrels: Dict[str, int], k: int) -> float:
    """Fraction of relevant documents that show up in the top-k."""
    relevant = {d for d, s in qrels.items() if s > 0}
    if not relevant:
        return 0.0
    hits = sum(1 for d in retrieved[:k] if d in relevant)
    return hits / len(relevant)


def mrr_at_k(retrieved: List[str], qrels: Dict[str, int], k: int) -> float:
    """Reciprocal rank of the first relevant document within the top-k."""
    relevant = {d for d, s in qrels.items() if s > 0}
    for i, d in enumerate(retrieved[:k], start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: List[str], qrels: Dict[str, int], k: int) -> float:
    """Normalized DCG@k using graded relevance from qrels (gain = 2^rel - 1)."""
    gains = [qrels.get(d, 0) for d in retrieved[:k]]
    dcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted(qrels.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0
