"""Ranking metrics: Recall@K, NDCG@K, Average Precision, Reciprocal Rank.

All functions take a ranked list of recommended ids and a set of relevant
(held-out) ids, and return a float in [0, 1].
"""
import numpy as np


def recall_at_k(ranked: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for item in ranked[:k] if item in relevant)
    return hits / min(len(relevant), k)


def ndcg_at_k(ranked: list, relevant: set, k: int) -> float:
    dcg = sum(1.0 / np.log2(i + 2)
              for i, item in enumerate(ranked[:k]) if item in relevant)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0


def average_precision(ranked: list, relevant: set, k: int | None = None) -> float:
    if not relevant:
        return 0.0
    ranked = ranked[:k] if k else ranked
    hits, score = 0, 0.0
    for i, item in enumerate(ranked):
        if item in relevant:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(relevant), len(ranked))


def reciprocal_rank(ranked: list, relevant: set, k: int | None = None) -> float:
    ranked = ranked[:k] if k else ranked
    for i, item in enumerate(ranked):
        if item in relevant:
            return 1.0 / (i + 1)
    return 0.0
