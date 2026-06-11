"""Offline evaluation of the full two-stage pipeline on the held-out test set.

Stage 1 (retrieval) : Recall@50 of FAISS candidates vs. test interactions
Stage 2 (ranking)   : NDCG@10, MAP, MRR of the ranked Top-10 list

Also produces per-genre quality and a retrieval-vs-ranking comparison that
the dashboard renders. Already-read (train) stories are excluded from every
recommendation list, matching serving behaviour.
"""
import pickle

import numpy as np
import pandas as pd

from src import config
from src.evaluation.metrics import (average_precision, ndcg_at_k, recall_at_k,
                                    reciprocal_rank)
from src.ranking.features import FeatureBuilder
from src.ranking.train_ranker import RANKER_PATH, score_candidates
from src.retrieval.faiss_store import CandidateRetriever
from src.utils.io import load_npy, save_json, setup_logger

log = setup_logger("eval")

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False


def main() -> None:
    train = pd.read_parquet(config.PROCESSED_DIR / "train.parquet")
    test = pd.read_parquet(config.PROCESSED_DIR / "test.parquet")
    stories = pd.read_csv(config.RAW_DIR / "stories.csv")
    users = pd.read_csv(config.RAW_DIR / "users.csv")

    user_vecs = load_npy(config.PROCESSED_DIR / "user_tower_vectors.npy")
    user_ids = pd.read_csv(config.PROCESSED_DIR / "user_index.csv")["user_id"].tolist()
    user_pos = {u: i for i, u in enumerate(user_ids)}

    retriever = CandidateRetriever()
    fb = FeatureBuilder(train, stories, users)
    with open(RANKER_PATH, "rb") as f:
        ranker = pickle.load(f)

    test_sets = test.groupby("user_id")["story_id"].apply(set).to_dict()
    seen_sets = train.groupby("user_id")["story_id"].apply(set).to_dict()
    user_pref = users.set_index("user_id")["preferred_genre"].to_dict()

    recalls, ndcgs, maps, mrrs = [], [], [], []
    sim_ndcgs = []                       # retrieval-order baseline for comparison
    per_genre: dict[str, list] = {}

    for uid, relevant in test_sets.items():
        if uid not in user_pos:
            continue
        cand_ids, sims = retriever.retrieve(
            user_vecs[user_pos[uid]], k=config.TOP_K_CANDIDATES,
            exclude=seen_sets.get(uid, set()))
        if not cand_ids:
            continue

        # ---- Stage 1 metric: retrieval quality
        recalls.append(recall_at_k(cand_ids, relevant, config.RECALL_K))
        sim_ndcgs.append(ndcg_at_k(cand_ids, relevant, config.NDCG_K))

        # ---- Stage 2: re-rank candidates and measure final list quality
        feats = fb.build(uid, cand_ids, sims)
        scores = score_candidates(ranker, feats)
        order = np.argsort(-scores)
        ranked = [cand_ids[i] for i in order]

        ndcg = ndcg_at_k(ranked, relevant, config.NDCG_K)
        ndcgs.append(ndcg)
        maps.append(average_precision(ranked, relevant, k=config.TOP_K_CANDIDATES))
        mrrs.append(reciprocal_rank(ranked, relevant))
        per_genre.setdefault(user_pref.get(uid, "Unknown"), []).append(ndcg)

    metrics = {
        "recall_at_50": float(np.mean(recalls)),
        "ndcg_at_10": float(np.mean(ndcgs)),
        "map": float(np.mean(maps)),
        "mrr": float(np.mean(mrrs)),
        "ndcg_at_10_retrieval_only": float(np.mean(sim_ndcgs)),
        "n_users_evaluated": len(ndcgs),
        "top_k_candidates": config.TOP_K_CANDIDATES,
    }
    genre_quality = {g: float(np.mean(v)) for g, v in sorted(per_genre.items())}

    save_json(metrics, config.REPORTS_DIR / "metrics.json")
    save_json(genre_quality, config.REPORTS_DIR / "genre_ndcg.json")

    if HAS_MLFLOW:
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
        with mlflow.start_run(run_name="offline-eval"):
            mlflow.log_metrics({k: v for k, v in metrics.items()
                                if isinstance(v, float)})

    log.info("=" * 52)
    log.info("Recall@50           : %.4f", metrics["recall_at_50"])
    log.info("NDCG@10  (ranked)   : %.4f", metrics["ndcg_at_10"])
    log.info("NDCG@10  (retrieval): %.4f", metrics["ndcg_at_10_retrieval_only"])
    log.info("MAP                 : %.4f", metrics["map"])
    log.info("MRR                 : %.4f", metrics["mrr"])
    log.info("users evaluated     : %d", metrics["n_users_evaluated"])
    log.info("=" * 52)


if __name__ == "__main__":
    main()
