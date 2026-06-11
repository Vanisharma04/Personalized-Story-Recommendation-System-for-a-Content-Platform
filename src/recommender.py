"""Serving facade: load all artifacts once, recommend for any user on demand.

Used by the Streamlit dashboard; equally usable from a notebook or API:

    from src.recommender import Recommender
    recs = Recommender().recommend("U00042")
"""
import pickle

import numpy as np
import pandas as pd

from src import config
from src.ranking.features import FeatureBuilder
from src.ranking.train_ranker import RANKER_PATH, score_candidates
from src.retrieval.faiss_store import CandidateRetriever
from src.utils.io import load_npy


class Recommender:
    def __init__(self):
        self.train = pd.read_parquet(config.PROCESSED_DIR / "train.parquet")
        self.stories = pd.read_csv(config.RAW_DIR / "stories.csv")
        self.users = pd.read_csv(config.RAW_DIR / "users.csv")

        self.user_vecs = load_npy(config.PROCESSED_DIR / "user_tower_vectors.npy")
        user_ids = pd.read_csv(config.PROCESSED_DIR / "user_index.csv")["user_id"]
        self.user_pos = {u: i for i, u in enumerate(user_ids)}

        self.retriever = CandidateRetriever()
        self.feature_builder = FeatureBuilder(self.train, self.stories, self.users)
        with open(RANKER_PATH, "rb") as f:
            self.ranker = pickle.load(f)

        self.story_meta = self.stories.set_index("story_id")
        self.seen = self.train.groupby("user_id")["story_id"].apply(set).to_dict()

    def recommend(self, user_id: str,
                  n: int = config.TOP_N_RECOMMENDATIONS) -> pd.DataFrame:
        """Two-stage recommendation: FAISS retrieval -> LightGBM ranking."""
        if user_id not in self.user_pos:
            raise KeyError(f"unknown user_id: {user_id}")

        cand_ids, sims = self.retriever.retrieve(
            self.user_vecs[self.user_pos[user_id]],
            k=config.TOP_K_CANDIDATES,
            exclude=self.seen.get(user_id, set()))

        feats = self.feature_builder.build(user_id, cand_ids, sims)
        scores = score_candidates(self.ranker, feats)
        order = np.argsort(-scores)

        rows, seen_titles = [], set()
        for i in order:
            if len(rows) == n:
                break
            sid = cand_ids[i]
            meta = self.story_meta.loc[sid]
            # the catalogue contains same-title stories by different authors;
            # keep the final list diverse by showing each title once
            if meta["title"] in seen_titles:
                continue
            seen_titles.add(meta["title"])
            rows.append({
                "rank": len(rows) + 1,
                "story_id": sid,
                "title": meta["title"],
                "genre": meta["genre"],
                "author": meta["author"],
                "similarity": round(sims[i], 4),
                "rank_score": round(float(scores[i]), 4),
                "description": meta["description"],
            })
        return pd.DataFrame(rows)

    def user_history(self, user_id: str, n: int = 15) -> pd.DataFrame:
        hist = (self.train[self.train["user_id"] == user_id]
                .sort_values("timestamp", ascending=False).head(n))
        return hist.merge(self.stories[["story_id", "title", "genre", "author"]],
                          on="story_id")
