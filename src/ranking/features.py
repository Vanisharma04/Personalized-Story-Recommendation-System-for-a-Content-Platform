"""Feature engineering for the ranking stage.

A single FeatureBuilder is fitted on historical interactions and produces
identical features at training and serving time:

User features        : reading frequency, mean completion, like rate, mean session
Story features       : popularity, mean completion, reading length, freshness
User x Story features: retrieval similarity, preferred-genre match,
                       historical genre affinity, author affinity
"""
import numpy as np
import pandas as pd

from src import config

FEATURE_COLUMNS = [
    "sim_score",
    "genre_match",
    "genre_affinity",
    "author_affinity",
    "story_popularity",
    "story_avg_completion",
    "story_reading_minutes",
    "story_freshness",
    "user_n_reads",
    "user_avg_completion",
    "user_like_rate",
    "user_avg_session",
]


class FeatureBuilder:
    def __init__(self, history: pd.DataFrame, stories: pd.DataFrame,
                 users: pd.DataFrame):
        self.stories = stories.set_index("story_id")
        self.users = users.set_index("user_id")

        # ---------------- story-level aggregates from history
        agg = history.groupby("story_id").agg(
            popularity=("user_id", "count"),
            avg_completion=("completion_rate", "mean"),
        )
        self.story_popularity = np.log1p(agg["popularity"]).to_dict()
        self.story_avg_completion = agg["avg_completion"].to_dict()
        self.global_avg_completion = float(history["completion_rate"].mean())

        # ---------------- user-level aggregates from history
        uagg = history.groupby("user_id").agg(
            n_reads=("story_id", "count"),
            avg_completion=("completion_rate", "mean"),
            like_rate=("likes", "mean"),
            avg_session=("reading_time", "mean"),
        )
        self.user_stats = uagg.to_dict("index")

        # ---------------- user genre / author affinities
        hist = history.merge(stories[["story_id", "genre", "author"]], on="story_id")
        self.user_genre_share = (
            hist.groupby(["user_id", "genre"]).size()
            / hist.groupby("user_id").size()
        ).to_dict()
        self.user_author_share = (
            hist.groupby(["user_id", "author"]).size()
            / hist.groupby("user_id").size()
        ).to_dict()

    def build(self, user_id: str, candidate_ids: list[str],
              sim_scores: list[float]) -> pd.DataFrame:
        """Feature matrix for one user's candidate list (rows align with input)."""
        ustats = self.user_stats.get(user_id,
                                     {"n_reads": 0, "avg_completion": self.global_avg_completion,
                                      "like_rate": 0.0, "avg_session": 0.0})
        preferred = self.users.at[user_id, "preferred_genre"] \
            if user_id in self.users.index else None

        rows = []
        for sid, sim in zip(candidate_ids, sim_scores):
            story = self.stories.loc[sid]
            genre, author = story["genre"], story["author"]
            rows.append({
                "sim_score": sim,
                "genre_match": float(genre == preferred),
                "genre_affinity": self.user_genre_share.get((user_id, genre), 0.0),
                "author_affinity": self.user_author_share.get((user_id, author), 0.0),
                "story_popularity": self.story_popularity.get(sid, 0.0),
                "story_avg_completion": self.story_avg_completion.get(
                    sid, self.global_avg_completion),
                "story_reading_minutes": story["avg_reading_minutes"],
                "story_freshness": 1.0 / (1.0 + story["publish_days_ago"] / 90.0),
                "user_n_reads": ustats["n_reads"],
                "user_avg_completion": ustats["avg_completion"],
                "user_like_rate": ustats["like_rate"],
                "user_avg_session": ustats["avg_session"],
            })
        return pd.DataFrame(rows, columns=FEATURE_COLUMNS)
