"""Content-based user embeddings from reading history.

A user's raw profile vector is the engagement-weighted average of the
embeddings of stories they read (train split only). Weights combine
completion rate, likes and recency, so a story the user finished and liked
last week influences their profile far more than one they abandoned a year
ago. These profiles feed the user tower of the retrieval model.
"""
import numpy as np
import pandas as pd

from src import config
from src.utils.io import load_npy, save_npy, setup_logger

log = setup_logger("user-emb")

RECENCY_HALF_LIFE_DAYS = 60.0


def compute_weights(df: pd.DataFrame, now: pd.Timestamp) -> np.ndarray:
    age_days = (now - df["timestamp"]).dt.total_seconds() / 86400.0
    recency = 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)
    engagement = 0.5 + df["completion_rate"] + 0.5 * df["likes"]
    return (recency * engagement).to_numpy(dtype=np.float32)


def build_user_profiles(train: pd.DataFrame, story_emb: np.ndarray,
                        story_pos: dict[str, int],
                        user_ids: list[str]) -> np.ndarray:
    now = train["timestamp"].max()
    weights = compute_weights(train, now)
    train = train.assign(_w=weights,
                         _pos=train["story_id"].map(story_pos))

    dim = story_emb.shape[1]
    profiles = np.zeros((len(user_ids), dim), dtype=np.float32)
    uid_pos = {u: i for i, u in enumerate(user_ids)}

    for uid, grp in train.groupby("user_id"):
        vecs = story_emb[grp["_pos"].to_numpy()]
        w = grp["_w"].to_numpy()[:, None]
        profile = (vecs * w).sum(axis=0) / (w.sum() + 1e-9)
        norm = np.linalg.norm(profile)
        if norm > 0:
            profile /= norm
        profiles[uid_pos[uid]] = profile
    return profiles


def main() -> None:
    train = pd.read_parquet(config.PROCESSED_DIR / "train.parquet")
    story_emb = load_npy(config.PROCESSED_DIR / "story_embeddings.npy")
    story_ids = pd.read_csv(config.PROCESSED_DIR / "story_index.csv")["story_id"].tolist()
    story_pos = {s: i for i, s in enumerate(story_ids)}

    users = pd.read_csv(config.RAW_DIR / "users.csv")
    user_ids = users["user_id"].tolist()

    profiles = build_user_profiles(train, story_emb, story_pos, user_ids)
    save_npy(profiles, config.PROCESSED_DIR / "user_profiles.npy")
    users[["user_id"]].to_csv(config.PROCESSED_DIR / "user_index.csv", index=False)
    log.info("saved user profiles: %s (non-zero: %d)",
             profiles.shape, int((np.abs(profiles).sum(axis=1) > 0).sum()))


if __name__ == "__main__":
    main()
