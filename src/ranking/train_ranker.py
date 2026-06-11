"""Stage-2 ranker: LightGBM LambdaRank over FAISS candidates.

Training data is built leak-free by re-splitting the *train* interactions
chronologically per user: the older 80% ("history") drives features, the
newest 20% ("ranker labels") provides positives. For every user we retrieve
Top-K candidates from FAISS; a candidate gets label 1 if the user actually
read it in the label window, else 0. LambdaRank then optimises NDCG directly
over each user's candidate list. The held-out test split is never touched.
"""
import pickle

import numpy as np
import pandas as pd

from src import config
from src.ranking.features import FEATURE_COLUMNS, FeatureBuilder
from src.retrieval.faiss_store import CandidateRetriever
from src.utils.io import load_npy, save_json, setup_logger

log = setup_logger("ranker")

# NOTE: src/__init__.py imports torch first so lightgbm/faiss share its
# OpenMP runtime — required on macOS to avoid a segfault at model load.
try:
    import lightgbm as lgb
    HAS_LGB = True
except (ImportError, OSError):  # OSError: missing libomp on bare macOS
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAS_LGB = False

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

RANKER_PATH = config.MODELS_DIR / "ranker.pkl"


def temporal_resplit(train: pd.DataFrame):
    """Per-user 80/20 chronological split of the train set."""
    train = train.sort_values(["user_id", "timestamp"])
    rank = train.groupby("user_id").cumcount()
    size = train.groupby("user_id")["story_id"].transform("size")
    label_mask = rank >= (size * 0.8).astype(int)
    return train[~label_mask], train[label_mask]


def build_training_table(history, labels, stories, users):
    user_vecs = load_npy(config.PROCESSED_DIR / "user_tower_vectors.npy")
    user_ids = pd.read_csv(config.PROCESSED_DIR / "user_index.csv")["user_id"].tolist()
    user_pos = {u: i for i, u in enumerate(user_ids)}

    retriever = CandidateRetriever()
    fb = FeatureBuilder(history, stories, users)

    label_sets = labels.groupby("user_id")["story_id"].apply(set).to_dict()
    seen_sets = history.groupby("user_id")["story_id"].apply(set).to_dict()

    frames, y, groups = [], [], []
    for uid, positives in label_sets.items():
        cand_ids, sims = retriever.retrieve(
            user_vecs[user_pos[uid]], k=config.TOP_K_CANDIDATES,
            exclude=seen_sets.get(uid, set()))
        labels_vec = np.array([1 if c in positives else 0 for c in cand_ids])
        if labels_vec.sum() == 0:
            continue  # no positive retrieved -> nothing for LambdaRank to order
        frames.append(fb.build(uid, cand_ids, sims))
        y.append(labels_vec)
        groups.append(len(cand_ids))

    X = pd.concat(frames, ignore_index=True)
    return X, np.concatenate(y), groups, fb


def main() -> None:
    train = pd.read_parquet(config.PROCESSED_DIR / "train.parquet")
    stories = pd.read_csv(config.RAW_DIR / "stories.csv")
    users = pd.read_csv(config.RAW_DIR / "users.csv")

    history, labels = temporal_resplit(train)
    log.info("history rows: %d | label rows: %d", len(history), len(labels))

    X, y, groups, _ = build_training_table(history, labels, stories, users)
    log.info("ranker table: %s | positives: %d | users: %d",
             X.shape, int(y.sum()), len(groups))

    if HAS_LGB:
        model = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=20,
            eval_at=[10],
            random_state=config.SEED,
            verbosity=-1,
        )
        model.fit(X, y, group=groups)
        importance = dict(zip(FEATURE_COLUMNS,
                              model.feature_importances_.astype(float)))
    else:
        model = HistGradientBoostingClassifier(max_iter=400,
                                               random_state=config.SEED)
        model.fit(X, y)
        importance = {}

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RANKER_PATH, "wb") as f:
        pickle.dump(model, f)
    save_json(importance, config.REPORTS_DIR / "feature_importance.json")

    if HAS_MLFLOW:
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
        with mlflow.start_run(run_name="lgbm-ranker"):
            mlflow.log_params({"model": type(model).__name__,
                               "n_rows": len(X), "n_users": len(groups)})
            mlflow.log_metric("n_positives", float(y.sum()))

    log.info("ranker saved to %s (%s)", RANKER_PATH, type(model).__name__)


def score_candidates(model, features: pd.DataFrame) -> np.ndarray:
    """Uniform scoring API across LGBMRanker and sklearn fallback."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    return model.predict(features)


if __name__ == "__main__":
    main()
