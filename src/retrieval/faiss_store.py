"""FAISS index over item-tower vectors for fast candidate generation.

Vectors are L2-normalised, so IndexFlatIP performs exact cosine-similarity
search. At this catalogue size exact search is instant; the module exposes
the same interface you would keep when swapping in IVF/HNSW at scale.
"""
import faiss
import numpy as np
import pandas as pd

from src import config
from src.utils.io import load_npy, setup_logger

log = setup_logger("faiss")

INDEX_PATH = config.FAISS_DIR / "stories.index"


def build_index() -> None:
    item_vecs = load_npy(config.PROCESSED_DIR / "item_tower_vectors.npy")
    index = faiss.IndexFlatIP(item_vecs.shape[1])
    index.add(item_vecs)
    config.FAISS_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    log.info("FAISS index built: %d vectors, dim=%d", index.ntotal, item_vecs.shape[1])


class CandidateRetriever:
    """Loads the FAISS index once and serves Top-K candidate queries."""

    def __init__(self):
        self.index = faiss.read_index(str(INDEX_PATH))
        self.story_ids = pd.read_csv(
            config.PROCESSED_DIR / "story_index.csv")["story_id"].to_numpy()

    def retrieve(self, user_vec: np.ndarray,
                 k: int = config.TOP_K_CANDIDATES,
                 exclude: set[str] | None = None):
        """Return (story_ids, scores) of top-k candidates for one user vector."""
        query = np.ascontiguousarray(user_vec.reshape(1, -1), dtype=np.float32)
        # over-fetch so we can drop already-read stories and still return k
        fetch = k + (len(exclude) if exclude else 0)
        scores, idx = self.index.search(query, min(fetch, self.index.ntotal))
        ids, sims = [], []
        for i, s in zip(idx[0], scores[0]):
            sid = self.story_ids[i]
            if exclude and sid in exclude:
                continue
            ids.append(sid)
            sims.append(float(s))
            if len(ids) == k:
                break
        return ids, sims

    def retrieve_batch(self, user_vecs: np.ndarray, k: int = config.TOP_K_CANDIDATES):
        """Vectorised top-k for many users at once (no exclusion)."""
        scores, idx = self.index.search(
            np.ascontiguousarray(user_vecs, dtype=np.float32), k)
        return self.story_ids[idx], scores


if __name__ == "__main__":
    build_index()
