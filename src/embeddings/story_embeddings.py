"""Semantic story embeddings via Sentence Transformers.

Each story is encoded from a composite text of title + genre + description,
producing a dense 384-d vector (all-MiniLM-L6-v2). Vectors are L2-normalised
so that inner product == cosine similarity everywhere downstream.
"""
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src import config
from src.utils.io import save_npy, setup_logger

log = setup_logger("story-emb")


def build_story_text(stories: pd.DataFrame) -> list[str]:
    return (
        stories["title"] + ". Genre: " + stories["genre"] + ". " + stories["description"]
    ).tolist()


def main() -> None:
    stories = pd.read_csv(config.RAW_DIR / "stories.csv")
    texts = build_story_text(stories)

    log.info("encoding %d stories with %s", len(texts), config.SENTENCE_MODEL)
    model = SentenceTransformer(config.SENTENCE_MODEL)
    emb = model.encode(texts, batch_size=128, show_progress_bar=True,
                       normalize_embeddings=True)
    emb = np.asarray(emb, dtype=np.float32)

    save_npy(emb, config.PROCESSED_DIR / "story_embeddings.npy")
    stories[["story_id"]].to_csv(config.PROCESSED_DIR / "story_index.csv", index=False)
    log.info("saved story embeddings: %s", emb.shape)


if __name__ == "__main__":
    main()
