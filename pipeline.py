"""End-to-end training pipeline.

    python pipeline.py            # run everything
    python pipeline.py --from 3   # resume from step 3

Steps:
  1. generate synthetic dataset
  2. temporal train/test split
  3. story embeddings (Sentence Transformers)
  4. user profile embeddings
  5. two-tower retrieval model
  6. FAISS index
  7. LightGBM ranker
  8. offline evaluation
"""
import argparse
import time

from src.utils.io import setup_logger

log = setup_logger("pipeline")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="start", type=int, default=1)
    args = parser.parse_args()

    from src.preprocessing import generate_data, split
    from src.embeddings import story_embeddings, user_embeddings
    from src.retrieval import two_tower, faiss_store
    from src.ranking import train_ranker
    from src.evaluation import evaluate

    steps = [
        ("generate dataset", generate_data.main),
        ("train/test split", split.main),
        ("story embeddings", story_embeddings.main),
        ("user embeddings", user_embeddings.main),
        ("two-tower training", two_tower.train_model),
        ("FAISS index", faiss_store.build_index),
        ("ranker training", train_ranker.main),
        ("offline evaluation", evaluate.main),
    ]

    t0 = time.time()
    for i, (name, fn) in enumerate(steps, start=1):
        if i < args.start:
            log.info("[%d/8] %s  (skipped)", i, name)
            continue
        log.info("[%d/8] %s ...", i, name)
        t = time.time()
        fn()
        log.info("[%d/8] %s done in %.1fs", i, name, time.time() - t)

    log.info("pipeline complete in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
