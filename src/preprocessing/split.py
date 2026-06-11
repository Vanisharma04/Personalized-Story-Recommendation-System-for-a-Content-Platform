"""Temporal train/test split.

For each user, the most recent 20% of interactions (min 1, only for users
with >= 5 interactions) are held out as the test set. This mirrors the
production setting: predict what a user reads *next*, never train on the
future.
"""
import pandas as pd

from src import config
from src.utils.io import setup_logger

log = setup_logger("split")


def main() -> None:
    interactions = pd.read_csv(config.RAW_DIR / "interactions.csv",
                               parse_dates=["timestamp"])
    interactions = interactions.sort_values(["user_id", "timestamp"])

    def _split(group: pd.DataFrame) -> pd.Series:
        n = len(group)
        if n < 5:
            return pd.Series(False, index=group.index)
        n_test = max(1, int(n * 0.2))
        flags = pd.Series(False, index=group.index)
        flags.iloc[-n_test:] = True
        return flags

    test_mask = interactions.groupby("user_id", group_keys=False).apply(
        _split, include_groups=False)
    interactions["is_test"] = test_mask

    train = interactions[~interactions["is_test"]].drop(columns="is_test")
    test = interactions[interactions["is_test"]].drop(columns="is_test")

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_parquet(config.PROCESSED_DIR / "train.parquet", index=False)
    test.to_parquet(config.PROCESSED_DIR / "test.parquet", index=False)

    log.info("train: %d rows | test: %d rows | users in test: %d",
             len(train), len(test), test["user_id"].nunique())


if __name__ == "__main__":
    main()
