"""Two-Tower retrieval model (PyTorch).

User tower : engagement-weighted history profile (384-d) + behaviour stats -> 128-d
Item tower : sentence-transformer story embedding (384-d) + side features  -> 128-d

Side features (popularity, completion, freshness / reading frequency, like
rate) are standardised and concatenated to the content embeddings before the
towers, letting retrieval capture exposure effects that pure content vectors
cannot — the usual production design.

Trained with in-batch sampled softmax: for a batch of (user, read-story)
pairs, every other story in the batch acts as a negative. Both towers emit
L2-normalised vectors, so similarity is cosine and the item tower output can
be indexed directly in FAISS.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src import config
from src.utils.io import load_npy, save_json, save_npy, setup_logger

log = setup_logger("two-tower")

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False


class Tower(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class TwoTowerModel(nn.Module):
    def __init__(self, user_dim: int, item_dim: int,
                 hidden: int = config.TOWER_HIDDEN,
                 out_dim: int = config.TOWER_DIM):
        super().__init__()
        self.user_tower = Tower(user_dim, hidden, out_dim)
        self.item_tower = Tower(item_dim, hidden, out_dim)

    def forward(self, user_x: torch.Tensor, item_x: torch.Tensor):
        return self.user_tower(user_x), self.item_tower(item_x)


class InteractionDataset(Dataset):
    """(user_profile, positive_story_embedding) pairs, engagement-weighted."""

    def __init__(self, train: pd.DataFrame, user_profiles: np.ndarray,
                 story_emb: np.ndarray, user_pos: dict, story_pos: dict):
        u = train["user_id"].map(user_pos).to_numpy()
        s = train["story_id"].map(story_pos).to_numpy()
        # over-sample strong engagements: completed+liked reads appear twice
        strong = ((train["completion_rate"] > 0.7) & (train["likes"] == 1)).to_numpy()
        self.u_idx = np.concatenate([u, u[strong]])
        self.s_idx = np.concatenate([s, s[strong]])
        self.user_profiles = user_profiles
        self.story_emb = story_emb

    def __len__(self):
        return len(self.u_idx)

    def __getitem__(self, i):
        return (self.user_profiles[self.u_idx[i]],
                self.story_emb[self.s_idx[i]],
                self.s_idx[i])


def _standardize(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-9)


def build_tower_inputs(train: pd.DataFrame, user_profiles: np.ndarray,
                       story_emb: np.ndarray, user_ids: list[str],
                       story_ids: list[str], stories: pd.DataFrame):
    """Concatenate standardized behavioural side features to content vectors."""
    # ---- item side: log-popularity, mean completion, freshness
    agg = train.groupby("story_id").agg(pop=("user_id", "count"),
                                        comp=("completion_rate", "mean"))
    meta = stories.set_index("story_id")
    item_side = np.array([
        [np.log1p(agg["pop"].get(s, 0)),
         agg["comp"].get(s, train["completion_rate"].mean()),
         1.0 / (1.0 + meta.at[s, "publish_days_ago"] / 90.0)]
        for s in story_ids], dtype=np.float32)

    # ---- user side: log reading frequency, mean completion, like rate
    uagg = train.groupby("user_id").agg(n=("story_id", "count"),
                                        comp=("completion_rate", "mean"),
                                        like=("likes", "mean"))
    user_side = np.array([
        [np.log1p(uagg["n"].get(u, 0)),
         uagg["comp"].get(u, train["completion_rate"].mean()),
         uagg["like"].get(u, 0.0)]
        for u in user_ids], dtype=np.float32)

    item_x = np.hstack([story_emb, _standardize(item_side)]).astype(np.float32)
    user_x = np.hstack([user_profiles, _standardize(user_side)]).astype(np.float32)
    return user_x, item_x


def sampled_softmax_loss(u_vec, i_vec, item_ids, temperature):
    """In-batch softmax with masking of accidental duplicate positives."""
    logits = u_vec @ i_vec.T / temperature                     # [B, B]
    labels = torch.arange(len(u_vec), device=u_vec.device)
    # mask other occurrences of the same item so they aren't false negatives
    dup = item_ids[:, None] == item_ids[None, :]
    dup.fill_diagonal_(False)
    logits = logits.masked_fill(dup, float("-inf"))
    return F.cross_entropy(logits, labels)


def train_model() -> None:
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu")
    log.info("training on %s", device)

    train = pd.read_parquet(config.PROCESSED_DIR / "train.parquet")
    user_profiles = load_npy(config.PROCESSED_DIR / "user_profiles.npy")
    story_emb = load_npy(config.PROCESSED_DIR / "story_embeddings.npy")
    user_ids = pd.read_csv(config.PROCESSED_DIR / "user_index.csv")["user_id"].tolist()
    story_ids = pd.read_csv(config.PROCESSED_DIR / "story_index.csv")["story_id"].tolist()
    user_pos = {u: i for i, u in enumerate(user_ids)}
    story_pos = {s: i for i, s in enumerate(story_ids)}

    stories = pd.read_csv(config.RAW_DIR / "stories.csv")
    user_x_all, item_x_all = build_tower_inputs(
        train, user_profiles, story_emb, user_ids, story_ids, stories)

    dataset = InteractionDataset(train, user_x_all, item_x_all, user_pos, story_pos)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True,
                        drop_last=True)

    model = TwoTowerModel(user_x_all.shape[1], item_x_all.shape[1]).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=config.EPOCHS * len(loader))

    if HAS_MLFLOW:
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
        mlflow.start_run(run_name="two-tower")
        mlflow.log_params({
            "tower_dim": config.TOWER_DIM, "hidden": config.TOWER_HIDDEN,
            "batch_size": config.BATCH_SIZE, "epochs": config.EPOCHS,
            "lr": config.LR, "temperature": config.TEMPERATURE,
            "n_pairs": len(dataset),
        })

    history = []
    for epoch in range(config.EPOCHS):
        model.train()
        total, steps = 0.0, 0
        for user_x, item_x, item_ids in loader:
            user_x = user_x.to(device).float()
            item_x = item_x.to(device).float()
            item_ids = item_ids.to(device)

            u_vec, i_vec = model(user_x, item_x)
            loss = sampled_softmax_loss(u_vec, i_vec, item_ids, config.TEMPERATURE)

            optim.zero_grad()
            loss.backward()
            optim.step()
            sched.step()
            total += loss.item()
            steps += 1

        avg = total / max(steps, 1)
        history.append(avg)
        log.info("epoch %d/%d  loss=%.4f", epoch + 1, config.EPOCHS, avg)
        if HAS_MLFLOW:
            mlflow.log_metric("train_loss", avg, step=epoch)

    # ------------------------------------------------ export final vectors
    model.eval()
    with torch.no_grad():
        item_vecs = model.item_tower(
            torch.from_numpy(item_x_all).float().to(device)).cpu().numpy()
        user_vecs = model.user_tower(
            torch.from_numpy(user_x_all).float().to(device)).cpu().numpy()

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), config.MODELS_DIR / "two_tower.pt")
    save_npy(item_vecs.astype(np.float32), config.PROCESSED_DIR / "item_tower_vectors.npy")
    save_npy(user_vecs.astype(np.float32), config.PROCESSED_DIR / "user_tower_vectors.npy")
    save_json({"loss_history": history}, config.REPORTS_DIR / "two_tower_history.json")

    if HAS_MLFLOW:
        mlflow.log_metric("final_loss", history[-1])
        mlflow.end_run()
    log.info("saved model + tower vectors (item %s, user %s)",
             item_vecs.shape, user_vecs.shape)


if __name__ == "__main__":
    train_model()
