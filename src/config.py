"""Central configuration for the recommendation engine."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FAISS_DIR = PROJECT_ROOT / "faiss_index"
REPORTS_DIR = PROJECT_ROOT / "reports"

SEED = 42

# ---------------------------------------------------------------- dataset
N_USERS = 1200
N_STORIES = 3000
TARGET_INTERACTIONS = 160_000

GENRES = [
    "Fantasy", "Romance", "Mystery", "Science Fiction", "Horror",
    "Historical Fiction", "Thriller", "Adventure", "Drama", "Mythology",
]

# ---------------------------------------------------------------- embeddings
SENTENCE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output size

# ---------------------------------------------------------------- two-tower
TOWER_DIM = 128          # final representation size of each tower
TOWER_HIDDEN = 256
BATCH_SIZE = 512
EPOCHS = 8
LR = 1e-3
TEMPERATURE = 0.07       # softmax temperature for in-batch sampled softmax

# ---------------------------------------------------------------- retrieval
TOP_K_CANDIDATES = 100

# ---------------------------------------------------------------- ranking
TOP_N_RECOMMENDATIONS = 10

# ---------------------------------------------------------------- evaluation
RECALL_K = 50
NDCG_K = 10

# ---------------------------------------------------------------- mlflow
MLFLOW_EXPERIMENT = "story-recsys"
MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
