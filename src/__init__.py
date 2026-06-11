# macOS: torch, faiss and lightgbm each bundle their own OpenMP runtime
# (libomp). Whichever native lib loads first provides the runtime for all of
# them — and if faiss loads before torch, lightgbm later segfaults when the
# ranker model is loaded. Importing torch at the package root guarantees the
# safe load order for every entry point (pipeline, dashboard, notebooks).
import torch  # noqa: F401
