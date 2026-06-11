"""Package init.

macOS only: torch, faiss and lightgbm each bundle their own OpenMP runtime
(libomp). Whichever native lib loads first provides the runtime for all of
them — and if faiss loads before torch, lightgbm later segfaults when the
ranker model is loaded. Importing torch here pins the safe load order.

On Linux (e.g. Streamlit Cloud) the clash doesn't occur — the system libgomp
is shared — so we skip the torch import entirely. The serving path
(FAISS + LightGBM over precomputed vectors) needs neither torch nor the
sentence-transformer, which keeps the deployed app's memory footprint small.
"""
import platform

if platform.system() == "Darwin":
    try:
        import torch  # noqa: F401
    except ImportError:
        pass
