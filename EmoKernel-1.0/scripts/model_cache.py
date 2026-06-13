"""
Shared model cache for sentence-transformers.
Single source of truth for model loading across all modules.
"""

import threading
import time
from typing import Optional

_model = None
_HAS_MODEL = False
_DOWNLOAD_ATTEMPTED = False
_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def ensure_model(timeout_s: float = 30.0) -> bool:
    """Try to load sentence-transformers model with timeout. Returns True if model is available."""
    global _model, _HAS_MODEL, _DOWNLOAD_ATTEMPTED
    if _model is not None:
        return True
    if _DOWNLOAD_ATTEMPTED:
        return False  # Already tried and failed; don't retry

    result: list = [None]

    def _load():
        try:
            from sentence_transformers import SentenceTransformer
            result[0] = SentenceTransformer(_MODEL_NAME)
        except Exception:
            result[0] = False

    t = threading.Thread(target=_load, daemon=True)
    t.start()
    t.join(timeout=timeout_s)

    if t.is_alive():
        _HAS_MODEL = False
        _DOWNLOAD_ATTEMPTED = True
        return False

    loaded = result[0]
    if loaded and loaded is not False:
        _model = loaded
        _HAS_MODEL = True
        return True

    _HAS_MODEL = False
    _DOWNLOAD_ATTEMPTED = True
    return False


def has_model() -> bool:
    """Check if sentence-transformers model is currently available."""
    return _HAS_MODEL and _model is not None


def get_model():
    """Get the loaded model instance, or None."""
    return _model


def compute_embeddings(labels: list, convert_to_numpy: bool = True):
    """Compute embeddings for a list of labels. Returns None if model unavailable."""
    if not has_model():
        return None
    return _model.encode(labels, convert_to_numpy=convert_to_numpy)