"""
py_odin — scODIN cell-type annotation toolkit.

Public API
----------
score(adata, ...)                       Per-cell RNA/ADT scoring.
score_clusters(adata, ...)              Cluster-level consensus labels.
simplify_labels(adata, ...)             Replace double-label strings with friendly names.
predict_unknown_cells_lgbm(adata, ...)  LightGBM-based prediction for 'unknown' cells.
                                         Requires optional deps: lightgbm, scikit-learn.
"""

from .py_odin import (
    score,
    score_clusters,
    simplify_labels,
    predict_unknown_cells_lgbm,
)

__all__ = [
    "score",
    "score_clusters",
    "simplify_labels",
    "predict_unknown_cells_lgbm",
]

__version__ = "0.1.0"
__author__  = "Siddharth S. Tomar"