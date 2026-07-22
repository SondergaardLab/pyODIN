"""
py_odin - ODIN cell-type annotation toolkit.

Entry points
------------
score(adata, ...)              - per-cell RNA/ADT scoring
score_clusters(adata, ...)     - cluster-level consensus labels
simplify_labels(adata, ...)    - replace double-label strings with friendly names
"""

import logging
import time
from typing import Optional

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

logger = logging.getLogger("scODIN")

#-----------------------
# Optional dependencies (lightgbm, scikit-learn)
# These are only required by predict_unknown_cells_lgbm().
#-----------------------

_LGBM_IMPORT_ERROR: Optional[ImportError] = None
try:
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report
    from sklearn.utils.class_weight import compute_sample_weight
except ImportError as exc:
    lgb = None
    train_test_split = None
    LabelEncoder = None
    classification_report = None
    compute_sample_weight = None
    _LGBM_IMPORT_ERROR = exc


def _require_lgbm_deps() -> None:
    """Raise an error if lightgbm/scikit-learn are not installed."""
    if _LGBM_IMPORT_ERROR is not None:
        raise ImportError(
            "predict_unknown_cells_lgbm() requires the optional dependencies "
            "'lightgbm' and 'scikit-learn'. Install them with:\n"
            "    pip install lightgbm scikit-learn\n"
            f"Original import error: {_LGBM_IMPORT_ERROR}"
        ) from _LGBM_IMPORT_ERROR


#-----------------------
# score
#-----------------------

def score(
    adata: sc.AnnData,
    gene_priority_table: pd.DataFrame,
    accepted_doubles_table: pd.DataFrame,
    *,
    core_cell_cutoff: float = 5.0,
    similarity_threshold: float = 1.0,
    cell_level: str = "Top",
    use_modality: str = "both",
    rna_layer: Optional[str] = "scaled",
    adt_obsm_key: str = "ADT",
    adt_map_col: str = "adt_id",
    adt_pos_only: bool = True,
) -> sc.AnnData:
    """
    Run scODIN per-cell scoring on *adata* and return it annotated.

    Parameters
    ----------
    adata:
        AnnData object to annotate.
    gene_priority_table:
        DataFrame with columns: gene_id, cell_type, gene_priority, tier,
        cell_level (and optionally gene_direction, adt_id).
    accepted_doubles_table:
        DataFrame with columns: cell_type1, cell_type2
        (and optionally simple_name for :func:`simplify_labels`).
    core_cell_cutoff:
        Scores below this value are zeroed out before labeling.
    similarity_threshold:
        If the top-two active scores differ by less than this, the cell is
        flagged as a potential double label.
    cell_level:
        Which ``cell_level`` row group to use from gene_priority_table.
    use_modality:
        One of ``"RNA"``, ``"ADT"``, or ``"both"``.
    rna_layer:
        Layer key for RNA.  Falls back to ``adata.X`` when absent.
    adt_obsm_key:
        Key in ``adata.obsm`` for the ADT matrix.
    adt_map_col:
        Column in *gene_priority_table* mapping rows to ADT feature names.
    adt_pos_only:
        When ``True``, only features whose ``gene_direction == 'pos'``
        contribute ADT signal.

    Results written to *adata*
    --------------------------
    adata.obs['single_labels']  — labels before double-label resolution
    adata.obs['double_labels']  — double-label pair strings (or 'not_double_label')
    adata.obs['final_labels']   — post-resolution labels
    adata.uns['odin_score_all'] — full (n_cells, n_cell_types) score matrix
    """
    if use_modality not in ("RNA", "ADT", "both"):
        raise ValueError(
            f"use_modality must be 'RNA', 'ADT', or 'both'; got {use_modality!r}"
        )

    start = time.time()
    logger.info(
        "score: starting for cell_level=%r, modality=%r", cell_level, use_modality
    )

    gp = _clean_priority_table(gene_priority_table, cell_level)

    rna_data = (
        _extract_rna(adata, gp, rna_layer)
        if use_modality in ("RNA", "both")
        else None
    )
    adt_df = (
        _extract_adt(adata, adt_obsm_key)
        if use_modality in ("ADT", "both")
        else None
    )

    expr_matrix, weight_matrix, feature_counts, cell_type_names = _build_matrices(
        adata, gp, rna_data, adt_df, adt_map_col, adt_pos_only
    )

    scores = _compute_scores(
        adata.obs_names, cell_type_names, expr_matrix, weight_matrix, feature_counts
    )
    scores = _apply_cutoff(scores, core_cell_cutoff)

    accepted_doubles_set = _build_accepted_doubles_set(accepted_doubles_table)
    tier_map = _build_tier_map(gp)

    # RNA-only: set final_labels to top scorer even for non-accepted doubles.
    # I need to validate RNA-only behaviour. Perhaps for next group meeting ?
    # ADT / both: leave final_labels as "unknown" for those cells.
    pre_resolution_labels, double_labels = _label_cells(
        scores,
        tier_map,
        similarity_threshold,
        set_final_on_double=(use_modality == "RNA"),
    )
    final_labels = _resolve_doubles(
        pre_resolution_labels, double_labels, accepted_doubles_set
    )

    adata.uns["odin_score_all"]  = scores
    adata.obs["single_labels"]   = pre_resolution_labels.values
    adata.obs["double_labels"]   = double_labels.values
    adata.obs["final_labels"]    = final_labels.values

    logger.info("score: done in %.2fs (~˘▾˘)~", time.time() - start)
    return adata


#-----------------------
# score_clusters
#-----------------------

def score_clusters(
    adata: sc.AnnData,
    clustering_column: str = "leiden",
) -> sc.AnnData:
    """
    Aggregate cell-level scODIN scores to assign a consensus label to each
    cluster.

    :func:`score` must have been called on *adata* first so that
    ``adata.obs['final_labels']`` and ``adata.uns['odin_score_all']`` exist.

    The enrichment score for a (cluster, label) pair is::

        result = cluster_total_score × √n_cells_with_label

    The label with the highest *result* per cluster wins.

    Results written to *adata*
    --------------------------
    adata.obs['odin_classification']  — consensus label per cell
    adata.uns['odin_cluster_summary'] — full summary DataFrame
    """
    if clustering_column not in adata.obs.columns:
        raise ValueError(
            f"Column '{clustering_column}' not found in adata.obs"
        )
    if "final_labels" not in adata.obs.columns or "odin_score_all" not in adata.uns:
        raise ValueError(
            "scODIN scores not found. Run score() first."
        )

    # cell counts per (cluster, label)
    cell_counts = (
        adata.obs[adata.obs["final_labels"] != "unknown"]
        .groupby([clustering_column, "final_labels"], observed=True)
        .size()
        .reset_index(name="ncells_label")
    )

    # sum odin scores per cluster via sparse matrix multiply
    # Perhaps I can check for Intel MKL implementation ? We are running it on threadripper
    # so it might have to wait.
    odin_scores: pd.DataFrame = adata.uns["odin_score_all"]
    score_arr   = odin_scores.values                    # (n_cells, n_types)
    col_names   = odin_scores.columns.values

    cluster_col                    = adata.obs[clustering_column].values
    unique_clusters, cluster_codes = np.unique(cluster_col, return_inverse=True)
    n_clusters = len(unique_clusters)
    n_cells    = score_arr.shape[0]
    n_types    = score_arr.shape[1]

    C = sp.csr_matrix(
        (np.ones(n_cells, dtype=np.float32),
         (cluster_codes, np.arange(n_cells))),
        shape=(n_clusters, n_cells),
    )
    cluster_sums = C @ score_arr                        # (n_clusters, n_types)

    k         = min(10, n_types)
    top_local = (
        np.argpartition(cluster_sums, -k, axis=1)[:, -k:]
        if k < n_types
        else np.tile(np.arange(n_types), (n_clusters, 1))
    )

    cluster_rep   = np.repeat(unique_clusters, k)
    type_idx_flat = top_local.ravel()
    label_rep     = col_names[type_idx_flat]
    score_rep     = cluster_sums[
        np.repeat(np.arange(n_clusters), k), type_idx_flat
    ]

    score_summary = pd.DataFrame({
        clustering_column:     cluster_rep,
        "final_labels":        label_rep,
        "cluster_total_score": score_rep,
    })

    # --- merge and compute enrichment ---
    merged = pd.merge(cell_counts, score_summary, on=[clustering_column, "final_labels"])
    merged["result"] = merged["cluster_total_score"] * np.sqrt(merged["ncells_label"])

    merged = merged.sort_values(
        [clustering_column, "result"], ascending=[True, False]
    ).reset_index(drop=True)

    result_vals  = merged["result"].values
    cluster_vals = merged[clustering_column].values

    boundary         = np.empty(len(merged), dtype=bool)
    boundary[:-1]    = cluster_vals[:-1] != cluster_vals[1:]
    boundary[-1]     = True

    second_best           = np.empty(len(merged))
    second_best[:-1]      = result_vals[1:]
    second_best[boundary] = np.nan

    merged["enrichment"] = result_vals / second_best

    # assign cluster labels
    cluster_assignments = (
        merged.sort_values("result", ascending=False)
        .groupby(clustering_column)
        .head(1)
        .set_index(clustering_column)["final_labels"]
        .to_dict()
    )

    adata.obs["odin_classification"] = (
        adata.obs[clustering_column]
        .map(cluster_assignments)
        .astype(str)
        .fillna("unknown")
        .astype(str)
    )
    adata.uns["odin_cluster_summary"] = merged

    logger.info("score_clusters: cluster-level classification complete.")
    return adata


#-----------------------
# simplify_labels
#-----------------------

def simplify_labels(
    adata: sc.AnnData,
    accepted_doubles_table: pd.DataFrame,
    cell_level: str,
) -> sc.AnnData:
    """
    Replace verbose double-label strings (e.g. ``"B_T"``) in
    ``adata.obs['final_labels']`` with human-readable names from
    *accepted_doubles_table*.

    Parameters
    ----------
    adata:
        AnnData object whose ``obs['final_labels']`` will be updated.
    accepted_doubles_table:
        Must contain columns ``cell_type1``, ``cell_type2``, ``simple_name``,
        and ``cell_level``.
    cell_level:
        Filters *accepted_doubles_table* to the relevant level.
    """
    if "final_labels" not in adata.obs.columns:
        raise KeyError("'final_labels' not found in adata.obs. Run score() first.")

    mapping = _build_simplify_mapping(accepted_doubles_table, cell_level)

    if not mapping:
        logger.warning(
            "simplify_labels: no mappings for cell_level=%r; nothing to simplify.",
            cell_level,
        )
        return adata

    logger.info(
        "simplify_labels: simplifying double labels for level=%r", cell_level
    )
    adata.obs["final_labels"] = (
        adata.obs["final_labels"]
        .astype(str)
        .replace(mapping)
        .astype("category")
    )
    return adata


#-----------------------
# predict_unknown_cells_lgbm
#-----------------------

def _get_lgbm_feature_matrix(
    adata: sc.AnnData,
    feature_layer: Optional[str],
) -> np.ndarray:
    """
    Resolve the dense feature matrix used to train/predict with LightGBM.

    feature_layer:
        None or "X"        — adata.X
        "odin_score_all"   — adata.uns['odin_score_all'] (written by score())
        any other string   — adata.layers[feature_layer]
    """
    if feature_layer is None or feature_layer == "X":
        mat = adata.X

    elif feature_layer == "odin_score_all":
        if "odin_score_all" not in adata.uns:
            raise KeyError(
                "feature_layer='odin_score_all' requires adata.uns['odin_score_all'], "
                "which is written by score(). Run score() first, or choose a "
                "different feature_layer."
            )
        scores = adata.uns["odin_score_all"]
        mat = scores.values if isinstance(scores, pd.DataFrame) else scores

    else:
        if feature_layer not in adata.layers:
            raise KeyError(
                f"feature_layer={feature_layer!r} not found in adata.layers. "
                f"Available layers: {list(adata.layers.keys())}"
            )
        mat = adata.layers[feature_layer]

    if hasattr(mat, "toarray"):
        mat = mat.toarray()

    return np.asarray(mat, dtype=float)


def predict_unknown_cells_lgbm(
    adata: sc.AnnData,
    label_col: str = "final_labels",
    unknown_str: str = "unknown",
    confidence_threshold: float = 0.65,
    feature_layer: Optional[str] = None,
) -> sc.AnnData:
    """
    Cell type prediction for 'unknown' cells using LightGBM,
    trained on the known cells within the same AnnData object.

    Requires the optional dependencies ``lightgbm`` and ``scikit-learn``;
    install with ``pip install lightgbm scikit-learn``.

    Parameters
    ----------
    adata:
        AnnData object containing *label_col* in ``adata.obs`` and the
        chosen feature matrix (see *feature_layer*).
    label_col:
        Column in ``adata.obs`` holding ground-truth / unknown labels.
    unknown_str:
        Value in *label_col* marking a cell as unlabeled / to be predicted.
    confidence_threshold:
        Minimum predicted-class probability required to accept a prediction
        outright; below this, the prediction is prefixed with ``"Uncertain_"``.
    feature_layer:
        Which feature matrix to train on. One of:
            None or "X"        — use ``adata.X`` (default).
            a key in adata.layers — use ``adata.layers[feature_layer]``.
            "odin_score_all"   — use the scODIN score matrix written to
                                  ``adata.uns['odin_score_all']`` by
                                  :func:`score`. Trains the classifier on
                                  per-cell-type scores rather than raw
                                  expression.

    Results written to *adata*
    --------------------------
    adata.obs['lgbm_pred_<label_col>'] — per-cell result, one of:
        [Original Label]      — ground truth for cells used in training
        [Predicted Label]     — model assignment for unknown cells (conf >= threshold)
        Uncertain_[Label]     — model's best guess for unknowns (conf < threshold)
        Singleton_Excluded    — known cells belonging to singleton classes (n=1)
        Excluded_Missing_Data — cells that were NaN/null in the original label column
    """
    _require_lgbm_deps()

    feature_matrix = _get_lgbm_feature_matrix(adata, feature_layer)

    y_raw = adata.obs[label_col].values
    is_unknown = (y_raw == unknown_str)
    is_nan = pd.isna(y_raw)
    is_known = (~is_unknown) & (~is_nan)

    unique_labels, counts = np.unique(y_raw[is_known], return_counts=True)
    valid_classes     = unique_labels[counts >= 2]
    singleton_classes = unique_labels[counts < 2]

    logger.info("predict_unknown_cells_lgbm:  Pipeline Setup ")
    logger.info("  Feature source:          %s", feature_layer or "X")
    logger.info("  Known training classes:  %d", len(valid_classes))
    logger.info("  Excluded singletons:     %d", len(singleton_classes))
    logger.info("  Missing/NaN cells:       %d", np.sum(is_nan))

    # Prepare training data 
    mask_trainable = np.isin(y_raw, valid_classes)
    X_trainable = feature_matrix[mask_trainable]
    y_trainable = y_raw[mask_trainable]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_trainable)

    # Validation pass to find best iteration 
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainable, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    weights_val = compute_sample_weight("balanced", y_train)

    base_params = {
        "objective":         "multiclass",
        "num_class":         len(le.classes_),
        "learning_rate":     0.03,
        "num_leaves":        31,
        "min_child_samples": 5,
        "feature_fraction":  0.2,
        "random_state":      2026,
        "n_jobs":            -1,
        "verbosity":         -1,
    }

    logger.info("predict_unknown_cells_lgbm: Step 1: Optimizing boosting rounds ")
    val_model = lgb.LGBMClassifier(**base_params, n_estimators=1000)
    val_model.fit(
        X_train, y_train,
        sample_weight=weights_val,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )
    best_iters = val_model.best_iteration_

    y_pred_val = val_model.predict(X_val)
    logger.info(
        "predict_unknown_cells_lgbm: VALIDATION REPORT (reference classes):\n%s",
        classification_report(
            y_val, y_pred_val,
            labels=np.arange(len(le.classes_)),
            target_names=le.classes_,
            zero_division=0,
        ),
    )

    # Final training on all known cells 
    logger.info(
        "predict_unknown_cells_lgbm: Step 2: Training production model (iters=%d) ",
        best_iters,
    )
    final_weights = compute_sample_weight("balanced", y_encoded)
    final_model = lgb.LGBMClassifier(**base_params, n_estimators=best_iters)
    final_model.fit(X_trainable, y_encoded, sample_weight=final_weights)

    # Results
    new_col = f"lgbm_pred_{label_col}"
    adata.obs[new_col] = adata.obs[label_col].astype(object)
    adata.obs.loc[np.isin(y_raw, singleton_classes), new_col] = "Singleton_Excluded"
    adata.obs.loc[is_nan, new_col] = "Excluded_Missing_Data"

    X_unknown = feature_matrix[is_unknown]
    if X_unknown.shape[0] > 0:
        logger.info(
            "predict_unknown_cells_lgbm: Step 3: Predicting %d cells ",
            X_unknown.shape[0],
        )
        probs      = final_model.predict_proba(X_unknown)
        max_probs  = np.max(probs, axis=1)
        preds      = np.argmax(probs, axis=1)
        decoded    = le.inverse_transform(preds)

        final_labels = [
            decoded[i] if max_probs[i] >= confidence_threshold else f"Uncertain_{decoded[i]}"
            for i in range(len(decoded))
        ]
        adata.obs.loc[is_unknown, new_col] = final_labels
        logger.info(
            "predict_unknown_cells_lgbm: predictions written to adata.obs['%s']",
            new_col,
        )
    else:
        logger.info(
            "predict_unknown_cells_lgbm: no cells found matching unknown_str=%r; skipping prediction.",
            unknown_str,
        )

    return adata


#-----------------------
# Private helpers
#-----------------------

def _clean_priority_table(
    gene_priority_table: pd.DataFrame,
    cell_level: str,
) -> pd.DataFrame:
    gp = gene_priority_table[
        gene_priority_table["cell_level"] == cell_level
    ].copy()

    if gp.empty:
        raise ValueError(
            f"No rows found in gene_priority_table for cell_level='{cell_level}'"
        )

    gp["gene_priority"] = pd.to_numeric(
        gp["gene_priority"].astype(str).str.replace(",", "."),
        errors="coerce",
    )
    return gp


def _build_tier_map(gp: pd.DataFrame) -> dict:
    return gp.groupby("tier")["cell_type"].unique().to_dict()


def _build_accepted_doubles_set(accepted_doubles_table: pd.DataFrame) -> set[str]:
    df = accepted_doubles_table
    p1 = df["cell_type1"].astype(str)
    p2 = df["cell_type2"].astype(str)
    sorted_pairs = pd.DataFrame({"a": p1, "b": p2}).apply(
        lambda r: f"{min(r.a, r.b)}_{max(r.a, r.b)}", axis=1
    )
    return set(sorted_pairs.tolist())


def _build_simplify_mapping(
    accepted_doubles_table: pd.DataFrame,
    cell_level: str,
) -> dict[str, str]:
    df   = accepted_doubles_table[
        accepted_doubles_table["cell_level"] == cell_level
    ]
    c1   = df["cell_type1"].astype(str)
    c2   = df["cell_type2"].astype(str)
    name = df["simple_name"].astype(str)
    fwd  = (c1 + "_" + c2).tolist()
    rev  = (c2 + "_" + c1).tolist()
    names_list = name.tolist()
    return dict(zip(fwd, names_list)) | dict(zip(rev, names_list))


def _extract_rna(
    adata: sc.AnnData,
    gp: pd.DataFrame,
    rna_layer: Optional[str],
) -> tuple[np.ndarray, np.ndarray]:
    priority_genes = gp["gene_id"].unique()
    present_mask   = np.isin(priority_genes, adata.var_names)
    present_genes  = priority_genes[present_mask]

    if rna_layer and rna_layer in adata.layers:
        mat = adata[:, present_genes].layers[rna_layer]
    else:
        logger.info("RNA layer %r not found; using adata.X.", rna_layer)
        mat = adata[:, present_genes].X

    if hasattr(mat, "toarray"):
        mat = mat.toarray()

    return np.asarray(mat, dtype=float), present_genes


def _extract_adt(
    adata: sc.AnnData,
    adt_obsm_key: str,
) -> pd.DataFrame:
    if adt_obsm_key not in adata.obsm:
        logger.warning(
            "ADT key '%s' not found in adata.obsm; ADT signal will be 0.",
            adt_obsm_key,
        )
        return pd.DataFrame()

    raw = adata.obsm[adt_obsm_key]
    if isinstance(raw, pd.DataFrame):
        return raw

    logger.warning(
        "obsm['%s'] is a numpy array; resolving feature names from "
        "adata.uns or falling back to integer indices.",
        adt_obsm_key,
    )
    col_names = adata.uns.get(f"{adt_obsm_key}_names", range(raw.shape[1]))
    return pd.DataFrame(raw, index=adata.obs_names, columns=col_names)


def _build_matrices(
    adata: sc.AnnData,
    gp: pd.DataFrame,
    rna_data: Optional[tuple[np.ndarray, np.ndarray]],
    adt_df: Optional[pd.DataFrame],
    adt_map_col: str,
    adt_pos_only: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the combined expression and weight matrices.

    One column per gp row (NOT per unique gene) so that shared marker genes
    (e.g. CD3D for T_cell and NKT_cell) contribute independently to each
    cell type.  Collapsing to unique gene IDs would silently destroy scores
    for all but the first cell type that shares a marker.

    Normalisation denominator uses the active row count per cell type
    (matching the original implementation).
    """
    cell_types       = gp["cell_type"].unique()
    cell_type_names  = np.asarray(cell_types)
    cell_type_index  = pd.Index(cell_types)
    n_cells          = adata.n_obs

    rna_mat, rna_gene_names = rna_data if rna_data is not None else (None, None)
    rna_gene_index = pd.Index(rna_gene_names) if rna_gene_names is not None else pd.Index([])
    rna_cols = set(map(str, rna_gene_names)) if rna_gene_names is not None else set()
    adt_cols = (
        set(map(str, adt_df.columns))
        if (adt_df is not None and not adt_df.empty)
        else set()
    )

    use_direction = (
        adt_pos_only
        and bool(adt_cols)
        and "gene_direction" in gp.columns
    )

    has_rna = gp["gene_id"].isin(rna_cols)
    if use_direction:
        has_adt = gp[adt_map_col].isin(adt_cols) & (gp["gene_direction"] == "pos")
    elif adt_map_col in gp.columns:
        has_adt = gp[adt_map_col].isin(adt_cols)
    else:
        has_adt = pd.Series(False, index=gp.index)

    # Dedup on (gene_id, cell_type): keeps independent columns for shared
    # marker genes across different cell types, but drops duplicate tier rows
    # for the same (gene, cell_type) pair.
    active = (
        gp[has_rna | has_adt]
        .drop_duplicates(subset=["gene_id", "cell_type"], keep="first")
        .reset_index(drop=True)
    )
    n_features  = len(active)
    expr_matrix = np.zeros((n_cells, n_features), dtype=float)
    gene_ids    = active["gene_id"].values

    if rna_cols:
        rna_active_mask  = np.isin(gene_ids, list(rna_cols))
        rna_active_genes = gene_ids[rna_active_mask]
        if len(rna_active_genes):
            rna_col_idx = rna_gene_index.get_indexer(rna_active_genes)
            expr_matrix[:, rna_active_mask] += rna_mat[:, rna_col_idx]

    if adt_cols and adt_map_col in active.columns:
        adt_ids = active[adt_map_col].values
        if use_direction:
            adt_active_mask = (
                np.isin(adt_ids, list(adt_cols))
                & (active["gene_direction"].values == "pos")
            )
        else:
            adt_active_mask = np.isin(adt_ids, list(adt_cols))
        active_adt_ids  = adt_ids[adt_active_mask]
        active_feat_idx = np.where(adt_active_mask)[0]
        if len(active_adt_ids):
            expr_matrix[:, active_feat_idx] += adt_df[active_adt_ids].values

    weight_matrix = np.zeros((n_features, len(cell_types)), dtype=float)
    fi_idx = np.arange(n_features)
    ti_idx = cell_type_index.get_indexer(active["cell_type"].values)
    weight_matrix[fi_idx, ti_idx] = active["gene_priority"].values

    feature_counts = np.bincount(ti_idx, minlength=len(cell_types)).astype(float)
    feature_counts[feature_counts == 0] = 1.0   # guard divide-by-zero

    return expr_matrix, weight_matrix, feature_counts, cell_type_names


def _compute_scores(
    obs_names: pd.Index,
    cell_type_names: np.ndarray,
    expr_matrix: np.ndarray,
    weight_matrix: np.ndarray,
    feature_counts: np.ndarray,
) -> pd.DataFrame:
    raw  = expr_matrix @ weight_matrix      # (n_cells, n_cell_types)
    raw /= np.sqrt(feature_counts)
    return pd.DataFrame(raw, index=obs_names, columns=cell_type_names)


def _apply_cutoff(scores: pd.DataFrame, core_cell_cutoff: float) -> pd.DataFrame:
    arr = scores.values
    arr[arr < core_cell_cutoff] = 0.0
    return scores


def _label_cells(
    scores: pd.DataFrame,
    tier_map: dict,
    similarity_threshold: float,
    set_final_on_double: bool = False,
) -> tuple[pd.Series, pd.Series]:
    """
    Tiered labeling: iterates over tiers in ascending order, assigning labels
    only to still-unlabeled cells.  Fully vectorised per tier.

    Parameters
    ----------
    set_final_on_double:
        When ``True``, non-accepted double-label cells still have
        ``final_labels`` set to the top-scoring type (RNA-only behaviour).
        When ``False``, they remain ``"unknown"``.
    """
    n_cells    = len(scores)
    final_arr  = np.full(n_cells, "unknown", dtype=object)
    double_arr = np.full(n_cells, "not_double_label", dtype=object)
    unlabeled  = np.ones(n_cells, dtype=bool)

    col_names  = np.asarray(scores.columns)
    scores_arr = scores.values

    for tier in sorted(tier_map.keys()):
        if not unlabeled.any():
            break

        type_idx = np.where(np.isin(col_names, tier_map[tier]))[0]
        ul_idx   = np.where(unlabeled)[0]
        arr      = scores_arr[np.ix_(ul_idx, type_idx)]    # (n_ul, n_types)
        n_types  = arr.shape[1]

        if n_types == 1:
            top1_local = np.zeros(len(ul_idx), dtype=int)
            top1_val   = arr[:, 0]
            top2_val   = np.zeros(len(ul_idx))
        else:
            part       = np.argpartition(arr, -2, axis=1)
            top2_local = part[:, -2]
            top1_local = part[:, -1]
            rows       = np.arange(len(ul_idx))
            top1_val   = arr[rows, top1_local]
            top2_val   = arr[rows, top2_local]

            # Fix any reversed top-2 slots from argpartition
            swap = top2_val > top1_val
            if swap.any():
                tmp                = top1_local[swap].copy()
                top1_local[swap]   = top2_local[swap]
                top2_local[swap]   = tmp
                tmp                = top1_val[swap].copy()
                top1_val[swap]     = top2_val[swap]
                top2_val[swap]     = tmp

            # tie-break: match pandas sort_values (stable/mergesort)
            tied = (top1_val == top2_val) & (top1_val > 0)
            if tied.any():
                tied_arr     = arr[np.ix_(np.where(tied)[0], np.arange(arr.shape[1]))]
                stable_order = np.argsort(-tied_arr, axis=1, kind="mergesort")
                top1_local[tied] = stable_order[:, 0]
                top2_local[tied] = stable_order[:, 1]
                tied_rows        = np.where(tied)[0]
                top1_val[tied]   = arr[tied_rows, top1_local[tied]]
                top2_val[tied]   = arr[tied_rows, top2_local[tied]]

        has_any   = top1_val > 0
        is_double = has_any & (top2_val > 0) & ((top1_val - top2_val) < similarity_threshold)
        is_single = has_any & ~is_double

        if is_single.any():
            abs_single            = ul_idx[is_single]
            final_arr[abs_single] = col_names[type_idx[top1_local[is_single]]]
            unlabeled[abs_single] = False

        if is_double.any():
            abs_double = ul_idx[is_double]
            t1_names   = col_names[type_idx[top1_local[is_double]]]
            t2_names   = col_names[type_idx[top2_local[is_double]]]

            swap_names   = t1_names > t2_names
            a            = np.where(swap_names, t2_names, t1_names)
            b            = np.where(swap_names, t1_names, t2_names)
            pair_strings = np.char.add(np.char.add(a, "_"), b)

            double_arr[abs_double] = pair_strings
            if set_final_on_double:
                final_arr[abs_double] = t1_names
            unlabeled[abs_double] = False

    idx = scores.index
    return pd.Series(final_arr, index=idx), pd.Series(double_arr, index=idx)


def _resolve_doubles(
    final_labels: pd.Series,
    double_labels: pd.Series,
    accepted_doubles_set: set[str],
) -> pd.Series:
    is_accepted = double_labels.isin(accepted_doubles_set)
    if not is_accepted.any():
        return final_labels
    final_labels                = final_labels.copy()
    final_labels[is_accepted]   = double_labels[is_accepted]
    return final_labels
