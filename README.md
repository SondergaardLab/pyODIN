# py_odin

ODIN cell-type annotation toolkit for the `scanpy` ecosystem.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Tutorial](#tutorial)
  1. [Get data preprocessed](#1-get-data-preprocessed)
  2. [Read gene priority tables](#2-read-gene-priority-tables)
  3. [Normalize ADT counts](#3-normalize-adt-counts)
  4. [Score cells and get a cluster-level consensus](#4-score-cells-and-get-a-cluster-level-consensus)
  5. [Refine a broad population at finer resolution](#5-refine-a-broad-population-at-finer-resolution)
  6. [Clean up label names](#6-clean-up-label-names)
  7. [Fill in remaining "unknown" cells (optional)](#7-optional-fill-in-remaining-unknown-cells)
- [Reference](#reference)
- [License](#license)

## Requirements

- Python >=3.9
- License: MIT
- Core dependencies (installed automatically when you install `py_odin`):
  `numpy`, `pandas`, `scanpy`, `scipy`, `igraph`, `lightgbm`, `leidenalg`,
  `muon`, `openpyxl`, `scikit-learn`.

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.
`uv add` runs inside a uv-managed project. If you don't already have one,
run `uv init` first. For example (this install uv and scvi-tools for getting tutorial dataset):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
mkdir pyODIN_tutorial
cd pyODIN_tutorial
uv init --python 3.10
uv add git+https://github.com/SondergaardLab/pyODIN.git
uv add scvi-tools
```

This installs `py_odin` along with every dependency declared in its
`pyproject.toml`.

(!Not recommended) If you are using macOS, and do not have homebrew installed, additional steps are needed to get `lightGBM` working:
```bash
uv pip uninstall lightgbm
uv pip install lightgbm --no-binary lightgbm --config-settings=cmake.define.USE_OPENMP=OFF
```

If you have homebrew installed: 
```bash
brew install cmake libomp
```


## Tutorial

This walks through a full annotation pass: preprocessing, per-cell scoring,
cluster-level consensus, and fine resolution classification.

### 1. Get data preprocessed

`py_odin` scores cells against marker panels. 
This tutorial uses scvi-tools' built-in CITE-seq PBMC dataset:

```python
import scvi
import scanpy as sc
import numpy as np
import pandas as pd
import muon as mu

from py_odin import score, score_clusters


# Get the single-cell object. Download it directly, or read a local copy:
adata = scvi.data.pbmc_seurat_v4_cite_seq(apply_filters=False)
# adata = sc.read_h5ad("./data/pbmc_seurat_v4.h5ad")


# Quality Control & Filtering
adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(
    adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
)

# Filter cells and genes
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
adata = adata[adata.obs.pct_counts_mt < 20, :].copy()

# Normalization and Log Transformation
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Highly Variable Gene (HVG) Selection
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)

# Scaling & PCA
sc.pp.scale(adata, max_value=10, zero_center=False)
sc.tl.pca(adata, svd_solver="arpack")

# Clustering
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=2, flavor='igraph')
```

### 2. Read gene priority tables

Gene priority tables will be available after journal publication.

```python
gene_priority_table    = pd.read_excel("marker_table.xlsx")
accepted_doubles_table = pd.read_excel("marker_table.xlsx", sheet_name="accepted_doubles")
```

### 3. Normalize ADT counts

`score()` expects already-normalized ADT values in `adata.obsm`. A standard
approach is CLR (centered log-ratio) normalization followed by scaling,
using `muon`:

```python
# Normalize the ADT
adt_adata = sc.AnnData(adata.obsm['protein_counts'])

adt_adata = mu.prot.pp.clr(adt_adata, inplace=False)
sc.pp.scale(adt_adata, zero_center=False, max_value=10)

adata.obsm['protein_counts_scaled'] = pd.DataFrame(
    adt_adata.X,
    columns=adt_adata.var_names,
    index=adt_adata.obs_names)
```

`protein_counts_scaled` is now in `adata.obsm`. Pass it to `score()` in the
next step via `use_modality="both"`, `adt_obsm_key`, and `adt_map_col`.

### 4. Score cells and get a cluster-level consensus

```python
adata = score(
    adata,
    gene_priority_table=gene_priority_table,
    accepted_doubles_table=accepted_doubles_table,
    cell_level="Top",         
    use_modality="both",
    adt_obsm_key="protein_counts_scaled",
    adt_map_col="adt_id",
    core_cell_cutoff=1,
    similarity_threshold=1,
)
adata = score_clusters(adata, clustering_column="leiden")

sc.pl.umap(adata, color="odin_classification")
```

`score()` writes per-cell `final_labels` (and the underlying score matrix
to `adata.uns['odin_score_all']`). `score_clusters()` smooths those into a
per-_cluster_ consensus in `adata.obs['odin_classification']`.

### 5. Refine a broad population at finer resolution

```python
broad_label = "CD4_T"
subset = adata[adata.obs["odin_classification"] == broad_label].copy()

subset = score(
    subset,
    gene_priority_table=gene_priority_table,
    accepted_doubles_table=accepted_doubles_table,
    cell_level="CD4_T",
    use_modality="both",
    core_cell_cutoff=1,
    similarity_threshold=1,
)

adata.obs["odin_classification"] = adata.obs["odin_classification"].astype(str)
adata.obs.loc[subset.obs_names, "odin_classification"] = subset.obs["final_labels"].astype(str).values
adata.obs["odin_classification"] = adata.obs["odin_classification"].astype("category")
```

Repeat for each broad population you want to break down further (B cells,
myeloid, etc.) — each subset call is independent, so you can loop over a
`{broad_label: cell_level}` dict rather than repeating this block by hand.

### 6. Clean up label names

Double-labels that survived resolution show up as `CellTypeA_CellTypeB`
strings. Swap them for the friendly names from `accepted_doubles_table`:

```python
from py_odin import simplify_labels

adata = simplify_labels(adata, accepted_doubles_table, cell_level="Top")
```

### 7. (Optional) Fill in remaining "unknown" cells

If cells are still labeled `"unknown"` after scoring, you can train a
classifier on the confidently-labeled cells and predict the rest. Be aware
that this process is extremely time consuming.:

```python
from py_odin import predict_unknown_cells_lgbm

adata = predict_unknown_cells_lgbm(
    adata,
    label_col="odin_classification",
    feature_layer="odin_score_all",   # train on scODIN scores rather than raw expression
)
```

This writes `adata.obs['lgbm_pred_odin_classification']`, with low-confidence
predictions prefixed `Uncertain_` so you can tell them apart from confident
calls at a glance.

---

### From source (editable, for development)

Use `uv sync`  to set up a local development environment:


## Reference

See the docstrings on the following for the full parameter reference:

- `score`
- `score_clusters`
- `simplify_labels`
- `predict_unknown_cells_lgbm`

## License

MIT License

Copyright (c) pyODIN authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
