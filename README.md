# Melbourne Airbnb — COMP20008 Assignment 2

**RQ**: Can superhost status be inferred from listing attributes and host operating behaviour once
Airbnb's own criteria are excluded, and is it more closely associated with listing fundamentals or
operational strategy?

## Run order

Run the notebooks in order; each reads what the previous one wrote. **Start Jupyter with the repo
root as the working directory** — the notebooks resolve paths from the current working directory, so
running one from inside `code/` will not find `data/`.

| # | Notebook | Reads | Writes |
|---|---|---|---|
| 1 | `code/1_preprocess.ipynb` | `data/listings.csv` | `data/clean.parquet`, `outputs/evidence.json` |
| 2 | `code/2_correlation.ipynb` | `data/clean.parquet` | `outputs/correlation_matrix.csv`, `outputs/fig_hlc_vs_y.png`, `outputs/fig_dsl_vs_dist.png`, `outputs/evidence_correlation.json` |
| 3 | `code/3_model.ipynb` | `data/clean.parquet` | `outputs/cv_results.csv`, `outputs/rq_experiments.csv`, `outputs/feature_ranking.csv`, `outputs/fs_validation.csv`, `outputs/fig_confusion.png`, `outputs/fig_importance.png`, `outputs/evidence_model.json` |
| 4 | `code/4_pca_cluster.ipynb` *(Stage C — not written yet)* | `data/clean.parquet` | `outputs/evidence_cluster.json`, `outputs/pca_table.csv`, `outputs/cluster_profiles.csv`, `outputs/fig_pca_clusters.png` |

Dependencies: `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `pyarrow`, plus `ipykernel` to run
the notebooks; notebook 4 also needs `fastcluster`. Notebook 3 takes about 3 minutes. Restart the
kernel and run all cells top to bottom — a cell run out of order will silently reuse a stale
variable.

## Rules that keep the report consistent

1. **Every number in the report comes from `outputs/`.** If the report and an `outputs/` file
   disagree, the file is right and the report changes.
2. **Notebooks 2–4 read only `data/clean.parquet`.** Do not re-clean `listings.csv` separately.
3. Each `code/README_*.md` explains what the keys in the matching `evidence_*.json` mean.

## Data

Retrieved from https://insideairbnb.com/get-the-data/ on 17/09/2026 (Melbourne, Victoria, Australia).
Both CSVs are stored with Git LFS. This is the raw Inside Airbnb download — the A1 dataset must not
be used (Amendment #1).

## Submission

One consolidated notebook (merge the four in `code/`) plus the Word report in `report/`.
