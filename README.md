# Melbourne Airbnb — COMP20008 Assignment 2

**RQ**: Can superhost status be inferred from listing attributes and host operating behaviour once
Airbnb's own criteria are excluded, and is it more closely associated with listing fundamentals or
operational strategy?

## Run order

Run each script from anywhere — each finds the project root itself. Run them in order; each reads
what the previous one wrote.

| # | Script | Reads | Writes |
|---|---|---|---|
| 1 | `new_resource_code/1_preprocess.py` | `data/listings.csv` | `data/clean.parquet`, `outputs/evidence.json` |
| 2 | `new_resource_code/2_correlation.py` | `data/clean.parquet` | `outputs/correlation_matrix.csv`, `outputs/fig_hlc_vs_y.png`, `outputs/fig_dsl_vs_dist.png`, `outputs/evidence_correlation.json` |
| 3 | `new_resource_code/3_model.py` | `data/clean.parquet` | `outputs/cv_results.csv`, `outputs/rq_experiments.csv`, `outputs/feature_ranking.csv`, `outputs/fs_validation.csv`, `outputs/fig_confusion.png`, `outputs/fig_importance.png`, `outputs/evidence_model.json` |
| 4 | `new_resource_code/4_pca_cluster.py` *(Stage C — not written yet)* | `data/clean.parquet` | `outputs/evidence_cluster.json`, `outputs/pca_table.csv`, `outputs/cluster_profiles.csv`, `outputs/fig_pca_clusters.png` |

Dependencies: `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `pyarrow`; step 4 also needs
`fastcluster`. Step 3 takes about 3 minutes — run it end to end in a terminal, not section by
section in VS Code, or variables left over from an earlier run will silently produce wrong numbers.

## Rules that keep the report consistent

1. **Every number in the report comes from `outputs/`.** If the report and an `outputs/` file
   disagree, the file is right and the report changes.
2. **Steps 2–4 read only `data/clean.parquet`.** Do not re-clean `listings.csv` separately.
3. Each `new_resource_code/README_*.md` explains what the keys in the matching `evidence_*.json` mean.

## Data

Retrieved from https://insideairbnb.com/get-the-data/ on 17/09/2026 (Melbourne, Victoria, Australia).
Both CSVs are stored with Git LFS. This is the raw Inside Airbnb download — the A1 dataset must not
be used (Amendment #1).

## Submission

One `code.ipynb` (assembled from the `# %%` cells of the four scripts) plus the Word report in
`report/`.
