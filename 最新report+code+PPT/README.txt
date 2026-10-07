COMP20008 Assignment 2 - Group W06G02 - Code README
=====================================================

Research question
-----------------
Can superhost status be inferred from listing attributes and host operating
behaviour once Airbnb's own criteria are excluded, and is it more closely
associated with listing fundamentals or operational strategy?


1. Files
--------
code.ipynb    The single notebook that produces every result in the report.
README.txt    This file.

Input (not submitted): data/listings.csv - Inside Airbnb, Melbourne, Victoria,
Australia (25,728 listings x 90 columns), http://insideairbnb.com/get-the-data/


2. How to run
-------------
1) Folder layout:
       code.ipynb
       data/listings.csv
   The notebook uses its own folder as the project root; it creates outputs/
   and writes data/clean.parquet itself.
2) Install the requirements (versions used for the reported results):
       Python 3.13, pandas 3.0.6, numpy 2.5.3, scikit-learn 1.9.1, scipy 1.18.1,
       matplotlib 3.11.2, seaborn 0.13.2, pyarrow (for the parquet file)
   e.g.  pip install pandas numpy scikit-learn scipy matplotlib seaborn pyarrow jupyter
3) Open code.ipynb and run Kernel -> Restart & Run All. Run the parts in order:
   Parts 2-4 read data/clean.parquet, which Part 1 writes.

Runtime: about 2 minutes on a laptop; Part 3 (k-NN tuning, permutation
importance, ablations) takes most of it. Memory: the Ward clustering in Part 4 runs on all
21,251 reviewed listings and peaks at about 4 GB RAM, so use a machine with 8 GB.
Reproducibility: every random step uses RANDOM_SEED = 20008 (Part 0), so a re-run
gives identical numbers.


3. Implementation
-----------------
Part 0  Setup: imports, paths, the global seed, a rounding helper.

Part 1  Preprocessing (all 25,728 rows; no row is dropped)
        - Parses price, amenity_count (json.loads) and the target y.
        - Step 1: days_since_last_review from last_review (reference date = latest
          review, 2026-06-28); the 4,477 zero-review listings get max + 1 = 4,594,
          plus a has_review flag.
        - Step 2: amenity_count; also measures the Assignment 1 comma-split error.
        - Step 3: price median imputation plus a price_missing indicator.
        - Derived feature dist_cbd (planar distance to Flinders Street Station).
        - Evidence for the three rejected candidate steps, plus host-level counts
          (the label belongs to the host) cited in the report.

Part 2  Correlation (21,251 reviewed listings)
        - 5 variables (host_listings_count, days_since_last_review,
          availability_365, dist_cbd, y) -> 10 pairs x Pearson, Spearman, MI, NMI.
        - MI/NMI on 10 equal-frequency bins (MI reported in nats and in bits);
          evidence for the binning choice.
        - Superhost rate by host_listings_count band; review recency by distance band.

Part 3  Supervised learning and feature selection (all 25,728 rows)
        - 80/20 split with StratifiedGroupKFold (grouped by host_id, stratified on y);
          class counts and the size of every CV validation fold.
        - 5-fold grouped stratified CV on the training set, scored by superhost-class F1:
          k-NN over k, decision tree over max_depth; majority-class baseline.
        - Test-set evaluation (per-class precision/recall/F1, macro-F1, training F1,
          confusion matrices, absolute gain over the baseline) and a paired
          host-level bootstrap (1,000 resamples) for 95% intervals.
        - Six feature-set experiments (fundamentals vs operational, with ablations),
          plus four more: without amenity_count, without price, and price moved
          between the two families.
        - Feature influence: tree impurity importance; permutation importance for both models.
        - Feature selection: filter = mutual_info_classif, embedded = tree importance;
          top 3 of each, re-validated by retraining.
        - Hard case: the tree's most confident false negative, against class medians,
          with its path through the tree and its percentile on the top-ranked feature.

Part 4  PCA and clustering (21,251 reviewed listings)
        - Three candidate feature sets compared; the 6-feature mixed set is used.
        - log1p on 3 skewed features, then StandardScaler.
        - PCA: explained variance and top-3 loadings of PC1-PC3; association of each
          component with superhost status.
        - K-Means elbow (SSE) and silhouette over k = 2..8 (k = 5); Ward hierarchical
          with the same k, compared with the single/complete/average linkages from the
          lectures;
          crosstab and ARI between the two, cluster profiles (medians, superhost rate).


4. Outputs and where they appear in the report
----------------------------------------------
All files are written to outputs/. Every number quoted in the report is stored in
one of the four evidence_*.json files, under the key named below.

Report item                                  Output file (key)                        Notebook
(section, table and figure numbers refer to the submitted report)
-------------------------------------------  ---------------------------------------  --------
2.1 / Table 1  Candidate preprocessing steps evidence_preprocess.json                 1.2-1.6
2.2  Binning choice, MI bin sensitivity      evidence_correlation.json                2.1
2.3  Split sizes and class balance           evidence_model.json (split,              3.1-3.1b
                                               split_details)
2.5 / Table 2  Candidate clustering sets     evidence_cluster.json                    4.5b
                                               (candidate_set_comparison)
2.5  k = 5: SSE drops and silhouette         evidence_cluster.json (kmeans_sweep)     4.5
2.5  Why Ward linkage                        evidence_cluster.json                    4.6b
                                               (linkage_comparison_sizes)
3.1 / Table 3  Before/after of the 3 steps   evidence_preprocess.json                 1.2-1.4
3.2 / Table 4  Correlation table (40 values) correlation_matrix.csv (mi = nats,       2.2
                                               mi_bits = bits)
3.2 / Figure 1 Superhost rate by host size   fig_hlc_vs_y.png (hlc_band)              2.4
3.3 / Tables 5-6  CV score per hyperparameter cv_results.csv                          3.3
3.3 / Table 7  Test precision/recall/F1      evidence_model.json (test, test_extra)   3.4-3.4b
3.3 / Table 8  Bootstrap 95% intervals       evidence_model.json (boot_*)             3.4-3.5
3.4 / Table 9  Feature ranking: MI,          feature_ranking.csv, evidence_model.json 3.6-3.7
               impurity, permutation           (permutation_importance)
3.4  Hard case 6812677                       evidence_model.json (hard_case,          3.8-3.8b
                                               false_negative_context)
3.5 / Table 10 PCA variance and loadings     pca_table.csv                            4.4
3.5 / Figure 2 PCA scatter, K-Means centroids fig_pca_clusters.png                    4.9
3.5 / Table 11 K-Means and Ward profiles     cluster_profiles.csv                     4.8
4.1  Ablations G, H                          rq_experiments_extra.csv                 3.5b
4.2  dsl ~ dist_cbd sign flip                fig_dsl_vs_dist.png (dist_band_trend)    2.5
4.3  Feature-set experiments A-F, TP/FP      rq_experiments.csv, evidence_model.json  3.4, 3.5
                                               (test_extra)
4.5  Components vs superhost status          evidence_cluster.json (pc_vs_superhost)  4.4b
4.5  K-Means vs Ward comparison              evidence_cluster.json (crosstab,          4.7
                                               kmeans_vs_hierarchical_ari)

Produced by the notebook but not shown in the report (kept as supporting evidence):
fig_elbow.png (elbow curve), fig_confusion.png (confusion matrices),
fig_importance.png (permutation importance chart), fs_validation.csv (top-3 retraining),
fig_dendrogram.png (Ward dendrogram, 2,000-row sample).

Note: cluster labels are 0-based in the notebook; the report numbers them 1-5.
