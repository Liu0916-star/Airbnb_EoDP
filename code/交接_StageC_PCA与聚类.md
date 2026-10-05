# Stage C handoff: PCA + clustering (K-Means & Hierarchical)

> For the team member handling PCA and clustering. Reading this file is enough to start work: where the data is read from, which features to use, what to do at each step, why each step is done that way, what to write in each report subsection, and how to write the Limitations. All of it is here.
> The code, results and report text for A (preprocessing + correlation) and B (supervised learning + feature selection) are already complete.

---

## 0. One-page overview

| Item | Decision |
|---|---|
| Data | `data/clean.parquet`, filtered to `has_review == 1` → **21,251 rows** (11,474 hosts) |
| Clustering features (6 continuous) | Fundamentals: `dist_cbd`, `accommodates`. Operational: `availability_365`, `amenity_count`, `days_since_last_review`, `host_listings_count` |
| Transform | `days_since_last_review`, `host_listings_count` and `dist_cbd` get `log1p` first, then all 6 columns go through `StandardScaler` |
| PCA | PCA on those 6 transformed columns. The spec requires reporting the explained variance ratio of the first 2–3 components, the 3 largest loadings for each component, and a **PC1 × PC2 scatter plot** (optionally coloured by the target variable y) |
| K-Means | k = 2…8, choosing k with elbow (inertia) + silhouette; `n_init=10, random_state=42` |
| Hierarchical | Ward linkage, **on all 21,251 rows**: use `fastcluster.linkage_vector` (`pip install fastcluster`), measured at 1.3 seconds, with results identical to scipy; cut into the same k as K-Means (a hard requirement of the spec) |
| Comparing the two methods | Adjusted Rand Index (ARI) + cross-tabulation + cluster sizes, all on the 21,251 rows (the rubric requires "actual cluster assignments and sizes" as evidence of whether the two differ) |
| Link to the RQ | **y (superhost) never takes part in clustering.** It is used only afterwards, to compute each cluster's superhost rate |
| Report length | **At most 1 page** (all four subsections together), 1 table + 1 figure |
| Code file | `新代码/pca_cluster.py`, in the same style as the other three files (see section 7) |

---

## 1. Background: our RQ and what has already been established

**RQ**: *Can superhost status be inferred from listing attributes and host operating behaviour once Airbnb's own criteria are excluded, and is it more closely associated with listing fundamentals or operational strategy?*

**Title**: *Operational Strategy or Listing Fundamentals? Predicting Superhost Status in the Melbourne Short-Term Rental Market*

**Wording rule (whole group)**: write only *associated with*, never *driven by / caused by / leads to*.

**What A and B have established** (your Discussion should speak to these):

| Source | Conclusion |
|---|---|
| Correlation | The strongest association with superhost is `days_since_last_review` (NMI 0.0785); the only fundamental variable, `dist_cbd`, is the weakest in the whole table (NMI 0.0047); `host_listings_count` is inverted-U shaped (3–5 listings peaks at 46.08%, over 100 listings only 16.92%) |
| Supervised learning | Fundamentals alone reach macro-F1 0.485 (baseline 0.408), operational alone 0.701, and operational still 0.670 after the review features are removed |
| Feature selection | The top 3 of both methods are entirely operational features; `days_since_last_review` and `host_listings_count` are chosen by both |

**The role of your part**: A and B are both **supervised**, using y. PCA and clustering are **unsupervised** and never look at y. They answer a complementary question:

> Without telling the algorithm who is a superhost, do listings fall naturally into groups mainly by operating behaviour or by fundamentals? And do those natural groups differ much in their superhost rate?

If the clusters separate mainly along operational dimensions, and the superhost rates differ a lot between clusters, that supports B's conclusion from an unsupervised angle. If they do not, that is also a finding worth discussing. **Do not presuppose a conclusion. Write what the data shows.**

---

## 2. Data: what to read, which rows to use

```
data/clean.parquet    25,728 rows × 98 columns, the group's single data entry point (produced by preprocess.py)
```

- Run `新代码/preprocess.py` first to generate it (about 10 seconds)
- **Use only the 21,251 rows with `has_review == 1`.** Reason: for the 4,477 zero-review listings, `days_since_last_review` is the artificial fill value 4,594, so they all pile up on one value and clustering would separate them into a **fake cluster**. The correlation part uses the same subset
- The superhost rate of these 21,251 rows is **35.72%** (30.65% for the full sample). Be careful which figure you quote

**Key numbers** (these must be quoted consistently):

| | Rows | Hosts | Superhost rate |
|---|---|---|---|
| Full sample | 25,728 | 14,113 | 30.65% |
| Reviewed subset (the one you use) | 21,251 | 11,474 | 35.72% |
| Zero-review listings (excluded) | 4,477 | — | 6.59% |

---

## 3. Feature set: three candidates, which one and why (the template requires this)

Template text: *Name 3 candidate feature sets for clustering, justify your choice, and apply it to both K-Means and Hierarchical clustering. Apply PCA to the same feature set.*

### 3.1 The three candidates

| Candidate | Features | Advantage | Disadvantage |
|---|---|---|---|
| **S1 operational only** | `availability_365`, `amenity_count`, `days_since_last_review`, `host_listings_count` | Clusters can be read directly as business models (active / dormant / large commercial operation, and so on) | No fundamentals, so it **cannot answer the question of what the data separates by** |
| **S2 fundamentals only** | `dist_cbd`, `accommodates` | Clusters correspond directly to location and size | Only 2 continuous variables, so PCA is meaningless (at most 2 components); part B already showed the fundamental signal is weak |
| **S3 mixed (chosen)** | S1 + S2, 6 in total | The PCA loadings show directly **whether the main axis of variation is operational or fundamental**; cluster profiles can compare both families | 4 operational against 2 fundamental, which is not balanced (write this into Limitations) |

**Why S3** (one or two sentences in the report): only a set containing both families lets the PCA loadings and cluster centres show which family the data's main structure runs along, and that is the core question of the RQ.

### 3.2 Why these columns are excluded (one sentence each in the report)

| Excluded | Reason |
|---|---|
| `y` / `host_is_superhost` | The target. Clustering must be unsupervised; y is used only **afterwards** to compute each cluster's superhost rate |
| `price_num` | Missing for **4,876 rows (22.9%)** in this subset. Imputing the median would pile a fake cluster on one value, and the skew is extreme (42.35, maximum $50,094 for one night) |
| `minimum_nights` | **87%** of values are ≤ 3 nights, skew 22.22, maximum 1,000. Almost no usable variation; it would only contribute outliers |
| `has_review` | Constant 1 in this subset, so no variation |
| `price_missing`, `room_type_3` | Binary or categorical. K-Means and Ward are both based on Euclidean distance, and mixing 0/1 variables with continuous ones makes the distance meaningless |
| Official criteria and review-derived columns (`review_scores_*`, `number_of_reviews`, and so on) | Excluded group-wide, without exception |

---

## 4. Preprocessing (before clustering)

### 4.1 Log transform

PCA and K-Means are both based on variance or distance, so **extreme values dominate the result**. Skew in the subset:

| Feature | Skew | Maximum | Treatment |
|---|---|---|---|
| `host_listings_count` | 4.44 | 667 | `np.log1p` |
| `days_since_last_review` | 2.19 | 4,593 | `np.log1p` |
| `dist_cbd` | 2.02 | 79.5 km | `np.log1p` |
| `accommodates` | 1.64 | 16 | unchanged (integer, narrow range) |
| `availability_365` | 0.13 | 365 | unchanged |
| `amenity_count` | −0.12 | 92 | unchanged |

`log1p(x) = log(1 + x)`, so it does not break when x = 0.

### 4.2 Standardisation

All 6 columns go through `StandardScaler()`: subtract the mean, divide by the standard deviation, per column. Without it, the variance of `availability_365` (0–365) would overwhelm `accommodates` (1–16) and the first principal component would simply be "days available".

**Note**: there is **no need** for a train/test split here. Clustering is descriptive, with no notion of a "test", so fitting directly on the 21,251 rows is correct.

---

## 5. Analysis steps

### 5.1 PCA

Spec text: *Apply PCA to the same feature set you used for clustering above and report the explained variance ratio for the first 2–3 components. For each of these components, identify the top 3 original variables contributing to it (by loading magnitude) and briefly interpret what that component appears to represent… Add a 2D scatter plot of the first two components, optionally coloured by your research question's target variable.*

1. Fit `PCA()` with no component limit on the 6 standardised columns
2. **Report the first 3 components**: `explained_variance_ratio_` and the cumulative value (the spec asks for the first 2–3; with 6 features, reporting 3 is safest)
3. For each component, list the **3 largest loadings by absolute value** (`components_`), keeping their signs
4. Name each component from those 3 loadings, for example "activity axis", "scale axis", "location axis". The rubric requires the interpretation to be "strictly grounded in its actual top 3 loading variables (not a guessed label)", so **the name must be readable straight off those 3 loadings**
5. **The sign is arbitrary**: a principal component can be flipped as a whole. Interpret by looking at which features share a sign and which oppose, never by saying "the positive direction is better"
6. **Scatter plot**: PC1 × PC2, **coloured by y (superhost / non-superhost)**. This is what the spec suggests, and it shows directly whether superhosts concentrate in one region of principal component space, which is the most direct link to the RQ

**The most important observation for the RQ at this step**: are the top 3 loadings of PC1 (the axis explaining the most variance) mostly operational features or mostly fundamental ones? And do the superhosts separate along one of the axes in the scatter plot?

### 5.2 K-Means

1. Cluster on the **6 standardised columns** (not on the principal components; those are only for plotting)
2. Try k from 2 to 8: `KMeans(n_clusters=k, n_init=10, random_state=42)`
3. Record inertia (elbow method) and the silhouette score for each k
   - Silhouette is O(n²) on the full data and very slow, so use `silhouette_score(..., sample_size=5000, random_state=42)`
4. Choose k by combining the elbow bend and the silhouette peak. **If the two disagree, choose the more interpretable one and say so in the report**
5. Refit the chosen k on all 21,251 rows to get a cluster label per listing

### 5.3 Hierarchical

1. **Run it on all 21,251 rows**, without sampling. The rubric requires it to be "correctly applied to your chosen feature set", and using a sample risks losing marks
2. The standard scipy `linkage` first stores a full pairwise distance table (about 1.8 GB), and machines with little memory get the process killed (I reproduced this on a 3 GB machine). So use **`fastcluster.linkage_vector(X, method="ward")`**:
   - Memory scales with the number of rows only; 21,251 rows ran in **1.3 seconds**
   - Compared against scipy on 2,000 rows with **ARI = 1.0**, identical
   - Install: `pip install fastcluster`. Write this dependency into the README
3. Plot the dendrogram: `scipy.cluster.hierarchy.dendrogram(Z, truncate_mode="lastp", p=30)`, otherwise 20,000 leaves are unreadable
4. Read the largest merge jumps off the dendrogram as a guide to choosing k
5. **Cut it into the same k as K-Means**: `fcluster(Z, t=k, criterion="maxclust")`. The spec requires both methods to use the same k
6. Why Ward: like K-Means, it minimises within-cluster variance, so differences between the two methods come from "hierarchical versus partitioning" rather than from a different objective. Single linkage tends to produce long chained clusters, and complete/average linkage are more sensitive to outliers

### 5.4 Comparing the two methods

Rubric text: *Explicit comparison of whether K-Means and Hierarchical clustering produced materially different groupings, evidenced by actual cluster assignments and sizes.*

- Both methods run on all 21,251 rows, so compare the labels directly
- **Cluster sizes**: how many listings are in each cluster of each method (the rubric names sizes explicitly)
- **Cross-tabulation** `pd.crosstab(kmeans_labels, hierarchical_labels)`: see which clusters correspond and which are split or merged (the rubric names assignments explicitly)
- **ARI** (`adjusted_rand_score`): 1 = identical, 0 = about as similar as random grouping
- Finish with an **explicit verdict**: "materially different" or "largely the same". The rubric requires "a clear verdict"
- Cluster numbering is arbitrary (K-Means cluster 0 does not necessarily correspond to hierarchical cluster 1), so pair them through the cross-tabulation rather than comparing numbers directly

### 5.5 Cluster profiles (required by the template)

For each cluster of each method, compute:
- Number of listings and its share
- The **median of the 6 features in their original units** (days, kilometres, listings, amenities), not the standardised values
- The **superhost rate** (the first point at which y appears; used only for description, never in the clustering itself)
- A descriptive name for each cluster, such as "active mid-sized host", "dormant listing", "large commercial operation"

**The two questions that link back to the RQ**:
1. How far apart are the superhost rates across clusters? (highest cluster against lowest)
2. Which family of features separates the clusters? Look at which features deviate most from 0 in the standardised cluster centres.

### 5.6 Figure (only one goes in the report)

**Scatter plot of PC1 × PC2, coloured by y (superhost)** (the spec requires this figure). Put the variance explained by PC1 and PC2 in the title, and name the components on the axis labels. If there are too many points, sample 5,000 with `alpha=0.3`. You can also project the K-Means cluster centres into PC space and draw them on the same figure with large markers, showing the target and the clusters in one chart.

The scree plot and dendrogram only need to be saved to files, not included in the report (there is no room).

---

## 6. How to write the report (1 page maximum in total)

In the template, "Dimensionality Reduction & Clustering (groups of four only)" appears once in each of the four subsections; delete the grey prompt text once you have written over it.

**Formatting rules (whole group)**: Calibri 11, single spacing; table numbering starts at **Table 6** and figures at **Figure 3** (Tables 1–5 and Figures 1–2 are already used by A and B); every paragraph that draws a conclusion must carry specific numbers.

### 6.1 Methodology (about 130 words)

- One sentence for each of the three candidate feature sets, one sentence on why S3 was chosen
- One sentence on the excluded columns (price missing 22.9%, `minimum_nights` 87% ≤ 3, binary/categorical unsuited to Euclidean distance, y used only for post-hoc description)
- One sentence on using only the 21,251 reviewed rows and why
- Which three columns get `log1p` plus standardisation, and why, in one sentence
- PCA: the criterion for retaining components
- K-Means: the range of k, the two criteria for choosing k, `n_init`
- Hierarchical: Ward, all 21,251 rows (via fastcluster), cut to the same k
- Comparison: cluster sizes + cross-tabulation + ARI
- **Methods only, no results**

### 6.2 Results (about 120 words + 1 table + 1 figure)

Template text: *Present explained variance per component, top-3 loadings per component, and cluster summaries from K-Means and Hierarchical clustering.*

**Suggested structure for Table 6** (squeezing PCA and the cluster profiles into one table):

- Upper part, PCA: one row per component, listing explained variance, cumulative explained variance, and the top 3 loadings (feature name + signed value)
- Lower part, cluster profiles: one row per cluster (listing both K-Means and Hierarchical), with the share, the median of the 6 features, and the superhost rate

**Figure 3**: the PC1 × PC2 scatter plot

The body text states numbers only: how much each of the first 3 components explains, individually and cumulatively; the chosen k and why (silhouette value, elbow bend); each method's cluster sizes and the ARI; and the superhost rates of the highest and lowest clusters. **No reasons.**

### 6.3 Discussion (about 200 words)

Template text: *Interpret what each component and cluster represents, and whether K-Means and Hierarchical produced materially different groupings.*

Three paragraphs:
1. **What each component represents**: name them from the loadings; is PC1 mainly operational or fundamental, and what does that mean for the RQ
2. **What each cluster represents**: describe each cluster's listings using the medians; how far the superhost rate varies across clusters; which family of features separates them. Compare against B's conclusion (operational > fundamentals) and against the inverted-U in `host_listings_count` from the correlation work: do the clusters support, add to, or challenge them?
3. **Whether the two methods differ materially**: cite the ARI and the cross-tabulation. If they differ a lot, a possible explanation is that K-Means assumes spherical clusters of similar size while Ward merges step by step and cannot revise earlier merges. If they differ little, the cluster structure is relatively stable

Use *associated with* throughout. **y is a post-hoc description**: a high superhost rate in a cluster does not mean that belonging to that cluster causes superhost status.

### 6.4 Limitations (about 100 words) — template prompt: *unexplained variance, scaling sensitivity, number of clusters*

**Write specific, numeric limitations, not empty statements like "the sample is small".** One or two sentences each, filling the bracketed values in from your actual results:

| # | Limitation | How to write it |
|---|---|---|
| 1 | **Unexplained variance** | The [n] retained components explain only [x]% of the variance, so the remaining [100 − x]% does not appear in the 2D figure or the interpretation. If PC1 explains little, that itself says no single structure dominates these 6 features |
| 2 | **Sensitivity to scaling and transform** | The result depends on the choice of `log1p` and standardisation. Without the log, the extreme values of `host_listings_count` (maximum 667) would take a cluster of their own. A sensitivity check is available: rerun without the log and report the ARI |
| 3 | **Uncertain cluster count** | The elbow bend is not sharp and the silhouette is only [s] (below 0.5 indicates weak structure), suggesting listings are closer to a continuum than to clearly separated groups. k is a choice made for interpretability, not the one true number in the data |
| 4 | **Ward's greedy merging** | Hierarchical clustering cannot undo a merge, so an early mistake persists; K-Means is sensitive to its initial centres (mitigated by `n_init=10`). If their ARI is only [x], the grouping is not unique |
| 5 | **Reviewed listings only** | The 4,477 zero-review listings (17.4%, superhost rate 6.59%) are excluded, so the clustering does not describe the never-reviewed group, and the cluster superhost rates run higher than the full sample (35.72% against 30.65%) |
| 6 | **Unequal feature families and no categorical variables** | 4 operational against 2 fundamental, and room type is excluded as categorical, so fundamentals are understated. An improvement would be Gower distance or k-prototypes for mixed-type variables |
| 7 | **Hosts are not independent** | 12% of rows come from hosts with more than 50 listings, and one host's listings are nearly identical, which can form tight small clusters and inflate their sizes. An improvement would be one listing per host, or aggregating to host level before clustering |
| 8 | **K-Means' shape assumption** | It assumes spherical clusters of similar size, so long or differently dense groups get cut apart |

**Considered and not used** (one sentence): DBSCAN (sensitive to its parameters, and density is hard to define in 6 dimensions); Gaussian Mixture (allows elliptical clusters, but there is no room in the page budget).

---

## 7. Code conventions (consistent with the other three files)

File: `新代码/pca_cluster.py`

```
"""
COMP20008 A2 — PCA + clustering (part C; run preprocess.py first)
Inputs: data/clean.parquet (only the 21,251 rows with has_review == 1)
Outputs: outputs/evidence_cluster.json, outputs/pca_table.csv, outputs/cluster_profiles.csv
         outputs/fig_pca_clusters.png (report Figure 3), outputs/fig_scree.png, outputs/fig_dendrogram.png
Run: python3 pca_cluster.py; each "# %%" corresponds to one cell of the final notebook
"""
```

Rules:

1. **Paths**: use `PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))`. Never write a relative path like `"outputs"`; running from another directory gives `Read-only file system`, which we have already been bitten by
2. **Sections**: start each section with `# %% Section N. xxx`, and use `# ---- N.1 xxx ----` sub-headings inside a section
3. **Spell variable names out in full**: `listings`, `scaled_features`, `kmeans_labels`, not `df`, `X2`, `km`
4. **A comment on every line**
5. **The ledger**: every number the report cites goes into an `evidence` dictionary, saved as `outputs/evidence_cluster.json`. Copy from there when writing the report
6. **Every random process** uses `random_state=42`
7. **Text in figures must be English**: the matplotlib default font cannot render Chinese and shows boxes instead
8. Save figures with `plt.savefig(...)` + `plt.close()`, **never `plt.show()`** (it hangs when run from a terminal)
9. When finished, **run it end to end from the terminal** (`python3 pca_cluster.py`), not cell by cell in VS Code: variables left in memory hide errors
10. Write a `README_cluster.md`, following the format of `README_model.md`

Existing code worth copying: section 0 of `model.py` (reading data, paths, the ledger) and section 6 (the horizontal bar chart and saving figures).

---

## 8. Things not to do

- ❌ Use y in the clustering or the PCA
- ❌ Use all 25,728 rows (the zero-review listings form a fake cluster)
- ❌ Use the A1 dataset (forbidden by Amendment #1)
- ❌ Use `price_filled` or `price_num` as clustering features
- ❌ Use scipy's `linkage` for hierarchical clustering on the full set (it runs out of memory); use fastcluster
- ❌ Use different k for the two methods
- ❌ Write causal sentences such as *superhost status is caused by cluster membership*
- ❌ Exceed one page in the report

---

## 9. Deliverables

| Deliverable | For |
|---|---|
| `新代码/pca_cluster.py` (runs end to end) | The whole group |
| `outputs/evidence_cluster.json` + 3 figures + 2 csv files | The whole group |
| `新代码/README_cluster.md` | The whole group |
| The four report subsections plus Table 6 and Figure 3, written straight into `A2_Report_MASTER.docx` | The whole group |

**Timing**: the report and code are due on 9 October. **A first draft by 6 October is recommended**, leaving 2 to 3 days for the group to unify formatting and cut pages. The report is currently about 8.5 pages, and with the Introduction and Conclusion added there is only about 1 page left for your part, so length is a hard constraint.

If any number or definition does not reconcile, `outputs/` json and csv files are the authority.
