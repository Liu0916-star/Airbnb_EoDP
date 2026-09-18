# Candidate research questions

For team discussion. Every number below was computed from the current
`data/listings.csv` — re-run them before quoting, but nothing here is inherited
from earlier drafts.

---

## 1. What the data actually contains

We are a group of **four**, so PCA and clustering are mandatory (2 marks of
Methods, 4 marks of Interpretation). Any RQ below has to support them.

`data/listings.csv` — **25,728 rows × 90 columns, 14,113 distinct hosts.**

### 1.1 It is two populations stitched together

`source` splits the file in two, and the two halves are not comparable:

| | `city scrape` | `previous scrape` |
|---|---|---|
| rows | 18,885 | 6,843 |
| scraped | 2026-06-17 | 2026-06-28 / 06-29 / 07-01 |
| `price` missing | **1.29%** | **92.20%** |
| superhost rate | 38.18% | 9.85% |
| `availability_365` median | 236 | **0** |
| `estimated_occupancy_l365d` median | 36 | **0** |
| `accommodates` median | 4 | 2 |

The two sets share **zero** listing IDs. `previous scrape` rows are listings
found in an earlier sweep that were *not* found in the current one — they are
delisted. The arithmetic confirms the mechanism: 6,843 × 0.922 = 6,309, and the
file has 6,553 missing prices in total. Essentially all price missingness is
these rows.

This matters because 6,553 / 25,728 = **25.5%** of the file has no price, no
availability and a superhost rate a quarter of the live half's. Leaving them in
means roughly a quarter of every model's training data is dead listings.

**Decision this forces:** restrict to `city scrape`, or keep everything and treat
`source` as a feature. The first is cleaner and leaves 18,885 rows with 1.29%
price missing. The second is defensible but needs the delisted population
handled explicitly rather than ignored. Either way it is a *quantified*
preprocessing decision, which the rubric rewards.

### 1.2 Fourteen columns are 100% empty

`neighborhood_overview`, `host_since`, `host_response_time`,
`host_response_rate`, `host_acceptance_rate`, `host_thumbnail_url`,
`host_neighbourhood`, `host_total_listings_count`, `host_verifications`,
`neighbourhood`, `neighbourhood_group_cleansed`, `calendar_updated`, `license`,
`instant_bookable`.

Worth knowing: `host_response_rate`, `host_acceptance_rate` and `host_since` are
all Airbnb *superhost criteria*. They are unusable here. Any RQ about superhost
status is building a model that cannot see the label's own definition.

Also `has_availability` has **1 distinct value** across the whole file — a
constant column, 0.6% null.

### 1.3 Missingness elsewhere

| column | null % | note |
|---|---|---|
| `price` | 25.5% | almost all from `previous scrape` |
| `bathrooms` | 31.7% | but `bathrooms_text` is only 0.1% null |
| `beds` | 27.4% | |
| `bedrooms` | 18.2% | |
| `review_scores_*` | 17.4% | null exactly when `number_of_reviews == 0` |
| `reviews_per_month` | 17.4% | same rows |

`last_review` is null for exactly the rows where `number_of_reviews == 0`
(verified: identical counts, 4,477 rows). Those rows have a superhost rate of
**6.59%** against **35.72%** for the rest — near-deterministic negatives, because
Airbnb requires a minimum number of stays to qualify.

### 1.4 Useful column facts

- `room_type`: Entire home/apt 18,829 · Private room 6,616 · Shared room 231 ·
  Hotel room 52
- `neighbourhood_cleansed`: 30 values · `property_type`: 82 values
- `calculated_host_listings_count`: median 2, max 319
- `minimum_nights`: median 2, max 1000. Only 426 listings (1.66%) require 30+
  nights.
- `estimated_occupancy_l365d` has **no nulls at all** (87 distinct values) —
  unusual, and potentially useful as a feature.

### 1.5 Price, for reference

Non-null prices (19,175 rows): median **243.67**, mean 317.65, p25 158.5,
p75 348, p90 510.6, p95 683.7, p99 1,590, **max 50,093.56**.

That right tail is roughly 200× the median. Any distance-based method will be
dominated by it unless it is handled deliberately.

---

## 2. Candidate A — Price tiers

> **Can Melbourne listings be sorted into price tiers, and do the tiers divide
> along property attributes or along how the host operates the listing?**

### Why it is strong

- **The skew is an asset, not a problem.** With p25 = 158.5, median = 243.67 and
  max = 50,093, Pearson and Spearman will disagree visibly on any pair involving
  price. The rubric wants a discussion of which methods agree and which diverge,
  and a real reason why — this hands us one.
- **Three tiers makes per-class precision/recall/F1 meaningful.** A binary target
  lets you get away with looking at one class. Three does not, and the rubric
  asks for per-class metrics on all models.
- **The preprocessing decision is already built in** (§1.1), with before/after
  numbers on the table.
- **Clustering has a natural feature set**, and it is one of the three the
  specification itself suggests (price, accommodates, bedrooms,
  review_scores_rating).
- **A real Melbourne hook:** location is the one attribute a host cannot change,
  so "does the tier follow the property or the operator?" is a question worth
  asking.

### What it costs us

- It is the most commonly attempted Airbnb question. Nothing wrong with that —
  the rubric scores depth of justification, not novelty — but we will not earn
  anything for originality.
- We must exclude `estimated_revenue_l365d`, which is `price × occupancy` and
  would leak the target outright.

### Open decisions

- Where the tier boundaries go. Quartiles balance the classes by construction,
  which makes the class-balance discussion trivial. Round numbers tied to the
  market (A$150 / A$350) give a mildly imbalanced 25/50/25 split and something
  real to justify. **Recompute the bands after the `source` filter is applied** —
  the percentiles above are on the unfiltered file.

---

## 3. Candidate B — Commercial operator detection

> **Can a commercial short-stay operator be told apart from someone letting a
> spare property, using the listing record alone?**

### Why it is strong

- **Policy relevance.** Victoria's short-stay levy debate is specifically about
  commercial operators. "Can you detect them from listing data?" is a real
  regulatory question.
- **The unit-of-analysis wrinkle is a good finding:** 55.76% of *listings* belong
  to a host with more than one listing, but only 2,730 of 14,113 *hosts* (19.3%)
  are multi-listing hosts. Two different base rates depending on whether you
  count listings or hosts — and the report has to pick one and say why.
- **Splitting by host is mandatory and provable**, not a judgement call: the
  label is constant within a host by construction, so a random split leaks
  deterministically.
- Works on the full file, so we are not throwing rows away.

### What it costs us

- **The signal is thin.** Mutual information between each candidate feature and
  a `>5 listings` label:

  | feature | median (pro) | median (non-pro) | MI |
  |---|---|---|---|
  | `availability_365` | 211 | 117 | 0.0503 |
  | `availability_30` | 16 | 9 | 0.0476 |
  | entire-home flag | 81.3% | 69.2% | 0.0385 |
  | amenity count | 40 | 34 | 0.0311 |
  | `accommodates` | 4 | 3 | 0.0262 |
  | `minimum_nights` | 2 | 2 | 0.0242 |
  | `bedrooms` | 2 | 2 | 0.0218 |
  | `price` | 258 | 232 | 0.0076 |

  Everything sits between 0.008 and 0.050. The strongest predictor is
  `availability_365`, which is arguably *part of what being a professional
  operator means* rather than an independent signal.
- Neighbourhood pro-rates are concentrated but not decisive: Melbourne 51.9%
  (n=8,375) down to Frankston 8.7% (n=263).
- **Mild circularity.** A full-time operator has high availability partly by
  definition. We would need to be careful about which side of that line each
  feature sits on.

---

## 4. Candidate C — Delisted vs live listings (recommend against)

> What distinguishes listings that have left the market from those still
> trading?

Tempting because §1.1 hands us the label for free (26.6% / 73.4%), but I would
not build the project on it:

- The label is an artefact of how Inside Airbnb assembles the file, not an
  observed market event. A listing could be missing from the city sweep for
  reasons unrelated to whether it is actually still trading.
- `price` missingness would be circular with the label (92.2% vs 1.29%), so our
  most interesting column is unusable as a predictor.
- It is much better used as **a preprocessing decision and a Limitations point**
  under whichever RQ we pick.

---

## 5. Recommendation

**Candidate A**, with B's operator angle kept as a *predictor* rather than the
target (split hosts into single-listing / small operator / large operator and let
it be a feature).

The reasoning is mark-driven. A is the only option that gives us something
substantive to say in *every* section the rubric scores: a quantified
preprocessing decision (§1.1), a genuine Pearson-vs-Spearman divergence driven by
the price tail, a three-class target that forces per-class metrics, and a
clustering feature set the specification already sanctions.

B is a better story and would be more interesting to present, but with every MI
below 0.05 the modelling section risks reading as "the models barely beat the
baseline, and we are not sure why". That is a harder report to write well.

---

## 6. Open questions for the team

1. **`source` filter — yes or no?** My view is yes (18,885 live listings, 1.29%
   price missing), but it costs us 27% of the rows.
2. **Tier boundaries** if we take A — quartiles, or market-meaningful round
   numbers?
3. **Do we want a third model?** The rubric allows Logistic Regression, SVM or
   Random Forest, but only earns marks if it is tuned and justified to the same
   standard as the other two. A three-class target makes an interpretable third
   model genuinely useful.
4. **Does anyone have a Melbourne angle we should build on instead?** The
   short-stay levy, the 2025 regulatory changes, or anything from the news would
   make the Introduction much stronger than a generic "Airbnb is big" opener.
