## Key Principles:

We value depth of analysis and clear justification over the number of techniques used. Every interpretive claim in your report must be supported by a specific value from your own output (a coefficient, a count, a row ID, a metric).

These requirements apply across every section below:

• Numbers in every interpretive paragraph. Every paragraph that draws a conclusion must cite at least one specific value from your own output.

• Cross-section consistency. Your report should be internally consistent — scores, figures, rates, and counts should agree wherever they recur.

Generic justification earns no marks. A justification that could be written without reference to your own computed output will receive no marks for that criterion, regardless of whether it happens to be correct.

## Methods — 5 marks (groups of 3) / 7 marks (groups of 4)

## Preprocessing (1 mark)

• Name 6 candidate preprocessing steps in total (from the provided list, of your own design, or a mix).

• Justify the 3 steps you selected over the other 3 you named, with reference to a property of the dataset relevant to your analysis.

• Your chosen steps should be logically useful and meaningful for your research question, not just individually correct.

• Correct implementation of your 3 chosen steps.

## Correlation (1 mark)

• Justify your chosen variable set based on your research question.

• Explain and justify any implementation choices you made when computing the coefficients (e.g. discretisation, encoding).

• Note where a method is not appropriate for a given pair, and why

• Correct computation of all four methods (Pearson, Spearman, Mutual Information, Normalised Mutual Information) across every variable pair in your set.

## Supervised Learning (2.5 marks)

• KNN and Decision Tree correctly set up: clear target, stratified split, 5-fold stratified CV for tuning.

• Train/test split strategy is justified against your target's actual class balance, not just present.

• A validation strategy for hyperparameter tuning (e.g., 5-fold stratified) is justified given your dataset's size and balance.

• Report scores for every hyperparameter value tried, not just the best one.

• Choose and justify hyperparameters that genuinely influence model behaviour.

• For every hyperparameter changed from its default: state the default, your chosen value, and its specific effect on your reported metric.

• Evaluation metrics beyond accuracy are justified as suited to your RQ, and per-class precision, recall, and F1-score (or another justified metric) are reported for all models.

• If used, justify the optional third model.

## Feature Selection (0.5 marks)

• One embedded method and one filter method correctly applied.

• Top 3 features from each method correctly derived.

## PCA — groups of 4 only (0.5 mark)

• Explained variance ratio reported for the first 2–3 components.

• Top 3 loading variables reported for each component.

## Clustering — groups of 4 only (1.5 marks)

• Name 3 candidate feature sets (from the provided list, of your own design, or a mix) and justify the one you selected.

• K-Means and Hierarchical clustering both correctly applied to your chosen feature set.

• For K-Means, k is chosen and justified (via the elbow method, or a value tied to the research question), not picked arbitrarily.

## Data Analysis & Interpretation — 11 marks (groups of 3) / 15 marks (groups of 4)

## Impact of Preprocessing on Results (1.5 marks)

• Explain how each of your chosen preprocessing steps measurably changed your dataset, correlations, model performance, or clusters — with a specific before/after value, not a general statement.

## Correlation Interpretation (2.5 marks)

• Report all four values (coefficients or scores, as applicable) for every variable pair in the full set, including the added target variable.

• Discuss which method(s) agree and which diverge, and why.

• Explain what the analysis reveals about the research question and what that suggests.

• Explain how these findings influenced later decisions

• Use non-causal language throughout; acknowledge possible confounders or biases.

## Model Evaluation & Comparison (2.5 marks)

• Quantify uncertainty in one reported metric for at least one model (e.g. bootstrap CI) and justify why that method was chosen over the alternatives.

• Report a justified baseline (e.g. majority-class) using the same metrics as your models, and state how much your model improves on it in absolute terms.

• Go beyond metrics: discuss feature influence, model behaviour and limitations, and what the results suggest about your research question.

## Feature Selection Interpretation (2 marks)

• Explain any disagreement between your embedded and filter method's top 3 features, and why the two methods might rank a feature differently.

• Identify one specific listing (by ID) where the two methods' top-ranked feature would lead to different predictions, or which is a genuine outlier on the top-ranked feature. Quote its actual attribute values and explain why it is a hard case.

• Discuss one concrete scenario in which the filter method could rank a feature highly while the embedded method ranks it low (or vice versa), grounded in a property of your own feature set.

## Limitations and Improvement Opportunities (1.5 marks)

• Limitations should cover each major section of the analysis, not just a generic closing paragraph. Specifically:

◦ preprocessing (e.g., information lost, assumptions made),

◦ correlation (e.g., sample size, outlier sensitivity, non-independence),

◦ supervised learning (e.g., class imbalance, overfitting risk, missing features),

◦ feature selection (e.g., correlated features splitting importance, method-specific bias),

◦ for groups of four: PCA (e.g., unexplained variance, scaling sensitivity) and clustering (e.g., feature choice, number of clusters, scaling).

• Limitations are specific to each group's methods/results, not generic boilerplate ("small sample size", "data could be biased") that could appear unchanged in any data-processing report.

• Discussion of alternative approaches or explanations not taken.

## PCA Interpretation — groups of 4 only (1.5 marks)

• Interpretation of what each reported component represents, grounded in its actual top 3 loading variables.

• Connection between the components and the research question.

## Clustering Interpretation — groups of 4 only (2.5 marks)

• Clear description and differentiation of clusters, with interpretation of what each represents.

• Discussion of what the clusters reveal about the research question and their usefulness in answering it.

• Explicit comparison of whether K-Means and Hierarchical clustering produced materially different groupings, evidenced by actual cluster assignments and sizes.

## Cross-Section Consistency (1 mark) - same for both group sizes

• Your report should be internally consistent: all values in your report (e.g., rates, counts, coefficients, metrics, etc) should agree with one another wherever the same quantity recurs across different sections.

## Report Quality — 2 marks

## Visualisation (1 mark):

• Figures and tables are clear, correctly labelled and captioned, and effectively support your analysis.

## Formatting & Presentation (1 mark):

• Follows the required page length, font, margins, and line spacing, with a clear, consistent, and professional presentation throughout.

To achieve an Excellent standard overall, quality must be demonstrated consistently across all major components of the report. Isolated strong elements are not sufficient if other parts are weak, superficial, inaccurate, or generic.

## Methods

<table><tr><td>Criteria</td><td>Excellent (1)</td><td colspan="2">Good (0.66)</td><td colspan="2">Basic (0.33)</td><td colspan="2">Not enough (0)</td></tr><tr><td>Data Preprocessing (1 mark)</td><td>All 6 candidates named and clearly distinct; the 3 selected are justified against a specific, quantified dataset property relevant to the RQ; all 3 implemented correctly and clearly support the RQ.</td><td colspan="2">6 candidates named; selection justified with reference to the dataset, though the RQ link or quantification is thinner in one or two cases; all 3 steps implemented correctly.</td><td colspan="2">Fewer than 6 candidates named, or selection justification is vague (no dataset-specific number); implementation correct but steps read as generic rather than RQ-driven.</td><td colspan="2">Steps not implemented or implemented incorrectly, or selection justification is generic</td></tr><tr><td>Correlation Analysis (1 mark)</td><td>Variable set justified specifically against the RQ; implementation choices (discretisation, encoding) stated and justified; all four methods computed correctly for every pair, including correct reasoning where a method is inappropriate.</td><td colspan="2">Variable set justified; all four methods computed correctly for nearly all pairs; one or two omission/minor error, or an implementation choice stated but not justified.</td><td colspan="2">Variable set justification thin or RQ-disconnected; some methods missing for some pairs or applied without checking appropriateness.</td><td colspan="2">Variable set not justified at all, or the four methods not correctly/fully computed across the set.</td></tr><tr><td></td><td>Excellent (2.5)</td><td>Very good (2)</td><td colspan="2">Good (1.25)</td><td colspan="2">Basic (0.75)</td><td>Not enough (0)</td></tr><tr><td>Supervised Learning Models (2.5 marks)</td><td>Both models correctly set up with clear target; split strategy justified; validation strategy justified against dataset size/balance; every hyperparameter value&#x27;s score reported; tuned hyperparameters genuinely influence behaviour, with default/chosen value/effect stated for each change; metrics beyond accuracy justified for the RQ, with per-class precision/recall/F1 for all models. If an optional third model is used, it is set up, tuned to the same standard as the two required models, and evaluated to the same standard and its inclusion is justified.</td><td>Both models correctly set up and tuned; split/validation mostly justified with one weak link; most (not all) hyperparameter values/effects reported; metrics appropriate and mostly justified. If a third model is used, its inclusion is justified and is set up and tuned to the same standard as the two required models, with same weak link as above.</td><td colspan="2">Both models correctly set up and tuned; split and validation strategies stated but justification is thin across more than one of them; hyperparameter sweep reported but effect discussion is partial; metrics go beyond accuracy, but the RQ-fit justification is weak. If a third model is used, it receives the same partial tuning/justification as the two required models.</td><td colspan="2">Models run and tuned, but tuning lacks justification (or tunes a low-influence parameter); split/validation choice stated but not justified against the data; only the best hyperparameter value reported; metrics are accuracy-only or unjustified beyond that. If a third model is used, it is included without justification or being tuned or evaluated to the same depth as the required models.</td><td>One or both required models missing/incorrect; no real hyperparameter tuning attempted; split, validation, or metric justification generic or absent. A third model, if present, is added untuned and unjustified.</td></tr></table>

<table><tr><td></td><td colspan="2">Excellent (0.5)</td><td colspan="2">Basic (0.25)</td><td colspan="2">Not enough (0)</td></tr><tr><td>Feature Selection (0.5 marks)</td><td colspan="2">One embedded and one filter method correctly applied to the full feature set; top 3 lists from each correctly and reproducibly derived from the group&#x27;s own output.</td><td colspan="2">Both methods applied, but with a technical error in one (e.g. filter method computed on the wrong feature set), or top 3 lists correct but not clearly traceable to the group&#x27;s own run.</td><td colspan="2">Only one method applied, both incorrect, or top 3 lists not actually derived from the group&#x27;s own output.</td></tr><tr><td></td><td>Excellent (1.5)</td><td colspan="2">Good (1)</td><td colspan="2">Basic (0.5)</td><td>Not enough (0)</td></tr><tr><td>Clustering - groups of 4 only (1.5 marks)</td><td>3 candidate feature sets named and compared, final choice justified against the RQ; K-Means and Hierarchical both correctly applied to the same feature set; k justified, Hierarchical using the same k.</td><td colspan="2">3 candidates named; both methods correctly applied with matching k; k justified but thinly, or feature-set comparison shallow.</td><td colspan="2">Fewer than 3 candidates compared, or k differs between methods without justification, or k visibly arbitrary.</td><td>One clustering method missing/incorrect, or the “chosen” feature set isn&#x27;t the connected to RQ.</td></tr><tr><td></td><td colspan="2">Excellent (0.5)</td><td colspan="2">Basic (0.25)</td><td colspan="2">Not enough (0)</td></tr><tr><td>PCA - groups of 4 only (0.5 marks)</td><td colspan="2">Applied to the same feature set used for clustering; explained variance ratio correctly reported for the first 2–3 components; top 3 loading variables correctly identified for each.</td><td colspan="2">Feature set doesn&#x27;t match clustering, or variance ratios reported without loadings (or vice versa), or a minor computational error is present.</td><td colspan="2">Not applied, or applied/reported incorrectly (wrong axis, wrong feature set, fabricated loadings).</td></tr></table>

## Data Analysis & Interpretation

<table><tr><td>Criteria</td><td>Excellent (1.5)</td><td colspan="2">Good (1)</td><td colspan="2">Basic (0.5)</td><td colspan="4">Not enough (0)</td></tr><tr><td>Impact of Preprocessing on Results (1.5 marks)</td><td>Every one of the 3 chosen steps has a specific before/after value (rows affected, distribution shift, downstream correlation / model / cluster change), tied to a measurable effect clearly connected to RQ, not a general “improved data quality” statement.</td><td colspan="2">Before/after values reported for most steps; one step described more generally without a specific number.</td><td colspan="2">Impact described but only qualitatively (missing a specific value) — capped here per the numbers rule even if the reasoning is sound.</td><td colspan="4">Impact of preprocessing not discussed or discussed in language that would apply to any group's choices.</td></tr><tr><td></td><td colspan="2">Excellent (2.5)</td><td colspan="2">Very good (2)</td><td colspan="2">Good (1.25)</td><td colspan="2">Basic (0.75)</td><td>Not enough (0)</td></tr><tr><td>Correlation Interpretation (2.5 mark)</td><td colspan="2">All four values reported for every pair including the added target; method agreement/divergence explained with a data-grounded reason; explicit link from strongest/weakest/absentrelationships to the RQ; clear statement of how results changed a later decision; non-causal language used throughout with confounders / biases acknowledged.</td><td colspan="2">All values reported; agreement/divergence noted correctly but explanation thinner; RQ connection anddownstream impact both present but one under-developed; non-causal language mostly consistent.</td><td colspan="2">All values reported; agreement/divergence and RQ connection are both addressed but at a surface level (state whathappened more than why); downstream impact is mentioned only in passing.</td><td colspan="2">Values reported but discussion of agreement/divergence or RQ relevance lacks a specific number(capped here); or downstream impact asserted without evidence it was acted on elsewhere.</td><td>Full pairs matrix incomplete; no discussion of method agreement/divergence; causal language</td></tr><tr><td></td><td colspan="2"></td><td colspan="2"></td><td colspan="2"></td><td colspan="2"></td><td>unqualified; or interpretation generic enough for any correlation table.</td></tr><tr><td>Model and evaluation Comparison (2.5 marks)</td><td colspan="2">Uncertainty quantified for one metric on one model, with resampling method chosen and justified; baseline reported on the same metrics with an explicit absolute-terms improvement; comparison covers feature influence, model behaviour/limitations, and RQ relevance.</td><td colspan="2">All four elements present and mostly own-data-grounded; one (commonly the uncertainty method's justification) under-justified.</td><td colspan="2">All four elements are attempted, but more than one is thinly justified (e.g. uncertainty quantified without a clear reason for the method,.).</td><td colspan="2">Elements present but at least one reads as generic; baseline present but improvement not stated in absolute terms.</td><td>Baseline comparison missing; uncertainty discussion absent; or any element generic enough to be interchangeable across groups.</td></tr><tr><td></td><td>Excellent (2)</td><td colspan="2">Very good (1.5)</td><td colspan="3">Good (1)</td><td colspan="2">Basic (0.5)</td><td>Not enough (0)</td></tr><tr><td>Feature Selection Interpretation (2 marks)</td><td>Disagreement between embedded/filter top-3 lists explained by why the methods weight that feature differently; a specific listing identified by ID as a hard case, with actual attribute values quoted and reasoned about; a concrete, feature-set-specific scenario given for opposite rankings.</td><td colspan="2">All three elements present and correct; hard-case row well-chosen but explanation thinner, or the scenario is somewhat generic but still tied to the group's features</td><td colspan="3">All three elements attempted and topically correct, but more than one is thin; e.g. the row is identified but its attribute values are only partially quoted, and the ranking-disagreement scenario leans generic rather than clearly tied to the group's own feature set</td><td colspan="2">Row ID given but attribute values not quoted, or disagreement explanation doesn't reference the group's actual rankings/values (missing numbers, capped here).</td><td>No specific row identified; disagreement explanation generic (would hold for any embedded-vs-filter comparison).</td></tr><tr><td></td><td colspan="2">Excellent (1.5)</td><td colspan="2">Good (1)</td><td colspan="3">Basic (0.5)</td><td colspan="2">Not enough (0)</td></tr><tr><td>Limitations and Improvement Opportunities (1.5 marks)</td><td colspan="2">Covers all required areas (preprocessing, correlation, supervised learning, feature selection, and for groups of 4: PCA, clustering); each limitation specific to the group's own methods/results; genuine discussion of alternatives not taken.</td><td colspan="2">All areas covered; most limitations specific, one or two closer to general statements.</td><td colspan="3">All areas nominally covered but limitations lean generic (“small sample size”, “data could be biased”) without tying back to the group's choices; alternatives mentioned but not discussed.</td><td colspan="2">One or more required areas missing entirely, or limitations are boilerplate that could appear unchanged in any group's report.</td></tr><tr><td></td><td>Excellent (2.5)</td><td colspan="2">Very good (2)</td><td colspan="3">Good (1.25)</td><td colspan="2">Basic (0.75)</td><td>Not enough (0)</td></tr><tr><td>Clustering Interpretation - groups of 4 only</td><td>Each cluster clearly described/differentiated using dominant features and interpreted; explicit discussion</td><td colspan="2">Clusters described, differentiated, and interpreted correctly; RQ relevance</td><td colspan="3">Clusters described and interpreted adequately, but RQ relevance discussion is generic; K-</td><td colspan="2">Clusters described but interpretation thin (labels without dominant-feature</td><td>Clusters presented as labels/visualisation with no interpretation; no explicit K-Means vs</td></tr><tr><td>(2.5 marks)</td><td>of RQ relevance and usefulness; K-Means vs Hierarchical explicitly compared using actual cluster assignments and sizes, with a clear verdict on whether groupings differ.</td><td colspan="2">discussed; K-Means/Hierarchical comparison present but with fewer specifics (e.g. sizes but not assignment overlap).</td><td colspan="3">Means/Hierarchical comparison is present but asserts a difference (or similarity) without clearly citing sizes or assignments.</td><td colspan="2">values); K-Means/Hierarchical comparison asserted without evidence (capped here).</td><td>Hierarchical comparison attempted.</td></tr><tr><td></td><td colspan="2">Excellent (1.5)</td><td colspan="2">Good (1)</td><td colspan="2">Basic (0.5)</td><td colspan="3">Not enough (0)</td></tr><tr><td>PCA Interpretation – groups of 4 only (1.5 marks)</td><td colspan="2">Each component interpreted strictly grounded in its actual top 3 loading variables (not a guessed label); clear, specific connection from components to the RQ.</td><td colspan="2">Components interpreted correctly and grounded in loadings; RQ connection present but more asserted than argued.</td><td colspan="2">Interpretation given but not clearly tied to the specific loading values reported (missing numbers, capped here).</td><td colspan="3">Interpretation doesn't match the actual reported loadings, or no RQ connection attempted.</td></tr></table>

## Cross-Section Consistency — same for both group sizes

<table><tr><td></td><td>Excellent (1)</td><td>Basic (0.5)</td><td>Not enough (0)</td></tr><tr><td>Cross-Section Consistency (1 mark)</td><td>Values that recur across sections (rates, counts, coefficients, metrics) agree wherever the same quantity appears.</td><td>A recurring value differs slightly in a way that doesn&#x27;t change the report&#x27;s conclusions (e.g. rounding, a stray older figure in one sentence)</td><td>A recurring value differs in a way that changes or undermines a conclusion drawn elsewhere.</td></tr></table>

## Report Quality

<table><tr><td>Criteria</td><td>Good (1)</td><td>Basic (0.5)</td><td>Not enough (0)</td></tr><tr><td>Visualisations (1 mark)</td><td>Visualisations are clear, well-designed, and easy to interpret; Correctly labelled and captioned; Effectively support and enhance the analysis</td><td>Visualisations are generally clear;Minor issues with labelling, captions, or clarity; Support the analysis but not always effectively</td><td>Visualisations are unclear, poorly labelled, missing, or misleading; Do not meaningfully support the analysis</td></tr><tr><td>Formatting &amp; Presentation (1 mark)</td><td>Fully follows formatting requirements; Clear, consistent, and professional throughout</td><td>Minor formatting inconsistencies;Generally clear and readable</td><td>Formatting requirements not followed;Unclear, inconsistent, or unprofessional presentation</td></tr></table>
