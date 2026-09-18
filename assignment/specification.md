## Uncovering Patterns in Melbourne Airbnb data: A Data Science Investigation

## 1. Overview

In this project, you will use the Melbourne Inside Airbnb dataset (listings.csv) made available through the Inside Airbnb platform. The dataset provides rich insights into short-term rental listings, host behaviour, pricing, and guest reviews across metropolitan Melbourne.

Through this project, you will:

• Formulate and investigate a research question (RQ) of your choice, exploring an aspect of the Melbourne short-term rental market that interests your group

• Perform data processing to clean and structure the dataset in a way that supports your question

• Conduct a correlation analysis to explore relationships relevant to you RQ (e.g., between listing and host attributes)

• Implement and compare supervised learning models to predict an outcome relevant to your question (e.g. host, listing, or pricing behaviour)

• Apply feature selection techniques to identify which attributes are most influential for your question

• (Groups of 4 only) Apply dimensionality reduction and clustering methods to uncover patterns (e.g. among listings or hosts) relevant to your question

Your findings will be summarised in a technical report, supported by data visualisations and code implementations. The audience of your report should be the teaching team, so you can assume that the basic technical terms are known. You will present your report and analysis in an oral presentation and then respond to questions.

## 2. Assignment Structure

## Key Principles:

We value depth of analysis and clear justification over the number of techniques used. Every interpretive claim in your report must be supported by a specific value from your own output (a coefficient, a count, a row ID, a metric).

These requirements apply across every section below:

0 Numbers in every interpretive paragraph. Every paragraph that draws a conclusion must cite at least one specific value from your own output.

◦ Cross-section consistency. Your report should be internally consistent — scores, figures, rates, and counts should agree wherever they recur.

◦ Generic justification earns no marks. A justification that could be written without reference to your own computed output will receive no marks for that criterion, regardless of whether it happens to be correct.

## 3. Data Analysis Task

## 3.1. Research Question

The research question clarifies the purpose of your analysis. It identifies the problem being addressed, sets the context, and explains why the analysis is being conducted.

Design a single research question of your own about the Melbourne short-term rental market, or choose from the six examples below if you'd prefer a starting point:

Can ‘host_is_superhost’ be predicted from listing and host attributes, and which features matter most?

Can multi-listing (“professional”) hosts be distinguished from single-listing hosts, and what operating differences show up in the data?

Can ‘room_type’ be predicted from listing attributes, and how does class imbalance affect model choice and evaluation?

• Can listings be grouped into price tiers, and what attributes distinguish tiers?

What listing and location attributes are associated with price variation across Melbourne neighbourhoods, and is price associated with perceived value?

• What distinguishes listings requiring a long minimum stay (30+ nights) from short-stay listings?

Your project must be driven by your chosen research question (RQ). This RQ should guide all parts of your analysis, not just the introduction. Each section should contribute to answering the same question:

• Preprocessing → prepare data relevant to the RQ

• Correlation analysis → explore relationships relevant to the RQ

• Modelling → test the RQ through prediction

• Feature selection → identify which attributes drive the RQ's outcome

• Clustering (if applicable) → provide additional insight related to the RQ

Example RQ:

"Can we predict whether a host is a ‘superhost’ based on listing attributes, review performance, and host activity?"

• Preprocessing: select/engineer features relevant to host activity and review performance

Correlation: identify variables associated with ‘superhost’ status

• Modelling: predict ‘superhost’ vs. ‘non-superhost’ using selected features

• Clustering: identify host operating profiles and relate them to ‘superhost’ status

A strong report shows a clear connection between all sections and the research question.

## 3.2. Data Pre-processing

Throughout this subject, you've learned various data preparation techniques, including handling missing values, reshaping data, scaling, encoding, discretising, merging datasets, and feature engineering.

Design your own preprocessing tasks suited to your research question or draw from the six examples below as a starting point; you're welcome to mix your own ideas with the list. Whichever path you take, implement 3 tasks in total, but name at least 6 candidates you considered, and explain why you selected your final 3 over the rest.

Review recency feature: parse ‘last_review’ into a date and engineer a ‘days_since_last_review’ feature.

• Minimum-stay discretisation: bucket ‘minimum_nights’ into short/medium/long-stay categories.

Distance-from-CBD discretisation: compute each listing's distance from Melbourne's CBD and bucket into inner/middle/outer bands.

• Missing-data strategy for bedrooms/beds: justify your approach with reference to ‘room_type’.

Property-type consolidation: reduce the 79 raw ‘property_type’ categories (41 of which have fewer than 10 listings each) into a manageable number of groups suitable for modelling, justified against your research question.

Host-listings-count flag: derive a single-listing vs. multi-listing host indicator from ‘host_listings_count’.

For each of the 3 tasks selected, state the alternative approach(es) you considered and why you did not choose them, referencing an actual number from the dataset (e.g. “X% of rows affected”), not a vague description.

For each of the 3 tasks you implement, you need to report its measurable impact with a specific before/after value — for example, rows affected, a distribution shift, or a resulting change in a downstream correlation, model metric, or cluster. A general statement that a step "improved data quality" is not sufficient.

Note any limitations of your chosen preprocessing steps in the Limitations and Improvement Opportunities section of your report.

## 3.3. Correlation Analysis

Choose a set of variables for your correlation analysis. You may create your own set or use one of the three example sets below as a starting point. Briefly justify why your chosen variables are relevant to your research question. Make sure your target variable (or a suitable proxy) is included in the set.

Set A (Listing size): 'accommodates', 'bedrooms', 'bathrooms' (the numeric variable your Assignment 1 pipeline already derived from 'bathrooms_text' — reuse it here rather than re-parsing). Relevant if your RQ concerns room type, price tiers, or long-stay listings, where listing size may play a role.

Set B (Price and value): 'price' (cleaned per Assignment 1), 'accommodates', 'bedrooms', 'review_scores_rating'. Relevant if your RQ concerns price tiers, price variation across neighbourhoods, or perceived value.

Set C (Host activity): 'host_listings_count', 'days_since_last_review', 'reviews_per_month', 'review_scores_rating'.

Relevant if your RQ concerns ‘superhost’ status or distinguishing professional (multi-listing) hosts from singlelisting hosts.

Apply all four methods covered in this subject (Pearson, Spearman, Mutual Information, and Normalised Mutual Information) to every pair of variables in the full set. If a method is not appropriate for a particular pair (e.g., because of the variable types), state this and explain why. Present your results clearly, for example in a table.

Example: Set A contains 3 variables. If you add your target variable, you will have 4 variables in total. This gives 6 unique pairs, and therefore up to 24 values (6 pairs x 4 methods).

Then discuss:

Agreement and differences: Which methods give similar conclusions, and which give different results? Why? Explain based on the properties of your actual data.

Connection to your research question: Which relationships are strongest, weakest, or absent? What do these results tell you about your research question?

Impact on your next steps: Explain how the results influenced your later analysis. For example, you might:

o drop one of two strongly related predictors before modelling;

o exclude a variable that shows little or no relationship where you expected one;

o perform additional preprocessing; or

o prioritise a variable during feature selection.

Note that you are not only examining relationships between predictors and the target. Relationships between predictor variables are also important; for example, they can reveal potential multicollinearity.

Also remember that correlation analysis identifies associations, not causal relationships. In your interpretations, you should:

Avoid causal language (e.g., “X causes Y”) unless clearly justified

• Use appropriate phrasing such as “is associated with”, “is correlated with”, or “may be related to”

• Acknowledge possible confounding variables, biases, or spurious relationships

Note any limitations of your chosen correlation analysis in the Limitations and Improvement Opportunities section of your report.

## 3.4 Feature Selection

Implement 2 feature selection methods: one embedded method (e.g., from your Decision Tree, or any other model you may have used as your optional third model) and one filter method (e.g. MI or NMI).

Report the top 3 features from each method and compare them:

Explain any disagreement between the two lists, and why the two methods might rank that feature differently.

Identify one specific row (by ID) from the dataset where the two methods' top-ranked feature would lead to different predictions, or whose value for the top-ranked feature is far outside the typical range seen in the rest of the dataset. Quote its actual attribute values and explain, in your own words, why this row is a hard case for your model.

Discuss one scenario in which the filter method could rank a feature highly while the embedded method ranks it low (or vice versa), grounded in a property of your own feature set

Note any limitations of your feature selection approach in the Limitations and Improvement Opportunities section of your report.

## 3.5. Supervised Learning Models and Evaluation

Machine learning models can be used to predict host, listing, or pricing outcomes based on dataset attributes.

You must implement K-Nearest Neighbours and Decision Tree, tune their hyperparameters and compare their performance.

If interested, you can additionally implement one of Logistic Regression, SVM, or Random Forest as an optional third model (other models outside this list can work as long as justified). This is not required. Marks for this section are earned through the quality and correctness of your comparison and interpretation, not through the number or sophistication of models used. If you choose to use one of these models, you need to justify your choice.

Design an evaluation approach that rigorously tests your models, covering each aspect below. For each one, choose a method suited to your data and RQ, and justify why you chose it over the alternatives.

Train/test split: Justify your split strategy against your target's actual class balance (e.g. stratified, random, time-based).

• Hyperparameter selection: Justify your validation strategy (e.g. 5-fold stratified) given your dataset size and balance. Report the score at every hyperparameter value tried, not just the one selected.

Hyperparameter justification: For every hyperparameter changed from its default, state the default, your chosen value, and its specific effect on your reported metric.

Evaluation metrics: Justify metrics beyond accuracy suited to your RQ (e.g. per-class precision/recall/F1, macro-F1, ROC-AUC, RMSE/MAE).

Uncertainty quantification: Quantify uncertainty in one reported metric for at least one model (e.g. bootstrap CI, repeated CV, or another resampling approach) and justify your choice.

• Baseline comparison: Compare each model against a justified baseline (e.g. the majority class (0R)) using the same metrics, with an explicit statement of improvement in absolute terms.

When comparing your models, go beyond reporting performance metrics. A meaningful comparison should include:

• Performance — compare using appropriate metrics and explain what they indicate in the context of your task.

Feature Influence — examine which features are important to each model and whether they rely on similar or different inputs.

Model Behaviour and Limitations — discuss strengths and weaknesses of each model (e.g., sensitivity to noise, overfitting, interpretability).

Interpretation of Results — explain what the model outcomes suggest about your research question, not just which model performs better.

Note any limitations of your modelling approach in the Limitations and Improvement Opportunities section of your report.


## 3.5 Dimensionality Reduction & Clustering (ONLY required for groups of FOUR)

## Clustering:

Choose a set of features that is relevant to your research question. You can design your own feature set, use one of the three examples below, or combine features from the examples with your own ideas. Compare at least three candidate feature sets you considered and justify your final choice.

• Geographic: latitude, longitude, amenity count, accommodates.

• Price-tier: price, accommodates, bedrooms, review_scores_rating.

• Host operating profile: host_listings_count, availability_365, review_scores_rating.

For K-Means, justify your choice of k (the number of clusters) using the elbow method or a value tied to the research question, rather than picking one arbitrarily. Apply Hierarchical clustering using the same number of clusters as your chosen k, so the two methods can be meaningfully compared. Your goal is to identify meaningful patterns in the data and discuss their implications. You should:

Clearly describe each cluster using its dominant features

• Explain how clusters differ from one another

• Interpret what each cluster represents

Discuss what the clusters reveal about your research question, and whether they are useful for answering it.

• Explicitly discuss whether K-Means and Hierarchical produced different groupings in your results

Note any limitations of your clustering approach in the Limitations and Improvement Opportunities section of your report. Simply presenting cluster labels or visualisations without interpretation is not sufficient.

## Principal Component Analysis:

Apply PCA to the same feature set you used for clustering above and report the explained variance ratio for the first 2–3 components.

For each of these components, identify the top 3 original variables contributing to it (by loading magnitude) and briefly interpret what that component appears to represent (e.g., a “listing size” axis or a “review quality” axis). Discuss what these components suggest in relation to your research question. Add a 2D scatter plot of the first two components, optionally coloured by your research question's target variable.

Note any limitations of your PCA in the Limitations and Improvement Opportunities section of your report.

## 4. Report

Your primary submission for this assignment is your report. The report should follow the structure of a technical paper. It should describe your approach and observations, both in data preparation and the machine learning algorithms you tried. Its main aim is to provide the reader with an understanding of the problem, particularly through a critical analysis of your results and findings.

The following is the expected structure of the report for this assignment:

1. Introduction

2. Methodology

3. Results Exploration and Analysis

4. Discussion and Interpretation

5. Limitations and Improvement Opportunities

6. Conclusion

7. References

The details of each section are covered in the provided template.

## 5. Interactive Oral Assessment

You need to conduct an interactive oral assessment that includes both presenting and answering questions. The detailed specification, rubrics and examples for this component are available on Canvas (Assignment 2: Interactive Oral Assessments).

## 6. Terms and Conditions

## 6.1 Data Acknowledgement

The data is provided by Inside Airbnb. You must properly cite the dataset in your report.

## Example citation:

Inside Airbnb. (2026). Melbourne, Victoria, Australia. Retrieved [7 July,2026] from http://insideairbnb.com/get-the-data/
