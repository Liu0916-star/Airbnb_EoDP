
# Stage 1 - Preprocessing

## Task
Implement 3 preprocessing tasks in total, but name at least 6 candidates you considered, and explain why you selected your final 3 over the rest.
For each of the 3 tasks selected, state the alternative approach(es) you considered and why you did not choose them, referencing an actual number from the dataset (e.g. “X% of rows affected”), not a vague description.

For each of the 3 tasks you implement, you need to report its measurable impact with a specific before/after value: e.g. rows affected, a distribution shift, or a resulting change in a downstream correlation, model metric, or cluster. A general statement that a step "improved data quality" is not sufficient. Note any limitations of your chosen preprocessing steps in the Limitations and Improvement Opportunities section of your report.

Example tasks:
- Review recency feature: parse ‘last_review’ into a date and engineer a ‘days_since_last_review’ feature.
- Minimum-stay discretisation: bucket ‘minimum_nights’ into short/medium/long-stay categories.
- Distance-from-CBD discretisation: compute each listing's distance from Melbourne's CBD and bucket into inner/middle/outer bands.
- Missing-data strategy for bedrooms/beds: justify your approach with reference to ‘room_type’.
- Property-type consolidation: reduce the 79 raw ‘property_type’ categories (41 of which have fewer than 10 listings each) into a manageable number of groups suitable for modelling, justified against your research question.
- Host-listings-count flag: derive a single-listing vs. multi-listing host indicator from ‘host_listings_count’.

## Our selected sensible 6
1. Keep only live listings (via `source`). 
From the data dictionary:
> Source: One of "neighbourhood search" or "previous scrape". "neighbourhood search" means that the listing was found by searching the city, while "previous scrape" means that the listing was seen in another scrape performed in the last 65 days, and the listing was confirmed to be still available on the Airbnb site.
So right now, the file is two piles put together: live listings