# Phase 17: Statistical Methodology

## Experimental Design
- **Unit of Analysis:** 50 distinct questions (queries).
- **Independent Variable:** Chunking Strategy (5 levels: `baseline_fixed_size`, `eluc_section_aware`, `recursive_character`, `paragraph_based`, `heading_based`).
- **Dependent Variables:** Recall@1, Recall@3, Recall@5, MRR@5.
- **Design:** Within-subjects (paired) design, as the exact same 50 questions were evaluated under all 5 conditions.

## Statistical Pipeline
Because the retrieval metrics are bounded in [0,1], zero-inflated, and highly discrete, the normality assumption of Repeated-Measures ANOVA is severely violated. We use the following non-parametric pipeline for each metric independently:

1. **Omnibus Test (Friedman Test):** Evaluates if there are significant differences in the repeated measurements (distributions) across the 5 strategies.
   - *Effect Size:* Kendall's W (coefficient of concordance).
2. **Post-Hoc Pairwise Test (Wilcoxon Signed-Rank Test):** If the Friedman test yields p < 0.05, we conduct pairwise Wilcoxon tests to identify which specific strategies differ.
   - Zero-difference pairs are dropped (`zero_method="wilcox"`) per standard SciPy configuration.
   - *Multiple Comparison Correction:* Holm-Bonferroni correction applied across the 10 pairwise comparisons to control the Family-Wise Error Rate (FWER).
   - *Effect Size:* Rank-Biserial Correlation ($r_{rb}$).
3. **Confidence Intervals:** 95% Confidence Intervals for the median difference are generated via bootstrapping (10,000 resamples, deterministic seed 42) on the paired observations. These provide descriptive uncertainty bounds.

## Invariants
- 250 exact Phase 16 observations were used.
- No inputs were mutated or re-evaluated.
