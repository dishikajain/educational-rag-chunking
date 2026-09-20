# Phase 19: Interpretation of Results

## 1. Research Objective
The fundamental research objective of this project is to evaluate whether and how different document chunking strategies affect retrieval performance in educational RAG (Retrieval-Augmented Generation) question answering.

## 2. Experimental Context
To address this objective, a controlled, deterministic experiment was conducted using an educational textbook (Chapters 1–5). A manually curated evaluation dataset of 50 questions was paired with a provenance-derived, source-grounded retrieval reference (250 total observations). Five chunking strategies were evaluated under fixed embedding and retrieval conditions (top_k=5, strict evaluation):
- Baseline Fixed-size
- ELUC / Section-aware
- Recursive character
- Paragraph-based
- Heading-based

## 3. Descriptive Findings
Descriptive performance across the five strategies reveals clear variations in observed retrieval metrics: Recall@1, Recall@3, Recall@5, and MRR@5.

- **Recall@1**: Heading-based showed the highest observed value (0.102), followed by Paragraph-based (0.073) and ELUC / Section-aware (0.071). Fixed-size and Recursive character showed the lowest observed values (0.024 and 0.025).
- **Recall@3**: Heading-based again produced the highest observed value (0.217). ELUC / Section-aware and Paragraph-based were similar (0.190 and 0.137, respectively). Fixed-size and Recursive character demonstrated lower observed retrieval performance (0.064 and 0.067).
- **Recall@5**: Heading-based achieved the highest observed value (0.274), with ELUC / Section-aware slightly lower (0.256). Paragraph-based achieved 0.215. Fixed-size and Recursive character were consistently lower (0.097 and 0.111).
- **MRR@5**: Heading-based yielded the highest observed MRR@5 (0.464), followed by Paragraph-based (0.434) and ELUC / Section-aware (0.393). Recursive character and Fixed-size showed smaller observed values (0.357 and 0.317).

## 4. Statistical Findings
Omnibus Friedman tests were conducted. The omnibus tests provide evidence against the null hypothesis of no difference across strategies for all four metrics:
- **Recall@1**: $\chi^2 = 44.85$, $p = 4.27 \times 10^{-9}$, Kendall's W = 0.224
- **Recall@3**: $\chi^2 = 70.02$, $p = 2.24 \times 10^{-14}$, Kendall's W = 0.350
- **Recall@5**: $\chi^2 = 81.61$, $p = 7.95 \times 10^{-17}$, Kendall's W = 0.408
- **MRR@5**: $\chi^2 = 18.77$, $p = 0.00087$, Kendall's W = 0.094

The effect sizes (Kendall's W) indicate a moderate effect for Recall@3 and Recall@5, and a small-to-moderate effect for Recall@1 and MRR@5.

## 5. Pairwise Findings
Post-hoc Wilcoxon signed-rank testing (Holm-adjusted) revealed specific strategy differences. Notable patterns (using Holm-adjusted $p < 0.05$ as the threshold for statistical significance):

- **Heading-based vs. Baseline Fixed-size**: Heading-based produced higher observed values than Fixed-size, with statistically significant paired differences across all four evaluated metrics (e.g., Recall@5 $p = 1.57 \times 10^{-5}$, $r_{rb} = -0.988$).
- **Paragraph-based vs. Baseline Fixed-size**: Statistically significant differences for Recall@1, Recall@3, and Recall@5, showing higher observed performance for Paragraph-based. The MRR@5 difference was marginally non-significant after Holm adjustment ($p = 0.0523$).
- **ELUC / Section-aware vs. Baseline Fixed-size**: Statistically significant for Recall@1, Recall@3, and Recall@5. MRR@5 was not significantly different after adjustment ($p = 0.234$).
- **Recursive character vs. Baseline Fixed-size**: No statistically significant differences across any metric (all adjusted $p > 0.25$).
- **Heading-based vs. Paragraph-based**: Significantly differed in Recall@3 and Recall@5 ($p = 0.002$ and $p = 0.011$ respectively), with Heading-based showing higher observed performance. MRR@5 differences were not statistically significant ($p = 0.756$).

While some comparisons reached statistical significance, the practical magnitudes of median differences were frequently bounded by the discrete nature of the data (e.g., median differences of exactly 0.000, despite significant rank distributions).

## 6. Cross-Metric Patterns
- Strategy performance rankings were broadly consistent across the $k$ cutoffs. Heading-based generally yielded the highest metrics, while fixed-size variations yielded the lowest.
- The Friedman effect-size estimates increased from Recall@1 through Recall@5, indicating that the degree of between-strategy rank differentiation observed in this dataset was larger at higher retrieval cutoffs.
- MRR@5 exhibited a much smaller omnibus effect size ($W = 0.094$) than the Recall metrics. Pairwise significance for MRR@5 was much rarer, indicating that while total relevant chunks retrieved varies by strategy, the rank of the *first* relevant chunk is less sensitive to chunking strategy in this dataset.

## 7. Structural Interpretation of Chunking Strategies
Structurally, the evaluated strategies differ in their reliance on arbitrary thresholds versus document boundaries (e.g., character counts vs. explicit headings/paragraphs).
- **Fixed-size and Recursive Character** generate splits primarily according to configured character-volume constraints rather than explicit document boundaries. These strategies were associated with lower observed retrieval performance.
- **Paragraph and Heading-based** chunking maintain discourse boundaries. The observed pattern is consistent with the possibility that preserving document-level discourse boundaries affects retrieval matching.
- **ELUC / Section-aware** merges sections up to a character limit. It showed lower observed performance than strict Heading-based chunking, suggesting a plausible, though untested, explanation that aggregating multiple sections might dilute relevant information compared to strict hierarchical slicing.

## 8. Limitations
- **Corpus**: Evaluation was restricted to Chapters 1–5 of a single educational textbook.
- **Dataset Size**: The 50-question manually curated dataset, while carefully constructed, is small.
- **Fixed Pipeline**: The experiment used a fixed embedding model and fixed retrieval parameters (top_k=5, cosine similarity). No reranking or alternative retrieval models were tested.
- **Evaluation Scope**: The analysis relies strictly on retrieval evaluation (Recall, MRR). It does not include an end-to-end evaluation of generated answer quality.

## 9. Threats to Validity
- **Internal Validity**: The deterministic provenance-derived retrieval reference provides traceability between the question source evidence and the evaluated chunks. However, if a strategy coincidentally returns alternative text containing the correct answer that was not the original provenance text, it is marked incorrect—a known structural limitation of strict source-grounded evaluation.
- **External Validity**: Generalizability to domains outside educational textbooks or to multi-hop reasoning question types is not supported by the existing methodology.
- **Construct Validity**: Recall and MRR effectively capture retrieval success, but the discrete, zero-inflated nature of the metrics restricts the granularity of practical effect sizes.
- **Statistical Conclusion Validity**: Non-parametric methods and family-wise error rate corrections (Holm) were strictly applied, ensuring robust p-values for the observed data.

## 10. Research Contribution
This study contributes a strictly controlled, reproducible, and deterministic comparison of five diverse chunking strategies on a shared corpus. It utilizes a deterministic provenance-derived, source-grounded retrieval reference methodology that avoids subjective relevance labeling and supports reproducible evaluation of chunking strategies under vector-similarity retrieval.

## 11. Practical Implications
For the structured educational text evaluated here, chunking approaches that preserve document structure were associated with higher observed retrieval recall. Implementers processing similar corpora might observe differences in retrieval depending on whether their preprocessing pipelines prioritize natural document hierarchies or arbitrary character limits.

## 12. Future Work
Subsequent research should investigate whether these observed retrieval differences are associated with measurable variations in downstream LLM generation quality. Additionally, evaluating the strategies across diverse domains and with alternative retrieval architectures (e.g., hybrid search, neural rerankers) would clarify the bounds of these findings.

## 13. Final Findings
1. **Main Observed Patterns**: Structurally aware chunking strategies (Heading-based, Paragraph-based) showed higher observed retrieval performance than arbitrary character-based strategies within the evaluated Chapters 1–5 corpus, 50-question dataset, embedding model, and top-k=5 retrieval configuration.
2. **Statistical Significance**: The omnibus tests provide evidence against the null hypothesis of no difference across strategies (Friedman $p < 0.001$).
3. **Important Differences**: Heading-based produced higher observed values than Fixed-size, with statistically significant paired differences across all four evaluated metrics. Fixed-size and Recursive Character chunking showed no significant differences from each other.
4. **Practical Implications**: For the structured educational text evaluated here, chunking approaches that preserve document structure were associated with higher observed retrieval recall.
5. **Limitations**: The study is strictly limited to one textbook, a 50-question dataset, and a fixed retrieval pipeline.
6. **Future Work**: End-to-end generation evaluation and cross-domain validation are recommended.
