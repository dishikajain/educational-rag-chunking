# Phase 14B: Retrieval Ground Truth Methodology

This document establishes the experimental methodology for generating the Phase 14B retrieval ground truth. It enforces a strict separation between the evaluation ground truth and the retrieval/embedding mechanisms being evaluated, ensuring independent, unbiased coverage metrics for subsequent phases.

## 1. Automated Candidate Discovery (Stage A)

To minimize error-prone manual searches across 3,929 formal chunks while preventing any bias from embeddings or vector databases, candidate chunks are discovered strictly via textual overlap and exact heuristic string matching. 

A deterministic script evaluates all chunks for each of the 5 chunking strategies and identifies candidate chunks containing necessary evidence based exclusively on:
- Phase 13 question and reference answer
- Frozen Phase 13 `source_passage`
- `source_page_range`
- Chunk textual content and `provenance` metadata
- Substring mapping (with whitespace/newline normalization)

### Candidate Artifact
The output of Stage A is stored in `data/ground_truth/annotation_candidates.jsonl`.
This artifact is strictly intermediate and serves only as a starting point. It uses the `annotation_status`: `"needs_manual_review"` and `annotated_by`: `"candidate_generator"`.

## 2. Human Verification (Stage B)

Automated candidate generation is never treated as the final annotator. For every one of the 250 question Ã— strategy combinations (50 questions Ã— 5 chunking strategies), a human must manually review and verify the candidates.

For each question-strategy pair, the annotator examines:
1. The question
2. The reference answer
3. The Phase 13 `source_passage`
4. The `source_page_range`
5. The actual chunk text of the candidate chunk(s)
6. The chunk provenance

The final output is written to `data/ground_truth/retrieval_ground_truth.jsonl`, where every record must hold `annotation_status`: `"verified"` and `annotated_by`: `"manual_annotation"`.

## 3. Definition of Relevance

A chunk is relevant **only** when its content provides necessary or supporting evidence required to answer the question or directly supports the reference answer. 

**Rejection Criteria:**
Chunks must NOT be included merely because:
- They are topically similar.
- They share keywords with the question/answer.
- They are adjacent to a relevant chunk.
- They physically overlap the source page range without containing the necessary evidence.
- They were retrieved highly by a semantic retriever/ChromaDB.

## 4. Ground-Truth Independence

To preserve the validity of evaluation metrics (Recall@K, MRR, nDCG), annotation must remain 100% independent from retrieval outputs. Annotators are forbidden from:
- Running retrievers to identify candidates.
- Viewing or using ChromaDB rankings or similarity scores.
- Relying on LLMs (e.g., GPT-4, Claude) to make the final relevance judgment.

## 5. Handling Evidence Scope

### Partial Evidence
If a chunk contains only part of the evidence required to answer a question, it is considered a relevant chunk if that partial evidence is necessary.

### Multi-Chunk Evidence
If multiple chunks are required to completely answer a question, **all necessary chunks** must be included in the ground truth. If one chunk is sufficient to answer the question, adjacent chunks with semantic overlap must not be included.

### Strategy-Specific Annotation
Chunk boundaries vary wildly between the 5 strategies (e.g., a single ELUC chunk may correspond to three recursive character chunks). Relevance is evaluated **independently** for each strategy based on its unique chunk boundaries.

## 6. Provenance and QC

During human verification, the annotator must manually verify the provenance of each candidate chunk against the Phase 13 `source_page_range` and physical text location. 

### Manual QC
If a candidate chunk's relevance is ambiguous, the annotator must inspect the chunk contents, neighbor chunks (to determine if evidence split), and the source PDF before establishing relevance. 

### Final Verification Status
Only question-strategy pairs explicitly marked with `annotation_status`: `"verified"` are permitted to be loaded during retrieval evaluation testing. Unverified candidate generation artifacts are rejected.

