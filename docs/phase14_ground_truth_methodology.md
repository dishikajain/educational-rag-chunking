# Phase 14: Retrieval Ground Truth Methodology

**Project:** A Comparative Study of Document Chunking Strategies in Retrieval-Augmented Generation for Educational Question Answering
**Corpus:** OpenStax Biology 2e, Chapters 1-5
**Phase 14 Goal:** Establish, for each of the 50 Phase 13 questions and each of the five chunking strategies, the set of chunk IDs whose text contains sufficient information to answer the question without any external knowledge.

---

## A. Status of All Five Chunking Strategies

| Strategy Key | Source Module | Config File | Chunk Data (formal/) | Status |
|---|---|---|---|---|
| baseline_fixed_size | src/chunking/chunker.py | config/baseline_chunking_config.yaml | EMPTY (.gitkeep only) | IMPLEMENTED, NOT YET RUN |
| eluc_section_aware | src/eluc/chunker.py | config/eluc_chunking_config.yaml | EMPTY (.gitkeep only) | IMPLEMENTED, NOT YET RUN |
| recursive_character | NOT IMPLEMENTED | NONE | EMPTY | MISSING |
| paragraph_based | NOT IMPLEMENTED | NONE | EMPTY | MISSING |
| heading_based | NOT IMPLEMENTED | NONE | EMPTY | MISSING |

### Critical Finding

Only two of the five strategies have source-code implementations:
- baseline_fixed_size -- implemented in src/chunking/chunker.py (fixed-size sliding window, 350-char chunks, 40-char overlap).
- eluc_section_aware -- implemented in src/eluc/chunker.py (structure-aware, heading/section boundary chunking, max 1400 chars).

The remaining three strategies have no implementation whatsoever. A search of all Python files under src/ found no references to recursive_char, paragraph_based, or heading_based strategy names in any module. The retriever (src/retrieval/retriever.py) currently only accepts "baseline_fixed_size" or "eluc_section_aware" as valid strategy arguments.

NOTE: Neither baseline_fixed_size nor eluc_section_aware has been run against the corpus yet. data/chunks/formal/ and data/chunks/dev/ are empty for all five strategies. ChromaDB was populated by scripts/index_corpus.py in a previous pipeline run, but no serialized chunk files exist.

Consequence: Phase 14 may be split:
- Phase 14a: Ground truth for baseline_fixed_size and eluc_section_aware (after running index_corpus.py to regenerate chunks).
- Phase 14b: Ground truth for the three remaining strategies after they are implemented.

---

## B. Relevance Criterion

### B.1 What Constitutes a Relevant Chunk

A chunk is relevant to a question if and only if:

1. Sufficiency: The chunk text contains enough information to fully or partially answer the question independently of any other chunk.
2. Source-grounding: The chunk covers the same source content as the question source_passage field, cross-verified using source_page_range.
3. Independence of retrieval: Relevance is determined by reading the chunk text against the question, NOT by whether the retriever ranked it highly.

A chunk is not relevant if:
- It contains content from the same section but not the answer-bearing passage.
- It is a heading-only chunk without body text.
- It is from a different chapter or section than the question section field.
- It contains only tangentially related material (same topic, different concept).

### B.2 Relevance Labels

| Label | Meaning |
|---|---|
| relevant | The chunk fully contains the answer or is sufficient to construct the correct answer. |
| partial | The chunk contains some answer-supporting content but not enough alone. |
| not_relevant | The chunk does not support the answer. |

For Phase 15/16 Recall@K evaluation, only relevant or partial chunks contribute to the ground-truth set R(q) for question q.

### B.3 Multi-Chunk Answers

Some questions answers span multiple source passages (e.g., relationship_comparison questions contrasting two concepts in different paragraphs). Rules:

1. Each contributing chunk is labeled relevant or partial individually.
2. At minimum one chunk must be relevant (fully sufficient or sufficient with the key claim).
3. No upper limit on relevant chunks, but the annotator must justify sets of more than 3.
4. If the source passage is entirely contained in a single chunk, that chunk is relevant; further chunks may be partial if they provide complementary content.

---

## C. How Ground Truth Is Created

### C.1 Source-Based, Not Retrieval-Based

Ground truth is established by:
1. Reading the question and its source_passage / source_page_range from data/questions/questions.jsonl.
2. Locating chunks in the strategy corpus that contain the source passage text.
3. Comparing chunk char ranges (char_start, char_end) against the question page range to identify candidates.
4. Reading each candidate chunk text to confirm relevance.

The retriever is never run as part of ground truth creation. Retrieved rankings are used only in Phase 15/16 evaluation.

### C.2 Page-Based Provenance

Every chunk produced by both existing strategies carries provenance_spans, mapping to (source_page, p2_block_index) pairs. These are used to:
- Filter candidate chunks to those overlapping the source_page_range from the question.
- Provide an objective, auditable basis for the relevance decision.
The annotator uses the source PDF as the authoritative reference.

### C.3 Human Verification Protocol

For each (question, strategy) pair:
1. Extract source_page_range and section from questions.jsonl.
2. Load all chunks from the strategy corpus.
3. Filter to chunks whose page span overlaps source_page_range +/- 1 page.
4. Read each filtered chunk and apply criterion B.1.
5. Record resulting relevant chunk IDs.
6. For any non-obvious case, document reasoning in annotation_notes.

No LLM, automated scoring function, or RAG-system output is used in this step.

---

## D. Ground Truth Artifact Schema

### D.1 Storage Location

  data/questions/ground_truth.jsonl

One JSON object per line, one per question (all five strategies in one record).

### D.2 JSON Schema per Record

{
  "question_id": "Q001",
  "question": "What is a hypothesis ...",
  "source_page_range": "29",
  "section": "1.1",
  "ground_truth": {
    "baseline_fixed_size": {
      "relevant_chunk_ids": ["openstax_biology2e_ch1_5_baseline_000012"],
      "partial_chunk_ids": [],
      "annotation_notes": "Chunk contains the exact source passage verbatim.",
      "annotator": "manual",
      "annotation_date": "YYYY-MM-DD"
    },
    "eluc_section_aware": {
      "relevant_chunk_ids": ["openstax_biology2e_ch1_5_eluc_000007"],
      "partial_chunk_ids": [],
      "annotation_notes": "Section 1.1 ELUC chunk fully contains the hypothesis definition.",
      "annotator": "manual",
      "annotation_date": "YYYY-MM-DD"
    },
    "recursive_character": {
      "relevant_chunk_ids": null,
      "partial_chunk_ids": null,
      "annotation_notes": "Strategy not yet implemented.",
      "annotator": null,
      "annotation_date": null
    },
    "paragraph_based": {
      "relevant_chunk_ids": null,
      "partial_chunk_ids": null,
      "annotation_notes": "Strategy not yet implemented.",
      "annotator": null,
      "annotation_date": null
    },
    "heading_based": {
      "relevant_chunk_ids": null,
      "partial_chunk_ids": null,
      "annotation_notes": "Strategy not yet implemented.",
      "annotator": null,
      "annotation_date": null
    }
  }
}

### D.3 Field Definitions

| Field | Type | Description |
|---|---|---|
| question_id | string | From questions.jsonl |
| question | string | Copied for readability |
| source_page_range | string | Copied from questions.jsonl for provenance tracing |
| section | string | Copied from questions.jsonl |
| ground_truth.<strategy>.relevant_chunk_ids | list[string] or null | null = strategy not yet available |
| ground_truth.<strategy>.partial_chunk_ids | list[string] or null | May be empty list |
| ground_truth.<strategy>.annotation_notes | string | Justification for the relevance decision |
| ground_truth.<strategy>.annotator | string or null | "manual" or initials |
| ground_truth.<strategy>.annotation_date | string or null | ISO 8601 date |

### D.4 No Phase 14 Fields in questions.jsonl

Relevant chunk IDs are stored exclusively in data/questions/ground_truth.jsonl. The questions.jsonl file is frozen after Phase 13 and will NOT be modified with any relevant_chunk_ids_* fields.

---

## E. How Phase 14 Supports Phase 15/16 Evaluation

### E.1 Recall@K

For each question q and strategy s:
  R(q, s) = relevant_chunk_ids(q, s) union partial_chunk_ids(q, s)

  Recall@K(q, s) = |{retrieved top-K chunk IDs} intersect R(q, s)| / |R(q, s)|

Macro-averaged Recall@K across all 50 questions is the primary retrieval metric.

### E.2 Supported K Values

The retrieval config has top_k: 5. Phase 15 will evaluate Recall@1, Recall@3, and Recall@5.

### E.3 Stratified Analysis

Because questions.jsonl contains question_type, difficulty, chapter, and section, Phase 15/16 can break down Recall@K by all of these dimensions.

---

## F. Consistency and Validation

### F.1 Chunk ID Format

- Baseline: openstax_biology2e_ch1_5_baseline_NNNNNN
- ELUC: openstax_biology2e_ch1_5_eluc_NNNNNN
- Future strategies: openstax_biology2e_ch1_5_{strategy_key}_NNNNNN

### F.2 Validation Script (scripts/validate_ground_truth.py)

Will:
1. Load ground_truth.jsonl and questions.jsonl.
2. For each annotated strategy, verify all listed chunk IDs exist in the corpus.
3. Verify no chunk ID appears in both relevant and partial lists.
4. Verify null is used consistently for unimplemented strategies.
5. Report questions with empty relevant_chunk_ids requiring justification.

---

## G. Files to be Created in Phase 14

| File | Purpose |
|---|---|
| data/questions/ground_truth.jsonl | Primary ground truth artifact (50 records) |
| data/questions/ground_truth_schema.json | JSON schema for validation |
| scripts/validate_ground_truth.py | Validation script |
| tests/test_ground_truth.py | Unit tests for schema and validation logic |

---

## H. Prerequisites Before Implementation

1. BLOCKER -- Three strategies (recursive_character, paragraph_based, heading_based) must be implemented before full Phase 14 can be completed.
2. BLOCKER -- The index_corpus.py pipeline must be run to regenerate/export chunk data for baseline_fixed_size and eluc_section_aware so chunk IDs and text are available for annotation.
3. A chunk export script should be created to dump all chunks from ChromaDB to JSONL for offline annotation (chunk_id, text, page_span per record).

---

## I. Summary of Key Decisions

| Decision | Rationale |
|---|---|
| Ground truth stored separately from questions | Preserves Phase 13 schema integrity |
| Source passage + page range as primary evidence | Objective, auditable, independent of retrieval |
| relevant / partial two-level labeling | Allows nuanced multi-chunk answers without loss |
| No upper limit on relevant chunk count | Some questions genuinely span multiple source locations |
| null for unimplemented strategies | Honest representation; distinguishes not-annotated from no-relevant-chunks |
| Recall@K as primary metric | Standard IR metric; directly measures whether answer source is retrieved |
