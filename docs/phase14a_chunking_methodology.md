# Phase 14A: Chunking Strategy Implementation Methodology

**Project:** A Comparative Study of Document Chunking Strategies in Retrieval-Augmented Generation for Educational Question Answering
**Corpus:** OpenStax Biology 2e, Chapters 1-5 (PDF pages 30-171, 142 pages)
**Phase 14A Goal:** Implement and freeze the three missing chunking strategies so that all five strategies produce formal chunk collections from the same preprocessing output. This completes the prerequisite for Phase 14B ground-truth annotation.

---

## A. Experimental Controls (Frozen Across All Five Strategies)

The following are identical for all five strategies. Nothing in Phase 14A may change them.

| Control | Value | Source |
|---|---|---|
| Source corpus | OpenStax Biology 2e | data/raw/openstax_biology2e.pdf |
| Corpus scope | Chapters 1-5 (PDF pages 30-171, 0-indexed 29-170) | scripts/index_corpus.py |
| doc_id | openstax_biology2e_ch1_5 | scripts/index_corpus.py |
| Preprocessing config | config/preprocessing_config.yaml (frozen) | Phase 3 |
| Embedding model | BAAI/bge-base-en-v1.5 | config/embedding_config.yaml |
| Embedding revision | a5beb1e3e68b9ab74eb54cfd186867f64f240e1a | config/embedding_config.yaml |
| Embedding dimension | 768 | config/embedding_config.yaml |
| Normalize embeddings | True | config/embedding_config.yaml |
| Distance metric | cosine | config/indexing_config.yaml, config/retrieval_config.yaml |
| ChromaDB storage | data/chroma_db | config/indexing_config.yaml |
| Question dataset | data/questions/questions.jsonl (50 questions, frozen) | Phase 13 |

---

## B. Existing Strategy Implementations (Frozen, No Changes)

### Strategy 1: baseline_fixed_size

- **Module:** src/chunking/chunker.py
- **Config:** config/baseline_chunking_config.yaml
- **Algorithm:** Fixed-size sliding window on doc.full_text (character-level). Structure-blind: does not use any block labels, gaps, or headings. Chunk boundaries are determined entirely by char position.
- **Frozen Parameters:**
  - chunk_size: 350 characters
  - chunk_overlap: 40 characters (stride = 310)
  - min_chunk_size: 70 characters
- **Chunk ID format:** {doc_id}_baseline_{index:06d}
- **Provenance:** _find_overlapping_provenance() maps each chunk's [char_start, char_end) back to ProvenanceSpan list. Page range derivable from provenance_spans.
- **Rationale for params:** Calibrated from Phase 3 output: chunk_size = round(p75 block length / 50) * 50 = 350 chars (p75 was 364 chars from 1416 blocks across Ch 1-5).

### Strategy 5: eluc_section_aware

- **Module:** src/eluc/chunker.py
- **Config:** config/eluc_chunking_config.yaml
- **Algorithm:** Four-pass structure-aware chunker. Pass 1: marks boundary_before using HEADING labels and is_section_break flags from PreprocessedBlock. Pass 2: accumulates blocks until max_chunk_chars, flushing on boundaries and overflow. Pass 3: overflow warnings. Pass 4: assembles ELUCChunk objects.
- **Frozen Parameters:**
  - max_chunk_chars: 1400 characters
  - min_block_chars: 5 characters
  - sidebar_atomic: True (sidebars never split mid-unit)
- **Chunk ID format:** {doc_id}_eluc_{index:06d}
- **Provenance:** char_start/char_end located via provenance; _find_overlapping_provenance() applies. Additional fields: source_blocks, heading_label, contains_sidebar, block_count, page_span.

---

## C. Missing Strategy Designs

### Design Principles for All Three New Strategies

1. All three receive the same PreprocessedDocument as baseline and ELUC: identical preprocessing output, identical full_text, identical provenance.
2. No chunking parameter may be tuned using the question dataset or retrieval results.
3. Parameters are calibrated from corpus statistics established in earlier phases and must be fixed before any retrieval evaluation.
4. Each strategy must produce a BaselineChunk-compatible object (chunk_id, doc_id, chunk_index, text, char_start, char_end, provenance_spans) so that the existing embed_document() and index_document() pipeline functions accept them without modification.
5. Chunk IDs must be unique within each strategy corpus and deterministic across runs.

---

### Strategy 2: recursive_character

#### Algorithm

Recursive character splitting with a priority-ordered list of separator strings. The algorithm attempts to split text using the highest-priority separator first; if any resulting piece is still larger than chunk_size, it recurses using the next separator. Overlapping windows of size chunk_overlap are added at each reassembly stage to maintain context continuity.

Detailed steps:
1. Start with doc.full_text (same character stream as baseline).
2. Define a separator priority list: ["\n\n", "\n", ". ", " ", ""]
3. Attempt to split full_text using the first separator ("\n\n").
4. For each resulting fragment:
   a. If len(fragment) <= chunk_size: keep as a terminal chunk candidate.
   b. If len(fragment) > chunk_size: recurse using the next separator in the list.
5. After splitting, reassemble into overlapping windows:
   a. Walk through terminal fragments in order.
   b. Accumulate into a window up to chunk_size characters.
   c. When window would exceed chunk_size, emit it as a chunk, then advance the window start by (chunk_size - chunk_overlap) characters worth of fragments.
6. Emit final window.
7. Discard any chunk shorter than min_chunk_size.

**Key distinction from baseline_fixed_size:** Splits prefer natural text boundaries (\n\n, \n, sentence, word) rather than cutting blindly at fixed character offsets.

#### Parameters

| Parameter | Value | Justification |
|---|---|---|
| chunk_size | 350 characters | Same as baseline_fixed_size so the two strategies differ ONLY in boundary-detection method, enabling a clean controlled comparison. |
| chunk_overlap | 40 characters | Same as baseline to match context-window behaviour across comparison pair. |
| min_chunk_size | 70 characters | Same as baseline. |
| separators | ["\n\n", "\n", ". ", " ", ""] | Standard recursive text split priority; double newline (paragraph) first, then single newline (line), then sentence period, then word, then character. |

**Chunk ID format:** {doc_id}_recursive_{index:06d}

**Provenance:** Use _find_overlapping_provenance(doc.provenance, char_start, char_end) exactly as baseline does. char_start and char_end are tracked per chunk by accumulating character offsets through the reassembly step.

**Output data model:** BaselineChunk (same dataclass as strategy 1). A BaselineChunkDocument with strategy="recursive_character".

**Determinism:** The separator list is fixed; no random element. Given the same full_text and separator list, the split and reassembly always produce the same chunks in the same order.

---

### Strategy 3: paragraph_based

#### Algorithm

Block-granularity chunking: accumulates whole PreprocessedBlock objects. A new chunk starts whenever any of the following signals is true:
- The current block has is_paragraph_break=True AND the accumulated char count >= paragraph_soft_min (prevents creating chunks of a single short sentence).
- The accumulated char count would exceed max_chunk_chars if the block were added.
- The current block has label in HEADING_LABELS (hard boundary: a heading always starts a new chunk).

Blocks are joined with "\n\n" (the same separator the preprocessor uses in full_text). The chunk's char_start and char_end are read from provenance spans.

**Key distinction from eluc_section_aware:** Paragraph-based splits only on is_paragraph_break (a gap-threshold signal from the preprocessor) rather than on structural heading labels. It does not require a heading to trigger a boundary. A section with many paragraphs will produce multiple paragraph chunks, whereas ELUC would keep them in one chunk up to max_chunk_chars.

#### Parameters

| Parameter | Value | Justification |
|---|---|---|
| max_chunk_chars | 1400 characters | Same as ELUC max_chunk_chars so any size difference between paragraph_based and eluc_section_aware is purely due to boundary selection criteria (paragraph gap vs. heading). |
| paragraph_soft_min | 150 characters | Prevents single-line paragraph boundaries from splitting trivially short content. 150 chars is approximately one body sentence in this corpus (body blocks average 252.7 chars; short blocks are ~50-120 chars). Any smaller threshold would produce many single-sentence chunks. |
| min_block_chars | 5 characters | Same as ELUC; rejects blank or near-blank blocks. |

**Chunk ID format:** {doc_id}_paragraph_{index:06d}

**Provenance:** Locate char_start from the first block's provenance span, char_end from the last block's provenance span, then call _find_overlapping_provenance() for the full range.

**Output data model:** BaselineChunk (same dataclass). A BaselineChunkDocument with strategy="paragraph_based".

**Determinism:** Driven entirely by is_paragraph_break and label fields computed deterministically by the preprocessor. No randomness.

---

### Strategy 4: heading_based

#### Algorithm

Heading-anchored chunking: every chunk must begin at a heading block (label in HEADING_LABELS) or at the start of the document. All body blocks following that heading are accumulated into the same chunk until the next heading is encountered or max_chunk_chars is exceeded.

Detailed steps:
1. Flatten all PreprocessedBlocks from doc.pages.
2. Walk blocks in reading order.
3. Whenever a HEADING block is encountered, flush the current chunk (if non-empty) and start a new chunk beginning with that heading.
4. Non-heading blocks are always appended to the current chunk.
5. If appending a block would exceed max_chunk_chars: flush the current chunk WITHOUT the new block, then start a new chunk beginning with the new block (overflow split). This prevents unbounded chunks in long sections.
6. At end of document, flush the final chunk.
7. Discard chunks shorter than min_chunk_chars.

HEADING_LABELS used: {"HEADING_1", "HEADING_2", "HEADING_3", "CHAPTER_TITLE", "CHAPTER_NUMBER", "CHAPTER_INTRO_SPAN"} -- identical to the set in src/eluc/chunker.py.

**Key distinction from eluc_section_aware:** Heading-based ONLY splits on headings (or overflow). It never splits mid-section at an is_section_break gap signal. This means sections with no internal headings remain as one chunk up to max_chunk_chars, whereas ELUC can split on section gap signals even without a heading. Each chunk is explicitly anchored to a heading text.

#### Parameters

| Parameter | Value | Justification |
|---|---|---|
| max_chunk_chars | 1400 characters | Same as ELUC, enabling a clean comparison: heading_based vs. eluc_section_aware differ only in which structural signal (heading-only vs. heading + section break) triggers a split. |
| min_chunk_chars | 70 characters | Same as baseline min_chunk_size; rejects stub chunks created by adjacent headings with no body text. |

**Chunk ID format:** {doc_id}_heading_{index:06d}

**Provenance:** Same approach as ELUC: locate char_start from the first block's provenance span, char_end from the last block's provenance span, then call _find_overlapping_provenance().

**Output data model:** BaselineChunk (same dataclass). A BaselineChunkDocument with strategy="heading_based".

**Determinism:** Driven by label field from preprocessor. No randomness.

---

## D. Strategy Comparison Summary

| Dimension | baseline_fixed_size | recursive_character | paragraph_based | heading_based | eluc_section_aware |
|---|---|---|---|---|---|
| Boundary signal | Character position | Text separators | is_paragraph_break gap | Heading labels | Heading labels + is_section_break |
| Corpus structure used | None | Partial (text only) | Block gaps | Block labels | Block labels + gaps |
| chunk_size / max_chars | 350 chars | 350 chars | 1400 chars | 1400 chars | 1400 chars |
| Overlap | 40 chars | 40 chars | None | None | None |
| Sidebar handling | None | None | None | Sidebars split normally | Sidebar atomic (never split) |
| Chunk ID prefix | _baseline_ | _recursive_ | _paragraph_ | _heading_ | _eluc_ |
| Source object | full_text | full_text | blocks | blocks | blocks |
| Model class | BaselineChunk | BaselineChunk | BaselineChunk | BaselineChunk | ELUCChunk |

---

## E. Parameter Calibration Basis

Parameters are calibrated from the Phase 3 corpus statistics already recorded in config/baseline_chunking_config.yaml:

- Full text length: 360,640 characters
- Body block count: 1416
- Block length p50: 120 chars (a typical paragraph sentence)
- Block length p75: 364 chars (basis for baseline chunk_size = 350)
- Block length p90: 662 chars
- Mean block length: 252.7 chars

Derived decisions:
- chunk_size = 350: matches p75 block length so baseline chunks contain roughly one body paragraph.
- max_chunk_chars = 1400: 4x the baseline chunk_size; allows paragraph_based, heading_based, and ELUC to include several body paragraphs under one heading without further splitting, matching realistic educational content units.
- chunk_overlap = 40: 11.4% of chunk_size; preserves enough trailing context for bridging questions while minimizing duplication.
- paragraph_soft_min = 150: approximately p25 block length (51 chars actual) rounded up to a conservative single-sentence minimum to avoid trivially short chunks.
- min_chunk_chars = 70: same as baseline (chunk_size / 5), consistent across all strategies.

---

## F. Files to Be Created

### New source modules

| File | Purpose |
|---|---|
| src/recursive_chunking/__init__.py | Package init |
| src/recursive_chunking/config.py | RecursiveChunkingConfig dataclass + load_config() |
| src/recursive_chunking/models.py | RecursiveChunkDocument (thin wrapper; chunks are BaselineChunk) |
| src/recursive_chunking/chunker.py | recursive_chunk() function |
| src/paragraph_chunking/__init__.py | Package init |
| src/paragraph_chunking/config.py | ParagraphChunkingConfig dataclass + load_config() |
| src/paragraph_chunking/models.py | ParagraphChunkDocument |
| src/paragraph_chunking/chunker.py | paragraph_chunk() function |
| src/heading_chunking/__init__.py | Package init |
| src/heading_chunking/config.py | HeadingChunkingConfig dataclass + load_config() |
| src/heading_chunking/models.py | HeadingChunkDocument |
| src/heading_chunking/chunker.py | heading_chunk() function |

### New config files

| File | Purpose |
|---|---|
| config/recursive_chunking_config.yaml | Frozen config for recursive_character strategy |
| config/paragraph_chunking_config.yaml | Frozen config for paragraph_based strategy |
| config/heading_chunking_config.yaml | Frozen config for heading_based strategy |

### Modified files

| File | Change Required |
|---|---|
| scripts/index_corpus.py | Add import and indexing steps for three new strategies |
| src/embedding/embedder.py | Add new chunk document types to the Union type annotation |
| src/retrieval/retriever.py | Add three new strategy names to the allowed list |

### New test files

| File | Purpose |
|---|---|
| tests/test_recursive_chunking.py | Unit tests for recursive_character strategy |
| tests/test_paragraph_chunking.py | Unit tests for paragraph_based strategy |
| tests/test_heading_chunking.py | Unit tests for heading_based strategy |

---

## G. No-Change Scope

The following files must NOT be modified in Phase 14A:

- src/chunking/* (baseline_fixed_size) -- frozen
- src/eluc/* (eluc_section_aware) -- frozen
- config/baseline_chunking_config.yaml -- frozen
- config/eluc_chunking_config.yaml -- frozen
- config/embedding_config.yaml -- frozen
- config/indexing_config.yaml -- frozen
- config/retrieval_config.yaml -- frozen
- config/preprocessing_config.yaml -- frozen
- data/questions/questions.jsonl -- frozen (Phase 13)
- data/questions/candidates.jsonl -- frozen (Phase 13)
- data/questions/questions_schema.json -- frozen (Phase 13)
- scripts/select_questions.py -- frozen (Phase 13)
- scripts/validate_questions.py -- frozen (Phase 13)
- tests/test_question_dataset.py -- frozen (Phase 13)
- docs/phase13_dataset_methodology.md -- frozen (Phase 13)
- config/question_dataset_config.yaml -- frozen (Phase 13)

---

## H. Chunk Output Artifacts (Phase 14A Deliverable)

After implementation and indexing:
- data/chroma_db/ will contain five ChromaDB collections, one per strategy.
- data/chunks/formal/{strategy}/ will contain exported JSONL files of all chunks (one JSON object per line: chunk_id, text, char_start, char_end, page_span) for offline annotation in Phase 14B.

Collection naming convention (from sanitize_collection_name()):
- baseline_fixed_size + BAAI/bge-base-en-v1.5 -> baseline_fixed_size_bge_base_en_v1_5
- recursive_character + BAAI/bge-base-en-v1.5 -> recursive_character_bge_base_en_v1_5
- paragraph_based + BAAI/bge-base-en-v1.5 -> paragraph_based_bge_base_en_v1_5
- heading_based + BAAI/bge-base-en-v1.5 -> heading_based_bge_base_en_v1_5
- eluc_section_aware + BAAI/bge-base-en-v1.5 -> eluc_section_aware_bge_base_en_v1_5

---

## I. Methodological Conflict Check with Phases 1-13

| Item | Assessment |
|---|---|
| Phase 3 preprocessing | No conflict. All five strategies consume the same PreprocessedDocument. |
| Phase 4 baseline chunking | No conflict. Strategy 1 config and code are frozen and unchanged. |
| Phase 7 embedding model | No conflict. All five strategies use the same frozen BGE model. |
| Phase 8/9 indexing/retrieval | Minor: retriever.py currently whitelists only 2 strategies. Must be extended to accept all 5. This is a permitted additive change (does not break existing strategy lookup). |
| Phase 13 questions | No conflict. Questions are not used to select or tune chunking parameters. |
| Phase 13 question schema | No conflict. No new fields added to questions.jsonl. |
| Phase 14 ground truth plan | Fully compatible. The ground_truth.jsonl schema allows null for strategies not yet annotated. |

No methodological conflicts found.

---

## J. Determinism and Reproducibility Checklist

| Property | recursive_character | paragraph_based | heading_based |
|---|---|---|---|
| Same input (PreprocessedDocument) always produces same output | Yes | Yes | Yes |
| No random number generation | Yes | Yes | Yes |
| No external data source | Yes | Yes | Yes |
| Chunk order is deterministic | Yes (reading order) | Yes (reading order) | Yes (reading order) |
| Chunk IDs are deterministic | Yes (sequential index) | Yes (sequential index) | Yes (sequential index) |
| Parameters fixed before any retrieval evaluation | Yes | Yes | Yes |
| Parameters not derived from question dataset | Yes | Yes | Yes |
