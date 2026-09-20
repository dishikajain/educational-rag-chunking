# Phase 13: Question Dataset — Authoritative Methodology Specification (Corpus-Corrected)

> Comparative Study of Document Chunking Strategies in RAG for Educational QA
> Corpus: OpenStax Biology 2e, Chapters 1–5
> Status: FINAL — for use as implementation specification
> Corpus reference: OpenStax Biology 2e (official edition)

---

## A. Purpose and Scope

Phase 13 constructs the **single, frozen question dataset** that will be used identically across all five chunking strategies in Phase 16. Its purpose is to define the independent stimulus that the RAG pipeline responds to, so that any observed differences in answer quality can be attributed solely to the chunking strategy and not to question selection artifacts.

The methodology is a **corpus-driven, manually curated process** with the following properties:

- Questions are derived from reading the source text directly, not from chunk boundaries or retrieved results.
- Every question must be answerable from Chapters 1–5 without outside knowledge.
- The dataset is balanced across chapters, question types, and difficulty levels, with balance treated as a target rather than a rigid requirement.
- Each question has a reference answer grounded to a specific verbatim or near-verbatim passage from the source text.
- The dataset is frozen before any strategy-comparative experiment runs.

Phase 13 produces the question and its source-grounded reference answer only. Chunk-level relevance annotation per strategy belongs to Phase 14.

---

## B. Dataset Size and Justification

### Final Target: **50 questions**

### Justification

50 questions provides 50 paired observations per strategy. This is a practical balance across three considerations:

**Experimental coverage:** Five chapters, five question types, and three difficulty levels are all represented with meaningful counts. No chapter or type is reduced to a trivially small sample.

**Annotation workload:** Each question requires careful reading of the source text, writing of a question, identification of the source passage, and construction of a grounded reference answer. At an estimated 10–20 minutes per question, 50 questions represents approximately 8–17 person-hours of annotation — appropriate for an MCA-level project.

**Later statistical analysis:** 50 paired observations per strategy provides a meaningful basis for whatever statistical comparisons are defined in Phase 17. Formal power analysis, effect size assumptions, and test selection will be addressed at that stage when the evaluation metrics are known. No power claims are made here.

### Acceptable Range

45–60 questions. Deviation from 50 requires documentation of the reason (e.g., difficulty finding sufficient corpus-grounded questions of a particular type). Fewer than 40 compromises coverage; more than 60 significantly increases annotation workload without proportionate benefit at this corpus scope.

---

## C. Verified Corpus Structure — OpenStax Biology 2e, Chapters 1–5

The following chapter and section structure is verified from the official OpenStax Biology 2e text. All question metadata must use these exact titles.

### Chapter 1: The Study of Life
| Section | Title |
|---------|-------|
| 1.1 | The Science of Biology |
| 1.2 | Themes and Concepts of Biology |

**Key content:** Nature and process of science; basic vs. applied science; scientific method and reasoning; properties of life (order, homeostasis, reproduction, energy processing); levels of biological organisation; diversity and phylogeny; subdisciplines of biology.

---

### Chapter 2: The Chemical Foundation of Life
| Section | Title |
|---------|-------|
| 2.1 | Atoms, Isotopes, Ions, and Molecules: The Building Blocks |
| 2.2 | Water |
| 2.3 | Carbon |

**Key content:** Atomic structure; isotopes; ions; chemical bonds (covalent, ionic, hydrogen); water properties (polarity, cohesion, adhesion, specific heat, solvent capacity); acids and bases; pH; carbon chemistry; functional groups; macromolecule synthesis and hydrolysis overview.

---

### Chapter 3: Biological Macromolecules
| Section | Title |
|---------|-------|
| 3.1 | Synthesis of Biological Macromolecules |
| 3.2 | Carbohydrates |
| 3.3 | Lipids |
| 3.4 | Proteins |
| 3.5 | Nucleic Acids |

**Key content:** Dehydration synthesis and hydrolysis reactions; monomers and polymers; monosaccharides, disaccharides, polysaccharides (starch, glycogen, cellulose, chitin); fats, oils, phospholipids, steroids; amino acids; peptide bonds; primary through quaternary protein structure; enzymes; DNA and RNA structure; nucleotides; double helix.

---

### Chapter 4: Cell Structure
| Section | Title |
|---------|-------|
| 4.1 | Studying Cells |
| 4.2 | Prokaryotic Cells |
| 4.3 | Eukaryotic Cells |
| 4.4 | The Endomembrane System and Proteins |
| 4.5 | The Cytoskeleton |
| 4.6 | Connections between Cells and Cellular Activities |

**Key content:** Microscopy; cell theory; prokaryotic cell features (nucleoid, plasmids, cell wall, pili, flagella); eukaryotic organelles (nucleus, ER, Golgi, mitochondria, chloroplasts, ribosomes, vacuoles, lysosomes, peroxisomes); endomembrane system and protein trafficking; cytoskeletal components (microtubules, microfilaments, intermediate filaments); cell junctions (tight, gap, plasmodesmata).

---

### Chapter 5: Structure and Function of Plasma Membranes
| Section | Title |
|---------|-------|
| 5.1 | Components and Structure |
| 5.2 | Passive Transport |
| 5.3 | Active Transport |
| 5.4 | Bulk Transport |

**Key content:** Fluid mosaic model; phospholipid bilayer; membrane proteins (integral, peripheral, glycoproteins); membrane fluidity; selective permeability; diffusion; facilitated diffusion; osmosis; tonicity (hypertonic, hypotonic, isotonic); sodium–potassium pump; electrochemical gradients; endocytosis (phagocytosis, pinocytosis, receptor-mediated); exocytosis.

---

## D. Chapter Distribution

### Target
Approximately 10 questions per chapter. Deviation of approximately ±2 per chapter is acceptable where chapter content density naturally produces more or fewer suitable questions.

| Chapter | Title | Target Count | Acceptable Range |
|---------|-------|--------------|-----------------|
| 1 | The Study of Life | 10 | 8–11 |
| 2 | The Chemical Foundation of Life | 10 | 9–12 |
| 3 | Biological Macromolecules | 10 | 9–12 |
| 4 | Cell Structure | 11 | 9–12 |
| 5 | Structure and Function of Plasma Membranes | 9 | 8–11 |

**Rationale for slight variation:** Chapters 2, 3, and 4 have more sections and denser, more concept-rich content, which naturally supports a wider variety of question types. Chapters 1 and 5 are somewhat narrower in scope, making the lower end of their ranges appropriate.

The final distribution must be documented in `docs/phase13_dataset_methodology.md` after curation is complete.

### Chapter-Type Coverage Requirement
Each chapter should contain representation of each question type **where the chapter naturally supports that type**. A question type that does not fit a chapter's content should not be artificially introduced to satisfy a distribution target. The annotator documents any type that is absent from a chapter and the reason.

---

## E. Question Taxonomy

Five question types are defined. Each type tests a distinct cognitive skill relevant to educational question answering and exercises the retrieval pipeline differently.

### Type 1: `factual_lookup`
**Definition:** A question with a single, discrete, verifiable answer directly stated in the text.
**Bloom's level:** Remember
**Corpus fit by chapter:**
- Ch 1: Science terminology, properties of life — well supported.
- Ch 2: Atomic structure, bond types, water properties — well supported.
- Ch 3: Macromolecule monomers, bond names — well supported.
- Ch 4: Organelle names and locations — well supported; richest chapter for this type.
- Ch 5: Transport mechanism names, membrane component names — well supported.

**Example (correctly located):** *"In which organelle does cellular respiration primarily occur in eukaryotic cells?"* → Chapter 4, Section 4.3

---

### Type 2: `definition`
**Definition:** A question requesting the formal definition of a biological term or concept as presented in the text.
**Bloom's level:** Remember / Understand
**Corpus fit by chapter:**
- Ch 1: Homeostasis, biology, inductive/deductive reasoning — moderate; 2–3 definitions available.
- Ch 2: Isotope, ion, covalent bond, cohesion, pH — well supported.
- Ch 3: Monomer, polymer, dehydration synthesis, hydrolysis, phospholipid, peptide bond — richest chapter for definitions.
- Ch 4: Cell theory, organelle definitions (nucleus, mitochondria, lysosome, etc.) — well supported.
- Ch 5: Selective permeability, tonicity, osmosis, endocytosis — well supported.

**Example (correctly located):** *"How does OpenStax Biology 2e define dehydration synthesis?"* → Chapter 3, Section 3.1

---

### Type 3: `explanation_conceptual`
**Definition:** A question requiring explanation of a process, mechanism, or concept involving multiple related steps or ideas.
**Bloom's level:** Understand
**Corpus fit by chapter:**
- Ch 1: Scientific method steps — moderate; limited process depth.
- Ch 2: How hydrogen bonds give water its properties; acid-base reactions — well supported.
- Ch 3: How protein structure determines function; how dehydration synthesis builds polymers — well supported.
- Ch 4: How the endomembrane system processes and routes proteins — well supported; richest chapter for this type.
- Ch 5: How osmosis works; how the sodium–potassium pump operates — well supported.

**Example (correctly located):** *"How does the endomembrane system coordinate the synthesis and secretion of proteins?"* → Chapter 4, Section 4.4

---

### Type 4: `relationship_comparison`
**Definition:** A question asking the learner to compare, contrast, or describe the relationship between two or more entities or concepts.
**Bloom's level:** Analyse
**Corpus fit by chapter:**
- Ch 1: Basic vs. applied science; inductive vs. deductive reasoning — limited but present.
- Ch 2: Ionic vs. covalent bonds; acids vs. bases — well supported.
- Ch 3: DNA vs. RNA; saturated vs. unsaturated fats; starch vs. cellulose — well supported.
- Ch 4: Prokaryotic vs. eukaryotic cells; smooth ER vs. rough ER — best chapter for this type.
- Ch 5: Passive vs. active transport; endocytosis vs. exocytosis; hypertonic vs. hypotonic solutions — well supported.

**Example (correctly located):** *"What are the key structural and functional differences between prokaryotic and eukaryotic cells?"* → Chapter 4, Sections 4.2 and 4.3

---

### Type 5: `application_inference`
**Definition:** A question requiring the learner to apply a concept from the text to a scenario or to make an inference grounded in the text but not directly restated.
**Bloom's level:** Apply / Analyse
**Corpus fit by chapter:**
- Ch 1: Limited — the chapter is introductory and does not provide sufficient mechanistic content for robust application questions.
- Ch 2: Moderate — pH and water property scenarios are feasible.
- Ch 3: Moderate — protein denaturation or macromolecule function scenarios are feasible.
- Ch 4: Well supported — organelle dysfunction scenarios, endomembrane disruption — richest chapter for this type.
- Ch 5: Well supported — tonicity scenarios (placing a cell in a hypo/hypertonic solution); transport inhibition scenarios.

**Example (correctly located):** *"If a eukaryotic cell's lysosomes were non-functional, what would be the most likely consequence for the cell, and why?"* → Chapter 4, Section 4.3

> **Note on Chapter 1:** `application_inference` and `relationship_comparison` questions may be sparse or absent from Chapter 1 without this being a defect. Chapter 1 is conceptually introductory, and forcing these types would produce artificial questions. This must be documented in the final distribution report.

### Excluded Types
The following question forms are excluded from Phase 13:

- Questions requiring synthesis across more than two chapters — the corpus scope does not support reliable multi-hop grounding.
- Opinion, evaluative, or normative questions — cannot be grounded or reference-answered from the text.
- Numerical or calculation questions — Chapters 1–5 do not contain sufficient quantitative problem content.
- Questions referencing figures, diagrams, or tables by number — visual content is not represented in the text pipeline.

### Type Distribution Target

| Type | Target Count | Acceptable Range |
|------|-------------|-----------------|
| `factual_lookup` | 10 | 8–12 |
| `definition` | 10 | 8–12 |
| `explanation_conceptual` | 12 | 10–14 |
| `relationship_comparison` | 10 | 8–12 |
| `application_inference` | 8 | 6–10 |
| **Total** | **50** | |

`application_inference` is intentionally smaller because it is the most difficult type to ground unambiguously within a 5-chapter corpus, and Chapter 1 provides limited support for it.

---

## F. Difficulty Levels

Three levels are defined based on the number of source passages required and the cognitive demand.

| Level | Code | Operational Definition |
|-------|------|----------------------|
| Easy | `easy` | The answer is contained within a single sentence or short consecutive passage; no synthesis across multiple passages is required. Maps to Bloom's Remember. |
| Medium | `medium` | The answer requires reading 2–4 connected sentences or a short paragraph; mild synthesis is needed to formulate a coherent answer. Maps to Bloom's Understand. |
| Hard | `hard` | The answer requires integrating information from multiple paragraphs or comparing distinct passages; meaningful synthesis is required. Maps to Bloom's Analyse or Apply. |

### Target Distribution

| Difficulty | Target Count | Acceptable Range |
|------------|-------------|-----------------|
| `easy` | 15 | 13–17 |
| `medium` | 20 | 18–22 |
| `hard` | 15 | 13–17 |

### Type × Difficulty Cross-tabulation (Target)

The following cross-tabulation is a design target, not a rigid requirement. Natural, corpus-supported questions take priority. Deviations of approximately ±1 per cell are acceptable and must be documented.

|                           | easy | medium | hard | Total |
|---------------------------|------|--------|------|-------|
| `factual_lookup`          | 8    | 2      | 0    | 10    |
| `definition`              | 5    | 5      | 0    | 10    |
| `explanation_conceptual`  | 2    | 8      | 2    | 12    |
| `relationship_comparison` | 0    | 4      | 6    | 10    |
| `application_inference`   | 0    | 1      | 7    | 8     |
| **Total**                 | 15   | 20     | 15   | **50** |

---

## G. Dataset Schema

### Format: JSON Lines (`.jsonl`)

**Rationale:** JSON Lines is chosen over CSV because:
- The `reference_answer` and `source_passage` fields contain multi-line text that creates quoting and escaping complexity in CSV.
- JSON is human-readable and natively parsed in Python without special handling.
- JSON Lines allows incremental streaming, per-record validation, and natural line-by-line processing.
- A companion JSON Schema file provides machine-readable validation.

### Record Schema

```json
{
  "question_id": "Q001",
  "question": "In which organelle does cellular respiration primarily occur in eukaryotic cells?",
  "chapter": 4,
  "section": "4.3",
  "section_title": "Eukaryotic Cells",
  "question_type": "factual_lookup",
  "difficulty": "easy",
  "reference_answer": "Cellular respiration primarily occurs in the mitochondria. These organelles are the site of cellular respiration, the process that produces ATP—the cell's primary energy currency.",
  "source_passage": "The mitochondria (singular = mitochondrion) are often called the 'powerhouses' or 'energy factories' of a cell because they are responsible for making adenosine triphosphate (ATP), the cell's main energy-carrying molecule.",
  "source_page_range": "96–97",
  "answer_grounding": "Directly stated in Section 4.3, in the passage describing mitochondria.",
  "requires_outside_knowledge": false,
  "created_by": "manual_annotation",
  "creation_date": "YYYY-MM-DD",
  "notes": ""
}
```

### Field Definitions

| Field | Type | Phase | Required | Definition |
|-------|------|-------|----------|------------|
| `question_id` | string | 13 | ✓ | Sequential identifier Q001–Q050. Stable and unchanging across all subsequent phases. |
| `question` | string | 13 | ✓ | The question text. Self-contained and phrased as a student would ask it. |
| `chapter` | integer | 13 | ✓ | Chapter number 1–5. |
| `section` | string | 13 | ✓ | Section number as it appears in the text (e.g., "4.3"). |
| `section_title` | string | 13 | ✓ | Section heading as it appears in the OpenStax Biology 2e text. |
| `question_type` | string | 13 | ✓ | One of the five defined types. |
| `difficulty` | string | 13 | ✓ | `easy`, `medium`, or `hard`. |
| `reference_answer` | string | 13 | ✓ | A concise, grounded answer derived solely from the source passage. 2–5 complete sentences. Human-written. |
| `source_passage` | string | 13 | ✓ | The verbatim or near-verbatim passage from the text that grounds the reference answer. |
| `source_page_range` | string | 13 | ✓ | PDF page range (e.g., "96–97") for provenance tracing. |
| `answer_grounding` | string | 13 | ✓ | A one-line description of where in the text the answer is found. |
| `requires_outside_knowledge` | boolean | 13 | ✓ | Must be `false` for all included questions. |
| `created_by` | string | 13 | ✓ | `manual_annotation` for all Phase 13 questions. |
| `creation_date` | string | 13 | ✓ | ISO 8601 date (YYYY-MM-DD). |
| `notes` | string | 13 | — | Optional annotation note. May be empty string. |

### Fields That Must NOT Appear in Phase 13

The following fields belong exclusively to Phase 14 and must not be present in `questions.jsonl`:

| Field | Phase |
|-------|-------|
| `relevant_chunk_ids_baseline_fixed_size` | 14 |
| `relevant_chunk_ids_recursive_character` | 14 |
| `relevant_chunk_ids_paragraph_based` | 14 |
| `relevant_chunk_ids_heading_based` | 14 |
| `relevant_chunk_ids_eluc_section_aware` | 14 |

The Phase 13 validation script and pytest tests must actively assert the absence of all `relevant_chunk_ids_*` fields from every record.

---

## H. Question Construction and Selection Procedure

### Principle of Independence
Questions must be created by reading the raw source text — the PDF or the Phase 3 preprocessed `full_text` — and writing questions based on what is read. Chunk boundaries, retrieval results, and ChromaDB collections must not be consulted at any point during Phase 13.

### Step 1: Source Reading
The annotator reads each chapter section from the original PDF (pages 30–171 corresponding to Chapters 1–5) or from the Phase 3 preprocessed `full_text`.

### Step 2: Candidate Pool Generation
Generate approximately **80–90 candidate questions** (a 1.6–1.8× surplus over the 50-question target). Each candidate is assigned all required metadata fields including `source_passage` and a human-written `reference_answer`. Candidates are written to `data/questions/candidates.jsonl`.

### Step 3: Stratified Random Selection
Selection follows a deterministic stratified random procedure:

1. Assign each candidate to a (chapter, type, difficulty) cell.
2. For cells with surplus candidates, select randomly using a fixed seed of **42**, consistent with the project-wide reproducibility seed established in `generation_config.yaml`.
3. Selection is performed by running `scripts/select_questions.py --seed 42`.
4. The script samples to meet the distribution targets defined in Sections D and F, respecting the ±2 per-chapter and ±1 per-cell tolerances.
5. The selected 50 questions are written to `data/questions/questions.jsonl`, with `question_id` values assigned sequentially (Q001–Q050) in chapter order.

### Step 4: Reference Answer Construction Protocol
Each reference answer must:
- Be written in 2–5 complete sentences.
- Draw only from the corresponding `source_passage` field.
- Not add definitions, clarifications, or elaborations from outside the source passage.
- Be written by the human annotator before and independently of any model output.
- Not be generated by Gemini or any other language model.

Reference answers grounded in model output would create circular evaluation and must be avoided entirely.

### Step 5: Question Phrasing Guidelines
Questions must:
- Be phrased as a biology student would naturally ask them.
- Be self-contained — answerable without reading surrounding questions.
- Not reference chunk boundaries, paragraph positions, or structural artifacts.
- Not presuppose information from chapters outside the scope (Chapters 1–5).
- Use terminology from the source text.

---

## I. Quality-Control Procedure

Quality control is performed in two passes: automated and manual.

### Automated Checks (`scripts/validate_questions.py`)

| Check | Criterion |
|-------|-----------|
| Schema validity | Every record validates against `data/questions/questions_schema.json` |
| `requires_outside_knowledge` | Must be `false` for all records |
| Required fields present | All required fields are non-empty strings or valid typed values |
| `question_id` uniqueness | No duplicate IDs |
| Phase 14 field absence | No `relevant_chunk_ids_*` fields present in any record |
| Chapter range | All `chapter` values are integers in {1, 2, 3, 4, 5} |
| Chapter balance | Each chapter has between 8 and 12 questions |
| Type representation | Each type has at least 7 questions |
| Difficulty balance | easy ≥ 13, medium ≥ 18, hard ≥ 13 |
| Section coverage | Each chapter has at least 3 distinct sections represented |
| Section title accuracy | `section_title` must match the known OpenStax Biology 2e section heading for the given `section` |
| Question length | Between 10 and 250 characters |
| Reference answer length | At least 50 characters |
| Source passage non-empty | Non-empty string |
| `creation_date` format | ISO 8601 (YYYY-MM-DD) |
| `question_type` enumeration | One of the five defined values |
| `difficulty` enumeration | One of `easy`, `medium`, `hard` |

### Known Valid Section Titles (for automated validation)

The following section titles are the only valid values for `section_title` in Phase 13 records:

```
Ch1: "The Science of Biology", "Themes and Concepts of Biology"
Ch2: "Atoms, Isotopes, Ions, and Molecules: The Building Blocks", "Water", "Carbon"
Ch3: "Synthesis of Biological Macromolecules", "Carbohydrates", "Lipids", "Proteins", "Nucleic Acids"
Ch4: "Studying Cells", "Prokaryotic Cells", "Eukaryotic Cells",
     "The Endomembrane System and Proteins", "The Cytoskeleton",
     "Connections between Cells and Cellular Activities"
Ch5: "Components and Structure", "Passive Transport", "Active Transport", "Bulk Transport"
```

### Manual Review Checklist
For every selected question, the annotator completes:

- [ ] **Answerability:** Reference answer is fully supported by `source_passage` without inference beyond the text.
- [ ] **No ambiguity:** The question has exactly one defensible answer. If two equally valid answers exist, the question is rewritten or dropped.
- [ ] **No near-duplicates:** The question is not semantically equivalent to another in the dataset.
- [ ] **No chunk-specific wording:** The question does not reference structural artifacts.
- [ ] **No outside-knowledge dependence:** The reference answer does not require knowledge beyond Chapters 1–5.
- [ ] **Difficulty consistency:** The assigned difficulty matches the operational definition.
- [ ] **Source passage accuracy:** `source_passage` is verbatim or near-verbatim from the text.
- [ ] **Chapter and section accuracy:** `chapter`, `section`, and `section_title` are verified against the actual OpenStax Biology 2e structure defined in Section C of this document.

---

## J. Phase Boundary Definitions

### Phase 13 — Question Dataset (current phase)

**Produces:**
- `data/questions/candidates.jsonl` — candidate pool (not frozen)
- `data/questions/questions.jsonl` — **FROZEN** selected dataset
- `data/questions/questions_schema.json` — JSON Schema draft-07
- `scripts/select_questions.py` — deterministic selection script
- `scripts/validate_questions.py` — automated QC script
- `tests/test_question_dataset.py` — pytest tests
- `config/question_dataset_config.yaml` — dataset parameters
- `docs/phase13_dataset_methodology.md` — methodology and final distribution report

**Does NOT produce:**
- ❌ Chunk-level relevance labels for any strategy
- ❌ Retrieval results from any strategy
- ❌ Model-generated answers or evaluations
- ❌ Any data requiring inspection of chunk boundaries

### Phase 14 — Retrieval Ground Truth
For each question and each of the five strategies: the list of chunk IDs that a human judge considers relevant. Fields `relevant_chunk_ids_*` are stored in a separate Phase 14 file. They are never added to `questions.jsonl`.
Requires: Phase 13 frozen questions + all five strategies indexed.

### Phase 15 — Evaluation Module
Code to score RAG outputs against Phase 13 reference answers and Phase 14 ground truth. Does not modify the question dataset or retrieval ground truth.

> [!IMPORTANT]
> Questions must be created and frozen **before** any chunking-strategy comparison is run and **before** Phase 14 annotation begins. Questions written after inspecting chunked output risk implicit bias toward whichever strategy's boundaries happened to align with the question wording.

---

## K. Required Project Files and Directory Structure

```
data/
└── questions/
    ├── candidates.jsonl           # ~80–90 candidate questions (pre-selection, not frozen)
    ├── questions.jsonl            # FROZEN: 50 selected, validated questions
    └── questions_schema.json      # JSON Schema draft-07 for validation

scripts/
    ├── select_questions.py        # Stratified random selection (--seed 42)
    └── validate_questions.py      # Automated QC checks against schema and targets

tests/
└── test_question_dataset.py       # pytest tests for dataset validity

docs/
└── phase13_dataset_methodology.md # Methodology documentation and final distribution report

config/
└── question_dataset_config.yaml   # Dataset parameters: target counts, seed, tolerances
```

### Frozen Artifact
`data/questions/questions.jsonl` is the sole frozen artifact of Phase 13. Once validated and committed, this file must not be modified. If a correction is required after freezing, it must be implemented as a documented revision (`questions_v2.jsonl`) with a justification entry in `docs/phase13_dataset_methodology.md`.

---

## L. Reproducibility Requirements

- The selection script must be deterministic: given the same `candidates.jsonl` and `--seed 42`, it must always produce the same `questions.jsonl`.
- All random operations inside `select_questions.py` must use `random.seed(42)` or `numpy.random.default_rng(42)` with no other random state dependencies.
- The `question_dataset_config.yaml` file records the seed, target counts, and tolerances so the selection parameters are explicit and auditable.
- The final distribution (actual per-chapter, per-type, per-difficulty counts) must be recorded in `docs/phase13_dataset_methodology.md` after curation, not before.

---

## M. Risks and Methodological Concerns

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Question leakage** — annotator phrases questions to match chunk boundaries | High | Annotator reads only the source PDF; chunk outputs are never consulted during Phase 13 |
| **Corpus mapping errors** — section numbers or titles taken from wrong edition or textbook | High | All section numbers and titles validated against the Section C table in this document; automated validator checks `section_title` against the known-valid list |
| **Chapter imbalance** — content-dense chapters (3, 4) attract disproportionate questions | Medium | Stratified selection enforces ±2 per-chapter tolerance; imbalance is documented |
| **Reference answer hallucination** — annotator introduces outside knowledge | High | Each answer cites `source_passage`; `requires_outside_knowledge` is validated; no model generation |
| **Difficulty inflation** — annotator labels everything as `medium` | Medium | Difficulty defined operationally; manual review checks each assignment |
| **Near-duplicate questions** — similar topics generate similar questions | Medium | Manual review within (chapter, type) pairs |
| **Artificial questions** — questions created solely to satisfy a distribution cell | Medium | Distribution targets are guidelines; natural corpus support takes priority; documented where types are absent |
| **Phase 14 contamination** — `relevant_chunk_ids_*` fields added prematurely | High | Schema test enforces absence; validation script checks all records |
| **Corpus scope drift** — question inadvertently references content beyond Chapter 5 | Medium | `requires_outside_knowledge: false` validation; annotator reads only Chapters 1–5 |
| **Model-generated reference answers** — introduces circular evaluation | High | Explicitly prohibited; all reference answers are human-written from the source passage |

---

## N. Erratum: Post-Freeze Metadata Corrections

Following a strict block-level corpus provenance audit during Phase 14B, it was discovered that three questions were manually annotated with off-by-one physical page numbers. The source_page_range metadata pointed to pages where the corresponding section (and source text) was entirely absent. The following in-place metadata corrections were made to the frozen dataset to preserve dataset viability while maintaining the original text:

- **Q022**: source_page_range corrected from "84" to "83" (Section 3.1).
- **Q033**: source_page_range corrected from "124" to "123" (Section 4.2).
- **Q050**: source_page_range corrected from "172" to "171" (Section 5.3).
