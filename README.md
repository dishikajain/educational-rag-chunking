# A Comparative Study of Document Chunking Strategies in Retrieval-Augmented Generation for Educational Question Answering

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/tests-172%20passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App%20Available-FF4B4B.svg)](src/app/streamlit_app.py)

---

## 📌 Executive Summary & Research Question

**Research Question:**  
> *Does preserving pedagogically coherent learning units during document chunking improve retrieval effectiveness and answer quality in Retrieval-Augmented Generation (RAG) systems for educational question answering?*

In educational question-answering systems powered by Retrieval-Augmented Generation (RAG), documents must be segmented into chunks before indexing and embedding. Standard pipelines typically employ arbitrary fixed-character windowing or generic recursive character splitting. However, educational content exhibits rich semantic hierarchies (chapters, modules, sub-sections, and conceptual boundaries). 

This repository implements an end-to-end, reproducible research study evaluating **five distinct document chunking strategies** on an educational biology textbook under strictly controlled experimental conditions.

---

## 🔬 Experimental Methodology & Control

To ensure a fair, reproducible comparison, all experimental variables are held strictly constant across all conditions:
- **Corpus:** Chapters 1–5 of OpenStax *Biology 2e* (The Study of Life, The Chemical Foundation of Life, Biological Macromolecules, Cell Structure, and Structure and Function of Plasma Membranes).
- **Preprocessing:** Identical PDF extraction via PyMuPDF, cleaning, text normalization, and structural boundary extraction.
- **Embedding Model:** Fixed `BAAI/bge-small-en-v1.5` (384-dimensional dense embeddings, normalized cosine similarity).
- **Vector Database:** ChromaDB with distinct collections per chunking strategy.
- **Retrieval Cutoff:** Fixed Top-$k = 5$.
- **Evaluation Dataset:** 50 validated, manually curated educational queries mapped to provenance-grounded reference text spans (yielding 250 evaluation observations across the 5 conditions).
- **Statistical Testing:** Non-parametric repeated-measures design (Friedman omnibus test + Holm-adjusted Wilcoxon signed-rank post-hoc pairwise tests, Kendall's $W$ effect sizes, and 10,000-sample bootstrap confidence intervals).

---

## 📊 Summary of Experimental Results

### 1. Overall Retrieval Performance Across Strategies (N = 50 queries)

| Strategy | Mean Recall@1 | Mean Recall@3 | Mean Recall@5 | Mean MRR@5 |
| :--- | :---: | :---: | :---: | :---: |
| **Heading-based** | **0.102** | **0.217** | **0.274** | **0.464** |
| **ELUC / Section-aware** | 0.071 | 0.190 | 0.256 | 0.393 |
| **Paragraph-based** | 0.073 | 0.137 | 0.215 | 0.434 |
| **Recursive character** | 0.025 | 0.067 | 0.111 | 0.357 |
| **Baseline Fixed-size** | 0.024 | 0.064 | 0.097 | 0.317 |

### 2. Statistical Omnibus Tests (Friedman Test)

| Metric | Friedman $\chi^2$ ($df=4$) | $p$-value | Kendall's $W$ (Effect Size) | Statistical Significance ($\alpha = 0.05$) |
| :--- | :---: | :---: | :---: | :---: |
| **Recall@1** | 44.852 | $4.27 \times 10^{-9}$ | 0.224 | **Yes** ($p < 0.001$) |
| **Recall@3** | 70.023 | $2.24 \times 10^{-14}$ | 0.350 | **Yes** ($p < 0.001$) |
| **Recall@5** | 81.607 | $7.95 \times 10^{-17}$ | 0.408 | **Yes** ($p < 0.001$) |
| **MRR@5** | 18.771 | $8.72 \times 10^{-4}$ | 0.094 | **Yes** ($p < 0.001$) |

### 3. Key Findings & Takeaways
1. **Document Structure Matters:** Strategies that respect structural discourse boundaries (**Heading-based**, **ELUC**, and **Paragraph-based**) significantly outperform arbitrary boundary strategies (**Fixed-size** and **Recursive Character**) across all recall metrics ($p < 0.05$ after Holm correction).
2. **Heading-based Performance:** Heading-based chunking achieved the highest observed retrieval recall and MRR@5, providing optimal unit granularity for concept-level educational queries.
3. **Arbitrary Splits Incur Loss:** Fixed-size and Recursive Character splitting often sever coherent definitions and explanations across chunk boundaries, yielding substantial drops in recall (mean Recall@5 of ~0.10 vs. ~0.27 for Heading-based).
4. **Rank Sensitivity:** Omnibus effect sizes ($W$) peaked at Recall@5 ($W = 0.408$), demonstrating that rank differentiation becomes pronounced at higher top-$k$ cutoffs.

---

## 🧩 Chunking Strategies Compared

| Strategy | Primary Boundary Signal | Relevant Size / Overlap Parameter | Total Formal Chunks Generated | Description |
| :--- | :--- | :--- | :---: | :--- |
| **Baseline Fixed-size** | Character position | `chunk_size=350`, `overlap=40` | 1,196 | Fixed-window chunking splitting strictly on character count regardless of words or sentences. |
| **Recursive character** | Text separators (`\n\n`, `\n`, ` `) | `chunk_size=350`, `overlap=40` | 1,302 | LangChain-style recursive separator hierarchy maintaining token windows. |
| **Paragraph-based** | Paragraph break gaps (`is_paragraph_break`) | `max_chars=1400` | 660 | Accumulates natural paragraph units until the character capacity threshold is reached. |
| **Heading-based** | Heading labels (`H1`, `H2`, `H3`) | `max_chars=1400` | 407 | Groups content under common hierarchical heading boundaries. |
| **ELUC / Section-aware** | Heading labels + `is_section_break` | `max_chars=1400` | 473 | **Educational Learning Unit Chunking**: Dynamically preserves pedagogical learning sections, sub-topics, and learning objectives. |

---

## 🏛️ System Architecture

The pipeline consists of modular, loosely-coupled components orchestrated through standard configurations:

```
[Raw Textbook PDF]
        │
        ▼ (PyMuPDF Extraction)
[ExtractedDocument & Pages]
        │
        ▼ (Text Normalization & Boundary Flagging)
[Preprocessed Spans & Tokens]
        │
        ▼ (5 Chunking Implementations)
[Chunked Collections (Fixed, Recursive, Paragraph, Heading, ELUC)]
        │
        ▼ (BAAI/bge-small-en-v1.5)
[384-d Dense Embeddings]
        │
        ▼ (ChromaDB Vector Store)
[ChromaDB Local Collections]
        │
        ▼ (Cosine Similarity Search, Top-k=5)
[Retriever Engine]
        │
        ├──▶ [Evaluation & Statistical Analysis (Recall@k, MRR@5, Friedman, Wilcoxon)]
        │
        ▼ (Prompt Engineering & Grounding)
[Gemini Generation Engine (Google GenAI)]
        │
        ▼
[Interactive Streamlit Application & Reports]
```

---

## 📁 Repository Structure

```
educational-rag-chunking/
├── app/                           # Web application components
├── config/                        # YAML configuration files for each component
│   ├── baseline_chunking_config.yaml
│   ├── eluc_chunking_config.yaml
│   ├── embedding_config.yaml
│   ├── generation_config.yaml
│   ├── heading_chunking_config.yaml
│   ├── indexing_config.yaml
│   ├── paragraph_chunking_config.yaml
│   ├── preprocessing_config.yaml
│   ├── question_dataset_config.yaml
│   ├── recursive_chunking_config.yaml
│   └── retrieval_config.yaml
├── data/                          # Data store (raw, processed, evaluation)
│   ├── chroma_db/                 # ChromaDB vector index stores
│   ├── chunks/                    # JSONL chunk exports (dev & formal)
│   ├── evaluation/                # Retrieval metric outputs & detailed logs
│   ├── ground_truth/              # Retrieval ground-truth pairings
│   ├── questions/                 # 50 validated query candidates and schema
│   └── raw/                       # OpenStax Biology 2e source PDF
├── docs/                          # Detailed research and methodology documentation
│   ├── phase13_dataset_methodology.md
│   ├── phase14_ground_truth_methodology.md
│   ├── phase17_statistical_methodology.md
│   └── phase19_interpretation.md
├── results/                       # Empirical outputs, statistical tables, and publication plots
│   ├── analysis/                  # Statistical CSVs & JSON summaries
│   ├── figures/                   # High-res PNG & PDF plots (boxplots, heatmaps, diagrams)
│   └── tables/                    # Markdown and CSV result tables
├── scripts/                       # CLI execution utilities
│   ├── verify_env.py              # Environment configuration check
│   ├── select_questions.py        # Question selection & curation
│   ├── validate_questions.py      # Question schema validation
│   ├── generate_ground_truth.py   # Provenance-grounded reference extraction
│   ├── index_corpus.py            # Complete offline indexing runner
│   ├── evaluate_retrieval.py      # Retrieval evaluation computation
│   ├── generate_phase16_results.py# Experiment runner across all 5 strategies
│   ├── run_phase17_analysis.py    # Non-parametric statistical analysis runner
│   └── generate_phase18_outputs.py# Table and plot generation runner
├── src/                           # Core library modules
│   ├── analysis/                  # Statistical calculators & bootstrap helpers
│   ├── app/                       # Streamlit multi-page web application
│   ├── chunking/                  # Baseline fixed-size chunking
│   ├── eluc/                      # ELUC section-aware chunking
│   ├── embedding/                 # Dense embedding generation (Sentence-Transformers)
│   ├── evaluation/                # IR evaluation metrics (Recall, MRR)
│   ├── experiment/                # Experiment execution harness
│   ├── extraction/                # PDF text extraction
│   ├── generation/                # Google Gemini LLM synthesis
│   ├── heading_chunking/          # Heading-based chunking
│   ├── indexing/                  # ChromaDB vector indexing
│   ├── paragraph_chunking/        # Paragraph-based chunking
│   ├── pipeline/                  # End-to-end RAG pipeline
│   ├── preprocessing/             # Document preprocessing & provenance tracking
│   ├── recursive_chunking/        # Recursive character chunking
│   ├── retrieval/                 # ChromaDB vector retriever
│   └── vectorstore/               # Low-level vectorstore wrappers
└── tests/                         # Full Pytest test suite (172 unit & integration tests)
```

---

## 🚀 Setup & Installation

### 1. Prerequisites
- Python 3.11 or higher
- Git
- (Optional) NVIDIA GPU with CUDA for faster embedding inference

### 2. Clone the Repository & Set Up Virtual Environment

```bash
git clone https://github.com/dishikajain/educational-rag-chunking.git
cd educational-rag-chunking

# Create virtual environment
python -m venv .venv

# Activate on Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# Activate on Linux / macOS:
source .venv/bin/activate
```

### 3. Install Dependencies

Install the locked or phase-specific requirements:
```bash
pip install -r requirements.txt
pip install -r requirements-phase7.txt   # sentence-transformers, torch
pip install -r requirements-phase8.txt   # chromadb
pip install -r requirements-phase10.txt  # google-genai
pip install -r requirements-phase12.txt  # streamlit
```

### 4. Configure Environment Variables

Copy `.env.example` to `.env` and fill in your credentials:
```bash
cp .env.example .env
```

Edit `.env`:
```ini
GOOGLE_API_KEY=your_gemini_api_key_here
CHROMA_PERSIST_PATH=./chromadb
CHROMA_DEV_PATH=./chromadb/dev
CHROMA_EXPERIMENT_PATH=./chromadb/experiment
DATA_RAW_PATH=./data/raw
DATA_EXTRACTED_PATH=./data/extracted
DATA_PREPROCESSED_PATH=./data/preprocessed
DATA_CHUNKS_DEV_PATH=./data/chunks/dev
DATA_CHUNKS_FORMAL_PATH=./data/chunks/formal
RESULTS_PATH=./results
LOG_LEVEL=INFO
```

### 5. Verify Installation

Run the verification script:
```bash
python scripts/verify_env.py
```

---

## 🏃 Execution Guide

### 1. Indexing the Corpus Across All 5 Strategies
To extract text from the textbook PDF, preprocess boundaries, generate embeddings, and build ChromaDB collections:
```bash
python scripts/index_corpus.py
```

### 2. Validating Questions & Ground Truth
To validate the 50 curated questions against schema and generate provenance ground truth:
```bash
python scripts/validate_questions.py
python scripts/generate_ground_truth.py
```

### 3. Running Controlled Retrieval Evaluation
Evaluate top-$k$ retrieval performance across the 5 strategies:
```bash
python scripts/evaluate_retrieval.py
```

### 4. Running Statistical Analysis (Phase 17)
Compute Friedman omnibus tests, post-hoc Wilcoxon signed-rank comparisons, Holm-adjusted $p$-values, effect sizes, and bootstrap confidence intervals:
```bash
python scripts/run_phase17_analysis.py
```

### 5. Generating Publication Tables and Figures (Phase 18)
Generate figures, correlation heatmaps, boxplots, and markdown tables:
```bash
python scripts/generate_phase18_outputs.py
```
Output files will be saved in `results/tables/` and `results/figures/`.

### 6. Running the Interactive Streamlit Web App
Launch the interactive web application to explore the study findings, inspect chunks, compare strategies side-by-side, and ask live questions:
```bash
streamlit run src/app/streamlit_app.py
```
Features included in the Streamlit app:
- **Overview & Research Pipeline:** Conceptual architecture and motivation.
- **Dataset & Methodology:** Inspection of questions, provenance spans, and chunking parameters.
- **Interactive Query Engine:** Side-by-side comparison of retrieved chunks and generated answers.
- **Empirical Results & Visualizations:** Interactive metric tables and high-resolution plots.

---

## 🧪 Testing

The repository includes a comprehensive test suite of **172 unit and integration tests** verifying data integrity, extraction accuracy, chunking correctness, vector store indexing, retrieval mechanics, generation, and statistical calculations.

Run the entire test suite:
```bash
# On Windows PowerShell:
$env:PYTHONPATH="."
pytest

# On Linux / macOS:
PYTHONPATH=. pytest
```

---

## 📋 Phase Roadmap & Implementation Status

| Phase | Description | Status |
| :---: | :--- | :---: |
| **1** | Environment Setup, Dependency Management & Verification | ✅ Completed |
| **2** | PDF Text Extraction with PyMuPDF & Metadata Preservation | ✅ Completed |
| **3** | Text Cleaning, Preprocessing & Heading/Paragraph Detection | ✅ Completed |
| **4** | Baseline Fixed-Size Chunking Implementation | ✅ Completed |
| **5** | Educational Learning Unit Chunking (ELUC) Design | ✅ Completed |
| **6** | ELUC Algorithm Implementation & Boundary Slicing | ✅ Completed |
| **7** | Sentence-Transformers Embedding Pipeline (`bge-small-en-v1.5`) | ✅ Completed |
| **8** | ChromaDB Vector Store Indexing & Metadata Tagging | ✅ Completed |
| **9** | Vector Similarity Retrieval & Top-$k$ Engine | ✅ Completed |
| **10** | Google Gemini LLM Integration & Generation Prompts | ✅ Completed |
| **11** | End-to-End Modular RAG Pipeline Integration | ✅ Completed |
| **12** | Interactive Multi-Page Streamlit Web Application | ✅ Completed |
| **13** | 50-Question Educational Query Dataset Curation & Validation | ✅ Completed |
| **14** | Deterministic Provenance-Grounded Retrieval Ground Truth | ✅ Completed |
| **15** | Evaluation Module (Recall@1/3/5, MRR@5) | ✅ Completed |
| **16** | Controlled Comparative Experiments (250 Observations) | ✅ Completed |
| **17** | Non-Parametric Statistical Analysis (Friedman, Wilcoxon, Holm) | ✅ Completed |
| **18** | Publication-Ready Tables & High-Resolution Figures | ✅ Completed |
| **19** | Research Findings Interpretation & Limitations Analysis | ✅ Completed |
| **20** | Academic Dissertation / Paper Synthesis | 📝 In Progress |

---

## 🛡️ Threats to Validity & Limitations

- **Corpus Breadth:** Evaluation is focused on Chapters 1–5 of OpenStax *Biology 2e*. While highly representative of undergraduate STEM textbooks, findings may vary on unstructured or narrative domains.
- **Strict Provenance Reference:** Ground truth is strictly grounded in the exact source passages from which questions were formulated. Alternative passages providing valid answers elsewhere in the textbook are not marked as hits in strict source evaluation.
- **Fixed Model Architecture:** Top-$k=5$, cosine similarity, and `bge-small-en-v1.5` were held fixed; future work includes evaluating neural rerankers and hybrid dense-sparse retrieval.

---

## 📜 License & Citation

This project is open-source under the [MIT License](LICENSE).

```bibtex
@misc{jain2026educationalragchunking,
  author = {Jain, Dishika},
  title = {A Comparative Study of Document Chunking Strategies in Retrieval-Augmented Generation for Educational Question Answering},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/dishikajain/educational-rag-chunking}}
}
```
