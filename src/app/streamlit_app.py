import streamlit as st
import os
import json
import pandas as pd
from pathlib import Path

from src.retrieval.config import load_config as load_retrieval_config
from src.retrieval.retriever import Retriever
from src.generation.config import load_config as load_generation_config
from src.generation.generator import GeminiGenerator
from src.pipeline.rag_pipeline import RAGPipeline
from src.pipeline.exceptions import PipelineError

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

st.set_page_config(
    page_title="RAG Chunking Study",
    page_icon="📚",
    layout="wide"
)

# ---------------------------------------------------------
# CACHED RESOURCES
# ---------------------------------------------------------
@st.cache_resource
def get_pipeline():
    retrieval_config = load_retrieval_config()
    retriever = Retriever(config=retrieval_config)
    generation_config = load_generation_config()
    generator = GeminiGenerator(config=generation_config)
    return RAGPipeline(retriever=retriever, generator=generator)

@st.cache_data
def load_json(filepath):
    path = ROOT_DIR / filepath
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_csv(filepath):
    path = ROOT_DIR / filepath
    if not path.exists():
        return None
    return pd.read_csv(path)

@st.cache_data
def load_markdown(filepath):
    path = ROOT_DIR / filepath
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# ---------------------------------------------------------
# PAGES
# ---------------------------------------------------------
def render_overview():
    st.title("A Comparative Study of Document Chunking Strategies in Retrieval-Augmented Generation for Educational Question Answering")
    st.subheader("Controlled evaluation of document chunking strategies for retrieval in educational RAG.")

    st.markdown("---")
    st.markdown("### Research Pipeline")
    st.info("Educational Textbook ➔ Text Extraction ➔ Preprocessing ➔ **5 Chunking Strategies** ➔ BGE Embeddings ➔ ChromaDB ➔ Top-k Retrieval ➔ Gemini Generation ➔ Retrieval Evaluation ➔ Statistical Analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Research Question")
        st.success("How does document chunking strategy affect retrieval performance in an educational RAG question-answering system?")
    with col2:
        st.markdown("### What makes this comparison controlled?")
        st.markdown("""
        - Same corpus
        - Same questions
        - Same preprocessing
        - Same embedding model
        - Same retrieval configuration
        - Same evaluation procedure
        - **Only chunking strategy varies**
        """)

    st.markdown("---")
    st.markdown("### Study at a Glance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Corpus", "Chapters 1–5")
    c2.metric("Questions", "50")
    c3.metric("Strategies", "5")
    c4.metric("Observations", "250")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Embedding", "BAAI/bge-base-en-v1.5")
    c6.metric("Retrieval", "Cosine similarity")
    c7.metric("Top-k", "5")
    c8.metric("Evaluation", "Recall@1, 3, 5, MRR@5")

def render_live_demo():
    st.title("LIVE RAG DEMONSTRATION")
    st.caption("This interactive demonstration is separate from the frozen 50-question experiment.")

    if not os.environ.get("GEMINI_API_KEY"):
        st.error("GEMINI_API_KEY environment variable is missing. Please set it before running the app.")
        return

    try:
        pipeline = get_pipeline()
    except (ValueError, RuntimeError) as e:
        st.error(f"Failed to initialize pipeline: {e}")
        return

    question = st.text_area("Question", placeholder="Enter your question here...")
    strategy_options = {
        "baseline_fixed_size": "Fixed-size",
        "eluc_section_aware": "ELUC / Section-aware",
        "recursive_character": "Recursive Character",
        "paragraph_based": "Paragraph-based",
        "heading_based": "Heading-based"
    }

    strategy_key = st.selectbox(
        "Strategy",
        options=list(strategy_options.keys()),
        format_func=lambda x: strategy_options[x]
    )

    if st.button("Run Retrieval / Generate Answer"):
        if not question or not question.strip():
            st.warning("Please enter a non-empty question.")
            return

        with st.spinner("Executing pipeline..."):
            try:
                result = pipeline.run(question=question, strategy=strategy_key)

                st.subheader("Generated Answer")
                st.info(result.generation.answer)

                st.subheader("Retrieval Summary")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Strategy", strategy_options[strategy_key])
                col2.metric("Retrieved Chunks", len(result.generation.retrieved_chunks))
                col3.metric("Top-k", 5)
                col4.metric("Similarity", "Cosine")

                st.subheader("Top-5 Retrieved Chunks")
                for i, chunk in enumerate(result.generation.retrieved_chunks):
                    with st.expander(f"Rank {i+1} - ID: {chunk.chunk_id} - Distance: {chunk.distance:.4f} - Page: {chunk.doc_id}"):
                        st.markdown(f"**Provenance:** `{chunk.provenance_spans}`")
                        st.markdown("**Text:**")
                        st.text(chunk.text)

            except PipelineError as e:
                st.error(f"Pipeline error: {str(e)}")

def render_chunking_strategies():
    st.title("Chunking Strategies")
    st.markdown("Chunking determines what textual unit is embedded and retrieved. Different boundary decisions can therefore change which source passages are returned for the same question.")

    with st.expander("FIXED-SIZE", expanded=True):
        st.write("**Boundary principle:** Splits the preprocessed document using a fixed character window with overlap, without explicitly using document structure.")
    with st.expander("ELUC / SECTION-AWARE", expanded=True):
        st.write("**Boundary principle:** Uses document structure such as headings, section boundaries and sidebar units while accumulating content up to a configured maximum chunk size.")
    with st.expander("RECURSIVE CHARACTER", expanded=True):
        st.write("**Boundary principle:** Recursively splits text using a hierarchy of separators while respecting configured chunk size and overlap.")
    with st.expander("PARAGRAPH-BASED", expanded=True):
        st.write("**Boundary principle:** Uses paragraph boundaries as the primary chunking units, with a fallback splitter for oversized paragraphs.")
    with st.expander("HEADING-BASED", expanded=True):
        st.write("**Boundary principle:** Uses detected document headings and section boundaries to create structure-aware chunks.")

def render_experimental_results():
    st.title("FROZEN EXPERIMENTAL RESULTS")
    st.caption("50 questions × 5 chunking strategies = 250 query-strategy observations")

    df = load_csv("results/tables/table1_overall_performance.csv")
    if df is not None:
        st.dataframe(df, hide_index=True, use_container_width=True)
    else:
        st.warning("Could not load table1_overall_performance.csv")

    st.markdown("### Metric Distributions")
    col1, col2 = st.columns(2)
    with col1:
        st.image(str(ROOT_DIR / "results/figures/recall_at_1_distribution.png"), caption="Recall@1 Distribution")
        st.image(str(ROOT_DIR / "results/figures/recall_at_5_distribution.png"), caption="Recall@5 Distribution")
    with col2:
        st.image(str(ROOT_DIR / "results/figures/recall_at_3_distribution.png"), caption="Recall@3 Distribution")
        st.image(str(ROOT_DIR / "results/figures/mrr_at_5_distribution.png"), caption="MRR@5 Distribution")

    st.image(str(ROOT_DIR / "results/figures/overall_metric_comparison.png"), caption="Overall Metric Comparison")
    st.image(str(ROOT_DIR / "results/figures/mrr_at_5_query_heatmap.png"), caption="MRR@5 Query Heatmap")

    st.info("These results describe the observed retrieval performance on the evaluated Chapters 1–5 corpus and 50-question dataset.")

def render_statistical_analysis():
    st.title("Statistical Analysis")

    st.subheader("Friedman Omnibus Tests")
    df_omnibus = load_csv("results/tables/table2_omnibus_statistics.csv")
    if df_omnibus is not None:
        st.dataframe(df_omnibus, hide_index=True, use_container_width=True)

    st.info("The omnibus tests provide evidence against the null hypothesis of no difference across strategies for all four metrics.")

    st.subheader("Pairwise Comparisons")
    df_pairwise = load_csv("results/tables/table3_pairwise_comparisons.csv")
    if df_pairwise is not None:
        metric = st.selectbox("Select Metric", options=df_pairwise['Metric'].unique())
        filtered_df = df_pairwise[df_pairwise['Metric'] == metric]
        st.dataframe(filtered_df, hide_index=True, use_container_width=True)

    st.caption("Paired Wilcoxon signed-rank tests were used for post-hoc comparisons with Holm correction for multiple comparisons.")

    st.markdown("---")
    st.markdown("### Statistical Definitions")
    st.write("**Friedman test:** Non-parametric alternative to repeated-measures ANOVA.")
    st.write("**Kendall's W:** Effect size for the Friedman test.")
    st.write("**Wilcoxon signed-rank test:** Non-parametric test for paired samples.")
    st.write("**Holm correction:** Controls family-wise error rate in multiple comparisons.")

def render_interpretation():
    st.title("Interpretation")
    markdown_content = load_markdown("docs/phase19_interpretation.md")
    if markdown_content:
        st.markdown(markdown_content)
    else:
        st.warning("Could not load interpretation document.")

def render_methodology():
    st.title("Methodology / Reproducibility")

    st.subheader("Experimental Pipeline")
    st.markdown("""
    1. Corpus
    2. Extraction
    3. Preprocessing
    4. Chunking
    5. Embedding
    6. Indexing
    7. Retrieval
    8. Ground-truth reference
    9. Evaluation
    10. Statistical analysis
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Variables")
        st.write("**Independent variable:** Chunking Strategy")
        st.write("**Dependent variables:** Recall@1, Recall@3, Recall@5, MRR@5")
    with col2:
        st.markdown("### Scope")
        st.write("**Experimental unit:** one question × one strategy")
        st.write("**Total observations:** 250")

    st.markdown("### Ground-Truth Reference")
    st.info("**Deterministic provenance-derived, source-grounded retrieval reference:** Relevant chunks are determined from the source provenance associated with each question rather than from retrieval rankings, embeddings, or subjective manual relevance labels.")

    st.markdown("### Metrics Explained")
    st.write("**Recall@k:** Fraction of provenance-defined relevant chunks retrieved within the top k results.")
    st.write("**MRR@5:** Reciprocal rank of the first relevant retrieved chunk, averaged across questions.")

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    st.sidebar.title("Navigation")

    pages = {
        "Overview": render_overview,
        "Live RAG Demo": render_live_demo,
        "Chunking Strategies": render_chunking_strategies,
        "Experimental Results": render_experimental_results,
        "Statistical Analysis": render_statistical_analysis,
        "Interpretation": render_interpretation,
        "Methodology / Reproducibility": render_methodology
    }

    selection = st.sidebar.radio("Go to", list(pages.keys()))

    pages[selection]()

if __name__ == "__main__":
    main()
