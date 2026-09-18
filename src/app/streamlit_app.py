import streamlit as st
import os

from src.retrieval.config import load_config as load_retrieval_config
from src.retrieval.retriever import Retriever
from src.generation.config import load_config as load_generation_config
from src.generation.generator import GeminiGenerator
from src.pipeline.rag_pipeline import RAGPipeline
from src.pipeline.exceptions import PipelineError

@st.cache_resource
def get_pipeline():
    retrieval_config = load_retrieval_config()
    retriever = Retriever(config=retrieval_config)
    
    generation_config = load_generation_config()
    generator = GeminiGenerator(config=generation_config)
    
    return RAGPipeline(retriever=retriever, generator=generator)

def main():
    st.title("Educational RAG Question Answering")
    
    if not os.environ.get("GEMINI_API_KEY"):
        st.error("GEMINI_API_KEY environment variable is missing. Please set it before running the app.")
        return

    try:
        pipeline = get_pipeline()
    except (ValueError, RuntimeError) as e:
        st.error(f"Failed to initialize pipeline: {e}")
        return

    question = st.text_area("Question", placeholder="Enter your question here...")
    strategy = st.selectbox(
        "Strategy",
        options=["baseline_fixed_size", "eluc_section_aware"]
    )
    
    if st.button("Run"):
        if not question or not question.strip():
            st.warning("Please enter a non-empty question.")
            return
            
        with st.spinner("Generating answer..."):
            try:
                result = pipeline.run(question=question, strategy=strategy)
                
                st.subheader("Answer")
                st.markdown(result.generation.answer)
                
                st.subheader("Retrieved Chunks")
                for i, chunk in enumerate(result.generation.retrieved_chunks):
                    with st.expander(f"Chunk {i+1} ({chunk.chunk_id}) - Distance: {chunk.distance:.4f}"):
                        st.markdown(f"**Strategy:** {chunk.strategy}")
                        st.markdown(f"**Document ID:** {chunk.doc_id}")
                        st.markdown(f"**Character Range:** {chunk.char_start} to {chunk.char_end}")
                        st.markdown(f"**Provenance Spans:** `{chunk.provenance_spans}`")
                        st.markdown("**Text:**")
                        st.text(chunk.text)
                        
            except PipelineError as e:
                st.error(f"Pipeline error: {str(e)}")

if __name__ == "__main__":
    main()
