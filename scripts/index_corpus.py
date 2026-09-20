import os
import sys
import time
import tempfile
import pymupdf as fitz
import logging
import json
from dataclasses import asdict

from src.extraction.extractor import extract_pdf
from src.preprocessing.preprocessor import preprocess
from src.preprocessing.config import load_config as load_prep_config

from src.chunking.chunker import chunk as baseline_chunk
from src.chunking.config import load_config as load_baseline_config

from src.eluc.chunker import eluc_chunk
from src.eluc.config import load_config as load_eluc_config

from src.recursive_chunking.chunker import recursive_chunk
from src.recursive_chunking.config import load_config as load_recursive_config

from src.paragraph_chunking.chunker import paragraph_chunk
from src.paragraph_chunking.config import load_config as load_paragraph_config

from src.heading_chunking.chunker import heading_chunk
from src.heading_chunking.config import load_config as load_heading_config

from src.embedding.embedder import embed_document
from src.embedding.config import load_config as load_embed_config

from src.indexing.indexer import index_document, sanitize_collection_name
from src.indexing.config import IndexingConfig


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PDF_PATH = "data/raw/openstax_biology2e.pdf"
CHROMA_DB_DIR = "data/chroma_db"

def save_chunks(doc, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"{doc.strategy}.jsonl")
    with open(out_file, "w", encoding="utf-8") as f:
        for chunk in doc.chunks:
            # We must serialize safely
            d = asdict(chunk)
            f.write(json.dumps(d) + "\n")

def run_indexing():
    logger.info("Starting offline corpus indexing runner")

    pdf_path = PDF_PATH
    chroma_db_dir = CHROMA_DB_DIR

    if not os.path.exists(pdf_path):
        logger.error(f"Source PDF not found at {pdf_path}")
        sys.exit(1)

    os.makedirs(chroma_db_dir, exist_ok=True)

    # Load configs
    try:
        prep_config = load_prep_config("config/preprocessing_config.yaml")
        baseline_config = load_baseline_config("config/baseline_chunking_config.yaml")
        eluc_config = load_eluc_config("config/eluc_chunking_config.yaml")
        recursive_config = load_recursive_config("config/recursive_chunking_config.yaml")
        paragraph_config = load_paragraph_config("config/paragraph_chunking_config.yaml")
        heading_config = load_heading_config("config/heading_chunking_config.yaml")
        embed_config = load_embed_config("config/embedding_config.yaml")
        index_config = IndexingConfig.from_yaml("config/indexing_config.yaml")
    except Exception as e:
        logger.error(f"Failed to load configurations: {e}")
        sys.exit(1)

    # PDF Slicing (Chapters 1-5 corresponding to PDF pages 29-175, i.e., 0-indexed 28 to 174)
    logger.info("Slicing OpenStax Biology 2e to Chapters 1-5 (PDF pages 29-175)...")
    temp_pdf_path = None
    try:
        with fitz.open(pdf_path) as src_doc:
            dest_doc = fitz.open()
            dest_doc.insert_pdf(src_doc, from_page=28, to_page=174)

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                temp_pdf_path = tmp.name

            dest_doc.save(temp_pdf_path)
            dest_doc.close()

        # 1. Extraction
        logger.info(f"Extracting sliced PDF ({temp_pdf_path})...")
        start_time = time.time()
        doc_id = "openstax_biology2e_ch1_5"
        extracted_doc = extract_pdf(temp_pdf_path, doc_id)
        logger.info(f"Extraction complete in {time.time() - start_time:.2f}s. Extracted {len(extracted_doc.pages)} pages.")

    finally:
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

    # 2. Preprocessing
    logger.info("Preprocessing extracted document...")
    start_time = time.time()
    preprocessed_doc = preprocess(extracted_doc, prep_config)
    logger.info(f"Preprocessing complete in {time.time() - start_time:.2f}s. Full text length: {len(preprocessed_doc.full_text)}")

    docs_to_index = []

    # 3. Baseline Chunking
    logger.info("Generating baseline chunks...")
    baseline_doc = baseline_chunk(preprocessed_doc, baseline_config)
    logger.info(f"Baseline chunking complete. Generated {len(baseline_doc.chunks)} chunks.")
    save_chunks(baseline_doc, "data/chunks/formal/fixed_char")
    docs_to_index.append(baseline_doc)

    # 4. ELUC Chunking
    logger.info("Generating ELUC chunks...")
    eluc_doc = eluc_chunk(preprocessed_doc, eluc_config)
    logger.info(f"ELUC chunking complete. Generated {len(eluc_doc.chunks)} chunks.")
    save_chunks(eluc_doc, "data/chunks/formal/eluc")
    docs_to_index.append(eluc_doc)

    # 5. Recursive Chunking
    logger.info("Generating Recursive chunks...")
    recursive_doc = recursive_chunk(preprocessed_doc, recursive_config)
    logger.info(f"Recursive chunking complete. Generated {len(recursive_doc.chunks)} chunks.")
    save_chunks(recursive_doc, "data/chunks/formal/recursive_char")
    docs_to_index.append(recursive_doc)

    # 6. Paragraph Chunking
    logger.info("Generating Paragraph chunks...")
    paragraph_doc = paragraph_chunk(preprocessed_doc, paragraph_config)
    logger.info(f"Paragraph chunking complete. Generated {len(paragraph_doc.chunks)} chunks.")
    save_chunks(paragraph_doc, "data/chunks/formal/paragraph")
    docs_to_index.append(paragraph_doc)

    # 7. Heading Chunking
    logger.info("Generating Heading chunks...")
    heading_doc = heading_chunk(preprocessed_doc, heading_config)
    logger.info(f"Heading chunking complete. Generated {len(heading_doc.chunks)} chunks.")
    save_chunks(heading_doc, "data/chunks/formal/heading")
    docs_to_index.append(heading_doc)

    # Embedding and Indexing loop
    for doc in docs_to_index:
        logger.info(f"Embedding {doc.strategy} chunks...")
        start_time = time.time()
        embedded_doc = embed_document(doc, embed_config)
        logger.info(f"{doc.strategy} embedding complete in {time.time() - start_time:.2f}s.")

        logger.info(f"Indexing {doc.strategy} collection into ChromaDB...")
        start_time = time.time()
        report = index_document(embedded_doc, index_config)
        logger.info(f"{doc.strategy} indexing complete in {time.time() - start_time:.2f}s. Collection: {report.collection_name}, Indexed items: {report.chunks_inserted}")

    logger.info("Offline corpus indexing complete!")

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    run_indexing()
