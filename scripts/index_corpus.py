import os
import sys
import time
import tempfile
import pymupdf as fitz
import logging



from src.extraction.extractor import extract_pdf
from src.preprocessing.preprocessor import preprocess
from src.preprocessing.config import load_config as load_prep_config
from src.chunking.chunker import chunk as baseline_chunk
from src.chunking.config import load_config as load_baseline_config
from src.eluc.chunker import eluc_chunk
from src.eluc.config import load_config as load_eluc_config
from src.embedding.embedder import embed_document
from src.embedding.config import load_config as load_embed_config
from src.indexing.indexer import index_document, sanitize_collection_name
from src.indexing.config import IndexingConfig


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PDF_PATH = "data/raw/openstax_biology2e.pdf"
CHROMA_DB_DIR = "data/chroma_db"

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
        embed_config = load_embed_config("config/embedding_config.yaml")
        index_config = IndexingConfig.from_yaml("config/indexing_config.yaml")
    except Exception as e:
        logger.error(f"Failed to load configurations: {e}")
        sys.exit(1)
        
    # PDF Slicing (Chapters 1-5 corresponding to PDF pages 31-171, i.e., 0-indexed 30 to 170)
    logger.info("Slicing OpenStax Biology 2e to Chapters 1-5 (PDF pages 31-171)...")
    temp_pdf_path = None
    try:
        with fitz.open(pdf_path) as src_doc:
            dest_doc = fitz.open()
            dest_doc.insert_pdf(src_doc, from_page=30, to_page=170)
            
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
    
    # 3. Baseline Chunking
    logger.info("Generating baseline chunks...")
    baseline_doc = baseline_chunk(preprocessed_doc, baseline_config)
    logger.info(f"Baseline chunking complete. Generated {len(baseline_doc.chunks)} chunks.")
    
    # 4. ELUC Chunking
    logger.info("Generating ELUC chunks...")
    eluc_doc = eluc_chunk(preprocessed_doc, eluc_config)
    logger.info(f"ELUC chunking complete. Generated {len(eluc_doc.chunks)} chunks.")
    
    # 5. Embedding
    logger.info("Embedding baseline chunks...")
    start_time = time.time()
    baseline_embedded = embed_document(baseline_doc, embed_config)
    logger.info(f"Baseline embedding complete in {time.time() - start_time:.2f}s.")
    
    logger.info("Embedding ELUC chunks...")
    start_time = time.time()
    eluc_embedded = embed_document(eluc_doc, embed_config)
    logger.info(f"ELUC embedding complete in {time.time() - start_time:.2f}s.")
    
    # 6. Indexing
    logger.info("Indexing baseline collection into ChromaDB...")
    start_time = time.time()
    baseline_collection_name = sanitize_collection_name("baseline_fixed_size", embed_config.model_name)
    baseline_report = index_document(baseline_embedded, index_config, baseline_collection_name, chroma_db_dir)
    logger.info(f"Baseline indexing complete in {time.time() - start_time:.2f}s. Collection: {baseline_collection_name}, Indexed items: {baseline_report.chunks_inserted}")
    
    logger.info("Indexing ELUC collection into ChromaDB...")
    start_time = time.time()
    eluc_collection_name = sanitize_collection_name("eluc_section_aware", embed_config.model_name)
    eluc_report = index_document(eluc_embedded, index_config, eluc_collection_name, chroma_db_dir)
    logger.info(f"ELUC indexing complete in {time.time() - start_time:.2f}s. Collection: {eluc_collection_name}, Indexed items: {eluc_report.chunks_inserted}")
    
    logger.info("Offline corpus indexing complete!")

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    run_indexing()
