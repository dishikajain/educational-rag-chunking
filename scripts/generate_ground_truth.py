import json
import re
import os
import argparse
from typing import List, Tuple
from collections import defaultdict
import tempfile



def load_chunks(chunks_dir: str, strategies: List[str]):
    chunks_data = {}
    for strat in strategies:
        file_path = os.path.join(chunks_dir, f"{strat}.jsonl")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            chunks_data[strat] = [json.loads(l) for l in f]
    return chunks_data

def get_preprocessed_document():
    # We must extract and preprocess the document identically to Phase 14A
    # so we can search the exact full_text.
    import pymupdf as fitz
    from src.extraction.extractor import extract_pdf
    from src.preprocessing.preprocessor import preprocess
    from src.preprocessing.config import load_config
    
    pdf_path = "data/raw/openstax_biology2e.pdf"
    
    with fitz.open(pdf_path) as src_doc:
        dest_doc = fitz.open()
        dest_doc.insert_pdf(src_doc, from_page=28, to_page=174)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            temp_pdf_path = tmp.name
        dest_doc.save(temp_pdf_path)
        dest_doc.close()
        
    raw_doc = extract_pdf(temp_pdf_path, 'doc')
    os.remove(temp_pdf_path)
    
    prep_config = load_config("config/preprocessing_config.yaml")
    preprocessed = preprocess(raw_doc, prep_config)
    return preprocessed

def generate_ground_truth():
    questions_file = "data/questions/questions.jsonl"
    chunks_dir = "data/chunks/formal"
    out_file = "data/ground_truth/retrieval_ground_truth.jsonl"
    
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    
    with open(questions_file, 'r', encoding='utf-8') as f:
        questions = [json.loads(l) for l in f]
        
    strategies = ["baseline_fixed_size", "recursive_character", "paragraph_based", "heading_based", "eluc_section_aware"]
    # Wait, the filenames in chunks/formal are: fixed_char, recursive_char, paragraph, heading, eluc
    strategy_files = {
        "baseline_fixed_size": "fixed_char",
        "recursive_character": "recursive_char",
        "paragraph_based": "paragraph",
        "heading_based": "heading",
        "eluc_section_aware": "eluc"
    }
    
    chunks_data = {}
    for strat, directory in strategy_files.items():
        with open(os.path.join(chunks_dir, directory, f"{strat}.jsonl"), 'r', encoding='utf-8') as f:
            chunks_data[strat] = [json.loads(l) for l in f]
            
    doc = get_preprocessed_document()
    
    records = []
    failures = []
    
    for q in questions:
        q_id = q["question_id"]
        
        # Identify expected pages based on source_page_range
        pages_in_range = [int(x) for x in re.findall(r'\d+', q["source_page_range"])]
        if len(pages_in_range) == 2:
            expected_pages = set(range(pages_in_range[0], pages_in_range[1] + 1))
        elif len(pages_in_range) == 1:
            expected_pages = {pages_in_range[0]}
        else:
            expected_pages = set()
        
        target_source_blocks = set()
        
        # Map physical expected_pages to doc blocks
        for expected_page in expected_pages:
            page_idx = expected_page - 28 # from_page=28
            local_page_number = expected_page - 28 + 1
            if 0 <= page_idx < len(doc.pages):
                page = doc.pages[page_idx]
                for blk in page.blocks:
                    target_source_blocks.add((local_page_number, blk.p2_block_index))
                    
        if not target_source_blocks:
            failures.append(f"Zero blocks found on specified page range for {q_id}")
            continue
            
        for strat in strategies:
            strat_chunks = chunks_data[strat]
            relevant_ids = []
            
            for chunk in strat_chunks:
                chunk_blocks = set()
                for span in chunk["provenance_spans"]:
                    for blk in span["source_blocks"]:
                        chunk_blocks.add(tuple(blk))
                        
                if target_source_blocks.intersection(chunk_blocks):
                    relevant_ids.append(chunk["chunk_id"])
                    
            records.append({
                "question_id": q_id,
                "strategy": strat,
                "relevant_chunk_ids": relevant_ids
            })
            
    with open(out_file, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
            
    if failures:
        print("FAILURES:")
        for fail in failures:
            print(fail)
        raise ValueError(f"{len(failures)} questions failed mapping.")
            
    print(f"Generated {len(records)} ground truth records successfully.")
    
if __name__ == "__main__":
    generate_ground_truth()
