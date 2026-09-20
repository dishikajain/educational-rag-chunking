from typing import List
import logging

from src.preprocessing.models import PreprocessedDocument, PreprocessedBlock
from src.chunking.models import BaselineChunk
from src.chunking.chunker import _find_overlapping_provenance
from src.eluc.chunker import HEADING_LABELS, _locate_char_start, _locate_char_end
from src.paragraph_chunking.config import ParagraphChunkingConfig
from src.paragraph_chunking.models import ParagraphChunkDocument
from src.recursive_chunking.chunker import _split_text_with_separators

logger = logging.getLogger(__name__)

SEPARATOR = "\n\n"

def paragraph_chunk(doc: PreprocessedDocument, config: ParagraphChunkingConfig) -> ParagraphChunkDocument:
    all_blocks: List[PreprocessedBlock] = []
    for page in doc.pages:
        for block in page.blocks:
            all_blocks.append(block)
            
    if not all_blocks:
        return ParagraphChunkDocument(
            doc_id=doc.doc_id,
            strategy=config.strategy,
            config_used=config,
            chunks=[]
        )
        
    chunk_candidates: List[List[PreprocessedBlock]] = []
    current_blocks: List[PreprocessedBlock] = []
    current_char_count = 0
    
    for i in range(len(all_blocks)):
        b = all_blocks[i]
        is_heading = b.label in HEADING_LABELS
        is_paragraph_break = b.is_paragraph_break is True
        
        b_len = len(b.text)
        sep_len = len(SEPARATOR) if current_blocks else 0
        
        should_flush = False
        
        if current_blocks:
            if is_heading:
                should_flush = True
            elif is_paragraph_break and current_char_count >= config.paragraph_soft_min:
                should_flush = True
            elif current_char_count + sep_len + b_len > config.max_chunk_chars:
                should_flush = True
                
        if should_flush:
            chunk_candidates.append(current_blocks)
            current_blocks = []
            current_char_count = 0
            sep_len = 0
            
        current_blocks.append(b)
        current_char_count += b_len + sep_len
        
    if current_blocks:
        chunk_candidates.append(current_blocks)
        
    # Process candidates
    chunks: List[BaselineChunk] = []
    chunk_index = 0
    
    for group in chunk_candidates:
        # Filter min block chars
        meaningful = [b for b in group if len(b.text) >= config.min_block_chars]
        if not meaningful:
            continue
            
        char_start = _locate_char_start(doc.provenance, meaningful[0])
        char_end = _locate_char_end(doc.provenance, meaningful[-1])
        
        if char_start != -1 and char_end != -1:
            group_text = doc.full_text[char_start:char_end]
        else:
            group_text = SEPARATOR.join(b.text for b in meaningful)
            
        # Oversized paragraph fallback
        if len(group_text) > config.max_chunk_chars:
            logger.warning(f"Oversized paragraph ({len(group_text)} chars). Using recursive split.")
            # From frozen config of recursive_chunking
            sub_texts = _split_text_with_separators(group_text, ["\n\n", "\n", ". ", " ", ""], 350)
            
            # Reassemble into overlapping chunks
            # chunk_size=350, overlap=40
            char_offset = char_start if char_start != -1 else 0
            
            # Use the same sliding window logic as recursive_chunk
            fragment_starts = []
            curr_start = char_offset
            for frag in sub_texts:
                fragment_starts.append(curr_start)
                curr_start += len(frag)
                
            i = 0
            while i < len(sub_texts):
                current_chunk_text = ""
                current_char_start = fragment_starts[i]
                j = i
                
                while j < len(sub_texts) and len(current_chunk_text) + len(sub_texts[j]) <= 350:
                    current_chunk_text += sub_texts[j]
                    j += 1
                    
                if j == i:
                    current_chunk_text = sub_texts[j]
                    j += 1
                    
                current_char_end = current_char_start + len(current_chunk_text)
                
                if len(current_chunk_text) >= 70:
                    chunk_id = f"{doc.doc_id}_paragraph_{chunk_index:06d}"
                    prov = _find_overlapping_provenance(doc.provenance, current_char_start, current_char_end)
                    chunks.append(BaselineChunk(
                        chunk_id=chunk_id, doc_id=doc.doc_id, chunk_index=chunk_index,
                        text=current_chunk_text, char_start=current_char_start, char_end=current_char_end,
                        provenance_spans=prov
                    ))
                    chunk_index += 1
                    
                if j == len(sub_texts):
                    break
                    
                next_i = j - 1
                while next_i > i:
                    overlap_text = "".join(sub_texts[next_i:j])
                    if len(overlap_text) > 40:
                        next_i += 1
                    else:
                        break
                if next_i <= i:
                    next_i = i + 1
                i = next_i
        else:
            if len(group_text) >= config.min_chunk_chars:
                chunk_id = f"{doc.doc_id}_paragraph_{chunk_index:06d}"
                prov = _find_overlapping_provenance(doc.provenance, char_start, char_end) if char_start != -1 else []
                chunks.append(BaselineChunk(
                    chunk_id=chunk_id, doc_id=doc.doc_id, chunk_index=chunk_index,
                    text=group_text, char_start=char_start, char_end=char_end,
                    provenance_spans=prov
                ))
                chunk_index += 1

    return ParagraphChunkDocument(
        doc_id=doc.doc_id,
        strategy=config.strategy,
        config_used=config,
        chunks=chunks
    )
