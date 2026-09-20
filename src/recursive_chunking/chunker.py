from typing import List, Tuple
from src.preprocessing.models import PreprocessedDocument
from src.chunking.models import BaselineChunk
from src.chunking.chunker import _find_overlapping_provenance
from src.recursive_chunking.config import RecursiveChunkingConfig
from src.recursive_chunking.models import RecursiveChunkDocument

def _split_text_with_separators(text: str, separators: List[str], chunk_size: int) -> List[str]:
    """Recursively split text using the given separators."""
    if len(text) <= chunk_size:
        return [text]
    
    if not separators:
        return [text]
    
    separator = separators[0]
    next_separators = separators[1:]
    
    if separator == "":
        # Character-level split
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    splits = text.split(separator)
    final_splits = []
    
    for i, s in enumerate(splits):
        if i < len(splits) - 1:
            s += separator
        
        if len(s) <= chunk_size:
            final_splits.append(s)
        else:
            if next_separators:
                final_splits.extend(_split_text_with_separators(s, next_separators, chunk_size))
            else:
                final_splits.append(s)
                
    # Merge consecutive small splits if possible to avoid too many tiny fragments
    merged_splits = []
    current_split = ""
    for s in final_splits:
        if len(current_split) + len(s) <= chunk_size:
            current_split += s
        else:
            if current_split:
                merged_splits.append(current_split)
            current_split = s
    if current_split:
        merged_splits.append(current_split)
        
    return merged_splits

def recursive_chunk(doc: PreprocessedDocument, config: RecursiveChunkingConfig) -> RecursiveChunkDocument:
    full_text = doc.full_text
    chunk_size = config.chunk_size
    chunk_overlap = config.chunk_overlap
    min_chunk_size = config.min_chunk_size
    separators = config.separators
    doc_id = doc.doc_id
    
    chunks: List[BaselineChunk] = []
    
    if not full_text:
        return RecursiveChunkDocument(
            doc_id=doc_id,
            strategy=config.strategy,
            config_used=config,
            chunks=[]
        )
        
    fragments = _split_text_with_separators(full_text, separators, chunk_size)
    
    # We now have a list of fragments. We need to build overlapping chunks.
    # We'll use a sliding window over the fragments.
    
    char_start = 0
    chunk_index = 0
    
    # Track the exact start offset for each fragment
    fragment_starts = []
    current_start = 0
    for frag in fragments:
        fragment_starts.append(current_start)
        current_start += len(frag)
        
    i = 0
    while i < len(fragments):
        current_chunk_text = ""
        current_char_start = fragment_starts[i]
        j = i
        
        while j < len(fragments) and len(current_chunk_text) + len(fragments[j]) <= chunk_size:
            current_chunk_text += fragments[j]
            j += 1
            
        # If a single fragment is larger than chunk_size (should only happen if out of separators),
        # we still add it so we don't drop text.
        if j == i:
            current_chunk_text = fragments[j]
            j += 1
            
        current_char_end = current_char_start + len(current_chunk_text)
        
        if len(current_chunk_text) >= min_chunk_size:
            chunk_id = f"{doc_id}_recursive_{chunk_index:06d}"
            provenance_spans = _find_overlapping_provenance(
                doc.provenance, current_char_start, current_char_end
            )
            chunks.append(BaselineChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                chunk_index=chunk_index,
                text=current_chunk_text,
                char_start=current_char_start,
                char_end=current_char_end,
                provenance_spans=provenance_spans
            ))
            chunk_index += 1
            
        # Determine next 'i' to satisfy chunk_overlap
        if j == len(fragments):
            break
            
        # Move 'i' forward such that the overlap between the current chunk and the next chunk is <= chunk_overlap
        next_i = j - 1
        while next_i > i:
            overlap_text = "".join(fragments[next_i:j])
            if len(overlap_text) > chunk_overlap:
                next_i += 1
            else:
                break
        
        if next_i <= i:
            next_i = i + 1
            
        i = next_i

    return RecursiveChunkDocument(
        doc_id=doc_id,
        strategy=config.strategy,
        config_used=config,
        chunks=chunks
    )
