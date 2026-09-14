from typing import List, Optional, Set
import logging
from src.preprocessing.models import PreprocessedDocument, PreprocessedBlock, ProvenanceSpan
from src.eluc.config import ELUCChunkingConfig
from src.eluc.models import ELUCChunk, ELUCChunkDocument

# Shared utility: see src/chunking/chunker.py
from src.chunking.chunker import _find_overlapping_provenance

logger = logging.getLogger(__name__)

HEADING_LABELS = frozenset({
    "HEADING_1",
    "HEADING_2",
    "HEADING_3",
    "CHAPTER_TITLE",
    "CHAPTER_NUMBER",
    "CHAPTER_INTRO_SPAN",
})

SEPARATOR = "\n\n"

def _locate_char_start(provenance: List[ProvenanceSpan], block: PreprocessedBlock) -> int:
    key = (block.source_page, block.p2_block_index)
    for span in provenance:
        if key in span.source_blocks:
            return span.char_start
    logger.warning(f"Block {key} not found in provenance; char_start set to -1")
    return -1

def _locate_char_end(provenance: List[ProvenanceSpan], block: PreprocessedBlock) -> int:
    key = (block.source_page, block.p2_block_index)
    for span in provenance:
        if key in span.source_blocks:
            return span.char_end
    logger.warning(f"Block {key} not found in provenance; char_end set to -1")
    return -1

def eluc_chunk(doc: PreprocessedDocument, config: ELUCChunkingConfig) -> ELUCChunkDocument:
    # Pass 0 — Flatten
    all_blocks: List[PreprocessedBlock] = []
    for page in doc.pages:
        for block in page.blocks:
            all_blocks.append(block)

    if not all_blocks:
        return ELUCChunkDocument(
            doc_id=doc.doc_id,
            strategy="eluc_section_aware",
            config_used=config,
            chunks=[]
        )

    # Pass 1 — Mark Section Boundaries
    boundary_before = [False] * len(all_blocks)
    for i in range(len(all_blocks)):
        b = all_blocks[i]
        prev = all_blocks[i - 1] if i > 0 else None

        cond_a = (i == 0)
        cond_b = (b.label in HEADING_LABELS)
        cond_c = (b.is_section_break is True)
        cond_d = (
            b.sidebar_unit_id is not None and
            (prev is None or prev.sidebar_unit_id != b.sidebar_unit_id)
        )

        boundary_before[i] = cond_a or cond_b or cond_c or cond_d

    # Pass 2 — Accumulate into Candidate Chunks
    chunk_candidates: List[List[PreprocessedBlock]] = []
    current_blocks: List[PreprocessedBlock] = []
    current_char_count = 0

    for i in range(len(all_blocks)):
        b = all_blocks[i]
        
        # Step A
        if boundary_before[i] and current_blocks:
            chunk_candidates.append(current_blocks)
            current_blocks = []
            current_char_count = 0

        # Step B
        prev = all_blocks[i - 1] if i > 0 else None
        is_mid_sidebar = (
            b.sidebar_unit_id is not None and
            i > 0 and
            prev is not None and prev.sidebar_unit_id == b.sidebar_unit_id
        )
        would_overflow = (current_char_count + len(b.text) > config.max_chunk_chars)
        
        if would_overflow and current_blocks and not is_mid_sidebar:
            chunk_candidates.append(current_blocks)
            current_blocks = []
            current_char_count = 0

        # Step C
        current_blocks.append(b)
        current_char_count += len(b.text)

    if current_blocks:
        chunk_candidates.append(current_blocks)

    # Pass 3 — Overflow Warning (Splitting is handled by Pass 2)
    # Pass 2 already flushes candidates before they overflow, EXCEPT for sidebars
    # and single blocks that are inherently larger than max_chunk_chars.
    final_chunk_groups = chunk_candidates

    for group in final_chunk_groups:
        first_sid = group[0].sidebar_unit_id
        is_sidebar_candidate = (
            first_sid is not None and
            all(b.sidebar_unit_id == first_sid for b in group)
        )
        total_chars = sum(len(b.text) for b in group)
        
        if total_chars > config.max_chunk_chars:
            if is_sidebar_candidate:
                logger.warning(
                    f"Sidebar unit {first_sid} exceeds max_chunk_chars ({total_chars} chars); "
                    "emitting whole per sidebar atomicity rule."
                )
            else:
                source_page = group[0].source_page
                p2_idx = group[0].p2_block_index
                logger.warning(
                    f"Non-sidebar candidate at source ({source_page}, {p2_idx}) has "
                    f"{total_chars} chars with no splittable paragraph breaks; emitting whole. "
                    f"max_chunk_chars={config.max_chunk_chars}"
                )

    # Pass 4 — Assemble ELUCChunk Objects
    chunks: List[ELUCChunk] = []
    chunk_index = 0

    for group in final_chunk_groups:
        first_idx = -1
        last_idx = -1
        for i, b in enumerate(group):
            if len(b.text) >= config.min_block_chars:
                if first_idx == -1:
                    first_idx = i
                last_idx = i
                
        if first_idx == -1:
            continue

        meaningful = group[first_idx:last_idx + 1]

        char_start = _locate_char_start(doc.provenance, meaningful[0])
        char_end = _locate_char_end(doc.provenance, meaningful[-1])

        provenance_spans = []
        if char_start != -1 and char_end != -1:
            text = doc.full_text[char_start:char_end]
            provenance_spans = _find_overlapping_provenance(doc.provenance, char_start, char_end)
        else:
            text = SEPARATOR.join(b.text for b in meaningful)

        source_blocks = []
        for b in meaningful:
            source_blocks.extend(b.source_blocks)

        heading_label = next(
            (b.label for b in meaningful if b.label in HEADING_LABELS),
            None
        )

        chunk_id = f"{doc.doc_id}_eluc_{chunk_index:06d}"
        
        contains_sidebar = any(b.sidebar_unit_id is not None for b in meaningful)
        
        page_min = min(b.source_page for b in meaningful)
        page_max = max(b.source_page for b in meaningful)
        page_span = (page_min, page_max)

        chunks.append(ELUCChunk(
            chunk_id=chunk_id,
            doc_id=doc.doc_id,
            chunk_index=chunk_index,
            text=text,
            char_start=char_start,
            char_end=char_end,
            provenance_spans=provenance_spans,
            source_blocks=source_blocks,
            heading_label=heading_label,
            contains_sidebar=contains_sidebar,
            block_count=len(meaningful),
            page_span=page_span,
        ))
        chunk_index += 1

    return ELUCChunkDocument(
        doc_id=doc.doc_id,
        strategy="eluc_section_aware",
        config_used=config,
        chunks=chunks
    )
