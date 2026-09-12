import bisect
from typing import List

from src.preprocessing.models import PreprocessedDocument, ProvenanceSpan
from src.chunking.config import BaselineChunkingConfig
from src.chunking.models import BaselineChunk, BaselineChunkDocument


def _find_overlapping_provenance(
    provenance: List[ProvenanceSpan],
    char_start: int,
    char_end: int,
) -> List[ProvenanceSpan]:
    """
    Return all ProvenanceSpans that overlap the character range [char_start, char_end).

    A span S overlaps if: S.char_start < char_end AND S.char_end > char_start.

    Uses bisect on char_start values for O(log n) left boundary, then linear scan
    forward until S.char_start >= char_end.
    """
    if not provenance:
        return []

    # Build a sorted list of char_start values for bisect
    starts = [s.char_start for s in provenance]

    # Find the rightmost span whose char_start < char_end
    right_bound = bisect.bisect_left(starts, char_end)

    overlapping = []
    # Walk backwards from right_bound-1 to 0; stop when S.char_end <= char_start
    # (spans to the left are fully before our range)
    # More efficient: walk forward from the first possible overlap
    # First possible: any span whose char_end > char_start
    # Start search from index where char_start could first reach our range.
    left_bound = bisect.bisect_left(starts, char_start)
    # Also check left_bound-1 in case a span straddles char_start from the left
    search_start = max(0, left_bound - 1)

    for i in range(search_start, right_bound):
        s = provenance[i]
        if s.char_end > char_start and s.char_start < char_end:
            overlapping.append(s)

    return overlapping


def chunk(
    doc: PreprocessedDocument,
    config: BaselineChunkingConfig,
) -> BaselineChunkDocument:
    """
    Pure, deterministic fixed-size sliding-window chunker.

    Operates exclusively on doc.full_text (flat character stream).
    Does NOT use any structural metadata from doc.pages or doc.provenance
    to influence chunk boundaries — the baseline is structure-blind.

    Provenance spans from doc.provenance are attached to chunks as metadata only,
    after chunk boundaries are determined.
    """
    full_text = doc.full_text
    stride = config.stride
    chunk_size = config.chunk_size
    min_chunk_size = config.min_chunk_size
    doc_id = doc.doc_id

    chunks: List[BaselineChunk] = []

    if not full_text:
        return BaselineChunkDocument(
            doc_id=doc_id,
            strategy=config.strategy,
            config_used=config,
            chunks=[],
        )

    char_start = 0
    chunk_index = 0

    while char_start < len(full_text):
        char_end = min(char_start + chunk_size, len(full_text))
        chunk_text = full_text[char_start:char_end]

        # Emit only if at or above minimum size
        if len(chunk_text) >= min_chunk_size:
            chunk_id = f"{doc_id}_baseline_{chunk_index:06d}"
            provenance_spans = _find_overlapping_provenance(
                doc.provenance, char_start, char_end
            )
            chunks.append(BaselineChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                chunk_index=chunk_index,
                text=chunk_text,
                char_start=char_start,
                char_end=char_end,
                provenance_spans=provenance_spans,
            ))
            chunk_index += 1

        # If we've reached the end, stop
        if char_end == len(full_text):
            break

        char_start += stride

    return BaselineChunkDocument(
        doc_id=doc_id,
        strategy=config.strategy,
        config_used=config,
        chunks=chunks,
    )
