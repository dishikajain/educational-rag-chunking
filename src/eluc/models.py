from dataclasses import dataclass
from typing import List, Tuple, Optional
from src.preprocessing.models import ProvenanceSpan
from src.eluc.config import ELUCChunkingConfig

@dataclass
class ELUCChunk:
    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    char_start: int
    char_end: int
    provenance_spans: List[ProvenanceSpan]
    source_blocks: List[Tuple[int, int]]
    heading_label: Optional[str]
    contains_sidebar: bool
    block_count: int
    page_span: Tuple[int, int]

@dataclass
class ELUCChunkDocument:
    doc_id: str
    strategy: str
    config_used: ELUCChunkingConfig
    chunks: List[ELUCChunk]
