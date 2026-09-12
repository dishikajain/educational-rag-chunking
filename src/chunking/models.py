from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from src.chunking.config import BaselineChunkingConfig
from src.preprocessing.models import ProvenanceSpan


@dataclass
class BaselineChunk:
    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    char_start: int
    char_end: int
    provenance_spans: List[ProvenanceSpan]


@dataclass
class BaselineChunkDocument:
    doc_id: str
    strategy: str
    config_used: BaselineChunkingConfig
    chunks: List[BaselineChunk]
