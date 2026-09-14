from dataclasses import dataclass
from typing import List
from src.preprocessing.models import ProvenanceSpan

@dataclass
class EmbeddedChunk:
    chunk_id: str
    doc_id: str
    text: str
    embedding: List[float]
    strategy: str
    char_start: int
    char_end: int
    provenance_spans: List[ProvenanceSpan]

@dataclass
class EmbeddedChunkDocument:
    doc_id: str
    strategy: str
    model_name: str
    model_revision: str
    embedding_dimension: int
    chunks: List[EmbeddedChunk]
