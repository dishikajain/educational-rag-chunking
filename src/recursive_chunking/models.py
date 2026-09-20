from dataclasses import dataclass
from typing import List
from src.chunking.models import BaselineChunk
from src.recursive_chunking.config import RecursiveChunkingConfig

@dataclass
class RecursiveChunkDocument:
    doc_id: str
    strategy: str
    config_used: RecursiveChunkingConfig
    chunks: List[BaselineChunk]
