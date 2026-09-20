from dataclasses import dataclass
from typing import List
from src.chunking.models import BaselineChunk
from src.heading_chunking.config import HeadingChunkingConfig

@dataclass
class HeadingChunkDocument:
    doc_id: str
    strategy: str
    config_used: HeadingChunkingConfig
    chunks: List[BaselineChunk]
