from dataclasses import dataclass
from typing import List
from src.chunking.models import BaselineChunk
from src.paragraph_chunking.config import ParagraphChunkingConfig

@dataclass
class ParagraphChunkDocument:
    doc_id: str
    strategy: str
    config_used: ParagraphChunkingConfig
    chunks: List[BaselineChunk]
