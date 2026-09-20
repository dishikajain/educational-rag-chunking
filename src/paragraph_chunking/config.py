import yaml
from dataclasses import dataclass
from typing import List

@dataclass
class ParagraphChunkingConfig:
    strategy: str
    max_chunk_chars: int
    paragraph_soft_min: int
    min_chunk_chars: int
    min_block_chars: int
    corpus_name: str
    corpus_chapters: List[int]

def load_config(path: str) -> ParagraphChunkingConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return ParagraphChunkingConfig(**data)
