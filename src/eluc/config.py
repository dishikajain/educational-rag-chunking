import os
import yaml
from dataclasses import dataclass
from typing import List

@dataclass
class ELUCChunkingConfig:
    strategy: str
    max_chunk_chars: int
    min_block_chars: int
    sidebar_atomic: bool
    corpus_name: str
    corpus_chapters: List[int]

    def __post_init__(self):
        if self.sidebar_atomic is not True:
            raise ValueError("sidebar_atomic must be True")
        if self.max_chunk_chars <= 0:
            raise ValueError("max_chunk_chars must be positive")
        if self.min_block_chars <= 0:
            raise ValueError("min_block_chars must be positive")

def load_config(path: str) -> ELUCChunkingConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return ELUCChunkingConfig(**data)
