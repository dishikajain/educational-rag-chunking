import yaml
from dataclasses import dataclass
from typing import List

@dataclass
class RecursiveChunkingConfig:
    strategy: str
    chunk_size: int
    chunk_overlap: int
    min_chunk_size: int
    separators: List[str]
    corpus_name: str
    corpus_chapters: List[int]

def load_config(path: str) -> RecursiveChunkingConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return RecursiveChunkingConfig(**data)
