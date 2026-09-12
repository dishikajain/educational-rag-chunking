import os
import yaml
from dataclasses import dataclass
from typing import List


@dataclass
class BaselineChunkingConfig:
    strategy: str
    chunk_size: int
    chunk_overlap: int
    min_chunk_size: int
    corpus_name: str
    corpus_chapters: List[int]

    def __post_init__(self):
        """Validate parameter consistency on load."""
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.min_chunk_size <= 0:
            raise ValueError(f"min_chunk_size must be positive, got {self.min_chunk_size}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be strictly less than "
                f"chunk_size ({self.chunk_size}); stride would be <= 0"
            )

    @property
    def stride(self) -> int:
        """Number of characters to advance per step."""
        return self.chunk_size - self.chunk_overlap


def load_config(config_path: str) -> BaselineChunkingConfig:
    """Load BaselineChunkingConfig from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Chunking configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return BaselineChunkingConfig(
        strategy=str(data["strategy"]),
        chunk_size=int(data["chunk_size"]),
        chunk_overlap=int(data["chunk_overlap"]),
        min_chunk_size=int(data["min_chunk_size"]),
        corpus_name=str(data["corpus_name"]),
        corpus_chapters=list(data["corpus_chapters"]),
    )
