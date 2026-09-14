from dataclasses import dataclass
import os
from typing import Any

import yaml


@dataclass
class IndexingConfig:
    embedding_dimension: int = 768
    distance_metric: str = "cosine"
    index_batch_size: int = 256
    reset_collection_on_run: bool = True
    storage_path: str = "data/chroma_db"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.embedding_dimension, int)
            or isinstance(self.embedding_dimension, bool)
            or self.embedding_dimension != 768
        ):
            raise ValueError(
                f"embedding_dimension must be exactly 768, "
                f"got {self.embedding_dimension!r}"
            )

        if (
            not isinstance(self.distance_metric, str)
            or self.distance_metric != "cosine"
        ):
            raise ValueError(
                f"distance_metric must be 'cosine', "
                f"got {self.distance_metric!r}"
            )

        if (
            not isinstance(self.index_batch_size, int)
            or isinstance(self.index_batch_size, bool)
            or self.index_batch_size != 256
        ):
            raise ValueError(
                f"index_batch_size must be exactly 256, "
                f"got {self.index_batch_size!r}"
            )

        if not isinstance(self.reset_collection_on_run, bool):
            raise ValueError("reset_collection_on_run must be boolean")

        if (
            not isinstance(self.storage_path, str)
            or not self.storage_path.strip()
        ):
            raise ValueError(
                "storage_path must be a non-empty string"
            )

    @classmethod
    def from_yaml(cls, path: str) -> "IndexingConfig":
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Indexing configuration file not found: {path}"
            )

        with open(path, "r", encoding="utf-8") as f:
            data: Any = yaml.safe_load(f)

        if data is None:
            data = {}

        if not isinstance(data, dict):
            raise TypeError(
                "Indexing configuration YAML must contain a mapping/object."
            )

        allowed_fields = {
            "embedding_dimension",
            "distance_metric",
            "index_batch_size",
            "reset_collection_on_run",
            "storage_path",
        }

        unknown_fields = set(data) - allowed_fields
        if unknown_fields:
            raise ValueError(
                f"Unknown indexing configuration field(s): "
                f"{sorted(unknown_fields)}"
            )

        return cls(**data)