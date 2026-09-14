import os
import yaml
from dataclasses import dataclass


@dataclass
class EmbeddingConfig:
    model_name: str
    model_revision: str
    embedding_dimension: int
    normalize_embeddings: bool
    batch_size: int


def load_config(path: str) -> EmbeddingConfig:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"Invalid or empty config file: {path}")

    required_fields = [
        "model_name",
        "model_revision",
        "embedding_dimension",
        "normalize_embeddings",
        "batch_size",
    ]

    missing = [field for field in required_fields if field not in data]
    if missing:
        raise ValueError(f"Missing required config fields: {missing}")

    if data["embedding_dimension"] <= 0:
        raise ValueError("embedding_dimension must be positive")

    if data["batch_size"] <= 0:
        raise ValueError("batch_size must be positive")

    return EmbeddingConfig(
        model_name=data["model_name"],
        model_revision=data["model_revision"],
        embedding_dimension=data["embedding_dimension"],
        normalize_embeddings=data["normalize_embeddings"],
        batch_size=data["batch_size"],
    )