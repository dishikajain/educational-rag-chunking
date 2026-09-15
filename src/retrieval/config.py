import os
import yaml
from pydantic import BaseModel, field_validator
from pathlib import Path

class RetrievalConfig(BaseModel):
    model_name: str
    model_revision: str
    query_instruction: str
    top_k: int
    distance_metric: str
    batch_size: int
    storage_path: str

    @field_validator('model_name')
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        if v != "BAAI/bge-base-en-v1.5":
            raise ValueError(f"model_name must be exactly 'BAAI/bge-base-en-v1.5', got '{v}'")
        return v

    @field_validator('model_revision')
    @classmethod
    def validate_model_revision(cls, v: str) -> str:
        if v != "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a":
            raise ValueError(f"model_revision must be exactly 'a5beb1e3e68b9ab74eb54cfd186867f64f240e1a', got '{v}'")
        return v

    @field_validator('query_instruction')
    @classmethod
    def validate_query_instruction(cls, v: str) -> str:
        if v != "Represent this sentence for searching relevant passages: ":
            raise ValueError("Invalid BGE query_instruction")
        return v

    @field_validator('distance_metric')
    @classmethod
    def validate_distance_metric(cls, v: str) -> str:
        if v != "cosine":
            raise ValueError(f"distance_metric must be 'cosine', got '{v}'")
        return v

    @field_validator('batch_size')
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        if v != 1:
            raise ValueError(f"batch_size must be exactly 1, got {v}")
        return v

    @field_validator('top_k')
    @classmethod
    def validate_top_k(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"top_k must be a positive integer, got {v}")
        return v

def load_config(path: str = "config/retrieval_config.yaml") -> RetrievalConfig:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")
        
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        
    if not data or not isinstance(data, dict):
        raise ValueError(f"Invalid or empty YAML configuration in {path}")
        
    return RetrievalConfig(**data)
