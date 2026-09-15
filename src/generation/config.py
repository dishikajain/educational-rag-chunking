import os
import yaml
from pydantic import BaseModel, field_validator

class GenerationConfig(BaseModel):
    model_config = {"extra": "forbid", "strict": True}
    
    model_name: str
    temperature: float
    thinking_budget: int
    max_output_tokens: int
    system_instruction: str

    @field_validator('model_name')
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        if v != "gemini-2.5-flash":
            raise ValueError(f"model_name must be exactly 'gemini-2.5-flash', got {v}")
        return v

    @field_validator('temperature')
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if v != 0.0:
            raise ValueError(f"temperature must be exactly 0.0, got {v}")
        return v

    @field_validator('thinking_budget')
    @classmethod
    def validate_thinking_budget(cls, v: int) -> int:
        if v != 0:
            raise ValueError(f"thinking_budget must be exactly 0, got {v}")
        return v

    @field_validator('max_output_tokens')
    @classmethod
    def validate_max_output_tokens(cls, v: int) -> int:
        if v != 1024:
            raise ValueError(f"max_output_tokens must be exactly 1024, got {v}")
        return v


def load_config(path: str = "config/generation_config.yaml") -> GenerationConfig:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")
        
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        
    if not data or not isinstance(data, dict):
        raise ValueError("Invalid or empty YAML configuration")
        
    return GenerationConfig(**data)
