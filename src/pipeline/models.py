from pydantic import BaseModel
from src.generation.models import GenerationResult

class RAGResult(BaseModel):
    question: str
    strategy: str
    generation: GenerationResult
