from pydantic import BaseModel
from typing import List, Any, Dict
from src.retrieval.models import RetrievedChunk

class GenerationResult(BaseModel):
    original_question: str
    answer: str
    model_identifier: str
    prompt_used: str
    retrieved_chunks: List[RetrievedChunk]
    metadata: Dict[str, Any]
