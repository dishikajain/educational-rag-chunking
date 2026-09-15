from pydantic import BaseModel, Field
from typing import Dict, Any, List

class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    strategy: str
    text: str
    distance: float
    char_start: int
    char_end: int
    provenance_spans: List[Dict[str, Any]]
    metadata: Dict[str, Any] = Field(default_factory=dict)
