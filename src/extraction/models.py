from dataclasses import dataclass
from typing import List, Optional

@dataclass
class TextBlock:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    font_size: Optional[float] = None
    font_name: Optional[str] = None

@dataclass
class ExtractedPage:
    page_number: int
    blocks: List[TextBlock]

@dataclass
class ExtractedDocument:
    doc_id: str
    pages: List[ExtractedPage]
