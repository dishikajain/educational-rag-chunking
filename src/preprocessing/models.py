from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from src.preprocessing.config import PreprocessingConfig

@dataclass
class ProvenanceSpan:
    char_start: int
    char_end: int
    source_blocks: List[Tuple[int, int]]
    label: str

@dataclass
class PreprocessedBlock:
    source_page: int
    p2_block_index: int
    reading_order_index: int
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    label: str
    font_size: Optional[float] = None
    font_name: Optional[str] = None
    is_bold: bool = False
    zone: str = "FULL_WIDTH"
    page_type: str = "SINGLE_COLUMN"
    gap_before_pt: Optional[float] = None
    is_paragraph_break: Optional[bool] = None
    is_section_break: Optional[bool] = None
    sidebar_unit_id: Optional[str] = None
    sidebar_type: Optional[str] = None
    source_blocks: Optional[List[Tuple[int, int]]] = None

    def __post_init__(self):
        if self.source_blocks is None:
            self.source_blocks = [(self.source_page, self.p2_block_index)]

@dataclass
class PreprocessedPage:
    page_number: int
    page_type: str
    blocks: List[PreprocessedBlock]

@dataclass
class PreprocessedDocument:
    doc_id: str
    pages: List[PreprocessedPage]
    full_text: str
    provenance: List[ProvenanceSpan]
    config_used: PreprocessingConfig
