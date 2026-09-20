import pytest
from src.recursive_chunking.config import RecursiveChunkingConfig
from src.recursive_chunking.chunker import recursive_chunk
from src.paragraph_chunking.config import ParagraphChunkingConfig
from src.paragraph_chunking.chunker import paragraph_chunk
from src.heading_chunking.config import HeadingChunkingConfig
from src.heading_chunking.chunker import heading_chunk
from src.preprocessing.models import PreprocessedDocument, PreprocessedPage, PreprocessedBlock, ProvenanceSpan
from src.chunking.chunker import _find_overlapping_provenance

def test_recursive_deterministic():
    config = RecursiveChunkingConfig("recursive_character", 50, 10, 10, ["\n\n", " "], "test", [1])
    prov = [ProvenanceSpan(0, 100, [(1,0)], "BODY")]
    doc = PreprocessedDocument("test_doc", [], "A" * 100, prov, None)
    res1 = recursive_chunk(doc, config)
    res2 = recursive_chunk(doc, config)
    assert len(res1.chunks) > 0
    assert res1.chunks == res2.chunks

def test_paragraph_deterministic():
    config = ParagraphChunkingConfig("paragraph_based", 100, 20, 10, 5, "test", [1])
    b1 = PreprocessedBlock(1, 0, 0, "A" * 30, 0, 0, 0, 0, "BODY", is_paragraph_break=False)
    b2 = PreprocessedBlock(1, 1, 1, "B" * 30, 0, 0, 0, 0, "BODY", is_paragraph_break=True)
    page = PreprocessedPage(1, "SINGLE_COLUMN", [b1, b2])
    text = "A" * 30 + "\n\n" + "B" * 30
    prov = [ProvenanceSpan(0, len(text), [(1,0), (1,1)], "BODY")]
    doc = PreprocessedDocument("test_doc", [page], text, prov, None)
    res1 = paragraph_chunk(doc, config)
    res2 = paragraph_chunk(doc, config)
    assert len(res1.chunks) > 0
    assert res1.chunks == res2.chunks
    
def test_heading_deterministic():
    config = HeadingChunkingConfig("heading_based", 100, 10, "test", [1])
    b1 = PreprocessedBlock(1, 0, 0, "HEADING", 0, 0, 0, 0, "HEADING_1")
    b2 = PreprocessedBlock(1, 1, 1, "BODY TEXT", 0, 0, 0, 0, "BODY")
    page = PreprocessedPage(1, "SINGLE_COLUMN", [b1, b2])
    text = "HEADING\n\nBODY TEXT"
    prov = [ProvenanceSpan(0, len(text), [(1,0), (1,1)], "BODY")]
    doc = PreprocessedDocument("test_doc", [page], text, prov, None)
    res1 = heading_chunk(doc, config)
    res2 = heading_chunk(doc, config)
    assert len(res1.chunks) > 0
    assert res1.chunks == res2.chunks
