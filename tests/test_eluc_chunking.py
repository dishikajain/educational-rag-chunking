import os
import pytest
import json
from dataclasses import asdict

from src.extraction.models import ExtractedDocument, ExtractedPage, TextBlock
from src.preprocessing.models import PreprocessedDocument, PreprocessedBlock, PreprocessedPage, ProvenanceSpan
from src.eluc.config import load_config, ELUCChunkingConfig
from src.eluc.models import ELUCChunk, ELUCChunkDocument
from src.eluc.chunker import eluc_chunk, HEADING_LABELS

CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "config", "eluc_chunking_config.yaml"
)

@pytest.fixture
def config() -> ELUCChunkingConfig:
    return load_config(CONFIG_PATH)

def test_load_eluc_config():
    config = load_config(CONFIG_PATH)
    assert config.strategy == "eluc_section_aware"
    assert config.max_chunk_chars == 1400
    assert config.min_block_chars == 5
    assert config.sidebar_atomic is True

def test_eluc_config_sidebar_atomic_false_raises():
    with pytest.raises(ValueError):
        ELUCChunkingConfig(
            strategy="eluc_section_aware",
            max_chunk_chars=1400,
            min_block_chars=5,
            sidebar_atomic=False,
            corpus_name="test",
            corpus_chapters=[1]
        )

def test_eluc_config_max_chunk_chars_positive():
    with pytest.raises(ValueError):
        ELUCChunkingConfig(
            strategy="eluc_section_aware",
            max_chunk_chars=0,
            min_block_chars=5,
            sidebar_atomic=True,
            corpus_name="test",
            corpus_chapters=[1]
        )

def test_eluc_config_min_block_chars_positive():
    with pytest.raises(ValueError):
        ELUCChunkingConfig(
            strategy="eluc_section_aware",
            max_chunk_chars=1400,
            min_block_chars=0,
            sidebar_atomic=True,
            corpus_name="test",
            corpus_chapters=[1]
        )

def make_doc(blocks):
    provenance = []
    char_count = 0
    for i, b in enumerate(blocks):
        # Phase 3 convention: \n\n on all blocks except the last
        is_last = (i == len(blocks) - 1)
        sep = "" if is_last else "\n\n"
        text_with_sep = b.text + sep
        start = char_count
        end = char_count + len(text_with_sep)
        provenance.append(ProvenanceSpan(
            char_start=start,
            char_end=end,
            source_blocks=[(b.source_page, b.p2_block_index)],
            label=b.label
        ))
        char_count = end

    full_text = "".join(b.text + ("" if i == len(blocks) - 1 else "\n\n") for i, b in enumerate(blocks))
    
    return PreprocessedDocument(
        doc_id="test_doc",
        pages=[PreprocessedPage(page_number=1, page_type="SINGLE_COLUMN", blocks=blocks)],
        full_text=full_text,
        provenance=provenance,
        config_used=None
    )

@pytest.fixture
def dummy_doc():
    blocks = [
        PreprocessedBlock(source_page=1, p2_block_index=0, reading_order_index=0, text="Header 1", x0=0, y0=0, x1=10, y1=10, label="HEADING_1"),
        PreprocessedBlock(source_page=1, p2_block_index=1, reading_order_index=1, text="Body paragraph 1", x0=0, y0=20, x1=100, y1=30, label="BODY", is_paragraph_break=True),
        PreprocessedBlock(source_page=1, p2_block_index=2, reading_order_index=2, text="Body paragraph 2", x0=0, y0=40, x1=100, y1=50, label="BODY", is_paragraph_break=True),
        PreprocessedBlock(source_page=1, p2_block_index=3, reading_order_index=3, text="Sidebar Start", x0=10, y0=60, x1=50, y1=70, label="SIDEBAR_BODY", sidebar_unit_id="side1"),
        PreprocessedBlock(source_page=1, p2_block_index=4, reading_order_index=4, text="Sidebar End", x0=10, y0=80, x1=50, y1=90, label="SIDEBAR_BODY", sidebar_unit_id="side1"),
        PreprocessedBlock(source_page=1, p2_block_index=5, reading_order_index=5, text="Heading 2", x0=0, y0=100, x1=10, y1=110, label="HEADING_2"),
        PreprocessedBlock(source_page=1, p2_block_index=6, reading_order_index=6, text="Body paragraph 3", x0=0, y0=120, x1=100, y1=130, label="BODY", is_paragraph_break=True),
        # Degenerate block
        PreprocessedBlock(source_page=1, p2_block_index=7, reading_order_index=7, text="12", x0=0, y0=140, x1=10, y1=150, label="BODY"),
    ]
    return make_doc(blocks)

def test_eluc_basic_chunking(dummy_doc, config):
    doc = eluc_chunk(dummy_doc, config)
    assert len(doc.chunks) > 0
    assert doc.chunks[0].chunk_id == "test_doc_eluc_000000"
    
def test_boundary_at_index_zero(dummy_doc, config):
    doc = eluc_chunk(dummy_doc, config)
    assert doc.chunks[0].source_blocks[0] == (1, 0)
    
def test_heading_opens_boundary(dummy_doc, config):
    doc = eluc_chunk(dummy_doc, config)
    assert len(doc.chunks) == 3
    assert doc.chunks[0].heading_label == "HEADING_1"
    assert doc.chunks[0].block_count == 3
    assert doc.chunks[1].contains_sidebar is True
    assert doc.chunks[1].block_count == 2
    assert doc.chunks[2].heading_label == "HEADING_2"
    # Block count is 2 because the degenerate block at index 7 is trimmed!
    assert doc.chunks[2].block_count == 2
    
def test_section_break_true_opens_boundary(config):
    blocks = [
        PreprocessedBlock(source_page=1, p2_block_index=0, reading_order_index=0, text="Header", x0=0, y0=0, x1=10, y1=10, label="BODY"),
        PreprocessedBlock(source_page=1, p2_block_index=1, reading_order_index=1, text="Break here", x0=0, y0=0, x1=10, y1=10, label="BODY", is_section_break=True),
    ]
    doc = make_doc(blocks)
    out = eluc_chunk(doc, config)
    assert len(out.chunks) == 2

def test_section_break_none_no_boundary(config):
    blocks = [
        PreprocessedBlock(source_page=1, p2_block_index=0, reading_order_index=0, text="Header", x0=0, y0=0, x1=10, y1=10, label="BODY"),
        PreprocessedBlock(source_page=1, p2_block_index=1, reading_order_index=1, text="No break here", x0=0, y0=0, x1=10, y1=10, label="BODY", is_section_break=None),
    ]
    doc = make_doc(blocks)
    out = eluc_chunk(doc, config)
    assert len(out.chunks) == 1
    
def test_section_break_false_no_boundary(config):
    blocks = [
        PreprocessedBlock(source_page=1, p2_block_index=0, reading_order_index=0, text="Header", x0=0, y0=0, x1=10, y1=10, label="BODY"),
        PreprocessedBlock(source_page=1, p2_block_index=1, reading_order_index=1, text="No break here", x0=0, y0=0, x1=10, y1=10, label="BODY", is_section_break=False),
    ]
    doc = make_doc(blocks)
    out = eluc_chunk(doc, config)
    assert len(out.chunks) == 1

def test_multiple_signals_single_boundary(config):
    blocks = [
        PreprocessedBlock(source_page=1, p2_block_index=0, reading_order_index=0, text="Header", x0=0, y0=0, x1=10, y1=10, label="BODY"),
        PreprocessedBlock(source_page=1, p2_block_index=1, reading_order_index=1, text="Multiple signals", x0=0, y0=0, x1=10, y1=10, label="HEADING_1", is_section_break=True),
    ]
    doc = make_doc(blocks)
    out = eluc_chunk(doc, config)
    assert len(out.chunks) == 2

def test_heading_after_sidebar_opens_boundary(dummy_doc, config):
    doc = eluc_chunk(dummy_doc, config)
    assert len(doc.chunks) == 3

def test_section_boundary_flush_before_overflow(config):
    config.max_chunk_chars = 20
    blocks = [
        PreprocessedBlock(source_page=1, p2_block_index=0, reading_order_index=0, text="A very long body text block", x0=0, y0=0, x1=10, y1=10, label="BODY"),
        PreprocessedBlock(source_page=1, p2_block_index=1, reading_order_index=1, text="Next body", x0=0, y0=0, x1=10, y1=10, label="HEADING_1"),
    ]
    doc = make_doc(blocks)
    out = eluc_chunk(doc, config)
    assert len(out.chunks) == 2
    assert out.chunks[0].text == "A very long body text block\n\n"
    assert out.chunks[1].text == "Next body"

def test_overflow_splits_non_sidebar(config):
    config.max_chunk_chars = 15
    blocks = [
        PreprocessedBlock(source_page=1, p2_block_index=0, reading_order_index=0, text="Block 1 text", x0=0, y0=0, x1=10, y1=10, label="BODY"),
        PreprocessedBlock(source_page=1, p2_block_index=1, reading_order_index=1, text="Block 2 text", x0=0, y0=0, x1=10, y1=10, label="BODY", is_paragraph_break=True),
    ]
    doc = make_doc(blocks)
    out = eluc_chunk(doc, config)
    assert len(out.chunks) == 2

def test_mid_sidebar_never_overflows(config):
    config.max_chunk_chars = 10
    blocks = [
        PreprocessedBlock(source_page=1, p2_block_index=0, reading_order_index=0, text="Side 1", x0=0, y0=0, x1=10, y1=10, label="SIDEBAR_BODY", sidebar_unit_id="side1"),
        PreprocessedBlock(source_page=1, p2_block_index=1, reading_order_index=1, text="Side 2", x0=0, y0=0, x1=10, y1=10, label="SIDEBAR_BODY", sidebar_unit_id="side1"),
        PreprocessedBlock(source_page=1, p2_block_index=2, reading_order_index=2, text="Side 3", x0=0, y0=0, x1=10, y1=10, label="SIDEBAR_BODY", sidebar_unit_id="side1"),
    ]
    doc = make_doc(blocks)
    out = eluc_chunk(doc, config)
    assert len(out.chunks) == 1
    assert out.chunks[0].block_count == 3

def test_heading_only_chunk(config):
    blocks = [
        PreprocessedBlock(source_page=1, p2_block_index=0, reading_order_index=0, text="Heading A", x0=0, y0=0, x1=10, y1=10, label="HEADING_1"),
        PreprocessedBlock(source_page=1, p2_block_index=1, reading_order_index=1, text="Heading B", x0=0, y0=0, x1=10, y1=10, label="HEADING_2"),
    ]
    doc = make_doc(blocks)
    out = eluc_chunk(doc, config)
    assert len(out.chunks) == 2
    assert out.chunks[0].block_count == 1
    assert out.chunks[1].block_count == 1

def test_sidebar_not_split_in_pass3(config):
    config.max_chunk_chars = 10
    blocks = [
        PreprocessedBlock(source_page=1, p2_block_index=0, reading_order_index=0, text="Side 1 long", x0=0, y0=0, x1=10, y1=10, label="SIDEBAR_BODY", sidebar_unit_id="side1"),
        PreprocessedBlock(source_page=1, p2_block_index=1, reading_order_index=1, text="Side 2 long", x0=0, y0=0, x1=10, y1=10, label="SIDEBAR_BODY", sidebar_unit_id="side1"),
    ]
    doc = make_doc(blocks)
    out = eluc_chunk(doc, config)
    assert len(out.chunks) == 1
    assert out.chunks[0].block_count == 2

def test_single_oversized_block_emitted_whole(config):
    config.max_chunk_chars = 10
    blocks = [
        PreprocessedBlock(source_page=1, p2_block_index=0, reading_order_index=0, text="A very long single block that cannot be split.", x0=0, y0=0, x1=10, y1=10, label="BODY"),
    ]
    doc = make_doc(blocks)
    out = eluc_chunk(doc, config)
    assert len(out.chunks) == 1
    assert out.chunks[0].block_count == 1

def test_char_start_end_valid_indices(dummy_doc, config):
    doc = eluc_chunk(dummy_doc, config)
    for c in doc.chunks:
        assert 0 <= c.char_start < c.char_end <= len(dummy_doc.full_text)

def test_chunk_text_length_invariant_P5(dummy_doc, config):
    # Every emitted ELUC chunk with valid provenance must exactly match the sliced full_text
    doc = eluc_chunk(dummy_doc, config)
    for c in doc.chunks:
        if c.char_start != -1 and c.char_end != -1:
            assert c.text == dummy_doc.full_text[c.char_start:c.char_end]

def test_missing_provenance_sentinel(config):
    blocks = [
        PreprocessedBlock(source_page=1, p2_block_index=0, reading_order_index=0, text="Header", x0=0, y0=0, x1=10, y1=10, label="BODY"),
    ]
    doc = PreprocessedDocument("doc", [PreprocessedPage(1, "SINGLE_COLUMN", blocks)], "Header", [], None)
    out = eluc_chunk(doc, config)
    assert out.chunks[0].char_start == -1
    assert out.chunks[0].char_end == -1
    assert out.chunks[0].provenance_spans == []
    assert out.chunks[0].text == "Header"

def test_eluc_chunk_id_format(dummy_doc, config):
    doc = eluc_chunk(dummy_doc, config)
    for i, c in enumerate(doc.chunks):
        assert c.chunk_id == f"test_doc_eluc_{i:06d}"

def test_eluc_json_serializable(dummy_doc, config):
    doc = eluc_chunk(dummy_doc, config)
    d = asdict(doc)
    s = json.dumps(d)
    assert s is not None
    
def test_doc_not_mutated(dummy_doc, config):
    import copy
    orig = copy.deepcopy(dummy_doc)
    eluc_chunk(dummy_doc, config)
    assert dummy_doc == orig

def test_short_block_between_meaningful_blocks(config):
    blocks = [
        PreprocessedBlock(source_page=1, p2_block_index=0, reading_order_index=0, text="First meaningful block.", x0=0, y0=0, x1=10, y1=10, label="BODY"),
        PreprocessedBlock(source_page=1, p2_block_index=1, reading_order_index=1, text="1", x0=0, y0=0, x1=10, y1=10, label="BODY"),
        PreprocessedBlock(source_page=1, p2_block_index=2, reading_order_index=2, text="Second meaningful block.", x0=0, y0=0, x1=10, y1=10, label="BODY"),
    ]
    doc = make_doc(blocks)
    out = eluc_chunk(doc, config)
    
    assert len(out.chunks) == 1
    chunk = out.chunks[0]
    
    assert chunk.text == doc.full_text[chunk.char_start:chunk.char_end]
    assert "1" in chunk.text
    assert chunk.block_count == 3
    assert (1, 1) in chunk.source_blocks
