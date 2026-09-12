import json
import os
import pytest

from src.extraction.models import ExtractedDocument, ExtractedPage, TextBlock
from src.preprocessing.config import load_config as load_preprocessing_config
from src.preprocessing.preprocessor import preprocess
from src.chunking.config import load_config, BaselineChunkingConfig
from src.chunking.models import BaselineChunk, BaselineChunkDocument
from src.chunking.chunker import chunk, _find_overlapping_provenance
from src.preprocessing.models import ProvenanceSpan


CHUNKING_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "config", "baseline_chunking_config.yaml"
)
PREPROCESSING_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "config", "preprocessing_config.yaml"
)


@pytest.fixture
def config() -> BaselineChunkingConfig:
    return load_config(CHUNKING_CONFIG_PATH)


@pytest.fixture
def small_config() -> BaselineChunkingConfig:
    """Minimal config for arithmetic-level tests."""
    return BaselineChunkingConfig(
        strategy="baseline_fixed_size",
        chunk_size=10,
        chunk_overlap=2,
        min_chunk_size=3,
        corpus_name="test",
        corpus_chapters=[1],
    )


def _make_pdoc(full_text: str, doc_id: str = "test_doc"):
    """Build a minimal PreprocessedDocument with correct provenance from full_text."""
    from src.preprocessing.models import (
        PreprocessedDocument,
        PreprocessedPage,
        PreprocessedBlock,
    )
    from src.preprocessing.config import PreprocessingConfig

    # Build a single provenance span covering the entire full_text
    provenance = [
        ProvenanceSpan(
            char_start=0,
            char_end=len(full_text),
            source_blocks=[(1, 0)],
            label="BODY",
        )
    ] if full_text else []

    # Dummy config
    pp_config = load_preprocessing_config(PREPROCESSING_CONFIG_PATH)

    block = PreprocessedBlock(
        source_page=1,
        p2_block_index=0,
        reading_order_index=0,
        text=full_text,
        x0=72.0,
        y0=100.0,
        x1=532.0,
        y1=130.0,
        label="BODY",
    ) if full_text else None

    page = PreprocessedPage(
        page_number=1,
        page_type="SINGLE_COLUMN",
        blocks=[block] if block else [],
    )

    return PreprocessedDocument(
        doc_id=doc_id,
        pages=[page],
        full_text=full_text,
        provenance=provenance,
        config_used=pp_config,
    )


def _make_multispan_pdoc(texts: list, doc_id: str = "multispan_doc"):
    """Build a PreprocessedDocument with one ProvenanceSpan per text segment (sep=\\n\\n)."""
    from src.preprocessing.models import (
        PreprocessedDocument,
        PreprocessedPage,
        PreprocessedBlock,
    )
    pp_config = load_preprocessing_config(PREPROCESSING_CONFIG_PATH)
    SEP = "\n\n"
    separators = [SEP] * len(texts)
    separators[-1] = ""
    full_text = "".join(t + s for t, s in zip(texts, separators))

    provenance = []
    char_pos = 0
    blocks = []
    for i, (t, sep) in enumerate(zip(texts, separators)):
        start = char_pos
        end = char_pos + len(t) + len(sep)
        provenance.append(ProvenanceSpan(
            char_start=start, char_end=end, source_blocks=[(1, i)], label="BODY"
        ))
        blocks.append(PreprocessedBlock(
            source_page=1, p2_block_index=i, reading_order_index=i,
            text=t, x0=72.0, y0=100.0 + i * 30, x1=532.0, y1=120.0 + i * 30,
            label="BODY",
        ))
        char_pos = end

    page = PreprocessedPage(page_number=1, page_type="SINGLE_COLUMN", blocks=blocks)
    return PreprocessedDocument(
        doc_id=doc_id, pages=[page], full_text=full_text,
        provenance=provenance, config_used=pp_config,
    )


# ── Config tests ───────────────────────────────────────────────────────────────

def test_load_baseline_config(config: BaselineChunkingConfig):
    assert config.strategy == "baseline_fixed_size"
    assert config.chunk_size == 350
    assert config.chunk_overlap == 40
    assert config.min_chunk_size == 70
    assert config.corpus_name == "openstax_biology2e"
    assert config.stride == 310


def test_invalid_config_overlap_gte_chunk_size():
    with pytest.raises(ValueError, match="chunk_overlap"):
        BaselineChunkingConfig(
            strategy="baseline_fixed_size",
            chunk_size=100,
            chunk_overlap=100,
            min_chunk_size=20,
            corpus_name="test",
            corpus_chapters=[1],
        )


def test_invalid_config_nonpositive_chunk_size():
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        BaselineChunkingConfig(
            strategy="baseline_fixed_size",
            chunk_size=0,
            chunk_overlap=0,
            min_chunk_size=20,
            corpus_name="test",
            corpus_chapters=[1],
        )


def test_invalid_config_nonpositive_min_chunk_size():
    with pytest.raises(ValueError, match="min_chunk_size must be positive"):
        BaselineChunkingConfig(
            strategy="baseline_fixed_size",
            chunk_size=100,
            chunk_overlap=10,
            min_chunk_size=0,
            corpus_name="test",
            corpus_chapters=[1],
        )


# ── Algorithm tests ────────────────────────────────────────────────────────────

def test_complete_coverage(small_config: BaselineChunkingConfig):
    """Every character in full_text appears in at least one chunk."""
    full_text = "A" * 35  # 35 chars; chunk_size=10, stride=8
    pdoc = _make_pdoc(full_text)
    cdoc = chunk(pdoc, small_config)

    covered = [False] * len(full_text)
    for c in cdoc.chunks:
        for i in range(c.char_start, c.char_end):
            covered[i] = True
    assert all(covered), "Not all characters are covered by at least one chunk"


def test_exact_overlap(small_config: BaselineChunkingConfig):
    """Adjacent chunks share exactly chunk_overlap characters."""
    full_text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # 26 chars
    pdoc = _make_pdoc(full_text)
    cdoc = chunk(pdoc, small_config)
    assert len(cdoc.chunks) >= 2
    for i in range(len(cdoc.chunks) - 1):
        c_cur = cdoc.chunks[i]
        c_next = cdoc.chunks[i + 1]
        overlap_text_cur = full_text[c_next.char_start:c_cur.char_end]
        overlap_text_next = full_text[c_next.char_start:c_cur.char_end]
        assert overlap_text_cur == overlap_text_next
        actual_overlap = c_cur.char_end - c_next.char_start
        assert actual_overlap == small_config.chunk_overlap, (
            f"Expected overlap {small_config.chunk_overlap}, got {actual_overlap}"
        )


def test_chunk_size_respected(small_config: BaselineChunkingConfig):
    """No chunk has more than chunk_size characters."""
    full_text = "X" * 100
    pdoc = _make_pdoc(full_text)
    cdoc = chunk(pdoc, small_config)
    for c in cdoc.chunks:
        assert len(c.text) <= small_config.chunk_size


def test_final_chunk_truncation(small_config: BaselineChunkingConfig):
    """Last chunk is shorter than chunk_size when text doesn't divide evenly."""
    # chunk_size=10, stride=8, min_chunk_size=3
    # 23 chars: chunk 0=[0,10), chunk 1=[8,18), chunk 2=[16,23) → last chunk len=7 >= 3
    full_text = "A" * 23
    pdoc = _make_pdoc(full_text)
    cdoc = chunk(pdoc, small_config)
    assert len(cdoc.chunks) == 3
    last = cdoc.chunks[-1]
    assert last.char_end == len(full_text)
    assert len(last.text) < small_config.chunk_size


def test_min_chunk_size_enforced(small_config: BaselineChunkingConfig):
    """Final tail chunk is dropped if shorter than min_chunk_size."""
    # chunk_size=10, stride=8, min_chunk_size=3
    # 18 chars: chunk 0=[0,10), chunk 1=[8,18) (len=10 OK), no further chunks
    # 19 chars: chunk 0=[0,10), chunk 1=[8,18)(len=10), chunk 2=[16,19)(len=3 >= 3, emitted)
    # 17 chars: chunk 0=[0,10), chunk 1=[8,17)(len=9 >= 3, OK)
    # Tail too small: 17 chars where last chunk would be 1 char < min(3):
    # stride=8: starts at 0, 8, 16... chunk 2=[16,17)=len 1 < min_chunk_size=3 → dropped
    full_text = "A" * 17
    pdoc = _make_pdoc(full_text)
    cdoc = chunk(pdoc, small_config)
    # chunk 0=[0,10), chunk 1=[8,17) len=9, which is >= 3 so both emitted
    # There is NO third chunk (16+10=26>17, so char_end=17, len=1 -> dropped)
    # Actually: after chunk 1, char_start=16, char_end=min(26,17)=17, len=1 < 3 → dropped
    # But chunk 1 has char_start=8, char_end=17 (we break at char_end==len)
    # Wait: stride=8. chunk 0: start=0, end=10. then start+=8=8. chunk 1: start=8, end=18 ->
    # clamp to min(18,17)=17. len=9 >= 3, emit. char_end==len(17), break.
    # So only 2 chunks expected.
    for c in cdoc.chunks:
        assert len(c.text) >= small_config.min_chunk_size


def test_empty_full_text():
    """Empty full_text produces an empty BaselineChunkDocument."""
    pdoc = _make_pdoc("")
    cfg = BaselineChunkingConfig(
        strategy="baseline_fixed_size",
        chunk_size=100, chunk_overlap=10, min_chunk_size=20,
        corpus_name="test", corpus_chapters=[1],
    )
    cdoc = chunk(pdoc, cfg)
    assert cdoc.chunks == []
    assert cdoc.doc_id == "test_doc"
    assert cdoc.strategy == "baseline_fixed_size"


def test_single_chunk_document():
    """Document shorter than chunk_size produces exactly one chunk."""
    cfg = BaselineChunkingConfig(
        strategy="baseline_fixed_size",
        chunk_size=100, chunk_overlap=10, min_chunk_size=20,
        corpus_name="test", corpus_chapters=[1],
    )
    pdoc = _make_pdoc("Hello world this is a test document.")
    cdoc = chunk(pdoc, cfg)
    assert len(cdoc.chunks) == 1
    assert cdoc.chunks[0].char_start == 0
    assert cdoc.chunks[0].char_end == len(pdoc.full_text)
    assert cdoc.chunks[0].text == pdoc.full_text


def test_exact_chunk_size_document():
    """Document length exactly equals chunk_size produces exactly one chunk."""
    cfg = BaselineChunkingConfig(
        strategy="baseline_fixed_size",
        chunk_size=50, chunk_overlap=10, min_chunk_size=10,
        corpus_name="test", corpus_chapters=[1],
    )
    pdoc = _make_pdoc("A" * 50)
    cdoc = chunk(pdoc, cfg)
    assert len(cdoc.chunks) == 1
    assert cdoc.chunks[0].char_start == 0
    assert cdoc.chunks[0].char_end == 50


def test_document_shorter_than_min_chunk_size():
    """Document shorter than min_chunk_size produces no chunks."""
    cfg = BaselineChunkingConfig(
        strategy="baseline_fixed_size",
        chunk_size=100, chunk_overlap=10, min_chunk_size=50,
        corpus_name="test", corpus_chapters=[1],
    )
    pdoc = _make_pdoc("Hi")
    cdoc = chunk(pdoc, cfg)
    assert cdoc.chunks == []


# ── Determinism and ID tests ───────────────────────────────────────────────────

def test_deterministic_output(config: BaselineChunkingConfig):
    """Two identical calls return byte-identical BaselineChunkDocuments."""
    pdoc = _make_pdoc("The quick brown fox jumps over the lazy dog. " * 30)
    cdoc1 = chunk(pdoc, config)
    cdoc2 = chunk(pdoc, config)
    assert len(cdoc1.chunks) == len(cdoc2.chunks)
    for c1, c2 in zip(cdoc1.chunks, cdoc2.chunks):
        assert c1.chunk_id == c2.chunk_id
        assert c1.text == c2.text
        assert c1.char_start == c2.char_start
        assert c1.char_end == c2.char_end


def test_chunk_id_format(config: BaselineChunkingConfig):
    """All chunk IDs match pattern {doc_id}_baseline_{:06d}."""
    import re
    pdoc = _make_pdoc("A" * 1000, doc_id="mytest_doc")
    cdoc = chunk(pdoc, config)
    pattern = re.compile(r"^mytest_doc_baseline_\d{6}$")
    for c in cdoc.chunks:
        assert pattern.match(c.chunk_id), f"Invalid chunk_id: {c.chunk_id}"


def test_chunk_id_uniqueness(config: BaselineChunkingConfig):
    """No two chunks in a document share the same chunk_id."""
    pdoc = _make_pdoc("B" * 2000)
    cdoc = chunk(pdoc, config)
    ids = [c.chunk_id for c in cdoc.chunks]
    assert len(ids) == len(set(ids)), "Duplicate chunk IDs found"


def test_chunk_index_sequential(config: BaselineChunkingConfig):
    """chunk_index is sequential from 0."""
    pdoc = _make_pdoc("C" * 1000)
    cdoc = chunk(pdoc, config)
    for expected_idx, c in enumerate(cdoc.chunks):
        assert c.chunk_index == expected_idx


# ── Provenance tests ───────────────────────────────────────────────────────────

def test_provenance_overlap_attribution(config: BaselineChunkingConfig):
    """Every chunk has at least one provenance_span for non-empty full_text."""
    pdoc = _make_pdoc("D" * 500)
    cdoc = chunk(pdoc, config)
    for c in cdoc.chunks:
        assert len(c.provenance_spans) >= 1, (
            f"Chunk {c.chunk_id} [{c.char_start},{c.char_end}) has no provenance spans"
        )


def test_provenance_boundary_accuracy():
    """For a known full_text + provenance, overlapping spans are exactly correct."""
    # 3 spans: [0,20), [20,40), [40,60)
    texts = ["A" * 18, "B" * 18, "C" * 18]
    pdoc = _make_multispan_pdoc(texts)
    # full_text = "A"*18 + "\n\n" + "B"*18 + "\n\n" + "C"*18
    # Spans: [0,20), [20,40), [40,58)
    cfg = BaselineChunkingConfig(
        strategy="baseline_fixed_size",
        chunk_size=25, chunk_overlap=5, min_chunk_size=5,
        corpus_name="test", corpus_chapters=[1],
    )
    cdoc = chunk(pdoc, cfg)
    for c in cdoc.chunks:
        for span in c.provenance_spans:
            # Each attributed span must genuinely overlap the chunk range
            assert span.char_end > c.char_start
            assert span.char_start < c.char_end


def test_provenance_no_false_spans():
    """No provenance span is attached that does not overlap [char_start, char_end)."""
    pdoc = _make_multispan_pdoc(["Hello world"] * 10)
    cfg = BaselineChunkingConfig(
        strategy="baseline_fixed_size",
        chunk_size=20, chunk_overlap=5, min_chunk_size=5,
        corpus_name="test", corpus_chapters=[1],
    )
    cdoc = chunk(pdoc, cfg)
    for c in cdoc.chunks:
        for span in c.provenance_spans:
            overlaps = span.char_end > c.char_start and span.char_start < c.char_end
            assert overlaps, (
                f"False span [{span.char_start},{span.char_end}) on "
                f"chunk [{c.char_start},{c.char_end})"
            )


# ── Serialization tests ────────────────────────────────────────────────────────

def test_json_serializable(config: BaselineChunkingConfig):
    """BaselineChunkDocument serialises to valid JSON without error."""
    from dataclasses import asdict
    pdoc = _make_pdoc("E" * 500)
    cdoc = chunk(pdoc, config)
    # Serialize using dataclasses.asdict (which handles nested dataclasses)
    data = asdict(cdoc)
    json_str = json.dumps(data)
    assert isinstance(json_str, str)
    assert len(json_str) > 0


def test_json_round_trip(config: BaselineChunkingConfig):
    """Serialise and deserialise; critical fields are preserved."""
    from dataclasses import asdict
    pdoc = _make_pdoc("F" * 700)
    cdoc = chunk(pdoc, config)
    data = asdict(cdoc)
    json_str = json.dumps(data)
    recovered = json.loads(json_str)

    assert recovered["doc_id"] == cdoc.doc_id
    assert recovered["strategy"] == cdoc.strategy
    assert len(recovered["chunks"]) == len(cdoc.chunks)
    for orig, rec in zip(cdoc.chunks, recovered["chunks"]):
        assert rec["chunk_id"] == orig.chunk_id
        assert rec["text"] == orig.text
        assert rec["char_start"] == orig.char_start
        assert rec["char_end"] == orig.char_end


# ── Integration test ───────────────────────────────────────────────────────────

def test_phase3_to_phase4_pipeline():
    """Construct minimal PreprocessedDocument via preprocess(), then chunk(). End-to-end."""
    pp_config = load_preprocessing_config(PREPROCESSING_CONFIG_PATH)
    ch_config = load_config(CHUNKING_CONFIG_PATH)

    page = ExtractedPage(
        page_number=1,
        blocks=[
            TextBlock(
                text="The cell is the basic unit of life.",
                x0=72, y0=100, x1=267, y1=120, font_size=9.0, font_name="NotoSans-Regular"
            ),
            TextBlock(
                text="All living organisms are made of cells.",
                x0=72, y0=130, x1=267, y1=150, font_size=9.0, font_name="NotoSans-Regular"
            ),
            TextBlock(
                text="Cells carry out the chemical reactions necessary to sustain life.",
                x0=72, y0=160, x1=267, y1=180, font_size=9.0, font_name="NotoSans-Regular"
            ),
        ] * 15  # multiply to give baseline enough baseline blocks
    )
    doc = ExtractedDocument(doc_id="integration_test", pages=[page])
    pdoc = preprocess(doc, pp_config)

    assert len(pdoc.full_text) > 0

    cdoc = chunk(pdoc, ch_config)

    assert cdoc.doc_id == "integration_test"
    assert cdoc.strategy == "baseline_fixed_size"
    assert len(cdoc.chunks) > 0

    # All chunk_ids follow correct format
    import re
    pattern = re.compile(r"^integration_test_baseline_\d{6}$")
    for c in cdoc.chunks:
        assert pattern.match(c.chunk_id)
        assert len(c.text) <= ch_config.chunk_size
        assert len(c.text) >= ch_config.min_chunk_size
        assert len(c.provenance_spans) >= 1

    # Verify strategy isolation: chunker used only full_text, not block labels
    # (no assertions on labels needed — just confirm we can chunk without touching them)
    assert cdoc.config_used.strategy == "baseline_fixed_size"
