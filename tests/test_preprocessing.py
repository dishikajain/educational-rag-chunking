import pytest
import os
from src.extraction.models import ExtractedDocument, ExtractedPage, TextBlock
from src.preprocessing.config import load_config, PreprocessingConfig
from src.preprocessing.preprocessor import preprocess


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "preprocessing_config.yaml")


@pytest.fixture
def config() -> PreprocessingConfig:
    return load_config(CONFIG_PATH)


def test_load_config(config: PreprocessingConfig):
    assert config.corpus_name == "openstax_biology2e"
    assert config.body_size_fallback == 9.0
    assert config.header_zone_height == 29.1
    assert config.footer_zone_height == 38.5
    assert "VISUAL CONNECTION" in config.sidebar_labels


def test_provenance_invariants_and_reconstruction(config: PreprocessingConfig):
    """Verify Provenance Invariants A-D and exact full_text reconstruction."""
    page1 = ExtractedPage(
        page_number=1,
        blocks=[
            TextBlock(text="Chapter 1 Title", x0=72, y0=100, x1=532, y1=130, font_size=34.0, font_name="NotoSans-Bold"),
            TextBlock(text="This is paragraph one.", x0=72, y0=150, x1=267, y1=180, font_size=9.0, font_name="NotoSans-Regular"),
            TextBlock(text="This is paragraph two.", x0=72, y0=190, x1=267, y1=220, font_size=9.0, font_name="NotoSans-Regular"),
        ],
    )
    doc = ExtractedDocument(doc_id="test_doc", pages=[page1])
    pdoc = preprocess(doc, config)

    # Invariant B: start is 0
    assert pdoc.provenance[0].char_start == 0
    # Invariant C: final span char_end == len(full_text)
    assert pdoc.provenance[-1].char_end == len(pdoc.full_text)

    # Invariant A: contiguous
    for i in range(len(pdoc.provenance) - 1):
        assert pdoc.provenance[i].char_end == pdoc.provenance[i + 1].char_start

    # Invariant D: exact reconstruction
    reconstructed = "".join(pdoc.full_text[s.char_start:s.char_end] for s in pdoc.provenance)
    assert reconstructed == pdoc.full_text


def test_dehyphenation_and_next_block_preservation(config: PreprocessingConfig):
    """Test de-hyphenation across blocks and verify next block text is unchanged."""
    page1 = ExtractedPage(
        page_number=1,
        blocks=[
            TextBlock(text="Thermal energy and thermo-", x0=72, y0=100, x1=267, y1=120, font_size=9.0, font_name="NotoSans-Regular"),
            TextBlock(text="dynamics are related.", x0=72, y0=130, x1=267, y1=150, font_size=9.0, font_name="NotoSans-Regular"),
        ],
    )
    doc = ExtractedDocument(doc_id="hyphen_doc", pages=[page1])
    pdoc = preprocess(doc, config)

    blocks = pdoc.pages[0].blocks
    assert len(blocks) == 2
    # block 0 trailing hyphen removed
    assert blocks[0].text == "Thermal energy and thermo"
    # block 1 text UNCHANGED
    assert blocks[1].text == "dynamics are related."

    # full_text joins them with zero-width separator
    assert "thermodynamics are related." in pdoc.full_text

    # Invariants hold
    assert pdoc.provenance[0].char_start == 0
    assert pdoc.provenance[-1].char_end == len(pdoc.full_text)
    assert pdoc.provenance[0].char_end == pdoc.provenance[1].char_start
    reconstructed = "".join(pdoc.full_text[s.char_start:s.char_end] for s in pdoc.provenance)
    assert reconstructed == pdoc.full_text


def test_heading_classification(config: PreprocessingConfig):
    """Test classification of headings based on font size ratios."""
    page = ExtractedPage(
        page_number=1,
        blocks=[
            TextBlock(text="Section 1.1", x0=72, y0=100, x1=267, y1=120, font_size=15.6, font_name="NotoSans-Bold"), # HEADING_2 ratio ~1.73
            TextBlock(text="Subsection Title", x0=72, y0=130, x1=267, y1=145, font_size=13.0, font_name="NotoSans-Bold"), # HEADING_3 ratio ~1.44
            TextBlock(text="Standard body text goes here.", x0=72, y0=150, x1=267, y1=180, font_size=9.0, font_name="NotoSans-Regular"), # BODY
            TextBlock(text="Bold key term", x0=72, y0=190, x1=267, y1=205, font_size=9.0, font_name="NotoSans-Bold"), # BODY_BOLD
            TextBlock(text="Figure 1.1 Caption", x0=72, y0=210, x1=267, y1=220, font_size=7.5, font_name="NotoSans-Regular"), # CAPTION
        ]
    )
    doc = ExtractedDocument(doc_id="heading_doc", pages=[page])
    pdoc = preprocess(doc, config)

    labels = [b.label for b in pdoc.pages[0].blocks]
    assert labels[0] == "HEADING_2"
    assert labels[1] == "HEADING_3"
    assert labels[2] == "BODY"
    assert labels[3] == "BODY_BOLD"
    assert labels[4] == "CAPTION"


def test_chapter_opener_behavior(config: PreprocessingConfig):
    """Test merged chapter opener block behavior."""
    page = ExtractedPage(
        page_number=1,
        blocks=[
            # Merged chapter title block at ratio ~1.73 (15.55pt / 9.0pt) -> HEADING_2
            TextBlock(text="CHAPTER 1 The Study of Life", x0=72, y0=100, x1=532, y1=130, font_size=15.55, font_name="NotoSans-Bold"),
            # High ratio numeral (ratio = 38.7/9.0 = 4.3 -> CHAPTER_NUMBER)
            TextBlock(text="1", x0=72, y0=150, x1=100, y1=200, font_size=38.7, font_name="NotoSans-Bold"),
            # Body blocks to ensure baseline 9.0 (mode)
            TextBlock(text="This is standard body text on the chapter opener page line 1.", x0=72, y0=210, x1=532, y1=240, font_size=9.0, font_name="NotoSans-Regular"),
            TextBlock(text="This is standard body text on the chapter opener page line 2.", x0=72, y0=250, x1=532, y1=280, font_size=9.0, font_name="NotoSans-Regular"),
            TextBlock(text="This is standard body text on the chapter opener page line 3.", x0=72, y0=290, x1=532, y1=320, font_size=9.0, font_name="NotoSans-Regular"),
        ]
    )
    doc = ExtractedDocument(doc_id="ch_doc", pages=[page])
    pdoc = preprocess(doc, config)

    blocks = pdoc.pages[0].blocks
    assert blocks[0].label == "HEADING_2"
    assert blocks[1].label == "CHAPTER_NUMBER"
    assert blocks[2].label == "BODY"


def test_header_footer_removal(config: PreprocessingConfig):
    """Test removal of running header and footer blocks outside content area."""
    page = ExtractedPage(
        page_number=2,
        blocks=[
            # Running header at y0=24.1 (top zone height is 29.1)
            TextBlock(text="CHAPTER 1 • THE STUDY OF LIFE", x0=72, y0=24.1, x1=532, y1=35.0, font_size=9.0, font_name="NotoSans-Regular"),
            # Main content
            TextBlock(text="Main body text of the chapter.", x0=72, y0=100, x1=267, y1=150, font_size=9.0, font_name="NotoSans-Regular"),
            # Footer at y1=760 (bottom zone y1 > 792 - 38.5 = 753.5)
            TextBlock(text="Access for free at openstax.org", x0=72, y0=758.5, x1=267, y1=770.0, font_size=9.0, font_name="NotoSans-Regular"),
        ]
    )
    doc = ExtractedDocument(doc_id="hf_doc", pages=[page])
    pdoc = preprocess(doc, config)

    # Header and footer must be removed from output blocks
    blocks = pdoc.pages[0].blocks
    assert len(blocks) == 1
    assert blocks[0].text == "Main body text of the chapter."


def test_two_column_reading_order(config: PreprocessingConfig):
    """Test drain-and-interleave reading order reconstruction on two-column page."""
    page = ExtractedPage(
        page_number=3,
        blocks=[
            # Top full width heading
            TextBlock(text="Full Width Header", x0=72, y0=100, x1=532, y1=120, font_size=15.6, font_name="NotoSans-Bold"),
            # Left col block 1
            TextBlock(text="Left 1", x0=72, y0=130, x1=267, y1=160, font_size=9.0, font_name="NotoSans-Regular"),
            # Right col block 1 (visually lower y0 than Left 2)
            TextBlock(text="Right 1", x0=317, y0=140, x1=532, y1=170, font_size=9.0, font_name="NotoSans-Regular"),
            # Left col block 2
            TextBlock(text="Left 2", x0=72, y0=170, x1=267, y1=200, font_size=9.0, font_name="NotoSans-Regular"),
            # Left col block 3
            TextBlock(text="Left 3", x0=72, y0=210, x1=267, y1=240, font_size=9.0, font_name="NotoSans-Regular"),
            # Right col block 2
            TextBlock(text="Right 2", x0=317, y0=180, x1=532, y1=210, font_size=9.0, font_name="NotoSans-Regular"),
            # Right col block 3
            TextBlock(text="Right 3", x0=317, y0=220, x1=532, y1=250, font_size=9.0, font_name="NotoSans-Regular"),
        ]
    )
    doc = ExtractedDocument(doc_id="twocol_doc", pages=[page])
    pdoc = preprocess(doc, config)

    blocks = pdoc.pages[0].blocks
    texts = [b.text for b in blocks]
    assert pdoc.pages[0].page_type == "TWO_COLUMN"

    # Full width header comes first
    assert texts[0] == "Full Width Header"
    # Left column drained before right column
    assert texts[1:4] == ["Left 1", "Left 2", "Left 3"]
    assert texts[4:7] == ["Right 1", "Right 2", "Right 3"]


def test_sidebar_callout_grouping(config: PreprocessingConfig):
    """Test detection and grouping of sidebars into SidebarUnits."""
    page = ExtractedPage(
        page_number=4,
        blocks=[
            # Sidebar header block
            TextBlock(text="VISUAL CONNECTION", x0=72, y0=100, x1=200, y1=115, font_size=10.8, font_name="NotoSans-Bold"),
            # Sidebar body block 1
            TextBlock(text="Look at this visual diagram carefully.", x0=72, y0=120, x1=532, y1=150, font_size=9.0, font_name="NotoSans-Regular"),
            # Sidebar body block 2
            TextBlock(text="What process is being illustrated?", x0=72, y0=160, x1=532, y1=180, font_size=9.0, font_name="NotoSans-Regular"),
            # Regular body after sidebar
            TextBlock(text="After answering the visual connection question, read on.", x0=72, y0=200, x1=267, y1=230, font_size=9.0, font_name="NotoSans-Regular"),
        ]
    )
    doc = ExtractedDocument(doc_id="sidebar_doc", pages=[page])
    pdoc = preprocess(doc, config)

    blocks = pdoc.pages[0].blocks
    assert len(blocks) == 2

    sidebar_block = blocks[0]
    assert sidebar_block.label == "SIDEBAR_BODY"
    assert sidebar_block.sidebar_type == "VISUAL CONNECTION"
    assert sidebar_block.sidebar_unit_id is not None
    assert "VISUAL CONNECTION\nLook at this visual diagram carefully.\nWhat process is being illustrated?" in sidebar_block.text
    assert len(sidebar_block.source_blocks) == 3

    assert blocks[1].text == "After answering the visual connection question, read on."


def test_gap_classification(config: PreprocessingConfig):
    """Test line, paragraph, and section gap classification."""
    page = ExtractedPage(
        page_number=5,
        blocks=[
            # Line continuation (gap = 100 - 90 = 10 -> line gap, gap_line_pt is 0.9, gap < 0.9 * 1.3 = 1.17 -> False)
            # Let's create specific tight gaps:
            TextBlock(text="Line 1", x0=72, y0=100.0, x1=267, y1=110.0, font_size=9.0, font_name="NotoSans-Regular"),
            TextBlock(text="Line 2", x0=72, y0=110.5, x1=267, y1=120.5, font_size=9.0, font_name="NotoSans-Regular"), # gap = 0.5pt -> line gap
            TextBlock(text="Paragraph 2", x0=72, y0=129.0, x1=267, y1=140.0, font_size=9.0, font_name="NotoSans-Regular"), # gap = 129 - 120.5 = 8.5pt -> para gap
            TextBlock(text="Section 2", x0=72, y0=160.0, x1=267, y1=175.0, font_size=15.6, font_name="NotoSans-Bold"), # gap = 160 - 140 = 20pt -> section gap
        ]
    )
    doc = ExtractedDocument(doc_id="gap_doc", pages=[page])
    pdoc = preprocess(doc, config)

    blocks = pdoc.pages[0].blocks
    assert blocks[0].gap_before_pt is None

    # Line 2
    assert blocks[1].gap_before_pt == pytest.approx(0.5)
    assert blocks[1].is_paragraph_break is False
    assert blocks[1].is_section_break is False

    # Paragraph 2
    assert blocks[2].gap_before_pt == pytest.approx(8.5)
    assert blocks[2].is_paragraph_break is True
    assert blocks[2].is_section_break is False

    # Section 2
    assert blocks[3].gap_before_pt == pytest.approx(20.0)
    assert blocks[3].is_paragraph_break is True
    assert blocks[3].is_section_break is True


def test_multipage_documents_and_determinism(config: PreprocessingConfig):
    """Test processing of multipage documents and verify output is 100% deterministic."""
    page1 = ExtractedPage(
        page_number=1,
        blocks=[
            TextBlock(text="VISUAL CONNECTION", x0=72, y0=100, x1=200, y1=115, font_size=10.8, font_name="NotoSans-Bold"),
            TextBlock(text="Page 1 content.", x0=72, y0=120, x1=532, y1=150, font_size=9.0, font_name="NotoSans-Regular"),
        ]
    )
    page2 = ExtractedPage(
        page_number=2,
        blocks=[
            TextBlock(text="Page 2 content.", x0=72, y0=100, x1=267, y1=130, font_size=9.0, font_name="NotoSans-Regular"),
        ]
    )
    doc = ExtractedDocument(doc_id="multi_doc", pages=[page1, page2])

    pdoc1 = preprocess(doc, config)
    pdoc2 = preprocess(doc, config)

    assert len(pdoc1.pages) == 2
    assert pdoc1.pages[0].page_number == 1
    assert pdoc1.pages[1].page_number == 2

    # Determinism test including exact sidebar_unit_id equality
    assert pdoc1.full_text == pdoc2.full_text
    assert pdoc1.provenance == pdoc2.provenance
    assert pdoc1.pages[0].blocks[0].sidebar_unit_id == pdoc2.pages[0].blocks[0].sidebar_unit_id
    assert pdoc1.pages[0].blocks[0].sidebar_unit_id == "sidebar_p1_b0"


def test_dehyphenation_does_not_cross_page_boundary(config: PreprocessingConfig):
    """Verify de-hyphenation does NOT cross page boundaries."""
    page1 = ExtractedPage(
        page_number=1,
        blocks=[
            TextBlock(text="Thermal energy and thermo-", x0=72, y0=100, x1=267, y1=120, font_size=9.0, font_name="NotoSans-Regular"),
        ],
    )
    page2 = ExtractedPage(
        page_number=2,
        blocks=[
            TextBlock(text="dynamics are related.", x0=72, y0=100, x1=267, y1=120, font_size=9.0, font_name="NotoSans-Regular"),
        ],
    )
    doc = ExtractedDocument(doc_id="cross_page_hyphen_doc", pages=[page1, page2])
    pdoc = preprocess(doc, config)

    # Hyphen must NOT be removed across page boundary
    assert pdoc.pages[0].blocks[0].text == "Thermal energy and thermo-"
    assert pdoc.pages[1].blocks[0].text == "dynamics are related."
    # Standard paragraph/page separator used (\n\n)
    assert pdoc.full_text == "Thermal energy and thermo-\n\ndynamics are related."

