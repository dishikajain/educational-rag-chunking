import pytest
import os
import sys
from unittest.mock import patch, MagicMock

@patch("scripts.index_corpus.extract_pdf")
@patch("scripts.index_corpus.preprocess")
@patch("scripts.index_corpus.baseline_chunk")
@patch("scripts.index_corpus.eluc_chunk")
@patch("scripts.index_corpus.recursive_chunk")
@patch("scripts.index_corpus.paragraph_chunk")
@patch("scripts.index_corpus.heading_chunk")
@patch("scripts.index_corpus.save_chunks")
@patch("scripts.index_corpus.embed_document")
@patch("scripts.index_corpus.index_document")
@patch("scripts.index_corpus.fitz.open")
def test_run_indexing_orchestration(mock_fitz_open, mock_index, mock_embed, mock_save_chunks, mock_heading, mock_paragraph, mock_recursive, mock_eluc, mock_baseline, mock_preprocess, mock_extract, tmp_path, monkeypatch):
    import scripts.index_corpus
    from scripts.index_corpus import run_indexing

    fake_pdf = tmp_path / "fake.pdf"
    fake_pdf.write_text("fake pdf")
    monkeypatch.setattr(scripts.index_corpus, "PDF_PATH", str(fake_pdf))

    mock_baseline.return_value.strategy = "baseline_fixed_size"
    mock_baseline.return_value.chunks = []

    mock_eluc.return_value.strategy = "eluc_section_aware"
    mock_eluc.return_value.chunks = []

    mock_recursive.return_value.strategy = "recursive_character"
    mock_recursive.return_value.chunks = []

    mock_paragraph.return_value.strategy = "paragraph_based"
    mock_paragraph.return_value.chunks = []

    mock_heading.return_value.strategy = "heading_based"
    mock_heading.return_value.chunks = []

    mock_index.side_effect = [MagicMock()] * 5

    try:
        run_indexing()
    except SystemExit:
        pass

    mock_fitz_open.assert_called()
    mock_extract.assert_called_once()
    mock_preprocess.assert_called_once()
    mock_baseline.assert_called_once()
    mock_eluc.assert_called_once()
    mock_recursive.assert_called_once()
    mock_paragraph.assert_called_once()
    mock_heading.assert_called_once()
    assert mock_save_chunks.call_count == 5

    src_doc_mock = mock_fitz_open.return_value.__enter__.return_value
    mock_fitz_open.return_value.insert_pdf.assert_called_once_with(
        src_doc_mock,
        from_page=28,
        to_page=174
    )

    assert mock_index.call_count == 5
    assert mock_embed.call_count == 5

def test_run_indexing_missing_pdf(tmp_path, monkeypatch):
    import scripts.index_corpus
    from scripts.index_corpus import run_indexing

    monkeypatch.setattr(scripts.index_corpus, "PDF_PATH", str(tmp_path / "nonexistent.pdf"))
    with pytest.raises(SystemExit) as exc:
        run_indexing()
    assert exc.value.code == 1
