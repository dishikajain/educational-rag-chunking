import pytest
import os
import sys
from unittest.mock import patch, MagicMock

@patch("scripts.index_corpus.extract_pdf")
@patch("scripts.index_corpus.preprocess")
@patch("scripts.index_corpus.baseline_chunk")
@patch("scripts.index_corpus.eluc_chunk")
@patch("scripts.index_corpus.embed_document")
@patch("scripts.index_corpus.index_document")
@patch("scripts.index_corpus.fitz.open")
def test_run_indexing_orchestration(mock_fitz_open, mock_index, mock_embed, mock_eluc, mock_baseline, mock_preprocess, mock_extract, tmp_path, monkeypatch):
    import scripts.index_corpus
    from scripts.index_corpus import run_indexing
    
    fake_pdf = tmp_path / "fake.pdf"
    fake_pdf.write_text("fake pdf")
    monkeypatch.setattr(scripts.index_corpus, "PDF_PATH", str(fake_pdf))
    
    try:
        run_indexing()
    except SystemExit:
        pass
        
    mock_fitz_open.assert_called()
    mock_extract.assert_called_once()
    mock_preprocess.assert_called_once()
    mock_baseline.assert_called_once()
    mock_eluc.assert_called_once()
    
    # 1. Assert fitz.open().insert_pdf() was called for Chapters 1-5 (Pages 31-171, i.e., 30-170 0-indexed)
    src_doc_mock = mock_fitz_open.return_value.__enter__.return_value
    mock_fitz_open.return_value.insert_pdf.assert_called_once_with(
        src_doc_mock, 
        from_page=30, 
        to_page=170
    )
    
    # 2. Assert both index_document calls received the persistent ChromaDB path
    assert mock_index.call_count == 2
    for call in mock_index.call_args_list:
        assert call.args[3] == "data/chroma_db"
    assert mock_embed.call_count == 2
    
def test_run_indexing_missing_pdf(tmp_path, monkeypatch):
    import scripts.index_corpus
    from scripts.index_corpus import run_indexing
    
    monkeypatch.setattr(scripts.index_corpus, "PDF_PATH", str(tmp_path / "nonexistent.pdf"))
    with pytest.raises(SystemExit) as exc:
        run_indexing()
    assert exc.value.code == 1
