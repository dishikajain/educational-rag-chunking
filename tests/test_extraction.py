import pytest
import pymupdf as fitz
from src.extraction.extractor import extract_pdf

@pytest.fixture
def dummy_pdf(tmp_path):
    pdf_path = tmp_path / "dummy.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Hello World", fontsize=12, fontname="helv")
    doc.save(str(pdf_path))
    doc.close()
    return str(pdf_path)

@pytest.fixture
def multipage_pdf(tmp_path):
    pdf_path = tmp_path / "multipage.pdf"
    doc = fitz.open()
    
    # Page 1
    page1 = doc.new_page()
    page1.insert_text((50, 50), "Page One Content", fontsize=12, fontname="helv")
    
    # Page 2
    page2 = doc.new_page()
    page2.insert_text((50, 50), "Page Two Content", fontsize=12, fontname="helv")
    
    doc.save(str(pdf_path))
    doc.close()
    return str(pdf_path)

@pytest.fixture
def corrupted_pdf(tmp_path):
    pdf_path = tmp_path / "corrupted.pdf"
    with open(pdf_path, "w") as f:
        f.write("This is not a valid PDF content.")
    return str(pdf_path)

def test_extract_pdf_success(dummy_pdf):
    extracted_doc = extract_pdf(dummy_pdf, "test_doc_01")
    assert extracted_doc.doc_id == "test_doc_01"
    assert len(extracted_doc.pages) == 1
    page = extracted_doc.pages[0]
    assert page.page_number == 1
    assert len(page.blocks) > 0
    block = page.blocks[0]
    assert "Hello World" in block.text
    assert block.font_size is not None

def test_extract_pdf_multipage(multipage_pdf):
    extracted_doc = extract_pdf(multipage_pdf, "multipage_doc")
    assert extracted_doc.doc_id == "multipage_doc"
    assert len(extracted_doc.pages) == 2
    
    page1 = extracted_doc.pages[0]
    assert page1.page_number == 1
    assert any("Page One" in b.text for b in page1.blocks)
    assert not any("Page Two" in b.text for b in page1.blocks)
    
    page2 = extracted_doc.pages[1]
    assert page2.page_number == 2
    assert any("Page Two" in b.text for b in page2.blocks)
    assert not any("Page One" in b.text for b in page2.blocks)

def test_extract_pdf_invalid_file():
    with pytest.raises(ValueError):
        extract_pdf("nonexistent_file.pdf", "test_doc_02")

def test_extract_pdf_corrupted_file(corrupted_pdf):
    with pytest.raises(ValueError) as excinfo:
        extract_pdf(corrupted_pdf, "corrupted_doc")
    assert "Failed to open or process PDF file" in str(excinfo.value)
