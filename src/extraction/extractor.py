import pymupdf as fitz  # PyMuPDF
from .models import ExtractedDocument, ExtractedPage, TextBlock

def extract_pdf(file_path: str, doc_id: str) -> ExtractedDocument:
    """
    Extracts structured text from a PDF file using PyMuPDF.
    Preserves the raw extraction order and text structure (lines are separated by \n).
    Reading order normalization and preprocessing are deferred to Phase 3.
    """
    try:
        with fitz.open(file_path) as doc:
            extracted_pages = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                blocks_data = page.get_text("dict").get("blocks", [])
                
                text_blocks = []
                for b in blocks_data:
                    if b.get("type") == 0:  # Text block
                        lines = []
                        font_size = None
                        font_name = None
                        
                        for line in b.get("lines", []):
                            line_text = ""
                            for span in line.get("spans", []):
                                line_text += span.get("text", "")
                                if font_size is None:
                                    font_size = span.get("size")
                                    font_name = span.get("font")
                            lines.append(line_text)
                        
                        block_text = "\n".join(lines).strip()
                        if block_text:
                            bbox = b.get("bbox", (0.0, 0.0, 0.0, 0.0))
                            text_blocks.append(TextBlock(
                                text=block_text,
                                x0=bbox[0],
                                y0=bbox[1],
                                x1=bbox[2],
                                y1=bbox[3],
                                font_size=font_size,
                                font_name=font_name
                            ))
                            
                extracted_pages.append(ExtractedPage(
                    page_number=page_num + 1,
                    blocks=text_blocks
                ))
            
            return ExtractedDocument(doc_id=doc_id, pages=extracted_pages)
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError(f"Failed to open or process PDF file {file_path}: {e}")
