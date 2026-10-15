import docx
import json
import sys

def analyze_docx(filepath):
    doc = docx.Document(filepath)
    info = {
        'sections': [],
        'paragraphs_sample': [],
        'styles': {}
    }
    
    for section in doc.sections:
        info['sections'].append({
            'page_width': section.page_width.inches if section.page_width else None,
            'page_height': section.page_height.inches if section.page_height else None,
            'left_margin': section.left_margin.inches if section.left_margin else None,
            'right_margin': section.right_margin.inches if section.right_margin else None,
            'top_margin': section.top_margin.inches if section.top_margin else None,
            'bottom_margin': section.bottom_margin.inches if section.bottom_margin else None,
        })
        
    for i, p in enumerate(doc.paragraphs):
        if p.text.strip():
            style_name = p.style.name if p.style else "No Style"
            if style_name not in info['styles']:
                font = p.style.font
                info['styles'][style_name] = {
                    'font_name': font.name,
                    'font_size': font.size.pt if font.size else None,
                    'bold': font.bold,
                    'italic': font.italic
                }
            if len(info['paragraphs_sample']) < 30:
                info['paragraphs_sample'].append({
                    'text': p.text[:100],
                    'style': style_name,
                    'alignment': str(p.alignment)
                })
                
    return info

if __name__ == "__main__":
    info = analyze_docx(sys.argv[1])
    print(json.dumps(info, indent=2))
