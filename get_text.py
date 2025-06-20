import fitz  # PyMuPDF
import os
import re
import unicodedata
from collections import OrderedDict

def extract_text_from_pdf(pdf_path):
    """Extract raw text from a PDF file, page by page."""
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            full_text += page.get_text("text")
        doc.close()
        return full_text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return None

def clean_extracted_pdf_text(text):
    """
    Normalize Unicode, fix hyphens, collapse excess blank lines,
    and remove stray control characters—preserving actual newlines.
    """
    if text is None:
        return None
    # 1) Unicode normalization
    text = unicodedata.normalize('NFKC', text)
    # 2) Rejoin hyphenated line-breaks (handles both LF and CRLF)
    text = re.sub(r'-\r?\n([A-Za-z])', r'\1', text)
    # 3) Collapse runs of 3+ blank lines into exactly two
    text = re.sub(r'(?:\r?\n){3,}', '\n\n', text)
    # 4) Remove stray control characters
    text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]', '', text)
    # 5) Collapse multiple spaces
    text = re.sub(r' +', ' ', text)
    return text.strip()

def split_into_chapters(text):
    """
    Split on each line that begins with the spaced-out 'C H A P T E R <number>'.
    Everything before the first such line is 'preamble'.
    """
    chapters = OrderedDict()

    # 1) Use lookahead to split before any line starting with 'C H A P T E R <n>'
    parts = re.split(
        r'(?m)(?=^C\s+H\s+A\s+P\s+T\s+E\s+R\s+\d+\b)',
        text
    )

    # 2) First part might be preamble if it doesn't start with C H A P T E R
    if parts and not parts[0].lstrip().startswith('C H A P T E R'):
        pre = parts.pop(0).strip()
        if pre:
            chapters['preamble'] = pre

    # 3) For each remaining part, extract the chapter number and text
    for part in parts:
        m = re.match(r'^\s*C\s+H\s+A\s+P\s+T\s+E\s+R\s+(\d+)\b', part)
        if not m:
            continue
        num = m.group(1)
        key = f'chapter_{num}'
        chapters[key] = part.strip()

    return chapters

def write_text_to_file(file_path, text_content):
    """Writes text content to a file, creating directories if needed."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text_content)

def process_pdfs_to_text_chapters(pdf_input_dir, text_output_dir):
    """
    Extracts text from PDFs in pdf_input_dir, cleans it,
    splits only on spaced-out 'C H A P T E R' headings,
    and writes each chunk to its own .txt file.
    """
    if not os.path.exists(pdf_input_dir):
        print(f"Error: PDF input directory '{pdf_input_dir}' not found.")
        return
    os.makedirs(text_output_dir, exist_ok=True)

    for fname in os.listdir(pdf_input_dir):
        if not fname.lower().endswith('.pdf'):
            continue
        pdf_path = os.path.join(pdf_input_dir, fname)
        raw = extract_text_from_pdf(pdf_path)
        if not raw:
            continue
        cleaned = clean_extracted_pdf_text(raw)
        chapters = split_into_chapters(cleaned)
        base = os.path.splitext(fname)[0]
        for key, content in chapters.items():
            out_name = f"{base}_{key}.txt"
            out_path = os.path.join(text_output_dir, out_name)
            write_text_to_file(out_path, content)
            print(f"Wrote {out_name}")

if __name__ == '__main__':
    process_pdfs_to_text_chapters('pdf_textbooks', 'output_chapters')
