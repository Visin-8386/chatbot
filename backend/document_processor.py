"""
Document Processor - Handles reading and chunking different document types.
Supports: PDF, DOCX, XLSX, TXT
"""
import os
from typing import List, Dict
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from openpyxl import load_workbook
from backend.config import CHUNK_SIZE, CHUNK_OVERLAP, SUPPORTED_EXTENSIONS


def extract_text_from_pdf(file_path: str) -> List[Dict]:
    """Extract text from PDF file, page by page."""
    pages = []
    reader = PdfReader(file_path)
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append({
                "text": text.strip(),
                "metadata": {"page": i + 1}
            })
    return pages


def extract_text_from_docx(file_path: str) -> List[Dict]:
    """Extract text from Word document."""
    doc = DocxDocument(file_path)
    full_text = []
    for para in doc.paragraphs:
        if para.text.strip():
            full_text.append(para.text.strip())

    # Also extract from tables
    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
            if row_text:
                full_text.append(row_text)

    text = "\n".join(full_text)
    if text:
        return [{"text": text, "metadata": {}}]
    return []


def extract_text_from_xlsx(file_path: str) -> List[Dict]:
    """Extract text from Excel file, sheet by sheet."""
    sheets = []
    wb = load_workbook(file_path, read_only=True, data_only=True)
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        rows = []
        for row in ws.iter_rows(values_only=True):
            row_text = " | ".join(str(cell) for cell in row if cell is not None)
            if row_text.strip():
                rows.append(row_text)
        if rows:
            text = "\n".join(rows)
            sheets.append({
                "text": text,
                "metadata": {"sheet": sheet_name}
            })
    wb.close()
    return sheets


def extract_text_from_txt(file_path: str) -> List[Dict]:
    """Extract text from plain text file."""
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc) as f:
                text = f.read().strip()
            if text:
                return [{"text": text, "metadata": {}}]
            return []
        except (UnicodeDecodeError, UnicodeError):
            continue
    return []


EXTRACTORS = {
    ".pdf": extract_text_from_pdf,
    ".docx": extract_text_from_docx,
    ".xlsx": extract_text_from_xlsx,
    ".txt": extract_text_from_txt,
}


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size

        # Try to break at a sentence or newline boundary
        if end < len(text):
            # Look for the last sentence-ending punctuation within the chunk
            for sep in ["\n\n", "\n", ". ", "! ", "? ", "; "]:
                last_sep = text[start:end].rfind(sep)
                if last_sep != -1 and last_sep > chunk_size * 0.3:
                    end = start + last_sep + len(sep)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks


def process_document(file_path: str) -> List[Dict]:
    """
    Process a document file and return a list of chunks with metadata.
    
    Returns:
        List of dicts: [{"text": "...", "metadata": {"source": "file.pdf", "page": 1, ...}}, ...]
    """
    ext = os.path.splitext(file_path)[1].lower()
    filename = os.path.basename(file_path)
    # Strip doc_id prefix (format: "abc12345_originalname.ext")
    if "_" in filename:
        filename = filename.split("_", 1)[1]

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {SUPPORTED_EXTENSIONS}")

    extractor = EXTRACTORS.get(ext)
    if not extractor:
        raise ValueError(f"No extractor for file type: {ext}")

    # Extract text sections
    sections = extractor(file_path)

    # Chunk each section
    all_chunks = []
    for section in sections:
        text = section["text"]
        meta = section["metadata"]
        meta["source"] = filename

        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            chunk_meta = {**meta, "chunk_index": i}
            all_chunks.append({
                "text": chunk,
                "metadata": chunk_meta
            })

    return all_chunks
