"""
Document Processor - Handles reading and chunking different document types.
Supports: PDF, DOCX, XLSX, TXT
"""
import os
import re
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


def _normalize_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph boundaries."""
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    paragraphs = []
    current = []

    for line in lines:
        if not line:
            if current:
                paragraphs.append(" ".join(current).strip())
                current = []
            continue
        current.append(line)

    if current:
        paragraphs.append(" ".join(current).strip())

    return "\n\n".join(paragraphs)


def _split_long_unit(unit: str, max_size: int) -> List[str]:
    """Split oversized text unit by sentence boundaries, fallback to hard split."""
    if len(unit) <= max_size:
        return [unit]

    sentence_parts = [part.strip() for part in re.split(r"(?<=[\.!\?;:])\s+", unit) if part.strip()]
    if len(sentence_parts) <= 1:
        return [unit[i:i + max_size].strip() for i in range(0, len(unit), max_size) if unit[i:i + max_size].strip()]

    result = []
    current = ""
    for sentence in sentence_parts:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_size:
            current = candidate
            continue

        if current:
            result.append(current)
        if len(sentence) <= max_size:
            current = sentence
        else:
            hard_parts = [sentence[i:i + max_size].strip() for i in range(0, len(sentence), max_size) if sentence[i:i + max_size].strip()]
            result.extend(hard_parts[:-1])
            current = hard_parts[-1] if hard_parts else ""

    if current:
        result.append(current)

    return result


def _build_chunks_from_units(units: List[str], chunk_size: int, overlap: int) -> List[str]:
    """Assemble units into overlap-aware chunks."""
    if not units:
        return []

    overlap_units = max(1, overlap // 120) if overlap > 0 else 0
    chunks = []
    current_units: List[str] = []
    current_len = 0

    for unit in units:
        unit_len = len(unit)
        additional_len = unit_len + (1 if current_units else 0)

        if current_units and current_len + additional_len > chunk_size:
            chunks.append(" ".join(current_units).strip())
            if overlap_units > 0:
                current_units = current_units[-overlap_units:]
                current_len = len(" ".join(current_units))
            else:
                current_units = []
                current_len = 0

        current_units.append(unit)
        current_len = len(" ".join(current_units))

    if current_units:
        chunks.append(" ".join(current_units).strip())

    return chunks


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into semantically coherent overlapping chunks."""
    normalized = _normalize_text(text)
    if not normalized:
        return []
    if len(normalized) <= chunk_size:
        return [normalized]

    paragraph_units = [p.strip() for p in re.split(r"\n\n+", normalized) if p.strip()]
    expanded_units: List[str] = []
    for unit in paragraph_units:
        expanded_units.extend(_split_long_unit(unit, chunk_size))

    return _build_chunks_from_units(expanded_units, chunk_size, overlap)


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
