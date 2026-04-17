"""Document parsing helpers for KnowFlow.

The parser module keeps all file handling away from the Streamlit UI. Each
function returns friendly errors instead of raising raw exceptions into the app.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


@dataclass
class ParsedDocument:
    """Text extracted from one uploaded or bundled source."""

    filename: str
    text: str
    error: str | None = None


def clean_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph boundaries."""
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_txt(file_obj: BinaryIO, filename: str) -> ParsedDocument:
    """Parse a plain text upload using forgiving UTF-8 decoding."""
    try:
        raw = file_obj.read()
        if isinstance(raw, str):
            text = raw
        else:
            text = raw.decode("utf-8", errors="ignore")
        text = clean_text(text)
        if not text:
            return ParsedDocument(filename, "", "The text file was empty.")
        return ParsedDocument(filename, text)
    except Exception:
        return ParsedDocument(filename, "", "Could not read this text file.")


def parse_pdf(file_obj: BinaryIO, filename: str) -> ParsedDocument:
    """Parse a PDF with PyMuPDF first, then pdfplumber as a fallback."""
    data = file_obj.read()
    if not data:
        return ParsedDocument(filename, "", "The PDF file was empty.")

    try:
        import fitz  # PyMuPDF

        pages: list[str] = []
        with fitz.open(stream=data, filetype="pdf") as doc:
            for page in doc:
                pages.append(page.get_text("text"))
        text = clean_text("\n\n".join(pages))
        if text:
            return ParsedDocument(filename, text)
    except Exception:
        pass

    try:
        import io
        import pdfplumber

        pages = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                pages.append(page.extract_text() or "")
        text = clean_text("\n\n".join(pages))
        if text:
            return ParsedDocument(filename, text)
    except Exception:
        pass

    return ParsedDocument(
        filename,
        "",
        "Could not extract text from this PDF. It may be scanned or image-only.",
    )


def parse_uploaded_file(uploaded_file) -> ParsedDocument:
    """Route a Streamlit upload to the right parser."""
    filename = getattr(uploaded_file, "name", "uploaded_file")
    extension = Path(filename).suffix.lower()
    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    if extension == ".txt":
        return parse_txt(uploaded_file, filename)
    if extension == ".pdf":
        return parse_pdf(uploaded_file, filename)
    return ParsedDocument(filename, "", "Unsupported file type. Please upload PDF or TXT.")


def parse_local_text_file(path: Path) -> ParsedDocument:
    """Read a bundled demo text file."""
    try:
        text = clean_text(path.read_text(encoding="utf-8", errors="ignore"))
        if not text:
            return ParsedDocument(path.name, "", "The sample file is empty.")
        return ParsedDocument(path.name, text)
    except Exception:
        return ParsedDocument(path.name, "", "Could not read the bundled sample document.")
