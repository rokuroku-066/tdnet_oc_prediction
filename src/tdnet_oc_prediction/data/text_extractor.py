from __future__ import annotations
import re
from pathlib import Path

class TextExtractor:
    """Extract and normalize text from HTML/TXT/PDF files."""

    def extract(self, file_path: str) -> str:
        p = Path(file_path)
        suffix = p.suffix.lower()
        if suffix in {".txt", ".html", ".htm"}:
            text = p.read_text(encoding="utf-8", errors="ignore")
        elif suffix == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError as e:
                raise ImportError("PDF extraction requires pypdf") from e
            reader = PdfReader(str(p))
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
        else:
            text = p.read_text(encoding="utf-8", errors="ignore")
        return re.sub(r"\s+", " ", text).strip()
