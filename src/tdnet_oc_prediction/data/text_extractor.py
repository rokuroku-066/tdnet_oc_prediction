from __future__ import annotations

import io
import re
from pathlib import Path


class TextExtractor:
    """Extract and normalize text from HTML/TXT/PDF files/bytes."""

    def extract_pdf_bytes(self, payload: bytes) -> str:
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise ImportError("PDF extraction requires pypdf") from e
        reader = PdfReader(io.BytesIO(payload))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
        return re.sub(r"\s+", " ", text).strip()

    def extract(self, file_path: str) -> str:
        p = Path(file_path)
        suffix = p.suffix.lower()
        if suffix in {".txt", ".html", ".htm"}:
            text = p.read_text(encoding="utf-8", errors="ignore")
        elif suffix == ".pdf":
            text = self.extract_pdf_bytes(p.read_bytes())
        else:
            text = p.read_text(encoding="utf-8", errors="ignore")
        return re.sub(r"\s+", " ", text).strip()
