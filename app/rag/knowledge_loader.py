"""
Extracts plain text from the different kinds of knowledge sources the
"Shared Knowledge & Memory" box needs to support:

  - Free-form documents (PDF / .txt)   -> Deepfake Knowledge Base, External Retrieval
  - Structured JSON entries            -> Manipulation Patterns DB, Model Registry,
                                           Forensic Rules, Past Cases & Feedback

Structured entries are flattened into a readable text string for
embedding, while the original JSON is preserved in the vector store's
metadata so it can be returned as-is on retrieval.
"""

import io
import json
from typing import Any, Dict

from pypdf import PdfReader


def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def extract_text_from_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore").strip()


def extract_text_from_upload(filename: str, file_bytes: bytes) -> str:
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if ext == "pdf":
        return extract_text_from_pdf(file_bytes)
    if ext in ("txt", "md"):
        return extract_text_from_txt(file_bytes)
    raise ValueError(f"Unsupported document type '.{ext}'. Use .pdf, .txt, or .md.")


def flatten_structured_entry(entry: Dict[str, Any]) -> str:
    """
    Turns a structured JSON entry (a manipulation pattern, forensic
    rule, model registry entry, or past case) into a readable text blob
    so it can be embedded and semantically searched, e.g.:

        {"pattern": "Reflection mismatch", "description": "...", "severity": "High"}

    becomes:

        "pattern: Reflection mismatch. description: Reflection does not
         match lighting. severity: High."
    """
    parts = []
    for key, value in entry.items():
        if isinstance(value, (list, tuple)):
            value = ", ".join(str(v) for v in value)
        elif isinstance(value, dict):
            value = json.dumps(value)
        parts.append(f"{key}: {value}")
    return ". ".join(parts)
