"""
rag_local.loaders

Standard Document Schema (Loader Output Contract)
-------------------------------------------------
Every loader MUST return a list of Document objects with at least:

- doc_id: str
    Stable string identifier for this document. Should remain the same across runs
    if the underlying document has not changed.

- text: str
    Full extracted text for the document (plaintext).

- source: str
    Filename or relative path indicating where the document came from.
    For this project, `source` should usually be relative to rag_local/Data/.

- meta: dict (optional)
    Optional dictionary for any extra metadata (page numbers, title, author, etc.)

Example Document object
-----------------------
doc = {
    "doc_id": "lecture01_nlp_intro_v1",
    "text": "Natural Language Processing (NLP) is ...",
    "source": "notes/lecture01_intro.md",
    "meta": {"course": "CMPE297", "week": 1}
}

Public API
----------
- load_documents(data_dir: str | Path = DEFAULT_DATA_DIR) -> list[Document]

Team rule:
----------
If you change the Document schema, you must also update any downstream modules that
consume it (chunking, indexing, RAG).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Union

from pypdf import PdfReader


# -----------------------------
# Types / Contract
# -----------------------------

class Document(TypedDict, total=False):
    """Represents document."""
    doc_id: str
    text: str
    source: str
    meta: Dict[str, Any]


DEFAULT_DATA_DIR = Path(__file__).parent / "Data"


# -----------------------------
# Public API
# -----------------------------

def load_documents(data_dir: Union[str, Path] = DEFAULT_DATA_DIR) -> List[Document]:
    """
    Walk `data_dir` recursively and load supported files into Document objects.

    Supported:
      - .txt
      - .md
      - .json
      - .pdf

    Determinism:
      - Discover files in deterministic order (relative path, case-insensitive)
      - Output docs sorted by `source`
    """
    root = Path(data_dir).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Data directory not found: {root}")

    all_files = [p for p in root.rglob("*") if p.is_file()]
    all_files.sort(key=lambda p: str(p.relative_to(root)).lower())

    docs: List[Document] = []
    for path in all_files:
        suf = path.suffix.lower()
        rel_source = _rel_source(path, root)

        if suf in {".txt", ".md"}:
            text = _read_text(path)
            text = _normalize_whitespace(text)
            if not text:
                _warn(f"Skipping empty {suf}: {rel_source}")
                continue

            meta = {
                "type": "txt" if suf == ".txt" else "markdown",
                "filename": path.name,
            }
            docs.append(normalize_document(source=rel_source, text=text, meta=meta))

        elif suf == ".json":
            obj = _read_json(path)
            text = _extract_text_from_json(obj)
            text = _normalize_whitespace(text)
            if not text:
                _warn(f"Skipping empty json: {rel_source}")
                continue

            meta = {"type": "json", "filename": path.name}
            docs.append(normalize_document(source=rel_source, text=text, meta=meta))

        elif suf == ".pdf":
            text, pages = _extract_text_from_pdf(path)
            text = _normalize_whitespace(text)
            if not text:
                _warn(f"Skipping empty/unenextractable pdf: {rel_source}")
                continue

            meta: Dict[str, Any] = {"type": "pdf", "filename": path.name}
            if pages is not None:
                meta["pages"] = pages

            docs.append(normalize_document(source=rel_source, text=text, meta=meta))

        else:
            # ignore other file types
            continue

    docs.sort(key=lambda d: d.get("source", ""))
    return docs


# -----------------------------
# Normalization helpers
# -----------------------------

def normalize_document(*, source: str, text: str, meta: Optional[Dict[str, Any]] = None) -> Document:
    """
    Produce a Document that matches the required schema.
    """
    doc: Document = {
        "doc_id": make_doc_id(source=source, text=text),
        "text": text,
        "source": source,
    }
    if meta:
        doc["meta"] = meta
    return doc


def make_doc_id(source: str, text: str) -> str:
    """
    Stable-ish ID derived from source + text.
    If file changes, doc_id changes deterministically.
    """
    h = hashlib.sha256()
    h.update(source.encode("utf-8", errors="ignore"))
    h.update(b"\n")
    h.update(text.encode("utf-8", errors="ignore"))
    return h.hexdigest()[:16]


def _normalize_whitespace(s: str) -> str:
    """
    Make whitespace deterministic (helps chunking + reproducibility).
    """
    return " ".join((s or "").split()).strip()


def _rel_source(path: Path, root: Path) -> str:
    """
    Return a stable relative path string for `source`.
    """
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def _warn(msg: str) -> None:
    """Internal helper for warn."""
    print(f"[warn] {msg}")


# -----------------------------
# File readers
# -----------------------------

def _read_text(path: Path) -> str:
    """Internal helper for read text."""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _read_json(path: Path) -> Any:
    """Internal helper for read json."""
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None


def _extract_text_from_json(obj: Any) -> str:
    """
    Minimal JSON text extraction.
    Accepts common keys; otherwise best-effort stringification.
    """
    if obj is None:
        return ""
    if isinstance(obj, dict):
        if isinstance(obj.get("text"), str):
            return obj["text"]
        if isinstance(obj.get("content"), str):
            return obj["content"]
        return json.dumps(obj, ensure_ascii=False, indent=2)
    if isinstance(obj, list):
        return json.dumps(obj, ensure_ascii=False, indent=2)
    if isinstance(obj, str):
        return obj
    return str(obj)


# -----------------------------
# PDF loader
# -----------------------------

def _extract_text_from_pdf(pdf_path: Path) -> tuple[str, Optional[int]]:
    """
    Extract text from a PDF using pypdf.
    Returns: (text, page_count)
    """
    try:
        reader = PdfReader(str(pdf_path))
    except Exception as e:
        _warn(f"Failed to open pdf: {pdf_path.name} ({e})")
        return "", None

    parts: List[str] = []
    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        page_text = page_text.strip()
        if page_text:
            parts.append(page_text)

    text = "\n\n".join(parts).strip()
    page_count = None
    try:
        page_count = len(reader.pages)
    except Exception:
        page_count = None

    return text, page_count
