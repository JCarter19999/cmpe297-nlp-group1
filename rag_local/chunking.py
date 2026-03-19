"""
rag_local.chunking

Chunk Schema (Output Contract)
------------------------------
Each chunk MUST include:
- chunk_id: str
    Stable identifier for the chunk (deterministic across runs if inputs unchanged).
- text: str
    Chunk text.
Optional:
- start: int
    Start character index within the original text (inclusive).
- end: int
    End character index within the original text (exclusive).
- doc_id: str
    Identifier of the parent document (propagated from input).
- source: str
    Source of the parent document (e.g., filename, URL).
- metadata: dict
    Optional metadata propagated from the parent document.

Example chunk object
--------------------
chunk = {
    "chunk_id": "doc123::chunk_0",
    "text": "This is the chunk text ...",
    "start": 0,
    "end": 500,
    "doc_id": "doc123",
    "source": "example.pdf",
    "metadata": {
        "author": "Jane Doe",
        "category": "lecture_notes"
    }
}

Public API
----------
- chunk_text(
      text: str,
      *,
      chunk_size: int = 800,
      overlap: int = 200,
      doc_id: str | None = None,
      source: str | None = None,
      metadata: dict | None = None,
      include_spans: bool = True
  ) -> list[Chunk]

Notes
-----
- No NLP libraries required (pure Python).
- Baseline implementation uses fixed-size, character-based chunking.
- You can later add token-based chunking, sentence boundaries, etc., while keeping the schema stable.
- Chunk order is deterministic across runs when inputs are unchanged.
- If doc_id is not provided, a deterministic hash-based chunk_id is used.
- The output schema is designed to remain stable as future enhancements
  (token-based chunking, sentence boundaries, etc.) are added.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict
import hashlib

# -----------------------------
# Chunk schema (contract)
# -----------------------------
class Chunk(TypedDict, total=False):
    """
    Standard chunk schema for chunking output.

    Required keys:
      - chunk_id: stable identifier string
      - text: chunk text

    Optional keys:
      - start: start char index in original text (inclusive)
      - end: end char index in original text (exclusive)
      - doc_id: document identifier (propagated from parent)
      - source: document source (propagated from parent)
      - metadata: optional metadata from parent document
    """
    chunk_id: str
    text: str
    start: int
    end: int
    doc_id: str
    source: str
    metadata: Dict[str, Any]

# -----------------------------
# Public entrypoint
# -----------------------------

def chunk_text(
    text: str,
    *,
    chunk_size: int = 800,
    overlap: int = 200,
    doc_id: Optional[str] = None,
    source: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    include_spans: bool = True,
) -> List[Chunk]:
    """
    Split `text` into overlapping chunks.

    Requirements satisfied (per ticket):
    - Fixed chunk size (chunk_size)
    - Configurable overlap (overlap)
    - Returns a list of chunk objects (Chunk schema)
    - Each chunk includes: chunk_id, text, optional start/end indices
    - No NLP libraries

    Parameters
    ----------
    text:
        Input document text.
    chunk_size:
        Maximum characters per chunk (baseline, char-based).
    overlap:
        Number of characters to overlap between consecutive chunks.
        Must be < chunk_size.
    doc_id:
        Optional document identifier used to make chunk_id stable across runs
        and unique across documents.

    source:
        Optional document source string (e.g., filename, URL, dataset key).
        If provided, it will be propagated into each chunk.
    metadata:
        Optional metadata dictionary from the parent document. If provided,
        it will be propagated into each chunk (shallow-copied).
    include_spans:
        If True, include start/end indices in each chunk.

    Returns
    -------
    list[Chunk]
        List of chunk objects in deterministic order.

    Raises
    ------
    ValueError:
        If chunk_size <= 0, overlap < 0, or overlap >= chunk_size.
    """
    _validate_params(chunk_size=chunk_size, overlap=overlap)

    if not text or not text.strip():
        return []

    chunks: List[Chunk] = []
    step = chunk_size - overlap

    # Deterministic traversal
    start = 0
    n = len(text)
    i = 0
    while start < n:
        end = min(start + chunk_size, n)
        chunk_str = text[start:end]

        # Optional: you can later add trimming rules here.
        # Keep baseline simple (do not alter text beyond slicing).
        chunk = _make_chunk(
            chunk_text=chunk_str,
            start=start,
            end=end,
            doc_id=doc_id,
            source=source,
            metadata=metadata,
            index=i,
            include_spans=include_spans,
        )
        chunks.append(chunk)

        if end == n:
            break
        start += step
        i += 1
    return chunks


# -----------------------------
# Helpers (modular building blocks)
# -----------------------------

def _validate_params(*, chunk_size: int, overlap: int) -> None:
    """Internal helper for validate params."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(f"overlap must be < chunk_size, got overlap={overlap}, chunk_size={chunk_size}")


def _make_chunk(
    *,
    chunk_text: str,
    start: int,
    end: int,
    doc_id: Optional[str],
    source: Optional[str],
    metadata: Optional[Dict[str, Any]],
    index: int,
    include_spans: bool,
) -> Chunk:
    """
    Normalize chunk output into the Chunk schema and generate a stable chunk_id.
    """
    chunk_id = make_chunk_id(doc_id=doc_id,index=index,start=start,end=end,chunk_text=chunk_text)
    chunk: Chunk = {
        "chunk_id": chunk_id,
        "text": chunk_text,
    }
    if include_spans:
        chunk["start"] = start
        chunk["end"] = end
    # Propagate document context (if provided)
    if doc_id is not None:
        chunk["doc_id"] = doc_id
    if source is not None:
        chunk["source"] = source
    if metadata is not None:
        chunk["metadata"] = dict(metadata) # shallow copy to avoid accidental mutation across chunks
    return chunk

def make_chunk_id(*, doc_id: Optional[str], index: int, start: int, end: int, chunk_text: str) -> str:
    """
    Create a deterministic chunk identifier.
    Strategy:
    - If doc_id is None: 
        Fall back to a deterministic Hash(doc_id + start/end + chunk_text). The id is still deterministic for the same text content
        but doc_id is recommended to avoid collisions across documents.
    - If doc_id is provided: 
        Use globally unique readable ID: "{doc_id}::chunk_{i}" (Deterministic given stable doc_id and chunking params.) 
    """
    if doc_id:
        return f"{doc_id}::chunk_{index}"
    h = hashlib.sha256()
    h.update("NO_DOC_ID".encode("utf-8"))
    h.update(b"\n")
    h.update(f"{start}:{end}".encode("utf-8"))
    h.update(b"\n")
    h.update(chunk_text.encode("utf-8", errors="ignore"))
    return h.hexdigest()[:16]


# -----------------------------
# Developer guide: how to extend
# -----------------------------

# ============================================================
# HOW TO EXTEND CHUNKING (Developer Guide)
# ============================================================
#
# Baseline today: character-based fixed-size chunks with overlap.
#
# Future upgrades you can add WITHOUT breaking downstream code:
# ------------------------------------------------------------
# 1) Sentence-aware chunking:
#    - Keep output schema the same (Chunk: chunk_id, text, start/end)
#    - Change chunk boundaries to respect sentence breaks.
#
# 2) Token-based chunking:
#    - Compute token spans, but still return chunk text + start/end (char indices)
#    - Add token metadata into an optional field later ONLY if needed
#
# 3) Metadata propagation:
#    - If you later chunk Documents instead of raw text, you can carry doc_id + source
#    - But keep `chunk_text(...)` stable as the low-level utility.
#
# Testing checklist before merging:
# ---------------------------------
# □ Deterministic output ordering
# □ overlap < chunk_size
# □ empty/whitespace text returns []
# □ chunk ids stable across runs (when doc_id + text unchanged)
# ============================================================
