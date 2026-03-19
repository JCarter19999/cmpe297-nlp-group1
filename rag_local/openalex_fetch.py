"""Core logic for openalex fetch."""
from __future__ import annotations


import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import requests


# ---------------------------------------------------------
# Optional PDF extraction backends
# ---------------------------------------------------------
_PDF_BACKEND = None

try:
    from pypdf import PdfReader  # type: ignore
    _PDF_BACKEND = "pypdf"
except Exception:
    try:
        from PyPDF2 import PdfReader  # type: ignore
        _PDF_BACKEND = "PyPDF2"
    except Exception:
        PdfReader = None
        _PDF_BACKEND = None


# ---------------------------------------------------------
# Data model
# ---------------------------------------------------------
@dataclass
class MaterializedPaper:
    """Represents materialized paper."""
    openalex_id: str
    title: str
    year: Optional[int]
    landing_url: str
    pdf_url: str
    query_slug: str
    paper_dir: str
    content_tier: str  # "fulltext" | "metadata_only"
    files_written: List[str]
    note: str = ""


# ---------------------------------------------------------
# Text / path helpers
# ---------------------------------------------------------
def _slugify(text: str, max_len: int = 80) -> str:
    """Internal helper for slugify."""
    text = (text or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    if not text:
        text = "untitled"
    return text[:max_len].rstrip("-")


def _query_slug(query: str) -> str:
    """Internal helper for query slug."""
    return _slugify(query or "openalex-search", max_len=60)


def _safe_filename(text: str, max_len: int = 120) -> str:
    """Internal helper for safe filename."""
    return _slugify(text or "paper", max_len=max_len)


def _ensure_dir(path: Path) -> None:
    """Internal helper for ensure dir."""
    path.mkdir(parents=True, exist_ok=True)


def _now_ts() -> str:
    """Internal helper for now ts."""
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _extract_host(url: str) -> str:
    """Internal helper for extract host."""
    try:
        return urlparse(url).netloc or ""
    except Exception:
        return ""


# ---------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; CMPE297-RAG/1.0; +local)",
    "Accept": "*/*",
}


def _is_probably_pdf_response(resp: requests.Response) -> bool:
    """Internal helper for is probably pdf response."""
    content_type = (resp.headers.get("Content-Type") or "").lower()
    if "application/pdf" in content_type:
        return True

    dispo = (resp.headers.get("Content-Disposition") or "").lower()
    if ".pdf" in dispo:
        return True

    url = str(resp.url or "").lower()
    if url.endswith(".pdf"):
        return True

    # Content sniffing fallback
    try:
        if resp.content[:5] == b"%PDF-":
            return True
    except Exception:
        pass

    return False


def _download_url(
    url: str,
    *,
    timeout_s: int = 60,
    allow_redirects: bool = True,
) -> requests.Response:
    """Internal helper for download url."""
    r = requests.get(
        url,
        headers=DEFAULT_HEADERS,
        timeout=timeout_s,
        allow_redirects=allow_redirects,
    )
    r.raise_for_status()
    return r


def _try_download_pdf(url: str, *, timeout_s: int = 60) -> Tuple[Optional[bytes], str]:
    """
    Returns:
      (pdf_bytes, message)
    Never raises.
    """
    if not url or not str(url).strip():
        return None, "empty pdf_url"

    try:
        r = _download_url(url, timeout_s=timeout_s, allow_redirects=True)
        if not _is_probably_pdf_response(r):
            ctype = r.headers.get("Content-Type", "")
            return None, f"response was not a PDF (content-type={ctype})"
        return r.content, "ok"
    except Exception as e:
        return None, str(e)


# ---------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------
def _extract_text_from_pdf_bytes(pdf_bytes: bytes) -> Tuple[str, str]:
    """
    Returns (text, note)
    """
    if not pdf_bytes:
        return "", "empty pdf bytes"

    if PdfReader is None:
        return "", "no PDF reader installed"

    try:
        import io

        bio = io.BytesIO(pdf_bytes)
        reader = PdfReader(bio)
        pages = []
        for page in reader.pages:
            try:
                txt = page.extract_text() or ""
            except Exception:
                txt = ""
            if txt.strip():
                pages.append(txt.strip())

        text = "\n\n".join(pages).strip()
        if not text:
            return "", f"parsed PDF with {_PDF_BACKEND}, but extracted no text"
        return text, f"parsed PDF with {_PDF_BACKEND}"
    except Exception as e:
        return "", f"PDF parse failed: {e}"


# ---------------------------------------------------------
# Materialization helpers
# ---------------------------------------------------------
def _write_text(path: Path, text: str) -> None:
    """Internal helper for write text."""
    path.write_text(text, encoding="utf-8", errors="ignore")


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    """Internal helper for write json."""
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _build_metadata_record(
    *,
    item: Dict[str, Any],
    content_tier: str,
    note: str,
    query: str,
    paper_dir: Path,
    pdf_saved: bool,
    txt_saved: bool,
) -> Dict[str, Any]:
    """Internal helper for build metadata record."""
    raw = dict(item.get("raw") or {})
    return {
        "source": "openalex",
        "openalex_id": item.get("openalex_id") or item.get("id") or "",
        "title": item.get("title") or "",
        "abstract": item.get("abstract") or "",
        "year": item.get("year"),
        "url": item.get("url") or "",
        "pdf_url": item.get("pdf_url") or "",
        "score": item.get("score"),
        "query": query,
        "content_tier": content_tier,
        "note": note,
        "paper_dir": str(paper_dir),
        "pdf_saved": bool(pdf_saved),
        "txt_saved": bool(txt_saved),
        "materialized_at": _now_ts(),
        "raw": raw,
    }


def _build_indexable_text(
    *,
    item: Dict[str, Any],
    fulltext: str,
    content_tier: str,
    note: str,
) -> str:
    """Internal helper for build indexable text."""
    title = (item.get("title") or "").strip()
    abstract = (item.get("abstract") or "").strip()
    year = item.get("year")
    url = (item.get("url") or "").strip()
    pdf_url = (item.get("pdf_url") or "").strip()
    openalex_id = (item.get("openalex_id") or item.get("id") or "").strip()

    header = [
        f"Title: {title}",
        f"OpenAlex ID: {openalex_id}",
        f"Year: {year}",
        f"Landing URL: {url}",
        f"PDF URL: {pdf_url}",
        f"Content Tier: {content_tier}",
        f"Note: {note}",
    ]

    parts = ["\n".join(header)]

    if abstract:
        parts.append("Abstract:\n" + abstract)

    if fulltext.strip():
        parts.append("Full Text:\n" + fulltext.strip())

    return "\n\n".join(parts).strip()


# ---------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------
def materialize_openalex_selected(
    *,
    selected: Sequence[Dict[str, Any]],
    data_dir: Path,
    query: str,
    timeout_s: int = 60,
    pause_s: float = 0.0,
) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str]]]:
    """
    Materialize selected OpenAlex papers into:
      rag_local/Data/openalex/<query_slug>/<paper_slug>/

    For each paper, attempts:
      1) direct PDF download
      2) PDF text extraction
      3) if PDF fails, save metadata/abstract only

    Returns:
      saved, skipped

    saved: list[dict]
      Each item includes content_tier = "fulltext" or "metadata_only"

    skipped: list[(paper_id, reason)]
      Only used when nothing usable could be saved at all.
    """
    data_dir = Path(data_dir).resolve()
    base_dir = data_dir / "openalex" / _query_slug(query)
    _ensure_dir(base_dir)

    saved: List[Dict[str, Any]] = []
    skipped: List[Tuple[str, str]] = []

    for item in selected:
        openalex_id = str(item.get("openalex_id") or item.get("id") or "").strip()
        title = str(item.get("title") or "").strip()
        abstract = str(item.get("abstract") or "").strip()
        landing_url = str(item.get("url") or "").strip()
        pdf_url = str(item.get("pdf_url") or "").strip()
        year = item.get("year")

        if not openalex_id and not title:
            skipped.append(("", "missing both id and title"))
            continue

        paper_slug = _safe_filename(f"{year or 'na'}-{title or openalex_id}", max_len=100)
        paper_dir = base_dir / paper_slug
        _ensure_dir(paper_dir)

        files_written: List[str] = []
        content_tier = "metadata_only"
        note = ""

        pdf_saved = False
        txt_saved = False

        # -------------------------------
        # Attempt full-text PDF fetch
        # -------------------------------
        fulltext = ""
        if pdf_url:
            pdf_bytes, fetch_msg = _try_download_pdf(pdf_url, timeout_s=timeout_s)
            if pdf_bytes is not None:
                pdf_path = paper_dir / "paper.pdf"
                pdf_path.write_bytes(pdf_bytes)
                files_written.append(str(pdf_path))
                pdf_saved = True

                extracted_text, parse_note = _extract_text_from_pdf_bytes(pdf_bytes)
                if extracted_text.strip():
                    fulltext = extracted_text
                    content_tier = "fulltext"
                    note = parse_note

                    txt_path = paper_dir / "paper.txt"
                    _write_text(
                        txt_path,
                        _build_indexable_text(
                            item=item,
                            fulltext=fulltext,
                            content_tier=content_tier,
                            note=note,
                        ),
                    )
                    files_written.append(str(txt_path))
                    txt_saved = True
                else:
                    # PDF downloaded, but extraction failed -> still keep metadata-only
                    content_tier = "metadata_only"
                    note = f"pdf_downloaded_but_text_extraction_failed: {parse_note}"
            else:
                content_tier = "metadata_only"
                note = f"pdf_download_failed: {fetch_msg}"
        else:
            content_tier = "metadata_only"
            note = "no pdf_url provided"

        # -------------------------------
        # Metadata-only fallback
        # -------------------------------
        if not txt_saved:
            if not title and not abstract:
                skipped.append((openalex_id or title or "unknown", f"nothing usable to save ({note})"))
                continue

            txt_path = paper_dir / "paper.txt"
            _write_text(
                txt_path,
                _build_indexable_text(
                    item=item,
                    fulltext="",
                    content_tier="metadata_only",
                    note=note,
                ),
            )
            files_written.append(str(txt_path))
            txt_saved = True

        # -------------------------------
        # Always write metadata JSON
        # -------------------------------
        meta_path = paper_dir / "metadata.json"
        _write_json(
            meta_path,
            _build_metadata_record(
                item=item,
                content_tier=content_tier,
                note=note,
                query=query,
                paper_dir=paper_dir,
                pdf_saved=pdf_saved,
                txt_saved=txt_saved,
            ),
        )
        files_written.append(str(meta_path))

        saved.append(
            {
                "openalex_id": openalex_id,
                "title": title,
                "year": year,
                "landing_url": landing_url,
                "pdf_url": pdf_url,
                "query_slug": _query_slug(query),
                "paper_dir": str(paper_dir),
                "content_tier": content_tier,
                "files_written": files_written,
                "note": note,
            }
        )

        if pause_s > 0:
            time.sleep(pause_s)

    return saved, skipped