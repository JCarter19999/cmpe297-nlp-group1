"""
rag_local.arxiv_fetch

Materialize arXiv search results (abstract-only) into rag_local/Data/arxiv/.

This module provides:
- fetch_arxiv_papers_to_data(...)  # main entrypoint
- CLI usage: python -m rag_local.arxiv_fetch "your topic" --max-results 10

Dataset contract:
rag_local/Data/arxiv/<arxiv_id>/
  - paper.txt   (plain text, includes title/authors/date/url/abstract)
  - meta.json   (machine-readable metadata)

Design goals:
- deterministic output ordering
- safe overwrite (idempotent runs)
- cache support via arxiv_client
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
import urllib.request
import urllib.error

from pypdf import PdfReader

from rag_local.arxiv_client import search_arxiv


def _safe_folder_name(arxiv_id: str) -> str:
    """
    Sanitize arXiv ID for filesystem folder naming.
    Example: "2401.01234v2" -> "2401.01234v2"
    For older IDs with slashes, replace / with _.
    """
    s = (arxiv_id or "").strip()
    s = s.replace("/", "_")
    # Keep it conservative
    s = re.sub(r"[^A-Za-z0-9._-]", "_", s)
    return s or "unknown_id"


def _format_paper_txt(p: Dict[str, Any]) -> str:
    """Internal helper for format paper txt."""
    title = p.get("title", "")
    authors = p.get("authors", []) or []
    published = p.get("published", "")
    updated = p.get("updated", "")
    arxiv_id = p.get("arxiv_id", "")
    url = p.get("url", "") or p.get("id_url", "")
    categories = p.get("categories", []) or []
    primary = p.get("primary_category", "")
    body_text = p.get("full_text", "") or p.get("abstract", "")


    header_lines = [
        f"Title: {title}",
        f"Authors: {', '.join(authors)}" if authors else "Authors: ",
        f"Published: {published}",
        f"Updated: {updated}" if updated else "Updated: ",
        f"arXiv ID: {arxiv_id}",
        f"URL: {url}",
        f"Primary Category: {primary}" if primary else "Primary Category: ",
        f"Categories: {', '.join(categories)}" if categories else "Categories: ",
        "",
        "Body Text:",
        body_text,
        "",
    ]
    # Ensure deterministic whitespace
    return "\n".join([line.rstrip() for line in header_lines]).strip() + "\n"

def _download_pdf(pdf_url: str, dest_path: Path, timeout_s: int = 60) -> None:
    """Internal helper for download pdf."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(pdf_url, headers={"User-Agent": "cmpe297-rag-tutor/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            dest_path.write_bytes(resp.read())
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to download PDF from {pdf_url}: {e}")


def _extract_text_from_pdf(pdf_path: Path) -> str:
    """Internal helper for extract text from pdf."""
    reader = PdfReader(str(pdf_path))
    parts = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = " ".join(text.split())
        if text:
            parts.append(text)
    return "\n\n".join(parts).strip()

def fetch_arxiv_papers_to_data(
    query: str,
    max_results: int = 3,
    data_dir: str | Path = "rag_local/Data",
    categories: Optional[List[str]] = None,
    sort_by: str = "relevance",
    sort_order: str = "descending",
    refresh: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Fetch arXiv results (abstract-only) and write them into:
      <data_dir>/arxiv/<arxiv_id>/paper.txt + meta.json

    Returns a summary dict (counts, paths).
    """
    base = Path(data_dir).expanduser().resolve()
    out_root = base / "arxiv"
    out_root.mkdir(parents=True, exist_ok=True)

    papers, info = search_arxiv(
        query=query,
        max_results = min(int(max_results), 3),  # safety cap to prevent large fetches
        categories=categories,
        sort_by=sort_by,
        sort_order=sort_order,
        refresh=refresh,
    )

    # Deterministic ordering (already sorted in client), but keep it explicit
    papers = list(papers)
    papers.sort(key=lambda p: (p.get("published", ""), p.get("arxiv_id", "")), reverse=True)

    written = 0
    skipped_empty = 0

    written_paths: List[str] = []

    for p in papers:
        arxiv_id = (p.get("arxiv_id") or "").strip()
        folder = out_root / _safe_folder_name(arxiv_id)

        abstract = (p.get("abstract") or "").strip()
        if not abstract:
            skipped_empty += 1
            if verbose:
                print(f"[warn] Skipping empty-abstract paper: {arxiv_id or p.get('id_url')}")
            continue

        folder.mkdir(parents=True, exist_ok=True)

        meta_path = folder / "meta.json"
        paper_path = folder / "paper.txt"

        meta_payload = {
            "source": "arxiv",
            "query": query,
            "fetched_info": info,
            "paper": p,
        }
        meta_path.write_text(json.dumps(meta_payload, indent=2, ensure_ascii=False), encoding="utf-8")

        pdf_url = (p.get("pdf_url") or "").strip()

        # Fallback: derive from id_url if needed
        if not pdf_url:
            id_url = (p.get("id_url") or "").strip()
            if "/abs/" in id_url:
                pdf_url = id_url.replace("/abs/", "/pdf/") + ".pdf"

        if pdf_url:
            pdf_path = folder / "paper.pdf"
            try:
                _download_pdf(pdf_url, pdf_path)
                full_text = _extract_text_from_pdf(pdf_path)
                if full_text:
                    p["full_text"] = full_text
                else:
                    # extraction empty; keep abstract fallback
                    p["full_text"] = (p.get("abstract") or "").strip()
            except Exception as e:
                # download/extract failed; keep abstract fallback
                p["full_text"] = (p.get("abstract") or "").strip()
        else:
            # no pdf url; fallback
            p["full_text"] = (p.get("abstract") or "").strip()

        paper_txt = _format_paper_txt(p)
        paper_path.write_text(paper_txt, encoding="utf-8")

        written += 1
        written_paths.append(str(paper_path))

    summary = {
        "query": query,
        "data_dir": str(base),
        "out_root": str(out_root),
        "requested": int(max_results),
        "returned": len(papers),
        "written": written,
        "skipped_empty": skipped_empty,
        "cache": info.get("cache"),
        "cache_path": info.get("cache_path"),
        "written_paths": written_paths[:10],  # preview only
    }

    if verbose:
        print(f"[arXiv] query={query!r} returned={len(papers)} written={written} skipped_empty={skipped_empty}")
        if info.get("cache"):
            print(f"[arXiv] cache={info.get('cache')} cache_path={info.get('cache_path')}")

    return summary


def _parse_args() -> argparse.Namespace:
    """Internal helper for parse args."""
    ap = argparse.ArgumentParser(description="Fetch arXiv papers into rag_local/Data/arxiv/")
    ap.add_argument("query", type=str, help="Search topic / keywords for arXiv")
    ap.add_argument("--max-results", type=int, default=3, help="Max number of papers to fetch")
    ap.add_argument("--data-dir", type=str, default="rag_local/Data", help="Root data directory")
    ap.add_argument("--categories", type=str, default="", help="Comma-separated categories e.g. cs.CL,cs.IR")
    ap.add_argument("--sort-by", type=str, default="relevance", help="relevance | submittedDate | lastUpdatedDate")
    ap.add_argument("--sort-order", type=str, default="descending", help="ascending | descending")
    ap.add_argument("--refresh", action="store_true", help="Bypass cache and refetch")
    return ap.parse_args()


def main() -> None:
    """Main."""
    args = _parse_args()
    cats = [c.strip() for c in args.categories.split(",") if c.strip()] if args.categories else None

    fetch_arxiv_papers_to_data(
        query=args.query,
        max_results=args.max_results,
        data_dir=args.data_dir,
        categories=cats,
        sort_by=args.sort_by,
        sort_order=args.sort_order,
        refresh=args.refresh,
        verbose=True,
    )


if __name__ == "__main__":
    main()
