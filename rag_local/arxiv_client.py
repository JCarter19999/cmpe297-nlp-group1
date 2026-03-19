"""
rag_local.arxiv_client

Minimal arXiv API client (Atom feed) using stdlib only.

Provides:
- search_arxiv(...) -> list[dict] where each dict includes metadata + abstract

Includes optional caching to:
rag_local/cache/arxiv_queries/<hash>.json

Notes:
- Uses the arXiv API endpoint: http://export.arxiv.org/api/query
- Returns abstract-only (no PDF text extraction).
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ARXIV_API_URL = "http://export.arxiv.org/api/query"

ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"

DEFAULT_CACHE_DIR = Path(__file__).parent / "cache" / "arxiv_queries"


def _norm_ws(s: str) -> str:
    """Internal helper for norm ws."""
    return " ".join((s or "").split()).strip()


def _sha256_hex(s: str) -> str:
    """Internal helper for sha256 hex."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _cache_key(params: Dict[str, Any]) -> str:
    # Deterministic key for params
    """Internal helper for cache key."""
    stable = json.dumps(params, sort_keys=True, ensure_ascii=False)
    return _sha256_hex(stable)


def _read_cache(cache_dir: Path, key: str) -> Optional[Dict[str, Any]]:
    """Internal helper for read cache."""
    path = cache_dir / f"{key}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_cache(cache_dir: Path, key: str, payload: Dict[str, Any]) -> Path:
    """Internal helper for write cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _build_query_url(
    query: str,
    max_results: int,
    start: int = 0,
    categories: Optional[List[str]] = None,
    sort_by: str = "relevance",
    sort_order: str = "descending",
) -> str:
    """
    query: keyword phrase, we map it to arXiv 'all:' fields.
    categories: optional list like ["cs.CL", "cs.IR"]
    """
    q = _norm_ws(query)
    if not q:
        raise ValueError("query cannot be empty")

    # arXiv expects: search_query=all:... optionally AND cat:...
    # Keep it simple: all:<terms> AND (cat:... OR cat:...)
    # Quote handling: arXiv API is forgiving but we keep it basic.
    base = f"all:{q}"
    if categories:
        cats = [c.strip() for c in categories if c.strip()]
        if cats:
            cat_expr = " OR ".join([f"cat:{c}" for c in cats])
            base = f"({base}) AND ({cat_expr})"

    params = {
        "search_query": base,
        "start": int(start),
        "max_results": int(max_results),
        "sortBy": sort_by,
        "sortOrder": sort_order,
    }
    return ARXIV_API_URL + "?" + urllib.parse.urlencode(params)


def _parse_atom(xml_text: str) -> List[Dict[str, Any]]:
    """
    Parse arXiv Atom XML feed and return list of dict papers.
    """
    root = ET.fromstring(xml_text)
    entries = root.findall(f"{ATOM_NS}entry")
    papers: List[Dict[str, Any]] = []

    for e in entries:
        # id is a URL like: http://arxiv.org/abs/XXXX.XXXXXvN
        id_url = _norm_ws((e.findtext(f"{ATOM_NS}id") or ""))
        arxiv_id = id_url.rsplit("/", 1)[-1] if id_url else ""

        title = _norm_ws(e.findtext(f"{ATOM_NS}title") or "")
        summary = _norm_ws(e.findtext(f"{ATOM_NS}summary") or "")
        published = _norm_ws(e.findtext(f"{ATOM_NS}published") or "")
        updated = _norm_ws(e.findtext(f"{ATOM_NS}updated") or "")

        authors = []
        for a in e.findall(f"{ATOM_NS}author"):
            name = _norm_ws(a.findtext(f"{ATOM_NS}name") or "")
            if name:
                authors.append(name)

        # Categories
        categories = []
        primary_category = ""
        pc = e.find(f"{ARXIV_NS}primary_category")
        if pc is not None and pc.attrib.get("term"):
            primary_category = pc.attrib["term"]

        for c in e.findall(f"{ATOM_NS}category"):
            term = (c.attrib.get("term") or "").strip()
            if term:
                categories.append(term)

        # Links
        url = ""
        pdf_url = ""
        for link in e.findall(f"{ATOM_NS}link"):
            rel = (link.attrib.get("rel") or "").strip()
            href = (link.attrib.get("href") or "").strip()
            link_type = (link.attrib.get("type") or "").strip().lower()
            title_attr = (link.attrib.get("title") or "").strip().lower()

            if rel == "alternate" and href and not url:
                url = href

            # PDF links can appear as:
            # - title="pdf"
            # - type="application/pdf"
            # - href ending in .pdf
            if href and (
                "pdf" in title_attr
                or link_type == "application/pdf"
                or href.endswith(".pdf")
                or "/pdf/" in href
            ):
                pdf_url = href

        papers.append(
            {
                "arxiv_id": arxiv_id,
                "id_url": id_url,
                "title": title,
                "authors": authors,
                "published": published,
                "updated": updated,
                "abstract": summary,
                "primary_category": primary_category,
                "categories": categories,
                "url": url or id_url,
                "pdf_url": pdf_url,
            }
        )

    return papers


def search_arxiv(
    query: str,
    max_results: int = 10,
    categories: Optional[List[str]] = None,
    sort_by: str = "relevance",
    sort_order: str = "descending",
    timeout_s: int = 30,
    cache_dir: Optional[str | Path] = None,
    refresh: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Search arXiv and return (papers, info).

    papers: list of dicts with keys:
      arxiv_id, title, authors, published, updated, abstract, categories, url, pdf_url, etc.

    info: metadata dict including cache hit/miss and cache path (if used)
    """
    if max_results <= 0:
        return [], {"cache": "skipped", "reason": "max_results<=0"}

    params = {
        "query": _norm_ws(query),
        "max_results": int(max_results),
        "categories": categories or [],
        "sort_by": sort_by,
        "sort_order": sort_order,
    }

    cdir = Path(cache_dir).expanduser().resolve() if cache_dir else DEFAULT_CACHE_DIR
    key = _cache_key(params)

    if not refresh:
        cached = _read_cache(cdir, key)
        if cached and "papers" in cached:
            return cached["papers"], {
                "cache": "hit",
                "cache_path": str(cdir / f"{key}.json"),
                "params": params,
            }

    url = _build_query_url(
        query=params["query"],
        max_results=params["max_results"],
        start=0,
        categories=params["categories"] or None,
        sort_by=params["sort_by"],
        sort_order=params["sort_order"],
    )

    req = urllib.request.Request(url, headers={"User-Agent": "cmpe297-rag-tutor/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            xml_text = resp.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as e:
        raise RuntimeError(f"arXiv API not reachable: {e}")

    papers = _parse_atom(xml_text)

    # Stable ordering (defensive): published desc then arxiv_id
    papers.sort(key=lambda p: (p.get("published", ""), p.get("arxiv_id", "")), reverse=True)

    payload = {
        "params": params,
        "fetched_at_unix": int(time.time()),
        "papers": papers,
    }
    path = _write_cache(cdir, key, payload)

    return papers, {
        "cache": "miss",
        "cache_path": str(path),
        "params": params,
        "url": url,
        "count": len(papers),
    }
