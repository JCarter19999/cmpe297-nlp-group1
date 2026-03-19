"""Core logic for wiki fetch."""
from __future__ import annotations


import json
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from rag_local.wiki_client import fetch_wikipedia_page


def _slugify(text: str, max_len: int = 100) -> str:
    """Internal helper for slugify."""
    text = (text or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    if not text:
        text = "untitled"
    return text[:max_len].rstrip("-")


def _query_slug(query: str) -> str:
    """Internal helper for query slug."""
    return _slugify(query or "wikipedia-search", max_len=60)


def materialize_wikipedia_selected(
    *,
    selected: Sequence[Dict],
    data_dir: Path,
    query: str,
) -> Tuple[List[Dict], List[Tuple[str, str]]]:
    """
    Save Wikipedia pages locally so they can be indexed by the existing RAG pipeline.

    Output:
      rag_local/Data/wikipedia/<query_slug>/<article_slug>/
        article.txt
        metadata.json
    """
    data_dir = Path(data_dir).resolve()
    base_dir = data_dir / "wikipedia" / _query_slug(query)
    base_dir.mkdir(parents=True, exist_ok=True)

    saved: List[Dict] = []
    skipped: List[Tuple[str, str]] = []

    for item in selected:
        page_id = str(item.get("id") or "").strip()
        title = str(item.get("title") or "").strip()

        if not page_id:
            skipped.append((title or "unknown", "missing wikipedia page id"))
            continue

        try:
            article = fetch_wikipedia_page(page_id)
            article_title = str(article.get("title") or title or page_id).strip()
            article_text = str(article.get("text") or "").strip()
            article_url = str(article.get("url") or item.get("url") or "").strip()

            if not article_title or not article_text:
                skipped.append((page_id, "empty title or empty article text"))
                continue

            article_slug = _slugify(article_title, max_len=100)
            article_dir = base_dir / article_slug
            article_dir.mkdir(parents=True, exist_ok=True)

            txt_path = article_dir / "article.txt"
            meta_path = article_dir / "metadata.json"

            txt_path.write_text(
                f"Title: {article_title}\n"
                f"Source: wikipedia\n"
                f"URL: {article_url}\n\n"
                f"{article_text}",
                encoding="utf-8",
                errors="ignore",
            )

            metadata = {
                "source": "wikipedia",
                "page_id": page_id,
                "title": article_title,
                "url": article_url,
                "query": query,
                "content_tier": "fulltext",
            }
            meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

            saved.append(
                {
                    "id": page_id,
                    "title": article_title,
                    "source": "wikipedia",
                    "article_dir": str(article_dir),
                    "content_tier": "fulltext",
                    "files_written": [str(txt_path), str(meta_path)],
                    "note": "saved wikipedia plaintext article",
                }
            )

        except Exception as e:
            skipped.append((page_id, str(e)))

    return saved, skipped