"""Core logic for wiki client."""
from __future__ import annotations


import re
from typing import Dict, List, Optional

import requests


WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {
    "User-Agent": "CMPE297-RAG-Assistant/1.0"
}


def _clean_html_snippet(text: str) -> str:
    """Internal helper for clean html snippet."""
    text = text or ""
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()


def search_wikipedia(query: str, limit: int = 10) -> List[Dict]:
    """
    Search Wikipedia using the official MediaWiki API.

    Returns a normalized lightweight result list:
      {
        "id": str,
        "title": str,
        "abstract": str,   # search snippet used as an abstract-like field
        "year": None,
        "url": str,
        "pdf_url": "",
        "source": "wikipedia",
        "raw": {...}
      }
    """
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": int(limit),
        "format": "json",
        "utf8": 1,
    }

    r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()

    results: List[Dict] = []
    for item in data.get("query", {}).get("search", []):
        page_id = str(item.get("pageid"))
        title = str(item.get("title") or "").strip()
        snippet = _clean_html_snippet(str(item.get("snippet") or ""))

        results.append(
            {
                "id": page_id,
                "title": title,
                "abstract": snippet,
                "year": None,
                "url": f"https://en.wikipedia.org/?curid={page_id}",
                "pdf_url": "",
                "source": "wikipedia",
                "raw": {
                    "pageid": page_id,
                    "title": title,
                    "snippet": snippet,
                },
            }
        )

    return results


def fetch_wikipedia_page(page_id: str | int) -> Dict:
    """
    Fetch full plaintext extract of a Wikipedia page.
    """
    params = {
        "action": "query",
        "prop": "extracts|info",
        "pageids": str(page_id),
        "inprop": "url",
        "explaintext": 1,
        "format": "json",
        "utf8": 1,
    }

    r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()

    pages = data.get("query", {}).get("pages", {})
    page = pages.get(str(page_id), {})

    return {
        "page_id": str(page_id),
        "title": str(page.get("title") or ""),
        "text": str(page.get("extract") or ""),
        "url": str(page.get("fullurl") or f"https://en.wikipedia.org/?curid={page_id}"),
    }