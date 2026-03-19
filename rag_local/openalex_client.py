"""Core logic for openalex client."""
from __future__ import annotations


from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import requests


def _reconstruct_abstract(abstract_inverted_index: Optional[Dict[str, List[int]]]) -> str:
    """
    OpenAlex sometimes provides abstracts as an inverted index:
      {"word":[pos,pos], "another":[pos], ...}
    We reconstruct a best-effort linear text.
    """
    if not abstract_inverted_index:
        return ""

    positions: Dict[int, str] = {}
    for word, idxs in abstract_inverted_index.items():
        for i in idxs:
            positions[i] = word

    if not positions:
        return ""

    max_i = max(positions.keys())
    words = [positions.get(i, "") for i in range(max_i + 1)]
    return " ".join(w for w in words if w).strip()


@dataclass
class OACandidate:
    """Represents oacandidate."""
    id: str
    title: str
    abstract: str
    year: Optional[int]
    url: Optional[str]
    pdf_url: Optional[str]


class OpenAlexClient:
    """Represents open alex client."""
    def __init__(self, base_url: str = "https://api.openalex.org", timeout_s: int = 30):
        """Initialize the instance."""
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "cmpe297-local-rag/1.0"})

    def search_works(
        self,
        query: str,
        *,
        limit: int = 25,
        mailto: Optional[str] = None,
        open_access_only: bool = True,
    ) -> List[OACandidate]:
        """
        Returns OpenAlex works for a query.

        If open_access_only=True, we filter on is_oa=true.
        """
        url = f"{self.base_url}/works"
        params: Dict[str, Any] = {
            "search": query,
            "per-page": int(limit),
        }
        if mailto:
            params["mailto"] = mailto

        if open_access_only:
            params["filter"] = "is_oa:true"

        r = self.session.get(url, params=params, timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json()
        results = data.get("results", [])

        out: List[OACandidate] = []
        for w in results:
            wid = w.get("id") or ""
            title = w.get("title") or ""

            # abstract can be either direct string or inverted index in "abstract_inverted_index"
            abstract = ""
            if isinstance(w.get("abstract"), str):
                abstract = w.get("abstract") or ""
            else:
                abstract = _reconstruct_abstract(w.get("abstract_inverted_index"))

            year = w.get("publication_year")

            # "primary_location" often has landing page + pdf
            url_landing = None
            pdf_url = None
            primary = w.get("primary_location") or {}
            if isinstance(primary, dict):
                url_landing = primary.get("landing_page_url") or primary.get("source", {}).get("homepage_url")
                pdf_url = primary.get("pdf_url")

            # fallback: best_oa_location
            best = w.get("best_oa_location") or {}
            if isinstance(best, dict):
                url_landing = url_landing or best.get("landing_page_url")
                pdf_url = pdf_url or best.get("pdf_url")

            out.append(
                OACandidate(
                    id=str(wid),
                    title=str(title),
                    abstract=str(abstract),
                    year=year,
                    url=url_landing,
                    pdf_url=pdf_url,
                )
            )

        return out