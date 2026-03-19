"""Core logic for paper rerank."""
from __future__ import annotations


import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests


# --------------------------------------------------------
# Data models
# --------------------------------------------------------
@dataclass
class RankedCandidate:
    """Represents ranked candidate."""
    id: str
    title: str
    abstract: str
    year: Optional[int]
    url: Optional[str]
    pdf_url: Optional[str]
    source: str
    score: float

    # debugging / UI visibility
    lexical_score: float = 0.0
    semantic_score: float = 0.0
    bm25_score: float = 0.0
    llm_score: float = 0.0
    audience_score: float = 0.0
    filter_notes: List[str] = field(default_factory=list)
    matched_topic_terms: List[str] = field(default_factory=list)
    matched_question_terms: List[str] = field(default_factory=list)
    inferred_level: str = "unknown"


@dataclass
class SearchProfile:
    """Represents search profile."""
    topic: str
    research_question: str = ""
    level: str = "undergraduate"   # high_school | undergraduate | masters | phd
    user_query: str = ""

    def effective_query(self) -> str:
        """Effective query."""
        parts = []
        if self.user_query.strip():
            parts.append(self.user_query.strip())
        if self.topic.strip() and self.topic.strip().lower() not in self.user_query.strip().lower():
            parts.append(self.topic.strip())
        if self.research_question.strip():
            parts.append(self.research_question.strip())
        return " | ".join(parts).strip()


# --------------------------------------------------------
# Basic math helpers
# --------------------------------------------------------
def _l2_norm(vec: Sequence[float]) -> float:
    """Internal helper for l2 norm."""
    return math.sqrt(sum(v * v for v in vec)) or 1.0


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity."""
    dot = sum(x * y for x, y in zip(a, b))
    return float(dot / (_l2_norm(a) * _l2_norm(b)))


def _safe_minmax_normalize(values: Sequence[float]) -> List[float]:
    """Internal helper for safe minmax normalize."""
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if math.isclose(vmax, vmin):
        return [0.0 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


def _clamp01(x: float) -> float:
    """Internal helper for clamp01."""
    return max(0.0, min(1.0, float(x)))


def _extract_score_from_text(text: str) -> Optional[float]:
    """
    Parse a numeric relevance score from free-form model output.

    Handles cases where the model returns additional explanation despite
    instructions to output only a number.
    """
    s = (text or "").strip()
    if not s:
        return None

    # Fast path for clean numeric output.
    try:
        return _clamp01(float(s))
    except Exception:
        pass

    # Handle fraction-style ratings first (e.g., "7/10" -> 0.7).
    frac = re.search(r"(?<!\d)(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)(?!\d)", s)
    if frac:
        try:
            num = float(frac.group(1))
            den = float(frac.group(2))
            if den != 0:
                return _clamp01(num / den)
        except Exception:
            pass

    # Fallback: capture numeric substrings (e.g., "0.2\n\nExplanation...").
    matches = re.findall(r"[-+]?\d*\.?\d+", s)
    if not matches:
        return None

    # Prefer a value already in [0,1] if present.
    for m in matches:
        try:
            val = float(m)
            if 0.0 <= val <= 1.0:
                return val
        except Exception:
            continue

    # Otherwise clamp the first parseable number.
    for m in matches:
        try:
            return _clamp01(float(m))
        except Exception:
            continue

    return None


# --------------------------------------------------------
# Text helpers
# --------------------------------------------------------
_TOKEN_RE = re.compile(r"\b[a-zA-Z0-9][a-zA-Z0-9\-_/]*\b")


def tokenize(text: str) -> List[str]:
    """Tokenize."""
    return _TOKEN_RE.findall((text or "").lower())


def unique_tokens(text: str) -> set[str]:
    """Unique tokens."""
    return set(tokenize(text))


def token_overlap_ratio(a: str, b: str) -> float:
    """Token overlap ratio."""
    ta = unique_tokens(a)
    tb = unique_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, len(ta))


def safe_text_for_rank(title: str, abstract: str, max_abs_chars: int = 2000) -> str:
    """Safe text for rank."""
    title = (title or "").strip()
    abstract = (abstract or "").strip()[:max_abs_chars]
    if title and abstract:
        return f"{title}\n\n{abstract}"
    return title or abstract


STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "with", "by",
    "how", "what", "why", "when", "is", "are", "was", "were", "be", "being",
    "from", "that", "this", "these", "those", "as", "at", "it", "its", "into",
    "about", "do", "does", "did", "can", "could", "should", "would", "than",
    "one", "two", "three", "using", "use", "used", "work", "works"
}


def content_terms(text: str) -> List[str]:
    """Content terms."""
    return [t for t in tokenize(text) if len(t) >= 3 and t not in STOPWORDS]


def matched_terms(query_text: str, doc_text: str, max_terms: int = 8) -> List[str]:
    """Matched terms."""
    q_terms = content_terms(query_text)
    d_terms = set(content_terms(doc_text))
    hits: List[str] = []
    seen = set()

    for t in q_terms:
        if t in d_terms and t not in seen:
            hits.append(t)
            seen.add(t)
        if len(hits) >= max_terms:
            break
    return hits


# --------------------------------------------------------
# Embedding client
# --------------------------------------------------------
class OllamaEmbedder:
    """Represents ollama embedder."""
    def __init__(self, host: str = "http://localhost:11434", model: str = "nomic-embed-text", timeout_s: int = 120):
        """Initialize the instance."""
        self.host = host.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.session = requests.Session()

    def _truncate(self, s: str, max_chars: int) -> str:
        """Internal helper for truncate."""
        return (s or "").strip()[:max_chars]

    def embed_batch(
        self,
        texts: List[str],
        *,
        batch_size: int = 8,
        max_chars_per_text: int = 4000,
        sleep_s: float = 0.0,
    ) -> List[List[float]]:
        """Embed batch."""
        if not texts:
            return []

        texts = [self._truncate(t, max_chars_per_text) for t in texts]
        out: List[List[float]] = []

        def try_embed_endpoint(batch: List[str]) -> Optional[List[List[float]]]:
            """Try embed endpoint."""
            url = f"{self.host}/api/embed"
            payload = {"model": self.model, "input": batch}
            try:
                r = self.session.post(url, json=payload, timeout=self.timeout_s)
            except Exception:
                return None
            if r.status_code != 200:
                return None
            data = r.json()
            if "embeddings" in data and isinstance(data["embeddings"], list):
                return data["embeddings"]
            if "data" in data and isinstance(data["data"], list):
                return [row["embedding"] for row in data["data"]]
            return None

        def fallback_embeddings(batch: List[str]) -> List[List[float]]:
            """Fallback embeddings."""
            url = f"{self.host}/api/embeddings"
            embs: List[List[float]] = []
            for t in batch:
                payload = {"model": self.model, "prompt": t}
                r = self.session.post(url, json=payload, timeout=self.timeout_s)
                r.raise_for_status()
                data = r.json()
                embs.append(data["embedding"])
                if sleep_s > 0:
                    time.sleep(sleep_s)
            return embs

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embs = try_embed_endpoint(batch)
            if embs is not None:
                out.extend(embs)
            else:
                out.extend(fallback_embeddings(batch))
            if sleep_s > 0:
                time.sleep(sleep_s)

        return out


# --------------------------------------------------------
# BM25
# --------------------------------------------------------
class BM25:
    """Represents bm25."""
    def __init__(self, docs: Sequence[str], k1: float = 1.5, b: float = 0.75):
        """Initialize the instance."""
        self.k1 = k1
        self.b = b
        self.docs = [self._tokenize(d) for d in docs]
        self.doc_lens = [len(d) for d in self.docs]
        self.avgdl = sum(self.doc_lens) / len(self.doc_lens) if self.doc_lens else 0.0
        self.tf = [Counter(d) for d in self.docs]

        df = Counter()
        for d in self.docs:
            df.update(set(d))
        self.df = df
        self.N = len(self.docs)

    def _tokenize(self, text: str) -> List[str]:
        """Internal helper for tokenize."""
        return tokenize(text)

    def score(self, query: str, doc_index: int) -> float:
        """Score."""
        if self.N == 0 or self.avgdl <= 0:
            return 0.0

        tokens = self._tokenize(query)
        score = 0.0
        dl = self.doc_lens[doc_index]
        tf_doc = self.tf[doc_index]

        for term in tokens:
            if term not in self.df:
                continue

            df = self.df[term]
            idf = math.log(1 + (self.N - df + 0.5) / (df + 0.5))

            tf = tf_doc.get(term, 0)
            denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            if denom == 0:
                continue

            score += idf * (tf * (self.k1 + 1) / denom)

        return score


# --------------------------------------------------------
# Optional LLM judge
# --------------------------------------------------------
def llm_relevance_score_safe(
    *,
    query: str,
    title: str,
    abstract: str,
    host: str = "http://localhost:11434",
    model: str = "llama3.1:8b",
    topic: str = "",
    research_question: str = "",
    level: str = "undergraduate",
    timeout_s: int = 45,
) -> Tuple[float, Optional[str]]:
    """Llm relevance score safe."""
    prompt = f"""
You are a relevance scorer for knowledge-source search.

User topic: {topic}
Research question: {research_question}
Education level: {level}
Search query: {query}

Candidate:
Title: {title}
Summary: {abstract}

Score this source from 0.0 to 1.0 for how useful it would be for this user.
Consider:
- topical relevance
- whether it seems appropriate for the user's level
- whether it helps answer the research question

Return ONLY a number between 0.0 and 1.0.
""".strip()

    url = f"{host.rstrip('/')}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}

    try:
        r = requests.post(url, json=payload, timeout=timeout_s)
        if r.status_code != 200:
            return 0.0, f"LLM reranker unavailable ({r.status_code}); continuing without it."
        data = r.json()
        text = str(data.get("response", "")).strip()
        parsed = _extract_score_from_text(text)
        if parsed is None:
            return 0.0, "LLM reranker unavailable (non-numeric model output); continuing without it."
        return parsed, None
    except Exception as e:
        return 0.0, f"LLM reranker unavailable ({type(e).__name__}); continuing without it."


# --------------------------------------------------------
# Audience / difficulty heuristics
# --------------------------------------------------------
BEGINNER_HINTS = {
    "tutorial", "review", "survey", "introduction", "introductory", "overview",
    "educational", "teaching", "didactic", "history", "basics"
}
ADVANCED_HINTS = {
    "benchmark", "optimization", "proof", "theorem", "ablation", "derivation",
    "formal", "novel", "state-of-the-art", "sota", "architecture", "scaling"
}


def audience_alignment_score(title: str, abstract: str, level: str) -> float:
    """Audience alignment score."""
    text = f"{title} {abstract}".lower()
    beginner_hits = sum(1 for kw in BEGINNER_HINTS if kw in text)
    advanced_hits = sum(1 for kw in ADVANCED_HINTS if kw in text)

    if level == "high_school":
        raw = 0.7 * (1 if beginner_hits > 0 else 0) + 0.3 * (0 if advanced_hits > 1 else 1)
        return _clamp01(raw)
    if level == "undergraduate":
        raw = 0.6 + 0.15 * min(beginner_hits, 2) - 0.1 * max(advanced_hits - 2, 0)
        return _clamp01(raw)
    if level == "masters":
        raw = 0.6 + 0.15 * min(advanced_hits, 2)
        return _clamp01(raw)
    if level == "phd":
        raw = 0.5 + 0.25 * min(advanced_hits, 2)
        return _clamp01(raw)

    return 0.5


def infer_document_level(title: str, abstract: str) -> str:
    """Infer document level."""
    text = f"{title} {abstract}".lower()
    beginner_hits = sum(1 for kw in BEGINNER_HINTS if kw in text)
    advanced_hits = sum(1 for kw in ADVANCED_HINTS if kw in text)

    if beginner_hits >= 2 and advanced_hits == 0:
        return "beginner-friendly"
    if advanced_hits >= 3:
        return "advanced"
    if advanced_hits >= 1:
        return "intermediate-advanced"
    if beginner_hits >= 1:
        return "intro-intermediate"
    return "general"


# --------------------------------------------------------
# Layered filtering + scoring
# --------------------------------------------------------
def candidate_hygiene_filter(c: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Source-aware filter:
    - OpenAlex: requires direct PDF
    - Wikipedia: requires URL and some text
    """
    notes: List[str] = []

    title = str(c.get("title") or "").strip()
    abstract = str(c.get("abstract") or "").strip()
    url = str(c.get("url") or "").strip()
    pdf_url = str(c.get("pdf_url") or "").strip()
    source = str(c.get("source") or "").strip().lower()

    if not title:
        return False, ["missing title"]

    if source == "openalex":
        if not pdf_url:
            return False, ["excluded: no direct pdf"]
        if len(abstract) < 120:
            notes.append("short abstract")
        return True, notes

    if source == "wikipedia":
        if not url:
            return False, ["excluded: missing article url"]
        if len(abstract) < 20:
            notes.append("short summary")
        notes.append("general-knowledge source")
        return True, notes

    # default fallback
    if not url and not pdf_url:
        return False, ["excluded: missing source url"]
    if len(abstract) < 40:
        notes.append("short summary")
    return True, notes


def lexical_prefilter_score(c: Dict[str, Any], profile: SearchProfile) -> float:
    """Lexical prefilter score."""
    title = str(c.get("title") or "")
    abstract = str(c.get("abstract") or "")
    doc_text = f"{title}\n{abstract}"

    topic_overlap = token_overlap_ratio(profile.topic, doc_text) if profile.topic.strip() else 0.0
    rq_overlap = token_overlap_ratio(profile.research_question, doc_text) if profile.research_question.strip() else 0.0
    user_overlap = token_overlap_ratio(profile.user_query, doc_text) if profile.user_query.strip() else 0.0

    title_topic_overlap = token_overlap_ratio(profile.topic, title) if profile.topic.strip() else 0.0
    title_rq_overlap = token_overlap_ratio(profile.research_question, title) if profile.research_question.strip() else 0.0

    score = (
        0.30 * user_overlap
        + 0.25 * topic_overlap
        + 0.25 * rq_overlap
        + 0.10 * title_topic_overlap
        + 0.10 * title_rq_overlap
    )
    return _clamp01(score)


def build_filtered_candidate_pool(
    *,
    candidates: List[Dict[str, Any]],
    profile: SearchProfile,
    lexical_keep_k: int = 25,
    lexical_min_score: float = 0.03,
) -> List[Dict[str, Any]]:
    """Build filtered candidate pool."""
    staged: List[Dict[str, Any]] = []

    for c in candidates:
        ok, notes = candidate_hygiene_filter(c)
        if not ok:
            continue

        lex = lexical_prefilter_score(c, profile)
        if lex < lexical_min_score:
            continue

        cc = dict(c)
        cc["_lexical_score"] = lex
        cc["_filter_notes"] = list(notes)
        cc["_audience_score"] = audience_alignment_score(
            str(c.get("title") or ""),
            str(c.get("abstract") or ""),
            profile.level,
        )
        staged.append(cc)

    staged.sort(key=lambda x: (x["_lexical_score"], x["_audience_score"]), reverse=True)
    return staged[:max(1, int(lexical_keep_k))]


def rerank_layered(
    *,
    profile: SearchProfile,
    candidates: List[Dict[str, Any]],
    ollama_host: str,
    embed_model: str = "nomic-embed-text",
    llm_model: str = "llama3.1:8b",
    use_llm_reranker: bool = False,
    lexical_keep_k: int = 25,
    top_n: int = 10,
    alpha_semantic: float = 0.50,
    beta_bm25: float = 0.20,
    gamma_lexical: float = 0.15,
    delta_llm: float = 0.10,
    epsilon_audience: float = 0.05,
    abstract_max_chars: int = 2000,
) -> Tuple[List[RankedCandidate], List[str]]:
    """Rerank layered."""
    warnings: List[str] = []

    pool = build_filtered_candidate_pool(
        candidates=candidates,
        profile=profile,
        lexical_keep_k=lexical_keep_k,
    )
    if not pool:
        return [], ["No candidates survived the lexical/intention filter."]

    query_text = profile.effective_query()
    embedder = OllamaEmbedder(host=ollama_host, model=embed_model)

    cand_texts = [
        safe_text_for_rank(
            str(c.get("title") or ""),
            str(c.get("abstract") or ""),
            max_abs_chars=abstract_max_chars,
        )
        for c in pool
    ]

    embs = embedder.embed_batch([query_text] + cand_texts, batch_size=8, max_chars_per_text=4000)
    if not embs or len(embs) != (1 + len(pool)):
        raise RuntimeError("Embedding call failed or returned the wrong number of vectors.")

    q_emb = embs[0]
    cand_embs = embs[1:]
    semantic_scores = [cosine_similarity(q_emb, e) for e in cand_embs]

    bm25 = BM25(cand_texts)
    raw_bm25 = [bm25.score(query_text, i) for i in range(len(pool))]
    bm25_norm = _safe_minmax_normalize(raw_bm25)

    llm_scores: List[float] = []
    llm_warning_emitted = False
    for c in pool:
        if use_llm_reranker and delta_llm > 0:
            score, warn = llm_relevance_score_safe(
                query=query_text,
                title=str(c.get("title") or ""),
                abstract=str(c.get("abstract") or ""),
                host=ollama_host,
                model=llm_model,
                topic=profile.topic,
                research_question=profile.research_question,
                level=profile.level,
            )
            llm_scores.append(score)
            if warn and not llm_warning_emitted:
                warnings.append(warn)
                llm_warning_emitted = True
        else:
            llm_scores.append(0.0)

    sem_norm = [0.5 * (s + 1.0) for s in semantic_scores]
    lex_scores = [float(c.get("_lexical_score", 0.0)) for c in pool]
    audience_scores = [float(c.get("_audience_score", 0.0)) for c in pool]

    total_w = alpha_semantic + beta_bm25 + gamma_lexical + delta_llm + epsilon_audience
    if total_w <= 0:
        total_w = 1.0

    ranked: List[RankedCandidate] = []
    for c, sem, bm, lex, llm, aud in zip(pool, sem_norm, bm25_norm, lex_scores, llm_scores, audience_scores):
        final_score = (
            (alpha_semantic / total_w) * sem
            + (beta_bm25 / total_w) * bm
            + (gamma_lexical / total_w) * lex
            + (delta_llm / total_w) * llm
            + (epsilon_audience / total_w) * aud
        )

        title = str(c.get("title") or "")
        abstract = str(c.get("abstract") or "")
        doc_text = f"{title}\n{abstract}"

        ranked.append(
            RankedCandidate(
                id=str(c.get("id") or ""),
                title=title,
                abstract=abstract,
                year=c.get("year"),
                url=c.get("url"),
                pdf_url=c.get("pdf_url"),
                source=str(c.get("source") or "unknown"),
                score=float(final_score),
                lexical_score=float(lex),
                semantic_score=float(sem),
                bm25_score=float(bm),
                llm_score=float(llm),
                audience_score=float(aud),
                filter_notes=list(c.get("_filter_notes", [])),
                matched_topic_terms=matched_terms(profile.topic, doc_text, max_terms=8),
                matched_question_terms=matched_terms(profile.research_question, doc_text, max_terms=8),
                inferred_level=infer_document_level(title, abstract),
            )
        )

    ranked.sort(key=lambda x: x.score, reverse=True)
    return ranked[:max(1, int(top_n))], warnings