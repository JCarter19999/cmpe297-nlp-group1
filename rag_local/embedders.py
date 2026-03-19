"""
rag_local.embedders

Concrete embedding implementations for the RAG pipeline.

Why this file exists
--------------------
Your local_index.py defines an Embedder Protocol (interface) but the project did not
yet include an implementation. This file closes that gap.

Primary embedder
----------------
- OllamaEmbedder: uses Ollama's /api/embeddings endpoint.

Fallback embedder
-----------------
- HashEmbedder: deterministic pseudo-embeddings from hashing (not semantically great,
  but unblocks testing + CI if embeddings aren't available).

Design rules
------------
- No heavyweight dependencies.
- Deterministic behavior where possible.
- Clear errors if Ollama is unreachable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Optional
import hashlib
import json
import math
import urllib.request
import urllib.error


def _l2_normalize(vec: List[float]) -> List[float]:
    """Internal helper for l2 normalize."""
    n = 0.0
    for x in vec:
        n += x * x
    if n <= 0.0:
        return vec
    inv = 1.0 / math.sqrt(n)
    return [x * inv for x in vec]


@dataclass
class OllamaEmbedder:
    """
    Embedder that calls Ollama embeddings endpoint.

    Endpoint:
      POST {host}/api/embeddings
      payload: {"model": "...", "prompt": "..."}

    Response:
      {"embedding": [float, float, ...], ...}

    Notes:
    - Some Ollama versions use different naming; this implementation targets the common API.
    - We L2-normalize vectors to make cosine similarity stable.
    """

    model: str = "nomic-embed-text"
    host: str = "http://localhost:11434"
    timeout_s: int = 60
    normalize: bool = True

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed texts."""
        return [self._embed_one(t) for t in texts]

    def embed_query(self, query: str) -> List[float]:
        """Embed query."""
        return self._embed_one(query)

    def _embed_one(self, text: str) -> List[float]:
        """Internal helper for embed one."""
        t = (text or "").strip()
        if not t:
            # Return a small deterministic vector for empty input
            v = [0.0] * 16
            return v

        url = self.host.rstrip("/") + "/api/embeddings"
        payload = {"model": self.model, "prompt": t}

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise RuntimeError(f"Ollama embeddings not reachable at {self.host}. Is Ollama running? ({e})")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Bad response from Ollama embeddings (invalid JSON): {e}")

        emb = data.get("embedding", None)
        if not isinstance(emb, list) or not emb:
            raise RuntimeError(f"Ollama embeddings response missing 'embedding' list. Got keys={list(data.keys())}")

        vec = [float(x) for x in emb]
        return _l2_normalize(vec) if self.normalize else vec


@dataclass
class HashEmbedder:
    """
    Deterministic fallback embedder (NOT semantic).

    Useful for:
    - CI runs
    - quick integration tests
    - situations where embeddings endpoint isn't installed

    Strategy:
    - hash text into a fixed-dimension vector using repeated SHA256 blocks
    - convert bytes to floats in [-1, 1]
    - L2 normalize
    """

    dim: int = 256
    normalize: bool = True

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed texts."""
        return [self._hash_to_vec(t) for t in texts]

    def embed_query(self, query: str) -> List[float]:
        """Embed query."""
        return self._hash_to_vec(query)

    def _hash_to_vec(self, text: str) -> List[float]:
        """Internal helper for hash to vec."""
        t = (text or "").encode("utf-8", errors="ignore")
        if not t:
            v = [0.0] * self.dim
            return v

        # generate enough bytes to fill dim
        needed = self.dim
        out: List[float] = []
        counter = 0
        while len(out) < needed:
            h = hashlib.sha256()
            h.update(t)
            h.update(counter.to_bytes(4, "little"))
            digest = h.digest()  # 32 bytes
            for b in digest:
                # map byte 0..255 to float -1..1
                out.append((b / 127.5) - 1.0)
                if len(out) >= needed:
                    break
            counter += 1

        v = out[:needed]
        return _l2_normalize(v) if self.normalize else v

@dataclass
class SentenceTransformersEmbedder:
    """
    SentenceTransformers embedder (recommended semantic embeddings).

    Notes:
    - Uses a local transformer model (downloads from Hugging Face on first run).
    - Import is inside __post_init__ to avoid hard dependency unless selected.
    - Returns L2-normalized vectors by default for stable cosine similarity.
    """

    model: str = "all-MiniLM-L6-v2"
    normalize: bool = True

    def __post_init__(self) -> None:
        """Internal helper for post init."""
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is not installed. Install with: pip install -e '.[rag]' "
                "or pip install sentence-transformers"
            ) from e

        self._model = SentenceTransformer(self.model)

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed texts."""
        vecs = self._model.encode(list(texts),normalize_embeddings=self.normalize,convert_to_numpy=True,show_progress_bar=False)
        return vecs.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Embed query."""
        vec = self._model.encode(query,normalize_embeddings=self.normalize,convert_to_numpy=True,show_progress_bar=False)
        return vec.tolist()

def make_embedder(
    *,
    backend: str,
    model: Optional[str] = None,
    host: str = "http://localhost:11434",
    timeout_s: int = 60,
) -> object:
    """
    Factory used by chat/rag.

    backend:
      - "ollama": OllamaEmbedder
      - "hash": HashEmbedder
      - "sbert": SentenceTransformersEmbedder
    """
    b = (backend or "").strip().lower()
    if b == "ollama":
        return OllamaEmbedder(model=model or "nomic-embed-text", host=host, timeout_s=timeout_s)
    if b == "hash":
        return HashEmbedder()
    if b == "sbert":
        return SentenceTransformersEmbedder(model=model or "all-MiniLM-L6-v2", normalize=True)
    raise ValueError(f"Unknown embedder backend: {backend!r}. Use 'ollama', 'hash', or 'sbert'.")
