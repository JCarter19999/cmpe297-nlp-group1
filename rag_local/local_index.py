"""
rag_local.local_index

Goal
----
Define a minimal *local vector index* interface that supports:
- add(chunks)
- search(query, top_k)
- save(path)
- load(path)

This ticket defines the API shape only — implementation can be simple.

Design principles
-----------------
- Stable contract for downstream code (rag.py, evaluation scripts)
- Pluggable embeddings (do NOT hard-code one provider here)
- Deterministic behavior (CI-friendly)
- Simple serialization format (JSON) for easy inspection

Index assumptions (baseline)
----------------------------
- You will store:
  1) chunks (text + metadata)
  2) embeddings (list[float] per chunk)

- search(query, top_k) will:
  1) embed the query
  2) compute similarity against stored embeddings
  3) return the best top_k results with scores

This module intentionally does not include any heavy dependencies.
"""
# -----------------------------
# Developer guide: extension points
# -----------------------------

# ============================================================
# HOW TO EXTEND THIS INDEX (Developer Guide)
# ============================================================
#
# 1) Faster search:
#    - Replace linear scan in search() with a real ANN library
#    - Keep public method signatures unchanged
#
# 2) Better persistence:
#    - Save vectors in a binary format for speed
#    - Keep save/load behavior compatible (or bump version)
#
# 3) Richer search results:
#    - Add "source" / "doc_id" fields to chunks upstream
#    - This module will automatically pass them through if you choose to
#
# Key rule:
# - Do not change add/search/save/load signatures without updating all callers.
# ============================================================

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, TypedDict, Union
import numpy as np
import json
import math

# Optional fast NN backend
try:
    from sklearn.neighbors import NearestNeighbors  # type: ignore
except Exception:
    NearestNeighbors = None  # type: ignore

# -----------------------------
# Types / Contracts
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
      - embedding: list[float]   # <-- important (stored with chunk metadata)
    """
    chunk_id: str
    text: str
  
    start: int
    end: int
    doc_id: str
    source: str
    metadata: Dict[str, Any]
    embedding: List[float]

class SearchResult(TypedDict, total=False):
    """
    Output contract for search results.
    """
    chunk_id: str
    text: str
    score: float
    # Optional fields (carried through if present)
    start: int
    end: int
    doc_id: str
    source: str
    metadata: Dict[str, Any]
    embedding: List[float]

class Embedder(Protocol):
    """
    Embedder interface so indexing doesn't care where embeddings come from.
    Implementations might use:
    - sentence-transformers
    - ollama embeddings endpoint
    - TF-IDF baseline
    """
    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed texts."""
        ...
    def embed_query(self, query: str) -> List[float]:
        """Embed query."""
        ...

# -----------------------------
# Public Index Interface
# -----------------------------
class LocalVectorIndex(Protocol):
    """
    Minimal index interface required by the project.
    """
    def add(self, chunks: Sequence[Chunk]) -> None:
        """Add."""
        ...
    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Search."""
        ...
    def save(self, path: Union[str, Path]) -> None:
        """Save."""
        ...
    @classmethod
    def load(cls, path: Union[str, Path], *, embedder: Embedder) -> "LocalVectorIndex":
        """Load."""
        ...

# -----------------------------
# Baseline Implementation
# -----------------------------
@dataclass
class SimpleLocalIndex:
    """
    A simple in-memory vector index with JSON persistence.
    Notes:
    - Intended as a baseline implementation.
    - Linear search (O(n)) is fine for class-scale datasets.
    - You can later swap to FAISS / annoy / etc behind the same API.
    """
    embedder: Embedder
    chunks: List[Chunk]
    vectors: List[List[float]]
    nn: Any  # sklearn NearestNeighbors instance or None

    def __init__(self, *, embedder: Embedder) -> None:
        """Initialize the instance."""
        self.embedder = embedder
        self.chunks = []
        self.vectors = []
        self.nn = None

    def add(self, chunks: Sequence[Chunk]) -> None:
        """
        Add chunks to the index.
        Contract:
        - One vector per chunk
        - chunks are appended in order provided (deterministic)
        """
        if not chunks:
            return

        texts = [c.get("text", "") for c in chunks]
        vecs = self.embedder.embed_texts(texts)

        if len(vecs) != len(chunks):
            raise ValueError("Embedder returned mismatched number of vectors.")

        # Store embedding with metadata
        for c, v in zip(chunks, vecs):
            c["embedding"] = v  

        self.chunks.extend(list(chunks))
        self.vectors.extend(vecs)

        # Rebuild NN structure for fast search (if available)
        self._rebuild_nn()

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Search the index.
        Returns:
        - top_k SearchResult objects with score in descending order
        Score definition:
        - Cosine similarity in [~0, 1] for typical embedding spaces
        """
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        if not self.chunks:
            return []
          
        # Same embedder used for both chunks and queries
        qvec = self.embedder.embed_query(query)
        k = min(top_k, len(self.chunks))
      
        # Fast path: sklearn NN (cosine distance)
        if self.nn is not None:
            Xq = np.array([qvec], dtype=np.float32)
            distances, indices = self.nn.kneighbors(Xq, n_neighbors=min(top_k, len(self.chunks)))
            results: List[SearchResult] = []
            for idx, dist in zip(indices[0].tolist(), distances[0].tolist()):
                c = self.chunks[idx]
                score = 1.0 - float(dist) # cosine distance = 1 - cosine similarity
                r: SearchResult = {
                    "chunk_id": c.get("chunk_id", ""),
                    "text": c.get("text", ""),
                    "score": score,
                }
                # carry optional metadata through
                for k in ("start", "end", "doc_id", "source", "metadata", "embedding"):
                    if k in c:
                        r[k] = c[k]  # type: ignore[index]
                results.append(r)
            return results
          
        # Fallback: deterministic linear scan (cosine similarity)
        scored: List[Tuple[int, float]] = []
        for i, vec in enumerate(self.vectors):
            score = cosine_similarity(qvec, vec)
            scored.append((i, score))

        scored.sort(key=lambda t: t[1], reverse=True)
        scored = scored[:top_k]

        results: List[SearchResult] = []
        for idx, score in scored:
            c = self.chunks[idx]
            r: SearchResult = {
                "chunk_id": c.get("chunk_id", ""),
                "text": c.get("text", ""),
                "score": float(score),
            }
          # Carry optional metadata through
            for k in ("start", "end", "doc_id", "source", "metadata", "embedding"):
              if k in c:
                r[k] = c[k]  # type: ignore[index] 
            results.append(r)
        return results

    def save(self, path: Union[str, Path]) -> None:
        """
        Save index contents to disk as JSON.
        Stored:
        - chunks
        - vectors
        Does NOT store the embedder (must be re-provided at load time).
        """
        p = Path(path).expanduser().resolve()
        payload = {
            "version": 1,
            "chunks": self.chunks,
            "vectors": self.vectors,
            "embedder": {  # Addition to payload to include embedder if we want to document the embedding method used
              "type": self.embedder.__class__.__name__,
              "model": getattr(self.embedder, "model", None),
              "normalize": getattr(self.embedder, "normalize", None),
              "vector_dim": (len(self.vectors[0]) if self.vectors else None),
            },
        }
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Union[str, Path], *, embedder: Embedder) -> "SimpleLocalIndex":
        """
        Load index contents from JSON and attach an embedder.
        """
        p = Path(path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Index file not found: {p}")
        payload = json.loads(p.read_text(encoding="utf-8"))
        idx = cls(embedder=embedder)
        idx.chunks = payload.get("chunks", [])
        idx.vectors = payload.get("vectors", [])

        # If vectors are missing, rebuild them from per-chunk embeddings:
        if not idx.vectors and idx.chunks:
            rebuilt: List[List[float]] = []
            for c in idx.chunks:
                emb = c.get("embedding")
                if emb is None:
                      raise ValueError("Index missing vectors and chunk embeddings; cannot rebuild.")
                rebuilt.append(list(emb))
            idx.vectors = rebuilt
          
        # Verify they match in length
        if len(idx.chunks) != len(idx.vectors):
            raise ValueError("Corrupt index: chunks and vectors length mismatch.")

        # Make load() rebuild the NN index automatically
        idx._rebuild_nn()
        return idx
      
    # -----------------------------
    # Internal helpers
    # -----------------------------
    def _rebuild_nn(self) -> None:
        """
        Build an internal nearest-neighbor structure for fast search.
        Uses sklearn NearestNeighbors if available, otherwise falls back to None (and search will use the baseline linear scan).
        """
        if NearestNeighbors is None:
            self.nn = None
            return
        if not self.vectors:
            self.nn = None
            return
            
        # Sanity: consistent dimensions
        d0 = len(self.vectors[0])
        if any(len(v) != d0 for v in self.vectors):
            raise ValueError("Inconsistent vector dimensions in index.")
            
        X = np.array(self.vectors, dtype=np.float32)

        # Since embeddings are L2-normalized, cosine similarity is well-behaved.
        nn = NearestNeighbors(metric="cosine", algorithm="auto")
        nn.fit(X)
        self.nn = nn
      
# -----------------------------
# Similarity (baseline)
# -----------------------------
def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """
    Compute cosine similarity between two vectors.
    Returns a float in [-1, 1] (typical embedding spaces yield ~[0, 1]).
    """
    if len(a) != len(b):
        raise ValueError("Vector dimension mismatch.")
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


