"""
rag_local.rag

Goal
----
Provide the baseline RAG pipeline functions:
- build_index(): load -> chunk -> index
- (later) answer_query(): retrieve -> generate

This version adds progress callback support for Streamlit rebuild progress.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from rag_local.loaders import load_documents, Document
from rag_local.chunking import chunk_text, Chunk
from rag_local.local_index import SimpleLocalIndex, Embedder
from rag_local.ollama_client import chat as ollama_chat


ProgressCallback = Optional[Callable[[float, str], None]]


@dataclass(frozen=True)
class BuildStats:
    """Represents build stats."""
    doc_count: int
    chunk_count: int


def _emit(progress_callback: ProgressCallback, frac: float, message: str) -> None:
    """Internal helper for emit."""
    if progress_callback is None:
        return
    frac = max(0.0, min(1.0, float(frac)))
    progress_callback(frac, message)


def build_index(
    *,
    data_dir: Union[str, Path],
    embedder: Embedder,
    chunk_size: int = 800,
    overlap: int = 200,
    progress_callback: ProgressCallback = None,
    embed_batch_size: int = 32,
) -> Tuple[SimpleLocalIndex, BuildStats]:
    """
    Build a local index with progress reporting.

    Progress phases:
      0.00-0.10  load documents
      0.10-0.35  chunk documents
      0.35-0.95  embed/add chunks
      0.95-1.00  finalize
    """
    data_dir = Path(data_dir).resolve()

    _emit(progress_callback, 0.01, "Scanning data directory...")
    docs: List[Document] = load_documents(data_dir)
    doc_count = len(docs)

    _emit(progress_callback, 0.10, f"Loaded {doc_count} document(s). Chunking...")

    all_chunks: List[Chunk] = []

    if doc_count == 0:
        index = SimpleLocalIndex(embedder=embedder)
        _emit(progress_callback, 1.00, "No documents found. Empty index created.")
        return index, BuildStats(doc_count=0, chunk_count=0)

    # Chunk per-document so progress can update incrementally
    for i, d in enumerate(docs, start=1):
        doc_text = d.get("text", "")
        source = d.get("source", "unknown")
        doc_id = d.get("doc_id", None)

        chunks = chunk_text(
            doc_text,
            chunk_size=chunk_size,
            overlap=overlap,
            doc_id=doc_id,
            source=source,
            metadata=d.get("metadata", d.get("meta", None)),
            include_spans=True,
        )
        all_chunks.extend(chunks)

        frac = 0.10 + 0.25 * (i / doc_count)
        _emit(progress_callback, frac, f"Chunking documents {i}/{doc_count}...")

    chunk_count = len(all_chunks)

    _emit(progress_callback, 0.35, f"Created {chunk_count} chunk(s). Embedding...")

    index = SimpleLocalIndex(embedder=embedder)

    if chunk_count == 0:
        _emit(progress_callback, 1.00, "No chunks created. Empty index built.")
        return index, BuildStats(doc_count=doc_count, chunk_count=0)

    # Add in batches so embedding/indexing progress can update
    for start in range(0, chunk_count, embed_batch_size):
        end = min(start + embed_batch_size, chunk_count)
        batch = all_chunks[start:end]
        index.add(batch)

        batch_progress = end / chunk_count
        frac = 0.35 + 0.60 * batch_progress
        _emit(progress_callback, frac, f"Embedding chunks {end}/{chunk_count}...")

    _emit(progress_callback, 0.98, "Finalizing index...")
    _emit(progress_callback, 1.00, f"Index build complete. {chunk_count} chunk(s) ready.")

    stats = BuildStats(doc_count=doc_count, chunk_count=chunk_count)
    return index, stats


def save_index(index: SimpleLocalIndex, path: Union[str, Path]) -> None:
    """Save index."""
    index.save(path)


def load_index(path: Union[str, Path], *, embedder: Embedder) -> SimpleLocalIndex:
    """Load index."""
    return SimpleLocalIndex.load(path, embedder=embedder)


def _format_retrieved_context(results: Sequence[Dict[str, Any]], *, max_chars: int = 6000) -> str:
    """
    Build a stable context block with source tags [S1], [S2], ... and a char budget.
    """
    parts: List[str] = []
    used = 0
    for i, r in enumerate(results, start=1):
        txt = (r.get("text") or "").strip()
        if not txt:
            continue
        src = r.get("source", "")
        chunk_id = r.get("chunk_id", "")
        score = float(r.get("score", 0.0))

        header = f"[S{i}] score={score:.3f} chunk_id={chunk_id}"
        if src:
            header += f" source={src}"
        block = header + "\n" + txt + "\n"

        if used + len(block) > max_chars:
            remain = max_chars - used
            if remain > 200:
                parts.append(block[:remain])
            break

        parts.append(block)
        used += len(block)

    return "\n".join(parts).strip()


def answer_query(
    *,
    query: str,
    index: SimpleLocalIndex,
    model: str,
    system_prompt: str,
    top_k: int = 5,
    max_context_chars: int = 6000,
) -> Dict[str, Any]:
    """
    Full RAG loop:
      1) retrieve top_k chunks from local index
      2) build prompt with citations [S1], [S2], ...
      3) call LLM
      4) return answer + sources
    """
    q = (query or "").strip()
    if not q:
        return {"answer": "(Empty query.)", "sources": [], "used_top_k": 0}

    results = index.search(q, top_k=top_k)
    context = _format_retrieved_context(results, max_chars=max_context_chars)

    rag_rules = (
        "You are an NLP tutor.\n"
        "Use the SOURCES to answer.\n"
        "Rules:\n"
        "- If the sources do not contain the answer, say so and ask a clarifying question.\n"
        "- Cite sources like [S1], [S2] for factual claims.\n"
        "- Be clear and step-by-step.\n"
    )

    user_content = f"SOURCES:\n{context}\n\nQUESTION:\n{q}\n"

    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "system", "content": rag_rules},
        {"role": "user", "content": user_content},
    ]

    answer = ollama_chat(messages, model=model)

    sources = []
    for i, r in enumerate(results, start=1):
        sources.append(
            {
                "source_id": f"S{i}",
                "chunk_id": r.get("chunk_id", ""),
                "score": float(r.get("score", 0.0)),
                "file": r.get("source", ""),
                "snippet": (r.get("text") or "")[:800],
            }
        )

    return {
        "answer": answer or "(No response.)",
        "sources": sources,
        "used_top_k": len(results),
    }