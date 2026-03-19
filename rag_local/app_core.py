"""
rag_local.app_core

Shared backend functions for:
  - CLI chatbot (rag_local/chat.py)
  - Streamlit UI (streamlit_app.py)

Streamlit should remain a thin UI layer; this module keeps all RAG/LLM logic
in rag_local/ so the CLI and GUI do not diverge.
"""

from __future__ import annotations

import time
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union

from rag_local.embedders import make_embedder
from rag_local.ollama_client import chat as ollama_chat
from rag_local.rag import build_index, load_index, save_index

Role = Literal["system", "user", "assistant"]
Message = Dict[str, str]

Cfg = Union[Mapping[str, Any], Any]  # dict-like (Streamlit) OR AppConfig-like (CLI)


# -----------------------------
# Config helpers (dict OR object)
# -----------------------------
def cfg_get(cfg: Cfg, key: str, default: Any = None) -> Any:
    """Cfg get."""
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def cfg_bool(cfg: Cfg, key: str, default: bool = False) -> bool:
    """Cfg bool."""
    return bool(cfg_get(cfg, key, default))


def cfg_int(cfg: Cfg, key: str, default: int) -> int:
    """Cfg int."""
    try:
        return int(cfg_get(cfg, key, default))
    except Exception:
        return int(default)


def cfg_str(cfg: Cfg, key: str, default: str) -> str:
    """Cfg str."""
    v = cfg_get(cfg, key, default)
    return default if v is None else str(v)


def cfg_model(cfg: Cfg, default: str = "llama3.1:8b") -> str:
    """
    Backward-compatible model lookup:
    - Streamlit uses `chat_model`
    - CLI config historically used `model`
    """
    chat_model = cfg_get(cfg, "chat_model", None)
    if chat_model:
        return str(chat_model)

    model = cfg_get(cfg, "model", None)
    if model:
        return str(model)

    return default


# -----------------------------
# Streamlit-friendly cfg merge
# -----------------------------
def cfg_with_overrides(cfg: dict, **overrides) -> dict:
    """
    Merge overrides into cfg (shallow). This keeps Streamlit flexible and prevents
    signature mismatch errors when we add new UI controls.
    """
    out = dict(cfg or {})
    for k, v in overrides.items():
        if v is not None:
            out[k] = v

    # keep both names synchronized for compatibility
    if "chat_model" in out and "model" not in out:
        out["model"] = out["chat_model"]
    if "model" in out and "chat_model" not in out:
        out["chat_model"] = out["model"]

    return out


# -----------------------------
# Index / embedder init
# -----------------------------
def init_embedder(cfg: Cfg):
    """Create the embedder used by the local JSON index."""
    return make_embedder(
        backend=cfg_str(cfg, "embed_backend", "ollama"),
        model=cfg_str(cfg, "embed_model", "nomic-embed-text"),
        host=cfg_str(cfg, "ollama_host", "http://localhost:11434"),
    )


def get_index_chunk_count(index: Any) -> int:
    """Get index chunk count."""
    try:
        chunks = getattr(index, "chunks", None)
        if chunks is None and isinstance(index, dict):
            chunks = index.get("chunks", [])
        return len(chunks or [])
    except Exception:
        return 0


def init_index(
    cfg: Cfg,
    *,
    force_rebuild: bool = False,
    progress_callback=None,
) -> Tuple[Optional[Any], Dict[str, Any]]:
    """Load (or build) the local JSON index.

    Returns (index, meta). If RAG is disabled, index is None.
    """
    rag_enabled = cfg_bool(cfg, "rag_enabled", True)
    if not rag_enabled:
        return None, {"rag_enabled": False, "reason": "rag_enabled is False"}

    embedder = init_embedder(cfg)

    data_dir = get_corpus_docs_dir(cfg)
    index_path = get_corpus_index_path(cfg)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    meta: Dict[str, Any] = {
        "rag_enabled": True,
        "data_dir": str(data_dir),
        "index_path": str(index_path),
        "built": False,
        "loaded": False,
    }

    # Load existing
    if index_path.exists() and not force_rebuild:
        if progress_callback:
            progress_callback(0.10, "Loading existing index...")
        index = load_index(index_path, embedder=embedder)
        if progress_callback:
            progress_callback(1.00, "Existing index loaded.")
        meta["loaded"] = True
        meta["chunk_count"] = get_index_chunk_count(index)
        return index, meta

    # Rebuild
    index, stats = build_index(
        data_dir=data_dir,
        embedder=embedder,
        chunk_size=cfg_int(cfg, "chunk_size", 800),
        overlap=cfg_int(cfg, "overlap", 200),
        progress_callback=progress_callback,
    )

    if progress_callback:
        progress_callback(0.99, "Saving index to disk...")
    save_index(index, index_path)
    if progress_callback:
        progress_callback(1.00, "Index saved.")

    meta.update(
        {
            "built": True,
            "doc_count": int(getattr(stats, "doc_count", 0)),
            "chunk_count": int(getattr(stats, "chunk_count", 0)),
        }
    )
    return index, meta


def refresh_index(
    cfg: Cfg,
    *,
    force_rebuild: bool = True,
    progress_callback=None,
) -> Tuple[Optional[Any], Dict[str, Any]]:
    """
    Thin wrapper used by the GUI when newly fetched documents need to become
    immediately searchable.
    """
    return init_index(cfg, force_rebuild=force_rebuild, progress_callback=progress_callback)


# -----------------------------
# Evaluation helpers
# -----------------------------
def run_eval_from_cfg(
    cfg: Cfg,
    index: Any,
    *,
    n_items: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Lazy import to avoid circular dependency:
      app_core -> eval_rag -> app_core
    """
    if index is None:
        return {"rows": [], "summary": {}, "error": "Index is not loaded."}

    from rag_local.eval_rag import default_eval_items, run_rag_eval

    if n_items is None:
        n_items = cfg_int(cfg, "rag_eval_n", 3)

    items = default_eval_items()[: max(0, int(n_items))]
    return run_rag_eval(cfg=cfg, index=index, items=items)

def run_conversation_eval_from_messages(messages: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    GUI-focused eval: score the actual conversation already stored in session state.
    Lazy import avoids circular dependency issues.
    """
    from rag_local.eval_rag import run_conversation_eval
    return run_conversation_eval(messages)
# -----------------------------
# Chat helpers
# -----------------------------
def _trim_history(history: List[Message], *, max_messages: int) -> List[Message]:
    """Internal helper for trim history."""
    if max_messages <= 0:
        return []
    if len(history) <= max_messages:
        return list(history)
    return list(history[-max_messages:])


def _format_retrieved_context(results: Sequence[Dict[str, Any]], *, max_chars: int = 6000) -> str:
    """Internal helper for format retrieved context."""
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


def _answer_with_rag(*, query: str, index: Any, cfg: Cfg) -> Tuple[str, List[Dict[str, Any]], Dict[str, float]]:
    """Internal helper for answer with rag."""
    q = (query or "").strip()
    if not q:
        return "(Empty query.)", [], {"retrieval_s": 0.0, "generation_s": 0.0, "total_s": 0.0}

    t0 = time.perf_counter()

    # Search
    top_k = cfg_int(cfg, "top_k", 5)
    t_retrieval_0 = time.perf_counter()
    results = index.search(q, top_k=top_k)
    retrieval_s = time.perf_counter() - t_retrieval_0

    # Build context
    max_context_chars = cfg_int(cfg, "max_context_chars", 6000)
    context = _format_retrieved_context(results, max_chars=max_context_chars)

    rag_rules = (
        "Use the SOURCES to answer.\n"
        "Rules:\n"
        "- If the sources do not contain the answer, say so.\n"
        "- Cite sources like [S1], [S2] for factual claims.\n"
        "- Be clear and step-by-step.\n"
    )

    system_prompt = cfg_str(cfg, "system_prompt", "You are a helpful assistant.").strip()
    user_content = f"SOURCES:\n{context}\n\nQUESTION:\n{q}\n"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": rag_rules},
        {"role": "user", "content": user_content},
    ]

    t_generation_0 = time.perf_counter()
    reply = ollama_chat(
        messages,
        model=cfg_model(cfg),
        host=cfg_str(cfg, "ollama_host", "http://localhost:11434"),
    )
    generation_s = time.perf_counter() - t_generation_0
    reply_text = reply or "(No response.)"

    sources: List[Dict[str, Any]] = []
    for i, r in enumerate(results, start=1):
        snippet = (r.get("text") or "").strip()
        if len(snippet) > 800:
            snippet = snippet[:800].rstrip() + "…"

        sources.append(
            {
                "source_id": f"S{i}",
                "chunk_id": r.get("chunk_id", ""),
                "score": float(r.get("score", 0.0)),
                "file": r.get("source", ""),
                "snippet": snippet,
            }
        )

    total_s = time.perf_counter() - t0
    trace = {
        "retrieval_s": round(retrieval_s, 4),
        "generation_s": round(generation_s, 4),
        "total_s": round(total_s, 4),
    }
    return reply_text, sources, trace


def answer_turn(
    *,
    history: List[Message],
    user_text: str,
    cfg: Cfg,
    index: Optional[Any] = None,
    return_trace: bool = False,
) -> Union[Tuple[str, List[Dict[str, Any]]], Tuple[str, List[Dict[str, Any]], Dict[str, float]]]:
    """Answer one user turn, returning (assistant_text, sources)."""
    q = (user_text or "").strip()
    if not q:
        trace = {"retrieval_s": 0.0, "generation_s": 0.0, "total_s": 0.0}
        return ("(Empty message.)", [], trace) if return_trace else ("(Empty message.)", [])

    rag_enabled = cfg_bool(cfg, "rag_enabled", True)

    # RAG path
    if rag_enabled and index is not None:
        reply_text, sources, trace = _answer_with_rag(query=q, index=index, cfg=cfg)
        return (reply_text, sources, trace) if return_trace else (reply_text, sources)

    # Non-RAG: normal chat completion with history.
    max_hist = cfg_int(cfg, "max_history_turns", 12)
    trimmed = _trim_history(history, max_messages=max_hist)

    system_prompt = cfg_str(cfg, "system_prompt", "You are a helpful assistant.").strip()
    messages = [{"role": "system", "content": system_prompt}] + trimmed
    messages.append({"role": "user", "content": q})

    t0 = time.perf_counter()
    reply = ollama_chat(
        messages,
        model=cfg_model(cfg),
        host=cfg_str(cfg, "ollama_host", "http://localhost:11434"),
    )
    total_s = time.perf_counter() - t0
    trace = {
        "retrieval_s": 0.0,
        "generation_s": round(total_s, 4),
        "total_s": round(total_s, 4),
    }
    out = (reply or "(No response.)"), []
    return (*out, trace) if return_trace else out

def sanitize_corpus_id(corpus_id: str) -> str:
    """Sanitize corpus id."""
    corpus_id = (corpus_id or "").strip().lower()
    safe = []
    for ch in corpus_id:
        if ch.isalnum() or ch in {"-", "_"}:
            safe.append(ch)
        else:
            safe.append("-")
    out = "".join(safe).strip("-")
    return out or "default"


def default_corpus_id() -> str:
    """Default corpus id."""
    return "chat-" + datetime.now().strftime("%Y%m%d-%H%M%S")


def get_corpus_root(cfg: Cfg) -> Path:
    """Get corpus root."""
    base_data_dir = Path(cfg_str(cfg, "data_dir", "rag_local/Data")).resolve()

    # If corpus mode is disabled, preserve old behavior
    use_corpus_mode = cfg_bool(cfg, "use_corpus_mode", True)
    if not use_corpus_mode:
        return base_data_dir.resolve()

    corpus_id = sanitize_corpus_id(cfg_str(cfg, "corpus_id", "default"))
    return (base_data_dir / "corpora" / corpus_id).resolve()


def get_corpus_docs_dir(cfg: Cfg) -> Path:
    """Get corpus docs dir."""
    root = get_corpus_root(cfg)
    if cfg_bool(cfg, "use_corpus_mode", True):
        return (root / "docs").resolve()
    return root.resolve()


def get_corpus_index_path(cfg: Cfg) -> Path:
    """Get corpus index path."""
    if cfg_bool(cfg, "use_corpus_mode", True):
        root = get_corpus_root(cfg)
        return (root / ".index" / "local_index.json").resolve()

    # fallback to legacy path
    return Path(cfg_str(cfg, "index_path", "rag_local/Data/.index/local_index.json")).resolve()


def delete_current_index(cfg: Cfg) -> bool:
    """Delete current index."""
    index_path = get_corpus_index_path(cfg)
    if index_path.exists():
        index_path.unlink()
        return True
    return False


def delete_current_corpus(cfg: Cfg) -> bool:
    """Delete current corpus."""
    docs_dir = get_corpus_docs_dir(cfg)
    corpus_root = get_corpus_root(cfg)

    # corpus mode only
    if not cfg_bool(cfg, "use_corpus_mode", True):
        return False

    if corpus_root.exists():
        shutil.rmtree(corpus_root)
        return True
    return False