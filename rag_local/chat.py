"""Core logic for chat."""
from __future__ import annotations


from pathlib import Path
from typing import Dict, List, Literal

from rag_local.config import get_config
from rag_local.embedders import make_embedder
from rag_local.eval_rag import default_eval_items, format_eval_report, run_rag_eval
from rag_local.rag import build_index, load_index, save_index

# IMPORTANT: use the shared core function (same as Streamlit)
from rag_local.app_core import answer_turn

Role = Literal["system", "user", "assistant"]
Message = Dict[str, str]


def _append_and_trim(history: List[Message], msg: Message, max_messages: int) -> None:
    """Internal helper for append and trim."""
    history.append(msg)
    if max_messages <= 0:
        history.clear()
        return
    if len(history) > max_messages:
        del history[:-max_messages]


def main() -> None:
    """Main."""
    cfg = get_config()

    history: List[Message] = []
    rag_enabled = bool(getattr(cfg, "rag_enabled", True))

    index = None  # lazily created

    if rag_enabled:
        embed_backend = getattr(cfg, "embed_backend", "ollama")
        embed_model = getattr(cfg, "embed_model", "nomic-embed-text")
        ollama_host = getattr(cfg, "ollama_host", "http://localhost:11434")

        embedder = make_embedder(backend=embed_backend, model=embed_model, host=ollama_host)
        print("[EMBED]", embed_backend, embedder.__class__.__name__, getattr(embedder, "model", None))

        index_path = Path(getattr(cfg, "index_path", "rag_local/Data/.index/local_index.json")).resolve()
        index_path.parent.mkdir(parents=True, exist_ok=True)

        if index_path.exists():
            index = load_index(index_path, embedder=embedder)
            print(f"[RAG] Loaded index: {index_path}")
        else:
            index, stats = build_index(
                data_dir=Path(getattr(cfg, "data_dir", "rag_local/Data")).resolve(),
                embedder=embedder,
                chunk_size=int(getattr(cfg, "chunk_size", 800)),
                overlap=int(getattr(cfg, "overlap", 200)),
            )
            save_index(index, index_path)
            print(f"[RAG] Built index: docs={stats.doc_count} chunks={stats.chunk_count} -> {index_path}")

        # Verify embeddings stored (safe)
        try:
            if index is not None and getattr(index, "chunks", None):
                c0 = index.chunks[0]
                # c0 might be a dict; handle both dict-like and object-like
                if isinstance(c0, dict):
                    emb = c0.get("embedding", [])
                    print(f"[EMBED] stored_in_chunk={'embedding' in c0} dim={len(emb)} keys={list(c0.keys())}")
                else:
                    emb = getattr(c0, "embedding", [])
                    print(f"[EMBED] stored_in_chunk={emb is not None} dim={len(emb) if emb else 0}")
        except Exception as e:
            print(f"[EMBED] verify skipped: {e}")

        # Optional startup eval (CLI only; no GUI wiring)
        if bool(getattr(cfg, "rag_eval_on_startup", True)):
            try:
                n = int(getattr(cfg, "rag_eval_n", 3))
                items = default_eval_items()[: max(0, n)]
                eval_result = run_rag_eval(cfg=cfg, index=index, items=items)
                print(format_eval_report(eval_result))
            except Exception as e:
                print(f"[EVAL] Startup eval skipped: {e}")

    print("Chatbot ready. Type 'exit' to quit.")

    while True:
        try:
            user = input("You: ").strip()
            if not user:
                continue

            if user.lower() in {"exit", "quit"}:
                print("Bye.")
                return

            _append_and_trim(
                history,
                {"role": "user", "content": user},
                max_messages=int(getattr(cfg, "max_history_turns", 12)),
            )

            reply, _sources = answer_turn(history=history, user_text=user, cfg=cfg, index=index)

            _append_and_trim(
                history,
                {"role": "assistant", "content": reply},
                max_messages=int(getattr(cfg, "max_history_turns", 12)),
            )

            print(f"\nBot: {reply}\n")

        except KeyboardInterrupt:
            print("\nBye.")
            return


if __name__ == "__main__":
    main()