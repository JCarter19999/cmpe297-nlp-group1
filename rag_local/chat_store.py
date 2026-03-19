"""Core logic for chat store."""
from __future__ import annotations


import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ChatRecord:
    """Represents chat record."""
    chat_id: str
    title: str
    created_at: str
    updated_at: str
    messages: List[Dict]


def _now_iso() -> str:
    """Internal helper for now iso."""
    return datetime.now().isoformat(timespec="seconds")


def default_chat_id() -> str:
    """Default chat id."""
    return "chat-" + datetime.now().strftime("%Y%m%d-%H%M%S")


def sanitize_chat_id(chat_id: str) -> str:
    """Sanitize chat id."""
    chat_id = (chat_id or "").strip().lower()
    safe = []
    for ch in chat_id:
        if ch.isalnum() or ch in {"-", "_"}:
            safe.append(ch)
        else:
            safe.append("-")
    out = "".join(safe).strip("-")
    return out or default_chat_id()


def get_chat_dir(corpus_root: Path) -> Path:
    """Get chat dir."""
    d = corpus_root / "chats"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_chat_path(corpus_root: Path, chat_id: str) -> Path:
    """Get chat path."""
    return get_chat_dir(corpus_root) / f"{sanitize_chat_id(chat_id)}.json"


def derive_chat_title(messages: List[Dict], fallback: str = "Untitled Chat") -> str:
    """Derive chat title."""
    for m in messages:
        if m.get("role") == "user":
            txt = (m.get("content") or "").strip().replace("\n", " ")
            if txt:
                return txt[:60]
    return fallback


def list_chats(corpus_root: Path) -> List[Dict]:
    """List chats."""
    chat_dir = get_chat_dir(corpus_root)
    rows: List[Dict] = []

    for p in sorted(chat_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            rows.append(
                {
                    "chat_id": data.get("chat_id", p.stem),
                    "title": data.get("title", p.stem),
                    "created_at": data.get("created_at", ""),
                    "updated_at": data.get("updated_at", ""),
                    "path": str(p),
                }
            )
        except Exception:
            continue

    rows.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return rows


def load_chat(corpus_root: Path, chat_id: str) -> Optional[ChatRecord]:
    """Load chat."""
    path = get_chat_path(corpus_root, chat_id)
    if not path.exists():
        return None

    data = json.loads(path.read_text(encoding="utf-8"))
    return ChatRecord(
        chat_id=data["chat_id"],
        title=data.get("title", data["chat_id"]),
        created_at=data.get("created_at", ""),
        updated_at=data.get("updated_at", ""),
        messages=list(data.get("messages", [])),
    )


def save_chat(
    corpus_root: Path,
    *,
    chat_id: str,
    messages: List[Dict],
    title: Optional[str] = None,
) -> ChatRecord:
    """Save chat."""
    chat_id = sanitize_chat_id(chat_id)
    path = get_chat_path(corpus_root, chat_id)

    existing = load_chat(corpus_root, chat_id)
    created_at = existing.created_at if existing else _now_iso()
    updated_at = _now_iso()

    if not title:
        title = derive_chat_title(messages, fallback=chat_id)

    payload = {
        "chat_id": chat_id,
        "title": title,
        "created_at": created_at,
        "updated_at": updated_at,
        "messages": messages,
    }

    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return ChatRecord(
        chat_id=chat_id,
        title=title,
        created_at=created_at,
        updated_at=updated_at,
        messages=messages,
    )


def delete_chat(corpus_root: Path, chat_id: str) -> bool:
    """Delete chat."""
    path = get_chat_path(corpus_root, chat_id)
    if path.exists():
        path.unlink()
        return True
    return False