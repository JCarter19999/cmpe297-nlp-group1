"""Core logic for ollama bootstrap."""
from __future__ import annotations


import subprocess
from dataclasses import dataclass
from typing import Optional, Tuple

import requests


@dataclass
class OllamaStatus:
    """Represents ollama status."""
    ok: bool
    message: str


def ollama_is_reachable(host: str, timeout_s: int = 2) -> bool:
    """Ollama is reachable."""
    host = host.rstrip("/")
    try:
        r = requests.get(f"{host}/api/tags", timeout=timeout_s)
        return r.status_code == 200
    except Exception:
        return False


def ensure_ollama_model(model: str) -> Tuple[bool, str]:
    """
    Ensures model is pulled locally.
    Uses `ollama list` and `ollama pull <model>` via subprocess.

    Returns (ok, message).
    """
    try:
        # `ollama list` -> returns models installed locally
        p = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=False)
        if p.returncode != 0:
            return False, f"`ollama list` failed: {p.stderr.strip() or p.stdout.strip()}"

        if model in p.stdout:
            return True, f"Model '{model}' is available."

        # pull it
        pull = subprocess.run(["ollama", "pull", model], capture_output=True, text=True, check=False)
        if pull.returncode != 0:
            return False, f"`ollama pull {model}` failed: {pull.stderr.strip() or pull.stdout.strip()}"

        return True, f"Pulled model '{model}'."
    except FileNotFoundError:
        return False, "Ollama CLI not found. Install Ollama and ensure `ollama` is on PATH."
    except Exception as e:
        return False, f"Failed to ensure model '{model}': {e}"


def preflight_ollama_for_embeddings(host: str, embed_model: str) -> OllamaStatus:
    """
    1) Verify host reachable
    2) Ensure embed model exists locally (auto-pull if not)
    """
    if not ollama_is_reachable(host):
        return OllamaStatus(
            ok=False,
            message=(
                f"Ollama not reachable at {host}. Start it first:\n"
                f"  ollama serve\n"
                f"Then reload the Streamlit app."
            ),
        )

    ok, msg = ensure_ollama_model(embed_model)
    if not ok:
        return OllamaStatus(ok=False, message=msg)

    return OllamaStatus(ok=True, message=f"Ollama OK. {msg}")