"""Core logic for ollama client."""
import json

import urllib.request
import urllib.error


def chat(messages, model="llama3.1:8b", host="http://localhost:11434", timeout_s=60):
    """
    Minimal Ollama chat call.
    messages: list of {"role": "...", "content": "..."} dicts
    returns: assistant text (str)
    """
    url = host.rstrip("/") + "/api/chat"
    payload = {"model": model, "messages": messages, "stream": False}

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("message", {}).get("content", "").strip()

    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama not reachable at {host}. Is it running? ({e})")

    except json.JSONDecodeError as e:
        raise RuntimeError(f"Bad response from Ollama (invalid JSON): {e}")
