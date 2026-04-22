from __future__ import annotations

import json
import logging
import os
import re
import subprocess

import httpx
from dotenv import load_dotenv


load_dotenv()

HF_TOKEN = (
    os.environ.get("HUGGING_FACE_HUB_TOKEN")
    or os.environ.get("HF_TOKEN")
    or os.environ.get("HUGGINGFACE_HUB_TOKEN")
)
if HF_TOKEN:
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", HF_TOKEN)
    os.environ.setdefault("HF_TOKEN", HF_TOKEN)
    os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", HF_TOKEN)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("summarizer")


def ollama_base_url() -> str:
    """Resolve Ollama host. Supports OLLAMA_BASE_URL / OLLAMA_HOST / OLLAMA_URL."""
    host = (
        os.environ.get("OLLAMA_BASE_URL")
        or os.environ.get("OLLAMA_HOST")
        or os.environ.get("OLLAMA_URL")
    )
    if host:
        if "/api/" in host:
            return host
        return host.rstrip("/") + "/api/generate"
    try:
        gateway = subprocess.check_output(
            ["ip", "route", "show", "default"],
            text=True,
        ).split()[2]
        return f"http://{gateway}:11434/api/generate"
    except Exception:
        return "http://localhost:11434/api/generate"


OLLAMA_URL = ollama_base_url()
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:e4b")
MAX_UPLOAD_BYTES = 500 * 1024 * 1024

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_TMP = os.path.join(BASE_DIR, "tmp")
os.makedirs(LOCAL_TMP, exist_ok=True)


def tail_text(text: str, limit: int = 500) -> str:
    cleaned = (text or "").strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[-limit:]


def sse(event: str, payload: dict) -> str:
    return f"data: {json.dumps({'event': event, 'payload': payload})}\n\n"


def combine_sources(transcript: str, chat: str) -> str:
    parts = []
    if transcript.strip():
        parts.append(f"=== Video/Audio Transcript ===\n{transcript.strip()}")
    if chat.strip():
        parts.append(f"=== Chat Messages ===\n{chat.strip()}")
    return "\n\n".join(parts)


def remove_repetitions(text: str, max_consecutive: int = 2) -> str:
    """Collapse consecutive identical lines — removes LLM hallucination loops."""
    lines = text.split("\n")
    result = []
    prev_content = None
    consecutive = 0
    for line in lines:
        content = re.sub(r"^\[\d{2}:\d{2}:\d{2}\]\s*", "", line).strip()
        if content and content == prev_content:
            consecutive += 1
            if consecutive <= max_consecutive:
                result.append(line)
        else:
            prev_content = content
            consecutive = 1
            result.append(line)
    return "\n".join(result)


def call_ollama(prompt: str, system: str = "", *, timeout: float = 900.0) -> str:
    response = httpx.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "system": system,
            "stream": False,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["response"]
