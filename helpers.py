from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from functools import lru_cache

import httpx
from dotenv import load_dotenv


check = load_dotenv()
print(f"Loaded .env: {check}")
print(f"HF_TOKEN: {os.environ.get('HF_TOKEN')}")
print(f"OLLAMA_BASE_URL: {os.environ.get('OLLAMA_BASE_URL')}")

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

_BENIGN_NEMO_CLASSLOAD_FRAGMENT = (
    "Error getting class at nemo.collections.asr.modules.transformer.get_nemo_transformer"
)
_BENIGN_NEMO_CLASSLOAD_DETAIL = "Located non-class of type 'function'"


def is_benign_nemo_transformer_log(message: str | None) -> bool:
    if not message:
        return False
    return (
        _BENIGN_NEMO_CLASSLOAD_FRAGMENT in message
        and _BENIGN_NEMO_CLASSLOAD_DETAIL in message
    )


class _BenignNeMoLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not is_benign_nemo_transformer_log(record.getMessage())


_benign_nemo_log_filter = _BenignNeMoLogFilter()
logging.getLogger().addFilter(_benign_nemo_log_filter)
logging.getLogger("nemo_logger").addFilter(_benign_nemo_log_filter)
for _handler in logging.getLogger().handlers:
    _handler.addFilter(_benign_nemo_log_filter)


def ollama_base_url() -> str:
    """Resolve Ollama host. Supports OLLAMA_BASE_URL / OLLAMA_HOST / OLLAMA_URL."""
    load_dotenv()
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
PROJECT_MEMORY_PATH = os.path.join(BASE_DIR, ".omx", "project-memory.json")


def tail_text(text: str, limit: int = 500) -> str:
    cleaned = (text or "").strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[-limit:]


def sse(event: str, payload: dict) -> str:
    return f"data: {json.dumps({'event': event, 'payload': payload})}\n\n"


def ollama_api_base() -> str:
    if "/api/" in OLLAMA_URL:
        return OLLAMA_URL.split("/api/", 1)[0]
    return OLLAMA_URL.rstrip("/")


def ollama_tags_url() -> str:
    return ollama_api_base().rstrip("/") + "/api/tags"


def ensure_ollama_ready(*models: str, timeout: float = 10.0) -> None:
    unique_models = []
    seen_models = set()
    for model in models:
        if not model or model in seen_models:
            continue
        unique_models.append(model)
        seen_models.add(model)

    try:
        response = httpx.get(ollama_tags_url(), timeout=timeout)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise RuntimeError(
            f"Ollama is not reachable at {ollama_api_base()} ({exc})"
        ) from exc

    data = response.json()
    available_models = {
        item.get("name", "").strip()
        for item in data.get("models", [])
        if item.get("name")
    }
    missing_models = [model for model in unique_models if model not in available_models]
    if missing_models:
        raise RuntimeError(
            "Ollama model(s) not found: "
            + ", ".join(missing_models)
            + ". Available models: "
            + ", ".join(sorted(available_models) or ["<none>"])
        )


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


NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "syalosovetskyi_subscribe_topic")
NTFY_URL = f"https://ntfy.sh/{NTFY_TOPIC}"


def _ntfy_payload(title: str, message: str) -> dict:
    return {
        "topic": NTFY_TOPIC,
        "title": title,
        "message": message,
        "priority": 3,
        "tags": ["white_check_mark"],
    }


def _desktop_notifications_available() -> bool:
    return bool(os.environ.get("DBUS_SESSION_BUS_ADDRESS")) and bool(
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    )


async def notify_done(title: str, message: str) -> None:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(NTFY_URL, json=_ntfy_payload(title, message))
            response.raise_for_status()
    except Exception as exc:
        log.warning("ntfy notification failed: %s", exc)
    if not _desktop_notifications_available():
        log.info("Desktop notifications unavailable in this session; skipping local toast")
        return
    try:
        from desktop_notifier import DesktopNotifier

        notifier = DesktopNotifier()
        await notifier.send(title=title, message=message)
    except Exception as exc:
        log.warning("Desktop notification failed: %s", exc)


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


@lru_cache(maxsize=1)
def load_user_profile() -> dict:
    profile = {
        "primary_name": "Сергей",
        "aliases": ["Сергей", "Сергій", "Serhii"],
    }
    try:
        with open(PROJECT_MEMORY_PATH, "r", encoding="utf-8") as file_handle:
            data = json.load(file_handle)
    except FileNotFoundError:
        return profile
    except Exception as exc:
        log.warning("Failed to read project memory: %s", exc)
        return profile

    user_profile = data.get("user_profile") or {}
    primary_name = str(user_profile.get("primary_name") or profile["primary_name"]).strip()
    aliases = user_profile.get("aliases") or profile["aliases"]
    aliases = [str(alias).strip() for alias in aliases if str(alias).strip()]
    if primary_name not in aliases:
        aliases.insert(0, primary_name)
    return {
        "primary_name": primary_name,
        "aliases": aliases,
    }
