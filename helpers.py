from __future__ import annotations

import json
import logging
import os
import re
from functools import lru_cache

import httpx

from config import settings
from tracing import start_observation

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

HF_TOKEN = settings.hf_token
OLLAMA_URL = settings.ollama_url
OLLAMA_MODEL = settings.ollama_model
OLLAMA_CLEAN_MODEL = settings.ollama_clean_model
MAX_UPLOAD_BYTES = settings.max_upload_bytes
LOCAL_TMP = str(settings.local_tmp)
PROJECT_MEMORY_PATH = str(settings.project_memory_path)
DEFAULT_USER_PROFILE = {
    "primary_name": settings.user_primary_name,
    "aliases": list(settings.user_aliases),
}


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


NTFY_TOPIC = settings.ntfy_topic
NTFY_URL = settings.ntfy_url


def _ntfy_payload(title: str, message: str) -> dict:
    return {
        "topic": NTFY_TOPIC,
        "title": title,
        "message": message,
        "priority": 3,
        "tags": ["white_check_mark"],
    }


def _desktop_notifications_available() -> bool:
    import os

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


def call_ollama(
    prompt: str,
    system: str = "",
    *,
    timeout: float | None = None,
    model: str | None = None,
    options: dict | None = None,
) -> str:
    request_model = model or OLLAMA_MODEL
    payload = {
        "model": request_model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": options or {},
    }
    with start_observation(
        "ollama.generate",
        as_type="generation",
        model=request_model,
        input={
            "prompt": prompt,
            "system": system,
            "options": payload["options"],
        },
        metadata={"provider": "ollama", "url": OLLAMA_URL},
    ) as generation:
        response = httpx.post(
            OLLAMA_URL,
            json=payload,
            timeout=timeout or settings.ollama_timeout_seconds,
        )
        response.raise_for_status()
        output = response.json()["response"]
        if generation is not None:
            generation.update(output=output)
        return output


@lru_cache(maxsize=1)
def load_user_profile() -> dict:
    profile = DEFAULT_USER_PROFILE.copy()
    if settings.user_profile_overridden:
        return profile
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
