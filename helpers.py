from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Iterable
import warnings
from functools import lru_cache

import httpx

from config import settings
from ollama_debug import save_text_request
from tracing import start_observation

warnings.filterwarnings("ignore", message="TensorFloat-32", module=r"pyannote\.audio")

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
_BENIGN_PYANNOTE_TF32_FRAGMENT = "TensorFloat-32 (TF32) has been disabled"


def is_benign_nemo_transformer_log(message: str | None) -> bool:
    if not message:
        return False
    return (
        (
            _BENIGN_NEMO_CLASSLOAD_FRAGMENT in message
            and _BENIGN_NEMO_CLASSLOAD_DETAIL in message
        )
        or _BENIGN_PYANNOTE_TF32_FRAGMENT in message
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


def ollama_ps_url() -> str:
    return ollama_api_base().rstrip("/") + "/api/ps"


def _ollama_warm_timeout_seconds(timeout: float) -> float:
    return max(timeout, min(float(settings.ollama_timeout_seconds), 60.0))


def _warm_ollama_model(model: str, *, timeout: float) -> None:
    payload = {
        "model": model,
        "prompt": "Reply with exactly one token: ok",
        "stream": False,
        "keep_alive": "0",
        "options": {"num_predict": 1, "temperature": 0},
    }
    response = httpx.post(OLLAMA_URL, json=payload, timeout=timeout)
    response.raise_for_status()


def _iter_ollama_model_names(model_info: dict) -> Iterable[str]:
    for key in ("name", "model"):
        value = model_info.get(key)
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned:
                yield cleaned


def _find_loaded_ollama_model(data: dict, model: str) -> dict | None:
    models = data.get("models", [])
    if not isinstance(models, list):
        return None
    for item in models:
        if not isinstance(item, dict):
            continue
        if model in set(_iter_ollama_model_names(item)):
            return item
    return None


def _ollama_processor_label(model_info: dict) -> str:
    details = model_info.get("details")
    candidates = [model_info]
    if isinstance(details, dict):
        candidates.append(details)
    for payload in candidates:
        value = payload.get("processor")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _ollama_size_vram_bytes(model_info: dict) -> int | None:
    value = model_info.get("size_vram")
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
    return None


def _ollama_uses_gpu(model_info: dict) -> bool:
    processor = _ollama_processor_label(model_info).lower()
    if "gpu" in processor:
        return True
    size_vram = _ollama_size_vram_bytes(model_info)
    return size_vram is not None and size_vram > 0


def _ollama_runtime_summary(model_info: dict) -> str:
    processor = _ollama_processor_label(model_info) or "<unknown>"
    size_vram = _ollama_size_vram_bytes(model_info)
    size_vram_text = str(size_vram) if size_vram is not None else "<unknown>"
    return f"processor={processor}, size_vram={size_vram_text}"


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

    if settings.ollama_device != "gpu":
        return

    warm_timeout = _ollama_warm_timeout_seconds(timeout)
    for model in unique_models:
        try:
            _warm_ollama_model(model, timeout=warm_timeout)
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Failed to warm Ollama model {model!r} on {ollama_api_base()} ({exc})"
            ) from exc

        try:
            response = httpx.get(ollama_ps_url(), timeout=timeout)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(
                "Ollama GPU verification failed because /api/ps is unavailable "
                f"at {ollama_api_base()} ({exc}). Upgrade Ollama or set OLLAMA_DEVICE=auto."
            ) from exc

        loaded_model = _find_loaded_ollama_model(response.json(), model)
        if loaded_model is None:
            raise RuntimeError(
                f"Ollama warmed model {model!r}, but it is not listed in /api/ps. "
                "Upgrade Ollama or set OLLAMA_DEVICE=auto to skip GPU verification."
            )

        if not _ollama_uses_gpu(loaded_model):
            raise RuntimeError(
                f"Ollama model {model!r} is not using the GPU "
                f"({_ollama_runtime_summary(loaded_model)}). "
                "Refusing to continue with CPU fallback. "
                "Fix the Ollama GPU setup or set OLLAMA_DEVICE=auto for debugging."
            )
        log.info("Ollama model %s verified on GPU (%s)", model, _ollama_runtime_summary(loaded_model))


def unload_ollama_models(*models: str, timeout: float = 10.0) -> None:
    unique_models = list(dict.fromkeys(m for m in models if m))
    for model in unique_models:
        try:
            payload = {"model": model, "keep_alive": "0"}
            httpx.post(OLLAMA_URL, json=payload, timeout=timeout)
            log.info("Ollama model %s unloaded from VRAM.", model)
        except Exception as exc:
            log.warning("Failed to unload Ollama model %s: %s", model, exc)


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
    resolved_options = options or {}
    resolved_timeout = timeout or settings.ollama_timeout_seconds
    save_text_request(
        prompt=prompt,
        system=system,
        model=request_model,
        url=OLLAMA_URL,
        options=resolved_options,
        timeout=resolved_timeout,
    )
    payload = {
        "model": request_model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": resolved_options,
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
            timeout=resolved_timeout,
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
