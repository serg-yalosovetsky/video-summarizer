from __future__ import annotations

import logging
import re
from contextlib import contextmanager, nullcontext
from functools import lru_cache
from typing import Any, Callable

from config import settings

log = logging.getLogger("summarizer.tracing")
_langfuse_client_initialized = False

_SECRET_VALUE_PATTERNS = (
    re.compile(r"\bsk-lf-[A-Za-z0-9-]+\b"),
    re.compile(r"\bpk-lf-[A-Za-z0-9-]+\b"),
    re.compile(r"\bhf_[A-Za-z0-9]+\b"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._-]+\b"),
)
_SECRET_KEY_NAMES = {
    "authorization",
    "api_key",
    "api-key",
    "hf_token",
    "hugging_face_hub_token",
    "huggingface_hub_token",
    "langfuse_public_key",
    "langfuse_secret_key",
    "public_key",
    "secret_key",
    "token",
}


def _mask_text(value: str) -> str:
    masked = value
    for pattern in _SECRET_VALUE_PATTERNS:
        masked = pattern.sub("[REDACTED]", masked)
    return masked


def mask_langfuse_data(data: Any, **_: Any) -> Any:
    if isinstance(data, str):
        return _mask_text(data)
    if isinstance(data, dict):
        masked: dict[Any, Any] = {}
        for key, value in data.items():
            if str(key).strip().lower() in _SECRET_KEY_NAMES:
                masked[key] = "[REDACTED]"
            else:
                masked[key] = mask_langfuse_data(value)
        return masked
    if isinstance(data, list):
        return [mask_langfuse_data(item) for item in data]
    if isinstance(data, tuple):
        return tuple(mask_langfuse_data(item) for item in data)
    return data


@lru_cache(maxsize=1)
def get_langfuse_client():
    global _langfuse_client_initialized
    if not settings.langfuse_enabled:
        return None

    try:
        from langfuse import Langfuse, get_client
    except ImportError:
        log.warning(
            "Langfuse keys are configured, but the langfuse package is not installed. "
            "Tracing will remain disabled."
        )
        return None

    Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_base_url,
        mask=mask_langfuse_data,
    )
    _langfuse_client_initialized = True
    return get_client()


def langfuse_is_enabled() -> bool:
    return settings.langfuse_enabled


def check_langfuse_auth() -> bool | None:
    client = get_langfuse_client()
    if client is None:
        return None
    try:
        return bool(client.auth_check())
    except Exception as exc:
        log.warning("Langfuse auth check failed: %s", exc)
        return False


def current_trace_context() -> dict[str, str] | None:
    client = get_langfuse_client()
    if client is None:
        return None

    trace_id = client.get_current_trace_id()
    observation_id = client.get_current_observation_id()
    if not trace_id:
        return None

    trace_context = {"trace_id": trace_id}
    if observation_id:
        trace_context["parent_span_id"] = observation_id
    return trace_context


@contextmanager
def start_observation(
    name: str,
    *,
    as_type: str = "span",
    trace_context: dict[str, str] | None = None,
    **kwargs: Any,
):
    client = get_langfuse_client()
    if client is None:
        with nullcontext() as ctx:
            yield ctx
        return

    with client.start_as_current_observation(
        name=name,
        as_type=as_type,
        trace_context=trace_context,
        **kwargs,
    ) as observation:
        yield observation


def flush_langfuse() -> None:
    if not _langfuse_client_initialized:
        return
    client = get_langfuse_client()
    if client is None:
        return
    try:
        client.flush()
    except Exception as exc:
        log.warning("Langfuse flush failed: %s", exc)


def trace_sync_call(
    name: str,
    fn: Callable[..., Any],
    *args: Any,
    as_type: str = "span",
    trace_context: dict[str, str] | None = None,
    output_builder: Callable[[Any], Any] | None = None,
    **observation_kwargs: Any,
) -> Any:
    with start_observation(
        name,
        as_type=as_type,
        trace_context=trace_context,
        **observation_kwargs,
    ) as observation:
        result = fn(*args)
        if observation is not None and output_builder is not None:
            output = output_builder(result)
            if output is not None:
                observation.update(output=output)
        return result
