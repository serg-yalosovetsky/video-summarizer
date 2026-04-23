from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value is not None else default


def _env_int(name: str, default: int) -> int:
    return int(_env_str(name, str(default)))


def _env_lower(name: str, default: str) -> str:
    return _env_str(name, default).strip().lower()


def _env_list(name: str, default: list[str]) -> tuple[str, ...]:
    raw = os.environ.get(name)
    if raw is None:
        values = default
    else:
        values = [item.strip() for item in raw.split(",")]
    cleaned = [item for item in values if item]
    return tuple(cleaned)


def _env_path(name: str, default: Path, *, base_dir: Path) -> Path:
    raw = os.environ.get(name)
    if not raw:
        path = default
    else:
        candidate = Path(raw).expanduser()
        path = candidate if candidate.is_absolute() else (base_dir / candidate)
    return path.resolve()


def _resolve_hf_token() -> str | None:
    return (
        os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )


def _resolve_ollama_url() -> str:
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


@dataclass(frozen=True)
class Settings:
    base_dir: Path
    local_tmp: Path
    artifacts_dir: Path
    project_memory_path: Path
    max_upload_bytes: int
    hf_token: str | None
    ollama_url: str
    ollama_model: str
    ollama_clean_model: str
    ollama_timeout_seconds: int
    canary_model: str
    canary_device: str
    pyannote_model: str
    pyannote_device: str
    canary_segment_batch_size: int
    frame_model: str
    frame_timestamps: tuple[int, ...]
    max_frames: int
    user_primary_name: str
    user_aliases: tuple[str, ...]
    user_profile_overridden: bool
    ntfy_topic: str
    ntfy_url: str
    ollama_summary_max_tokens: int
    ollama_clean_max_tokens: int
    max_visual_context_chars: int
    langfuse_public_key: str | None
    langfuse_secret_key: str | None
    langfuse_base_url: str
    langfuse_enabled: bool


def build_settings() -> Settings:
    load_dotenv()

    base_dir = Path(__file__).resolve().parent
    hf_token = _resolve_hf_token()
    if hf_token:
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", hf_token)
        os.environ.setdefault("HF_TOKEN", hf_token)
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", hf_token)

    ollama_url = _resolve_ollama_url()
    ollama_model = _env_str("OLLAMA_MODEL", "gemma4:e4b")
    ollama_clean_model = _env_str("OLLAMA_CLEAN_MODEL", ollama_model)
    frame_model = _env_str("FRAME_MODEL", ollama_model)
    ntfy_topic = _env_str("NTFY_TOPIC", "syalosovetskyi_subscribe_topic")
    langfuse_public_key = os.environ.get("LANGFUSE_PUBLIC_KEY") or None
    langfuse_secret_key = os.environ.get("LANGFUSE_SECRET_KEY") or None
    langfuse_base_url = _env_str(
        "LANGFUSE_BASE_URL",
        _env_str("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )
    user_primary_name = _env_str("USER_PRIMARY_NAME", "Сергей").strip() or "Сергей"
    user_aliases = list(_env_list("USER_ALIASES", ["Сергей", "Сергій", "Serhii"]))
    user_profile_overridden = (
        os.environ.get("USER_PRIMARY_NAME") is not None
        or os.environ.get("USER_ALIASES") is not None
    )
    if user_primary_name not in user_aliases:
        user_aliases.insert(0, user_primary_name)

    local_tmp = _env_path("LOCAL_TMP_DIR", base_dir / "tmp", base_dir=base_dir)
    local_tmp.mkdir(exist_ok=True)
    artifacts_dir = _env_path("ARTIFACTS_DIR", base_dir / "artifacts", base_dir=base_dir)
    artifacts_dir.mkdir(exist_ok=True)
    project_memory_path = _env_path(
        "PROJECT_MEMORY_PATH",
        base_dir / ".omx" / "project-memory.json",
        base_dir=base_dir,
    )

    return Settings(
        base_dir=base_dir,
        local_tmp=local_tmp,
        artifacts_dir=artifacts_dir,
        project_memory_path=project_memory_path,
        max_upload_bytes=max(1, _env_int("MAX_UPLOAD_MB", 500)) * 1024 * 1024,
        hf_token=hf_token,
        ollama_url=ollama_url,
        ollama_model=ollama_model,
        ollama_clean_model=ollama_clean_model,
        ollama_timeout_seconds=_env_int("OLLAMA_TIMEOUT_SECONDS", 900),
        canary_model=_env_str("CANARY_MODEL", "nvidia/canary-1b-v2"),
        canary_device=_env_lower("CANARY_DEVICE", "cuda"),
        pyannote_model=_env_str("PYANNOTE_MODEL", "pyannote/speaker-diarization-3.1"),
        pyannote_device=_env_lower("PYANNOTE_DEVICE", "auto"),
        canary_segment_batch_size=max(1, _env_int("CANARY_SEGMENT_BATCH_SIZE", 8)),
        frame_model=frame_model,
        frame_timestamps=tuple(int(item) for item in _env_list("FRAME_TIMESTAMPS", ["1", "2", "5", "10"])),
        max_frames=max(1, _env_int("MAX_FRAMES", 20)),
        user_primary_name=user_primary_name,
        user_aliases=tuple(user_aliases),
        user_profile_overridden=user_profile_overridden,
        ntfy_topic=ntfy_topic,
        ntfy_url=_env_str("NTFY_URL", f"https://ntfy.sh/{ntfy_topic}"),
        ollama_summary_max_tokens=_env_int("OLLAMA_SUMMARY_MAX_TOKENS", 1024),
        ollama_clean_max_tokens=_env_int("OLLAMA_CLEAN_MAX_TOKENS", 4096),
        max_visual_context_chars=_env_int("MAX_VISUAL_CONTEXT_CHARS", 2000),
        langfuse_public_key=langfuse_public_key,
        langfuse_secret_key=langfuse_secret_key,
        langfuse_base_url=langfuse_base_url,
        langfuse_enabled=bool(langfuse_public_key and langfuse_secret_key),
    )


settings = build_settings()
