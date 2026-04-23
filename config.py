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
    project_memory_path: Path
    max_upload_bytes: int
    hf_token: str | None
    ollama_url: str
    ollama_model: str
    ollama_clean_model: str
    canary_model: str
    canary_device: str
    pyannote_model: str
    pyannote_device: str
    canary_segment_batch_size: int
    frame_model: str
    frame_timestamps: tuple[int, ...]
    max_frames: int
    ntfy_topic: str
    ntfy_url: str
    ollama_summary_max_tokens: int
    ollama_clean_max_tokens: int
    max_visual_context_chars: int


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

    local_tmp = base_dir / "tmp"
    local_tmp.mkdir(exist_ok=True)

    return Settings(
        base_dir=base_dir,
        local_tmp=local_tmp,
        project_memory_path=base_dir / ".omx" / "project-memory.json",
        max_upload_bytes=500 * 1024 * 1024,
        hf_token=hf_token,
        ollama_url=ollama_url,
        ollama_model=ollama_model,
        ollama_clean_model=ollama_clean_model,
        canary_model=_env_str("CANARY_MODEL", "nvidia/canary-1b-v2"),
        canary_device=_env_lower("CANARY_DEVICE", "cuda"),
        pyannote_model=_env_str("PYANNOTE_MODEL", "pyannote/speaker-diarization-3.1"),
        pyannote_device=_env_lower("PYANNOTE_DEVICE", "auto"),
        canary_segment_batch_size=max(1, _env_int("CANARY_SEGMENT_BATCH_SIZE", 8)),
        frame_model=frame_model,
        frame_timestamps=(1, 2, 5, 10),
        max_frames=20,
        ntfy_topic=ntfy_topic,
        ntfy_url=f"https://ntfy.sh/{ntfy_topic}",
        ollama_summary_max_tokens=_env_int("OLLAMA_SUMMARY_MAX_TOKENS", 1024),
        ollama_clean_max_tokens=_env_int("OLLAMA_CLEAN_MAX_TOKENS", 4096),
        max_visual_context_chars=_env_int("MAX_VISUAL_CONTEXT_CHARS", 2000),
    )


settings = build_settings()
