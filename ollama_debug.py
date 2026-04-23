from __future__ import annotations

import base64
import json
import time
from pathlib import Path

_DEBUG_ROOT = Path(__file__).parent / "temp"
_counter = 0


def _next_dir(label: str) -> Path:
    global _counter
    _counter += 1
    ts = time.strftime("%Y%m%d_%H%M%S")
    folder = _DEBUG_ROOT / f"{ts}_{_counter:04d}_{label}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def save_text_request(
    *,
    prompt: str,
    system: str,
    model: str,
    url: str,
    options: dict,
    timeout: float,
) -> None:
    folder = _next_dir(f"text_{model.replace(':', '-')}")
    (folder / "prompt.txt").write_text(prompt, encoding="utf-8")
    if system:
        (folder / "system.txt").write_text(system, encoding="utf-8")
    config = {
        "type": "text",
        "model": model,
        "url": url,
        "timeout_seconds": timeout,
        "options": options,
    }
    (folder / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def save_vision_request(
    *,
    prompt: str,
    system: str,
    model: str,
    url: str,
    b64_image: str,
    schema: dict,
    timeout: float,
    keep_alive: str,
) -> None:
    folder = _next_dir(f"vision_{model.replace(':', '-')}")
    (folder / "prompt.txt").write_text(prompt, encoding="utf-8")
    if system:
        (folder / "system.txt").write_text(system, encoding="utf-8")
    try:
        image_bytes = base64.b64decode(b64_image)
        (folder / "image.jpg").write_bytes(image_bytes)
    except Exception:
        (folder / "image_b64.txt").write_text(b64_image[:200] + "...", encoding="utf-8")
    (folder / "schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    config = {
        "type": "vision",
        "model": model,
        "url": url,
        "timeout_seconds": timeout,
        "keep_alive": keep_alive,
    }
    (folder / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
