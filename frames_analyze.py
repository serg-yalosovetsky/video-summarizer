from __future__ import annotations

import asyncio
import base64
import os
import subprocess
import time

import httpx

from helpers import OLLAMA_MODEL, OLLAMA_URL, log
from prompts import FRAME_ANALYSIS_PROMPT, FRAME_ANALYSIS_SYSTEM


FRAME_TIMESTAMPS = [1, 2, 5, 10]
MAX_FRAMES = 20


def extract_single_frame(input_path: str, out_path: str, ts: int) -> bool:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        str(ts),
        "-i",
        input_path,
        "-frames:v",
        "1",
        "-f",
        "image2",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0 and os.path.exists(out_path)


def extract_frames(input_path: str, tmp_dir: str, duration_sec: float) -> list[str]:
    paths = []
    for ts in FRAME_TIMESTAMPS:
        if ts >= duration_sec:
            break
        out_path = os.path.join(tmp_dir, f"frame_{ts}s.jpg")
        if extract_single_frame(input_path, out_path, ts):
            paths.append(out_path)
    return paths


def extract_frames_at(input_path: str, tmp_dir: str, timestamps: list[int]) -> list[str]:
    paths = []
    for ts in timestamps:
        out_path = os.path.join(tmp_dir, f"frame_{ts}s.jpg")
        if extract_single_frame(input_path, out_path, ts):
            paths.append(out_path)
    return paths


def generate_frame_timestamps(done_ts: set[int], duration_sec: float, count: int) -> list[int]:
    """Return up to `count` new timestamps spread evenly across the video."""
    result = []
    seen = set(done_ts)
    step = max(1, int(duration_sec / (count + 1)))
    ts = step
    while ts < int(duration_sec) and len(result) < count:
        if ts not in seen and not any(abs(ts - done) < 2 for done in seen):
            result.append(ts)
            seen.add(ts)
        ts += step
    if len(result) < count:
        for ts in range(1, int(duration_sec)):
            if ts not in seen:
                result.append(ts)
                seen.add(ts)
                if len(result) >= count:
                    break
    return result[:count]


def is_context_sufficient(context: str) -> bool:
    if not context or len(context.strip()) < 80:
        return False
    lines = [line.strip() for line in context.split("\n") if line.strip()]
    if len(lines) < 2:
        return False
    no_info_phrases = {
        "no people",
        "no one",
        "nobody",
        "empty room",
        "no visible person",
        "no speaker",
        "no names",
        "cannot identify",
        "no individuals",
    }
    useful = sum(
        1
        for line in lines
        if not any(phrase in line.lower() for phrase in no_info_phrases)
    )
    return useful >= 2


def analyze_frame(image_path: str) -> str:
    with open(image_path, "rb") as file_handle:
        b64 = base64.b64encode(file_handle.read()).decode()
    response = httpx.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": FRAME_ANALYSIS_PROMPT,
            "system": FRAME_ANALYSIS_SYSTEM,
            "images": [b64],
            "stream": False,
        },
        timeout=120.0,
    )
    response.raise_for_status()
    return response.json()["response"]


def analyze_frames_with_progress(
    image_paths: list[str],
    async_q: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    start_index: int = 0,
    total_hint: int | None = None,
) -> str:
    results = []
    total = total_hint or (start_index + len(image_paths))
    for index, path in enumerate(image_paths, start=1):
        if index > 1:
            time.sleep(1)
        loop.call_soon_threadsafe(
            async_q.put_nowait,
            {"current": start_index + index, "total": total},
        )
        ts_label = os.path.basename(path).replace("frame_", "").replace(".jpg", "")
        try:
            desc = analyze_frame(path)
            results.append(f"[{ts_label}] {desc}")
            log.info("  [frames] analyzed %s (%d/%d)", ts_label, start_index + index, total)
        except Exception as exc:
            log.warning("  [frames] %s error: %s", ts_label, exc)
    loop.call_soon_threadsafe(async_q.put_nowait, None)
    return "\n".join(results)
