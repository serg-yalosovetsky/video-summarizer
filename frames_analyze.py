from __future__ import annotations

import asyncio
import base64
import subprocess
import time
from pathlib import Path

import httpx

from config import settings
from helpers import log
from models import FrameAnalysisResult, SpeakerFrameResult
from prompts import (
    FRAME_ANALYSIS_PROMPT,
    FRAME_ANALYSIS_SYSTEM,
    SPEAKER_FRAME_PROMPT_TEMPLATE,
    SPEAKER_FRAME_SYSTEM,
)

OLLAMA_URL = settings.ollama_url
FRAME_MODEL = settings.frame_model
FRAME_TIMESTAMPS = list(settings.frame_timestamps)
MAX_FRAMES = settings.max_frames


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
    return result.returncode == 0 and Path(out_path).exists()


def extract_frames(input_path: str, tmp_dir: str, duration_sec: float) -> list[str]:
    paths = []
    for ts in FRAME_TIMESTAMPS:
        if ts >= duration_sec:
            break
        out_path = str(Path(tmp_dir) / f"frame_{ts}s.jpg")
        if extract_single_frame(input_path, out_path, ts):
            paths.append(out_path)
    return paths


def extract_frames_at(input_path: str, tmp_dir: str, timestamps: list[int]) -> list[str]:
    paths = []
    for ts in timestamps:
        out_path = str(Path(tmp_dir) / f"frame_{ts}s.jpg")
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


def _ollama_vision_post(prompt: str, system: str, b64_image: str, schema: dict) -> str:
    """POST to Ollama vision endpoint with structured output; returns raw response string."""
    try:
        response = httpx.post(
            OLLAMA_URL,
            json={
                "model": FRAME_MODEL,
                "prompt": prompt,
                "system": system,
                "images": [b64_image],
                "stream": False,
                "format": schema,
            },
            timeout=120.0,
        )
    except httpx.TimeoutException as exc:
        log.warning("  [frames] ← TIMEOUT after 120s  url=%s  model=%s  exc=%s",
                    OLLAMA_URL, FRAME_MODEL, exc)
        raise
    except httpx.ConnectError as exc:
        log.warning("  [frames] ← CONNECT ERROR  url=%s  exc=%s", OLLAMA_URL, exc)
        raise
    log.info("  [frames] ← HTTP %d  (%.2fs)", response.status_code,
             response.elapsed.total_seconds())
    if not response.is_success:
        log.warning("  [frames] ← error body: %s", response.text[:500])
    response.raise_for_status()
    return response.json()["response"]


def analyze_frame(image_path: str) -> str:
    img_size = Path(image_path).stat().st_size
    log.info("  [frames] → POST %s  model=%s  image=%s  size=%dKB",
             OLLAMA_URL, FRAME_MODEL, Path(image_path).name, img_size // 1024)
    with open(image_path, "rb") as file_handle:
        b64 = base64.b64encode(file_handle.read()).decode()
    raw = _ollama_vision_post(
        FRAME_ANALYSIS_PROMPT,
        FRAME_ANALYSIS_SYSTEM,
        b64,
        FrameAnalysisResult.model_json_schema(),
    )
    try:
        return FrameAnalysisResult.model_validate_json(raw).to_context_str()
    except Exception:
        return raw




def analyze_speaker_frames(
    input_path: str,
    tmp_dir: str,
    diarization_segments: list[tuple[float, float, str]],
    async_q: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    start_index: int = 0,
    total_hint: int | None = None,
) -> str:
    """Extract and analyze one frame per unique speaker; falls back to next-longest segment if confidence is low."""
    from collections import defaultdict
    speaker_segs: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for start, end, spk in diarization_segments:
        speaker_segs[spk].append((start, end))
    for spk in speaker_segs:
        speaker_segs[spk].sort(key=lambda s: s[1] - s[0], reverse=True)

    results = []
    total = total_hint or (start_index + len(speaker_segs))
    for idx, spk in enumerate(sorted(speaker_segs), start=1):
        loop.call_soon_threadsafe(
            async_q.put_nowait,
            {"current": start_index + idx, "total": total},
        )
        result: SpeakerFrameResult | None = None
        used_ts = None
        for start, end in speaker_segs[spk]:
            ts = int((start + end) / 2)
            out_path = str(Path(tmp_dir) / f"frame_spk_{spk}_{ts}s.jpg")
            if not extract_single_frame(input_path, out_path, ts):
                continue
            prompt = SPEAKER_FRAME_PROMPT_TEMPLATE.format(speaker_id=spk, ts=ts)
            with open(out_path, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode()
            try:
                raw = _ollama_vision_post(
                    prompt,
                    SPEAKER_FRAME_SYSTEM,
                    b64,
                    SpeakerFrameResult.model_json_schema(),
                )
                candidate = SpeakerFrameResult.model_validate_json(raw)
            except Exception as exc:
                log.warning("  [frames] speaker frame error for %s @ %ds: %s", spk, ts, exc)
                continue
            if candidate.person_visible:
                result = candidate
                used_ts = ts
                break
            log.info("  [frames] low-confidence frame for %s @ %ds, trying next segment", spk, ts)
            result = candidate
            used_ts = ts

        if result and used_ts is not None:
            results.append(result.to_context_str(spk, used_ts))
            log.info("  [frames] speaker frame analyzed: %s @ %ds", spk, used_ts)
        else:
            log.warning("  [frames] no usable frame found for %s", spk)

    loop.call_soon_threadsafe(async_q.put_nowait, None)
    return "\n".join(results)


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
        ts_label = Path(path).name.replace("frame_", "").replace(".jpg", "")
        try:
            desc = analyze_frame(path)
            results.append(f"[{ts_label}] {desc}")
            log.info("  [frames] analyzed %s (%d/%d)", ts_label, start_index + index, total)
        except Exception as exc:
            log.warning("  [frames] %s error: %s", ts_label, exc)
    loop.call_soon_threadsafe(async_q.put_nowait, None)
    return "\n".join(results)
