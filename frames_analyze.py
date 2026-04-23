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
from tracing import start_observation

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
    with start_observation(
        "ollama.frame-analysis",
        as_type="generation",
        model=FRAME_MODEL,
        input={
            "prompt": FRAME_ANALYSIS_PROMPT,
            "system": FRAME_ANALYSIS_SYSTEM,
            "schema": FrameAnalysisResult.model_json_schema(),
        },
        metadata={
            "provider": "ollama",
            "image_name": Path(image_path).name,
            "image_size_bytes": img_size,
        },
    ) as generation:
        raw = _ollama_vision_post(
            FRAME_ANALYSIS_PROMPT,
            FRAME_ANALYSIS_SYSTEM,
            b64,
            FrameAnalysisResult.model_json_schema(),
        )
        try:
            result = FrameAnalysisResult.model_validate_json(raw).to_context_str()
        except Exception:
            result = raw
        if generation is not None:
            generation.update(output=result)
        return result




_NO_NAME_MARKERS = (
    "no visible",
    "no name",
    "unknown",
    "not visible",
    "no label",
    "no tag",
    "cannot see",
    "no person",
    "not identified",
)
_MAX_FRAMES_PER_SPEAKER = 3


def _is_no_name(name: str | None) -> bool:
    if not name:
        return True
    lower = name.lower()
    return any(marker in lower for marker in _NO_NAME_MARKERS)


def _has_name(candidate: "SpeakerFrameResult") -> bool:
    return not _is_no_name(candidate.caption_name) or not _is_no_name(candidate.active_panel_name)


def analyze_speaker_frames(
    input_path: str,
    tmp_dir: str,
    diarization_segments: list[tuple[float, float, str]],
    async_q: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    start_index: int = 0,
    total_hint: int | None = None,
) -> str:
    """Extract and analyze frames per speaker; prefers frames where a name is visible."""
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
        best_with_name: SpeakerFrameResult | None = None
        best_ts_with_name: int | None = None
        best_without_name: SpeakerFrameResult | None = None
        best_ts_without_name: int | None = None

        for start, end in speaker_segs[spk][:_MAX_FRAMES_PER_SPEAKER]:
            ts = int((start + end) / 2)
            out_path = str(Path(tmp_dir) / f"frame_spk_{spk}_{ts}s.jpg")
            if not extract_single_frame(input_path, out_path, ts):
                continue
            prompt = SPEAKER_FRAME_PROMPT_TEMPLATE.format(speaker_id=spk, ts=ts)
            with open(out_path, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode()
            try:
                with start_observation(
                    "ollama.speaker-frame-analysis",
                    as_type="generation",
                    model=FRAME_MODEL,
                    input={
                        "prompt": prompt,
                        "system": SPEAKER_FRAME_SYSTEM,
                        "schema": SpeakerFrameResult.model_json_schema(),
                    },
                    metadata={
                        "provider": "ollama",
                        "speaker_id": spk,
                        "timestamp_sec": ts,
                        "image_name": Path(out_path).name,
                    },
                ) as generation:
                    raw = _ollama_vision_post(
                        prompt,
                        SPEAKER_FRAME_SYSTEM,
                        b64,
                        SpeakerFrameResult.model_json_schema(),
                    )
                    candidate = SpeakerFrameResult.model_validate_json(raw)
                    if generation is not None:
                        generation.update(
                            output={
                                "person_visible": candidate.person_visible,
                                "caption_name": candidate.caption_name,
                                "active_panel_name": candidate.active_panel_name,
                                "appearance": candidate.appearance,
                                "position": candidate.position,
                            }
                        )
            except Exception as exc:
                log.warning("  [frames] speaker frame error for %s @ %ds: %s", spk, ts, exc)
                continue

            if candidate.person_visible:
                if _has_name(candidate):
                    best_with_name = candidate
                    best_ts_with_name = ts
                    name_found = candidate.caption_name or candidate.active_panel_name
                    log.info("  [frames] found name for %s @ %ds: %s", spk, ts, name_found)
                    break
                if best_without_name is None:
                    best_without_name = candidate
                    best_ts_without_name = ts
                    log.info("  [frames] person visible but no name for %s @ %ds, trying next", spk, ts)
            else:
                log.info("  [frames] no person visible for %s @ %ds, trying next", spk, ts)

        result = best_with_name or best_without_name
        used_ts = best_ts_with_name or best_ts_without_name

        if result and used_ts is not None:
            results.append(result.to_context_str(spk, used_ts))
            name_status = result.caption_name or result.active_panel_name or "no name"
            log.info("  [frames] speaker %s @ %ds → %s", spk, used_ts, name_status)
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
