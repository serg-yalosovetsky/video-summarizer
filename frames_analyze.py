from __future__ import annotations

import asyncio
import base64
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass, field
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


def _prompt_preview(prompt: str, limit: int = 180) -> str:
    compact = " ".join(prompt.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _schema_summary(schema: dict) -> str:
    properties = schema.get("properties")
    if not isinstance(properties, dict) or not properties:
        return "<none>"
    return ",".join(sorted(properties))


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
    timeout_seconds = 300.0
    started_at = time.monotonic()
    schema_summary = _schema_summary(schema)
    prompt_preview = _prompt_preview(prompt)
    approx_image_bytes = (len(b64_image) * 3) // 4
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
                "keep_alive": "10m",
            },
            timeout=timeout_seconds,
        )
    except httpx.TimeoutException as exc:
        elapsed = time.monotonic() - started_at
        log.warning(
            "  [frames] ← TIMEOUT after %.2fs  url=%s  model=%s  prompt_chars=%d  image_bytes~=%d  schema=%s  exc_type=%s  exc=%s",
            elapsed,
            OLLAMA_URL,
            FRAME_MODEL,
            len(prompt),
            approx_image_bytes,
            schema_summary,
            type(exc).__name__,
            exc,
        )
        log.warning("  [frames] ← prompt preview: %s", prompt_preview)
        raise
    except httpx.ConnectError as exc:
        log.warning(
            "  [frames] ← CONNECT ERROR  url=%s  model=%s  prompt_chars=%d  image_bytes~=%d  schema=%s  exc_type=%s  exc=%s",
            OLLAMA_URL,
            FRAME_MODEL,
            len(prompt),
            approx_image_bytes,
            schema_summary,
            type(exc).__name__,
            exc,
        )
        log.warning("  [frames] ← prompt preview: %s", prompt_preview)
        raise
    log.info("  [frames] ← HTTP %d  (%.2fs)", response.status_code,
             response.elapsed.total_seconds())
    if not response.is_success:
        log.warning("  [frames] ← error body: %s", response.text[:500])
        log.warning(
            "  [frames] ← failed request meta  model=%s  prompt_chars=%d  image_bytes~=%d  schema=%s  prompt=%s",
            FRAME_MODEL,
            len(prompt),
            approx_image_bytes,
            schema_summary,
            prompt_preview,
        )
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
_SPEAKER_FRAME_SCHEMA = SpeakerFrameResult.model_json_schema()


@dataclass
class _SpeakerFrameSelection:
    result: SpeakerFrameResult | None = None
    timestamp: int | None = None
    attempted_timestamps: list[int] = field(default_factory=list)
    failed_attempts: list[str] = field(default_factory=list)


def _is_no_name(name: str | None) -> bool:
    if not name:
        return True
    lower = name.lower()
    return any(marker in lower for marker in _NO_NAME_MARKERS)


def _has_name(candidate: "SpeakerFrameResult") -> bool:
    return not _is_no_name(candidate.caption_name) or not _is_no_name(candidate.active_panel_name)


def _group_segments_by_speaker(
    diarization_segments: list[tuple[float, float, str]],
) -> dict[str, list[tuple[float, float]]]:
    speaker_segments: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for start, end, speaker_id in diarization_segments:
        speaker_segments[speaker_id].append((start, end))
    for segments in speaker_segments.values():
        segments.sort(key=lambda segment: segment[1] - segment[0], reverse=True)
    return speaker_segments


def _speaker_frame_timestamp(start: float, end: float) -> int:
    return int((start + end) / 2)


def _speaker_frame_path(tmp_dir: str, speaker_id: str, ts: int) -> Path:
    return Path(tmp_dir) / f"frame_spk_{speaker_id}_{ts}s.jpg"


def _notify_progress(
    async_q: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    current: int,
    total: int,
) -> None:
    loop.call_soon_threadsafe(
        async_q.put_nowait,
        {"current": current, "total": total},
    )


def _analyze_speaker_frame_candidate(
    input_path: str,
    tmp_dir: str,
    speaker_id: str,
    ts: int,
) -> tuple[SpeakerFrameResult | None, str | None]:
    out_path = _speaker_frame_path(tmp_dir, speaker_id, ts)
    if not extract_single_frame(input_path, str(out_path), ts):
        log.warning(
            "  [frames] failed to extract speaker frame  speaker=%s  ts=%ds  image_path=%s",
            speaker_id,
            ts,
            out_path,
        )
        return None, f"{ts}s:extract_frame_failed"

    prompt = SPEAKER_FRAME_PROMPT_TEMPLATE.format(speaker_id=speaker_id, ts=ts)
    prompt_preview = _prompt_preview(prompt)
    img_size = out_path.stat().st_size
    log.info(
        "  [frames] → speaker POST  speaker=%s  ts=%ds  image=%s  size=%dKB  model=%s",
        speaker_id,
        ts,
        out_path.name,
        max(1, img_size // 1024),
        FRAME_MODEL,
    )
    with out_path.open("rb") as file_handle:
        b64 = base64.b64encode(file_handle.read()).decode()

    try:
        with start_observation(
            "ollama.speaker-frame-analysis",
            as_type="generation",
            model=FRAME_MODEL,
            input={
                "prompt": prompt,
                "system": SPEAKER_FRAME_SYSTEM,
                "schema": _SPEAKER_FRAME_SCHEMA,
            },
            metadata={
                "provider": "ollama",
                "speaker_id": speaker_id,
                "timestamp_sec": ts,
                "image_name": out_path.name,
            },
        ) as generation:
            raw = _ollama_vision_post(
                prompt,
                SPEAKER_FRAME_SYSTEM,
                b64,
                _SPEAKER_FRAME_SCHEMA,
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
        log.warning(
            "  [frames] speaker frame error  speaker=%s  ts=%ds  image=%s  image_path=%s  size=%dKB  prompt_chars=%d  exc_type=%s  exc=%s",
            speaker_id,
            ts,
            out_path.name,
            out_path,
            max(1, img_size // 1024),
            len(prompt),
            type(exc).__name__,
            exc,
        )
        log.warning(
            "  [frames] speaker frame prompt  speaker=%s  ts=%ds  prompt=%s",
            speaker_id,
            ts,
            prompt_preview,
        )
        return None, f"{ts}s:{type(exc).__name__}"

    return candidate, None


def _select_speaker_frame(
    input_path: str,
    tmp_dir: str,
    speaker_id: str,
    segments: list[tuple[float, float]],
) -> _SpeakerFrameSelection:
    selection = _SpeakerFrameSelection()
    fallback_result: SpeakerFrameResult | None = None
    fallback_ts: int | None = None

    for start, end in segments[:_MAX_FRAMES_PER_SPEAKER]:
        ts = _speaker_frame_timestamp(start, end)
        selection.attempted_timestamps.append(ts)

        candidate, failure = _analyze_speaker_frame_candidate(
            input_path=input_path,
            tmp_dir=tmp_dir,
            speaker_id=speaker_id,
            ts=ts,
        )
        if failure is not None:
            selection.failed_attempts.append(failure)
            continue
        if candidate is None:
            continue

        if not candidate.person_visible:
            log.info("  [frames] no person visible for %s @ %ds, trying next", speaker_id, ts)
            continue

        if _has_name(candidate):
            name_found = candidate.caption_name or candidate.active_panel_name
            log.info("  [frames] found name for %s @ %ds: %s", speaker_id, ts, name_found)
            selection.result = candidate
            selection.timestamp = ts
            return selection

        if fallback_result is None:
            fallback_result = candidate
            fallback_ts = ts
            log.info(
                "  [frames] person visible but no name for %s @ %ds, trying next",
                speaker_id,
                ts,
            )

    selection.result = fallback_result
    selection.timestamp = fallback_ts
    return selection


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
    speaker_segs = _group_segments_by_speaker(diarization_segments)

    results: list[str] = []
    total = total_hint or (start_index + len(speaker_segs))
    for idx, spk in enumerate(sorted(speaker_segs), start=1):
        _notify_progress(async_q, loop, start_index + idx, total)
        selection = _select_speaker_frame(
            input_path=input_path,
            tmp_dir=tmp_dir,
            speaker_id=spk,
            segments=speaker_segs[spk],
        )

        if selection.result is not None and selection.timestamp is not None:
            results.append(selection.result.to_context_str(spk, selection.timestamp))
            name_status = (
                selection.result.caption_name
                or selection.result.active_panel_name
                or "no name"
            )
            log.info(
                "  [frames] speaker %s @ %ds → %s",
                spk,
                selection.timestamp,
                name_status,
            )
        else:
            log.warning(
                "  [frames] no usable frame found for %s  attempted=%s  failures=%s",
                spk,
                selection.attempted_timestamps or ["<none>"],
                selection.failed_attempts or ["<none>"],
            )

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
