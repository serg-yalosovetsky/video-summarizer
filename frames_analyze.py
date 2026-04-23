from __future__ import annotations

import asyncio
import base64
import re
import subprocess
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

import httpx

from config import settings
from helpers import log
from models import (
    ActiveSpeakerDetection,
    CaptionExtraction,
    FrameAnalysisResult,
    SpeakerAppearance,
    SpeakerFrameResult,
    SpeakerPanelName,
)
from prompts import (
    ACTIVE_SPEAKER_DETECT_PROMPT,
    ACTIVE_SPEAKER_DETECT_SYSTEM,
    CAPTION_EXTRACT_PROMPT,
    CAPTION_EXTRACT_SYSTEM,
    FRAME_ANALYSIS_PROMPT,
    FRAME_ANALYSIS_SYSTEM,
    SPEAKER_APPEARANCE_PROMPT_TEMPLATE,
    SPEAKER_APPEARANCE_SYSTEM,
    SPEAKER_NAME_PROMPT_TEMPLATE,
    SPEAKER_NAME_SYSTEM,
)
from tracing import start_observation

T = TypeVar("T")
SpeakerSegment = tuple[float, float]
DiarizationSegment = tuple[float, float, str]

OLLAMA_URL = settings.ollama_url
FRAME_MODEL = settings.frame_model
FRAME_TIMESTAMPS = list(settings.frame_timestamps)
MAX_FRAMES = settings.max_frames

_FRAME_ANALYSIS_SCHEMA = FrameAnalysisResult.model_json_schema()
_ACTIVE_SPEAKER_SCHEMA = ActiveSpeakerDetection.model_json_schema()
_CAPTION_EXTRACTION_SCHEMA = CaptionExtraction.model_json_schema()
_SPEAKER_APPEARANCE_SCHEMA = SpeakerAppearance.model_json_schema()
_SPEAKER_PANEL_NAME_SCHEMA = SpeakerPanelName.model_json_schema()
_OLLAMA_TIMEOUT_SECONDS = 300.0
_OLLAMA_KEEP_ALIVE = "10m"
_MAX_FRAMES_PER_SPEAKER = 3
_NO_INFO_PHRASES = {
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
_VALID_POSITIONS = frozenset(
    {
        "top-left",
        "top-center",
        "top-right",
        "middle-left",
        "middle-center",
        "middle-right",
        "bottom-left",
        "bottom-center",
        "bottom-right",
    }
)
_CAPTION_ENTRY_NAME_RE = re.compile(r"([^\n:]{1,80})\s*:")


@dataclass(frozen=True)
class _EncodedImage:
    path: Path
    size_bytes: int
    b64: str

    @property
    def size_kb(self) -> int:
        return max(1, self.size_bytes // 1024)


@dataclass
class _SpeakerFrameSelection:
    result: SpeakerFrameResult | None = None
    timestamp: int | None = None
    attempted_timestamps: list[int] = field(default_factory=list)
    failed_attempts: list[str] = field(default_factory=list)


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


def _approx_image_bytes(b64_image: str) -> int:
    return (len(b64_image) * 3) // 4


def _timeline_frame_path(tmp_dir: str, ts: int) -> Path:
    return Path(tmp_dir) / f"frame_{ts}s.jpg"


def _speaker_frame_path(tmp_dir: str, speaker_id: str, ts: int) -> Path:
    return Path(tmp_dir) / f"frame_spk_{speaker_id}_{ts}s.jpg"


def _frame_label(image_path: str) -> str:
    return Path(image_path).stem.removeprefix("frame_")


def _resolve_total(start_index: int, total_hint: int | None, item_count: int) -> int:
    return total_hint or (start_index + item_count)


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


def _finish_progress(async_q: asyncio.Queue, loop: asyncio.AbstractEventLoop) -> None:
    loop.call_soon_threadsafe(async_q.put_nowait, None)


def _load_encoded_image(image_path: str | Path) -> _EncodedImage:
    path = Path(image_path)
    size_bytes = path.stat().st_size
    with path.open("rb") as file_handle:
        b64 = base64.b64encode(file_handle.read()).decode()
    return _EncodedImage(path=path, size_bytes=size_bytes, b64=b64)


def _extract_frames_for_timestamps(
    input_path: str,
    tmp_dir: str,
    timestamps: Iterable[int],
) -> list[str]:
    extracted_paths: list[str] = []
    for ts in timestamps:
        out_path = _timeline_frame_path(tmp_dir, ts)
        if extract_single_frame(input_path, str(out_path), ts):
            extracted_paths.append(str(out_path))
    return extracted_paths


def _candidate_default_timestamps(duration_sec: float) -> list[int]:
    timestamps: list[int] = []
    for ts in FRAME_TIMESTAMPS:
        if ts >= duration_sec:
            break
        timestamps.append(ts)
    return timestamps


def _can_use_timestamp(ts: int, seen: set[int]) -> bool:
    return ts not in seen and all(abs(ts - existing) >= 2 for existing in seen)


def _is_informative_context_line(line: str) -> bool:
    lower = line.lower()
    return not any(phrase in lower for phrase in _NO_INFO_PHRASES)


def _ollama_request_payload(
    prompt: str,
    system: str,
    b64_image: str,
    schema: dict,
) -> dict[str, object]:
    return {
        "model": FRAME_MODEL,
        "prompt": prompt,
        "system": system,
        "images": [b64_image],
        "stream": False,
        "format": schema,
        "keep_alive": _OLLAMA_KEEP_ALIVE,
    }


def _log_ollama_transport_error(
    label: str,
    exc: Exception,
    *,
    prompt: str,
    b64_image: str,
    schema: dict,
    started_at: float | None = None,
) -> None:
    elapsed_suffix = ""
    if started_at is not None:
        elapsed_suffix = f" after {time.monotonic() - started_at:.2f}s"
    log.warning(
        "  [frames] ← %s%s  url=%s  model=%s  prompt_chars=%d  image_bytes~=%d  schema=%s  exc_type=%s  exc=%s",
        label,
        elapsed_suffix,
        OLLAMA_URL,
        FRAME_MODEL,
        len(prompt),
        _approx_image_bytes(b64_image),
        _schema_summary(schema),
        type(exc).__name__,
        exc,
    )
    log.warning("  [frames] ← prompt preview: %s", _prompt_preview(prompt))


def _log_ollama_failed_response(
    response: httpx.Response,
    *,
    prompt: str,
    b64_image: str,
    schema: dict,
) -> None:
    log.warning("  [frames] ← error body: %s", response.text[:500])
    log.warning(
        "  [frames] ← failed request meta  model=%s  prompt_chars=%d  image_bytes~=%d  schema=%s  prompt=%s",
        FRAME_MODEL,
        len(prompt),
        _approx_image_bytes(b64_image),
        _schema_summary(schema),
        _prompt_preview(prompt),
    )


def _run_structured_vision_request(
    image: _EncodedImage,
    *,
    prompt: str,
    system: str,
    schema: dict,
    observation_name: str,
    metadata: dict[str, object],
    request_log_message: str,
    request_log_args: tuple[object, ...],
    parser: Callable[[str], T],
    observation_output: Callable[[T], object],
) -> T:
    log.info(request_log_message, *request_log_args)
    with start_observation(
        observation_name,
        as_type="generation",
        model=FRAME_MODEL,
        input={
            "prompt": prompt,
            "system": system,
            "schema": schema,
        },
        metadata=metadata,
    ) as generation:
        raw = _ollama_vision_post(prompt, system, image.b64, schema)
        result = parser(raw)
        if generation is not None:
            generation.update(output=observation_output(result))
        return result


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
    result = subprocess.run(cmd, capture_output=True, check=False)
    return result.returncode == 0 and Path(out_path).exists()


def extract_frames(input_path: str, tmp_dir: str, duration_sec: float) -> list[str]:
    return _extract_frames_for_timestamps(
        input_path,
        tmp_dir,
        _candidate_default_timestamps(duration_sec),
    )


def extract_frames_at(input_path: str, tmp_dir: str, timestamps: list[int]) -> list[str]:
    return _extract_frames_for_timestamps(input_path, tmp_dir, timestamps)


def generate_frame_timestamps(done_ts: set[int], duration_sec: float, count: int) -> list[int]:
    """Return up to `count` new timestamps spread evenly across the video."""
    result: list[int] = []
    seen = set(done_ts)
    step = max(1, int(duration_sec / (count + 1)))
    ts = step

    while ts < int(duration_sec) and len(result) < count:
        if _can_use_timestamp(ts, seen):
            result.append(ts)
            seen.add(ts)
        ts += step

    if len(result) < count:
        for ts in range(1, int(duration_sec)):
            if ts in seen:
                continue
            result.append(ts)
            seen.add(ts)
            if len(result) >= count:
                break

    return result[:count]


def is_context_sufficient(context: str) -> bool:
    if not context or len(context.strip()) < 80:
        return False

    lines = [line.strip() for line in context.splitlines() if line.strip()]
    if len(lines) < 2:
        return False

    useful_line_count = sum(1 for line in lines if _is_informative_context_line(line))
    return useful_line_count >= 2


def _ollama_vision_post(prompt: str, system: str, b64_image: str, schema: dict) -> str:
    """POST to Ollama vision endpoint with structured output; returns raw response string."""
    started_at = time.monotonic()
    try:
        response = httpx.post(
            OLLAMA_URL,
            json=_ollama_request_payload(prompt, system, b64_image, schema),
            timeout=_OLLAMA_TIMEOUT_SECONDS,
        )
    except httpx.TimeoutException as exc:
        _log_ollama_transport_error(
            "TIMEOUT",
            exc,
            prompt=prompt,
            b64_image=b64_image,
            schema=schema,
            started_at=started_at,
        )
        raise
    except httpx.ConnectError as exc:
        _log_ollama_transport_error(
            "CONNECT ERROR",
            exc,
            prompt=prompt,
            b64_image=b64_image,
            schema=schema,
        )
        raise

    log.info("  [frames] ← HTTP %d  (%.2fs)", response.status_code, response.elapsed.total_seconds())
    if not response.is_success:
        _log_ollama_failed_response(
            response,
            prompt=prompt,
            b64_image=b64_image,
            schema=schema,
        )
    response.raise_for_status()
    return response.json()["response"]


def _parse_frame_analysis(raw: str) -> str:
    try:
        return FrameAnalysisResult.model_validate_json(raw).to_context_str()
    except Exception:
        return raw


def analyze_frame(image_path: str) -> str:
    image = _load_encoded_image(image_path)
    return _run_structured_vision_request(
        image,
        prompt=FRAME_ANALYSIS_PROMPT,
        system=FRAME_ANALYSIS_SYSTEM,
        schema=_FRAME_ANALYSIS_SCHEMA,
        observation_name="ollama.frame-analysis",
        metadata={
            "provider": "ollama",
            "image_name": image.path.name,
            "image_size_bytes": image.size_bytes,
        },
        request_log_message="  [frames] → POST %s  model=%s  image=%s  size=%dKB",
        request_log_args=(OLLAMA_URL, FRAME_MODEL, image.path.name, image.size_kb),
        parser=_parse_frame_analysis,
        observation_output=lambda result: result,
    )


def _clean_text_value(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = " ".join(value.strip().split())
    return cleaned or None


def _is_no_name(name: str | None) -> bool:
    if not name:
        return True
    lower = name.lower()
    return any(marker in lower for marker in _NO_NAME_MARKERS)


def _clean_name(value: str | None) -> str | None:
    cleaned = _clean_text_value(value)
    if cleaned is None or _is_no_name(cleaned):
        return None
    return cleaned


def _normalise_position(position: str | None) -> str | None:
    cleaned = _clean_text_value(position)
    if cleaned is None:
        return None
    normalised = re.sub(r"[\s_]+", "-", cleaned.lower())
    return normalised if normalised in _VALID_POSITIONS else None


def _normalise_caption_name(value: str | None) -> str | None:
    if value is None:
        return None

    raw = value.strip()
    if not raw:
        return None

    parsed_names: list[str] = []
    for part in re.split(r"\n+|\s+\|\s+|(?<=[.!?])\s+(?=[A-ZА-ЯІЇЄҐ])", raw):
        match = _CAPTION_ENTRY_NAME_RE.match(part.strip())
        if match:
            parsed_name = _clean_name(match.group(1))
            if parsed_name is not None:
                parsed_names.append(parsed_name)

    if parsed_names:
        return parsed_names[-1]
    return _clean_name(raw)


def _candidate_rank(candidate: SpeakerFrameResult) -> int:
    if candidate.no_active_speaker and _clean_name(candidate.caption_name) is not None:
        return 3  # Teams self-speaker: no border + caption = highest confidence
    if _clean_name(candidate.active_panel_name) is not None:
        return 2
    if _clean_name(candidate.caption_name) is not None:
        return 1
    if candidate.person_visible:
        return 0
    return -1


def _speaker_frame_timestamp(start: float, end: float) -> int:
    candidate_ts = start + 3.0
    return int(candidate_ts) if candidate_ts < end else int(start)


def _run_speaker_frame_request(
    image: _EncodedImage,
    *,
    speaker_id: str,
    ts: int,
    step_name: str,
    prompt: str,
    system: str,
    schema: dict,
    parser: Callable[[str], T],
) -> T:
    return _run_structured_vision_request(
        image,
        prompt=prompt,
        system=system,
        schema=schema,
        observation_name=f"ollama.speaker-frame-{step_name}",
        metadata={
            "provider": "ollama",
            "speaker_id": speaker_id,
            "timestamp_sec": ts,
            "image_name": image.path.name,
            "step": step_name,
        },
        request_log_message=(
            "  [frames] → speaker %s  speaker=%s  ts=%ds  image=%s  size=%dKB  model=%s"
        ),
        request_log_args=(
            step_name,
            speaker_id,
            ts,
            image.path.name,
            image.size_kb,
            FRAME_MODEL,
        ),
        parser=parser,
        observation_output=(
            lambda result: result.model_dump() if hasattr(result, "model_dump") else result
        ),
    )


def _detect_active_speaker(
    image: _EncodedImage,
    *,
    speaker_id: str,
    ts: int,
) -> ActiveSpeakerDetection:
    return _run_speaker_frame_request(
        image,
        speaker_id=speaker_id,
        ts=ts,
        step_name="active-speaker",
        prompt=ACTIVE_SPEAKER_DETECT_PROMPT,
        system=ACTIVE_SPEAKER_DETECT_SYSTEM,
        schema=_ACTIVE_SPEAKER_SCHEMA,
        parser=ActiveSpeakerDetection.model_validate_json,
    )


def _extract_caption(
    image: _EncodedImage,
    *,
    speaker_id: str,
    ts: int,
) -> CaptionExtraction:
    result = _run_speaker_frame_request(
        image,
        speaker_id=speaker_id,
        ts=ts,
        step_name="caption",
        prompt=CAPTION_EXTRACT_PROMPT,
        system=CAPTION_EXTRACT_SYSTEM,
        schema=_CAPTION_EXTRACTION_SCHEMA,
        parser=CaptionExtraction.model_validate_json,
    )
    result.last_speaker_name = _normalise_caption_name(result.last_speaker_name)
    return result


def _describe_speaker(
    image: _EncodedImage,
    *,
    speaker_id: str,
    ts: int,
    position: str,
) -> SpeakerAppearance:
    return _run_speaker_frame_request(
        image,
        speaker_id=speaker_id,
        ts=ts,
        step_name="appearance",
        prompt=SPEAKER_APPEARANCE_PROMPT_TEMPLATE.format(position=position),
        system=SPEAKER_APPEARANCE_SYSTEM,
        schema=_SPEAKER_APPEARANCE_SCHEMA,
        parser=SpeakerAppearance.model_validate_json,
    )


def _read_speaker_panel_name(
    image: _EncodedImage,
    *,
    speaker_id: str,
    ts: int,
    position: str,
) -> SpeakerPanelName:
    result = _run_speaker_frame_request(
        image,
        speaker_id=speaker_id,
        ts=ts,
        step_name="speaker-name",
        prompt=SPEAKER_NAME_PROMPT_TEMPLATE.format(position=position),
        system=SPEAKER_NAME_SYSTEM,
        schema=_SPEAKER_PANEL_NAME_SCHEMA,
        parser=SpeakerPanelName.model_validate_json,
    )
    result.name = _clean_name(result.name)
    return result


def _build_speaker_frame_result(
    active_speaker: ActiveSpeakerDetection,
    caption: CaptionExtraction,
    appearance: SpeakerAppearance | None = None,
    panel_name: SpeakerPanelName | None = None,
) -> SpeakerFrameResult:
    position = _normalise_position(active_speaker.speaker_position) or ""
    appearance_text = _clean_text_value(appearance.appearance if appearance is not None else None) or ""
    caption_name = _normalise_caption_name(caption.last_speaker_name)
    active_panel_name = _clean_name(panel_name.name if panel_name is not None else None)

    if active_panel_name is not None:
        return SpeakerFrameResult(
            person_visible=True,
            caption_name=caption_name,
            active_panel_name=active_panel_name,
            appearance=appearance_text,
            position=position,
        )

    if caption_name is not None:
        if active_speaker.has_active_speaker and position:
            return SpeakerFrameResult(
                person_visible=True,
                caption_name=caption_name,
                appearance=appearance_text,
                position=position,
            )
        return SpeakerFrameResult(
            person_visible=True,
            caption_name=caption_name,
            no_active_speaker=True,
        )

    if active_speaker.has_active_speaker and position:
        return SpeakerFrameResult(
            person_visible=True,
            appearance=appearance_text,
            position=position,
        )

    return SpeakerFrameResult(person_visible=False)


def _group_segments_by_speaker(
    diarization_segments: list[DiarizationSegment],
) -> dict[str, list[SpeakerSegment]]:
    speaker_segments: dict[str, list[SpeakerSegment]] = defaultdict(list)
    for start, end, speaker_id in diarization_segments:
        speaker_segments[speaker_id].append((start, end))
    for segments in speaker_segments.values():
        segments.sort(key=lambda segment: segment[1] - segment[0], reverse=True)
    return speaker_segments


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

    image: _EncodedImage | None = None
    current_step = "load-image"
    current_prompt = ""

    try:
        image = _load_encoded_image(out_path)
        current_step = "active-speaker"
        current_prompt = ACTIVE_SPEAKER_DETECT_PROMPT
        active_speaker = _detect_active_speaker(
            image,
            speaker_id=speaker_id,
            ts=ts,
        )

        current_step = "caption"
        current_prompt = CAPTION_EXTRACT_PROMPT
        caption = _extract_caption(
            image,
            speaker_id=speaker_id,
            ts=ts,
        )

        appearance: SpeakerAppearance | None = None
        panel_name: SpeakerPanelName | None = None
        position = _normalise_position(active_speaker.speaker_position)
        if active_speaker.has_active_speaker and position:
            current_step = "appearance"
            current_prompt = SPEAKER_APPEARANCE_PROMPT_TEMPLATE.format(position=position)
            appearance = _describe_speaker(
                image,
                speaker_id=speaker_id,
                ts=ts,
                position=position,
            )

            current_step = "speaker-name"
            current_prompt = SPEAKER_NAME_PROMPT_TEMPLATE.format(position=position)
            panel_name = _read_speaker_panel_name(
                image,
                speaker_id=speaker_id,
                ts=ts,
                position=position,
            )

        candidate = _build_speaker_frame_result(
            active_speaker,
            caption,
            appearance=appearance,
            panel_name=panel_name,
        )
    except Exception as exc:
        image_name = image.path.name if image is not None else out_path.name
        image_size_kb = image.size_kb if image is not None else 0
        log.warning(
            "  [frames] speaker frame error  speaker=%s  ts=%ds  step=%s  image=%s  image_path=%s  size=%dKB  prompt_chars=%d  exc_type=%s  exc=%s",
            speaker_id,
            ts,
            current_step,
            image_name,
            out_path,
            image_size_kb,
            len(current_prompt),
            type(exc).__name__,
            exc,
        )
        log.warning(
            "  [frames] speaker frame prompt  speaker=%s  ts=%ds  step=%s  prompt=%s",
            speaker_id,
            ts,
            current_step,
            _prompt_preview(current_prompt) if current_prompt else "<none>",
        )
        return None, f"{ts}s:{current_step}:{type(exc).__name__}"

    return candidate, None


def _select_speaker_frame(
    input_path: str,
    tmp_dir: str,
    speaker_id: str,
    segments: list[SpeakerSegment],
) -> _SpeakerFrameSelection:
    selection = _SpeakerFrameSelection()
    best_result: SpeakerFrameResult | None = None
    best_timestamp: int | None = None
    best_rank = -1

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

        rank = _candidate_rank(candidate)
        if rank < 0:
            log.info("  [frames] no person visible for %s @ %ds, trying next", speaker_id, ts)
            continue

        if rank == 3:
            log.info(
                "  [frames] found self-speaker (no border + caption) for %s @ %ds: %s",
                speaker_id,
                ts,
                candidate.caption_name,
            )
        elif rank == 2:
            log.info(
                "  [frames] found active panel name for %s @ %ds: %s",
                speaker_id,
                ts,
                candidate.active_panel_name,
            )
        elif rank == 1:
            log.info(
                "  [frames] found caption fallback for %s @ %ds: %s",
                speaker_id,
                ts,
                candidate.caption_name,
            )
        else:
            log.info(
                "  [frames] person visible but no name for %s @ %ds, trying next",
                speaker_id,
                ts,
            )

        if rank > best_rank:
            best_result = candidate
            best_timestamp = ts
            best_rank = rank

    selection.result = best_result
    selection.timestamp = best_timestamp
    return selection


def analyze_speaker_frames(
    input_path: str,
    tmp_dir: str,
    diarization_segments: list[DiarizationSegment],
    async_q: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    start_index: int = 0,
    total_hint: int | None = None,
) -> str:
    """Extract and analyze frames per speaker; prefers frames where a name is visible."""
    speaker_segments = _group_segments_by_speaker(diarization_segments)
    results: list[str] = []
    total = _resolve_total(start_index, total_hint, len(speaker_segments))

    try:
        for index, speaker_id in enumerate(sorted(speaker_segments), start=1):
            _notify_progress(async_q, loop, start_index + index, total)
            selection = _select_speaker_frame(
                input_path=input_path,
                tmp_dir=tmp_dir,
                speaker_id=speaker_id,
                segments=speaker_segments[speaker_id],
            )

            if selection.result is not None and selection.timestamp is not None:
                results.append(selection.result.to_context_str(speaker_id, selection.timestamp))
                name_status = selection.result.preferred_name() or "no name"
                log.info(
                    "  [frames] speaker %s @ %ds → %s",
                    speaker_id,
                    selection.timestamp,
                    name_status,
                )
                continue

            log.warning(
                "  [frames] no usable frame found for %s  attempted=%s  failures=%s",
                speaker_id,
                selection.attempted_timestamps or ["<none>"],
                selection.failed_attempts or ["<none>"],
            )
    finally:
        _finish_progress(async_q, loop)

    return "\n".join(results)


def analyze_frames_with_progress(
    image_paths: list[str],
    async_q: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    start_index: int = 0,
    total_hint: int | None = None,
) -> str:
    results: list[str] = []
    total = _resolve_total(start_index, total_hint, len(image_paths))

    try:
        for index, image_path in enumerate(image_paths, start=1):
            if index > 1:
                time.sleep(1)
            _notify_progress(async_q, loop, start_index + index, total)

            ts_label = _frame_label(image_path)
            try:
                description = analyze_frame(image_path)
            except Exception as exc:
                log.warning("  [frames] %s error: %s", ts_label, exc)
                continue

            results.append(f"[{ts_label}] {description}")
            log.info("  [frames] analyzed %s (%d/%d)", ts_label, start_index + index, total)
    finally:
        _finish_progress(async_q, loop)

    return "\n".join(results)
