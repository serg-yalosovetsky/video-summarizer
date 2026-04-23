from __future__ import annotations

import math
import shutil
import wave
from functools import lru_cache
from typing import Any

import numpy as np

from config import settings
from helpers import HF_TOKEN, log
from tracing import start_observation
from transcribe_types import DiarizationSegment, LegacySegment

PYANNOTE_DEVICE = settings.pyannote_device
DIARIZATION_GAP_THRESHOLD_SEC = 1.0
DIARIZATION_MIN_SEGMENT_SEC = 0.5
DIARIZATION_MAX_SEGMENT_SEC = 20.0
SILENCE_PEAK_THRESHOLD_DBFS = -45.0
SILENCE_RMS_THRESHOLD_DBFS = -50.0
WAV_ACTIVITY_SAMPLE_DURATION_SEC = 15.0
WAV_ACTIVITY_SAMPLE_POSITIONS = (0.05, 0.5, 0.95)


def cuda_diagnostics(torch_module) -> str:
    return (
        f"torch={torch_module.__version__}, "
        f"torch.version.cuda={torch_module.version.cuda}, "
        f"torch.cuda.is_available()={torch_module.cuda.is_available()}, "
        f"nvidia-smi={'found' if shutil.which('nvidia-smi') else 'missing'}"
    )


def choose_pyannote_device():
    import torch

    if PYANNOTE_DEVICE not in {"cuda", "cpu", "auto"}:
        raise RuntimeError(
            f"Unsupported PYANNOTE_DEVICE={PYANNOTE_DEVICE!r}. Use one of: cuda, cpu, auto."
        )

    if PYANNOTE_DEVICE == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")

    if PYANNOTE_DEVICE == "auto":
        log.warning("CUDA is unavailable for pyannote, falling back to CPU.")
        return torch.device("cpu")

    raise RuntimeError(
        "CUDA is required for pyannote diarization but is unavailable. "
        f"{cuda_diagnostics(torch)}. "
        "Run install.bat/install.sh to install a CUDA-enabled PyTorch build and verify NVIDIA drivers, "
        "or set PYANNOTE_DEVICE=auto/cpu if you intentionally want CPU mode."
    )


@lru_cache(maxsize=1)
def get_diarizer():
    from pyannote.audio import Pipeline

    log.info("Loading pyannote diarization model...")
    model_id = settings.pyannote_model
    pipeline = Pipeline.from_pretrained(
        model_id,
        token=HF_TOKEN or True,
    )
    device = choose_pyannote_device()
    if hasattr(pipeline, "to"):
        pipeline.to(device)
    log.info("  [pyannote] device: %s", device)
    log.info("Diarizer ready.")
    return pipeline


def _segments_from_itertracks(annotation) -> list[LegacySegment]:
    return [
        (float(turn.start), float(turn.end), str(speaker))
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]


def _segments_from_payload(payload: Any) -> list[LegacySegment] | None:
    if not isinstance(payload, dict):
        return None

    for key in ("exclusive_diarization", "diarization", "segments"):
        items = payload.get(key)
        if items is None:
            continue
        return [
            (
                float(item["start"]),
                float(item["end"]),
                str(item["speaker"]),
            )
            for item in items
        ]

    return None


def _extract_diarization_segments(raw_output: Any) -> list[LegacySegment]:
    candidates = []
    for attr in ("exclusive_speaker_diarization", "speaker_diarization", "diarization"):
        if hasattr(raw_output, attr):
            candidate = getattr(raw_output, attr)
            if candidate is not None:
                candidates.append(candidate)
    candidates.append(raw_output)

    for candidate in candidates:
        if hasattr(candidate, "itertracks"):
            return _segments_from_itertracks(candidate)

        serialize = getattr(candidate, "serialize", None)
        if callable(serialize):
            serialized = _segments_from_payload(serialize())
            if serialized is not None:
                return serialized

        payload_segments = _segments_from_payload(candidate)
        if payload_segments is not None:
            return payload_segments

    raise TypeError(f"Unsupported diarization output type: {type(raw_output).__name__}")


def _dbfs(amplitude: float) -> float:
    if amplitude <= 0.0:
        return -120.0
    return 20.0 * math.log10(amplitude)


def _decode_pcm_frames(raw_frames: bytes, *, sample_width: int, channels: int) -> np.ndarray:
    dtype_by_width = {
        1: np.uint8,
        2: np.int16,
        4: np.int32,
    }
    dtype = dtype_by_width.get(sample_width)
    if dtype is None:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")

    samples = np.frombuffer(raw_frames, dtype=dtype)
    if samples.size == 0:
        return np.empty(0, dtype=np.float32)

    if sample_width == 1:
        normalized = (samples.astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        normalized = samples.astype(np.float32) / 32768.0
    else:
        normalized = samples.astype(np.float32) / 2147483648.0

    if channels > 1:
        normalized = normalized.reshape(-1, channels)
    return normalized.reshape(-1)


def _inspect_wav_activity(wav_path: str) -> dict[str, float | int | bool] | None:
    """
    Sample a few windows from a PCM WAV and estimate whether it is near-silent.
    """
    try:
        with wave.open(wav_path, "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            total_frames = wav_file.getnframes()
            if sample_rate <= 0 or total_frames <= 0:
                return None

            duration_sec = total_frames / sample_rate
            window_sec = min(WAV_ACTIVITY_SAMPLE_DURATION_SEC, duration_sec)
            window_frames = max(1, int(window_sec * sample_rate))
            max_start_sec = max(0.0, duration_sec - window_sec)

            if duration_sec <= window_sec:
                start_offsets = [0.0]
            else:
                starts = []
                half_window = window_sec / 2.0
                for position in WAV_ACTIVITY_SAMPLE_POSITIONS:
                    center_sec = duration_sec * position
                    start_sec = min(max(center_sec - half_window, 0.0), max_start_sec)
                    starts.append(round(start_sec, 3))
                start_offsets = sorted(set(starts))

            peak_levels: list[float] = []
            rms_levels: list[float] = []
            for start_sec in start_offsets:
                start_frame = min(total_frames, max(0, int(start_sec * sample_rate)))
                frames_to_read = min(window_frames, total_frames - start_frame)
                if frames_to_read <= 0:
                    continue

                wav_file.setpos(start_frame)
                raw_frames = wav_file.readframes(frames_to_read)
                samples = _decode_pcm_frames(
                    raw_frames,
                    sample_width=sample_width,
                    channels=channels,
                )
                if samples.size == 0:
                    continue

                peak_levels.append(_dbfs(float(np.max(np.abs(samples)))))
                rms_levels.append(_dbfs(float(np.sqrt(np.mean(samples * samples)))))

            if not peak_levels or not rms_levels:
                return None

            max_peak_dbfs = max(peak_levels)
            max_rms_dbfs = max(rms_levels)
            return {
                "sampled_windows": len(peak_levels),
                "max_peak_dbfs": max_peak_dbfs,
                "max_rms_dbfs": max_rms_dbfs,
                "likely_silent": (
                    max_peak_dbfs <= SILENCE_PEAK_THRESHOLD_DBFS
                    and max_rms_dbfs <= SILENCE_RMS_THRESHOLD_DBFS
                ),
            }
    except (OSError, ValueError, wave.Error) as exc:
        log.debug("  [pyannote] WAV activity inspection failed for %s: %s", wav_path, exc)
        return None


def run_diarization(wav_path: str) -> list[LegacySegment]:
    """Run speaker diarization and return sorted legacy segments."""
    with start_observation(
        "pyannote.diarization",
        input={"audio_file": wav_path.rsplit("/", 1)[-1]},
        metadata={"model": settings.pyannote_model},
    ) as span:
        pipeline = get_diarizer()
        raw = pipeline(wav_path)
        segments = _extract_diarization_segments(raw)
        sorted_segments = sorted(segments, key=lambda x: x[0])
        if not sorted_segments:
            activity = _inspect_wav_activity(wav_path)
            if activity is not None:
                if activity["likely_silent"]:
                    log.warning(
                        "  [pyannote] no speaker turns found; audio looks silent or near-silent "
                        "(peak=%.1f dBFS, rms=%.1f dBFS, sampled_windows=%d)",
                        activity["max_peak_dbfs"],
                        activity["max_rms_dbfs"],
                        activity["sampled_windows"],
                    )
                else:
                    log.warning(
                        "  [pyannote] no speaker turns found despite non-silent audio "
                        "(peak=%.1f dBFS, rms=%.1f dBFS, sampled_windows=%d)",
                        activity["max_peak_dbfs"],
                        activity["max_rms_dbfs"],
                        activity["sampled_windows"],
                    )
        if span is not None:
            span.update(
                output={
                    "segments_count": len(sorted_segments),
                    "speaker_count": len({speaker for _, _, speaker in sorted_segments}),
                }
            )
        return sorted_segments


def _segment_from_tuple(segment: LegacySegment) -> DiarizationSegment:
    start, end, speaker = segment
    return DiarizationSegment(start=float(start), end=float(end), speaker=str(speaker))


def _segment_to_tuple(segment: DiarizationSegment) -> LegacySegment:
    return (segment.start, segment.end, segment.speaker)


def normalize_diarization_segments(
    segments: list[LegacySegment],
    gap_threshold: float = DIARIZATION_GAP_THRESHOLD_SEC,
    min_duration: float = DIARIZATION_MIN_SEGMENT_SEC,
) -> list[LegacySegment]:
    """
    Normalize diarization spans into monotonic speaker segments.

    Args:
        segments: Raw `(start, end, speaker)` spans.
        gap_threshold: Max silent gap allowed when merging same-speaker spans.
        min_duration: Minimum duration that survives cleanup.

    Returns:
        Cleaned and merged legacy segments.
    """
    if not segments:
        return []

    cleaned: list[DiarizationSegment] = []
    for raw_segment in sorted((_segment_from_tuple(item) for item in segments), key=lambda item: item.start):
        start = max(0.0, raw_segment.start)
        end = max(start, raw_segment.end)
        if cleaned and start < cleaned[-1].end:
            start = cleaned[-1].end
        if (end - start) >= min_duration:
            cleaned.append(DiarizationSegment(start=start, end=end, speaker=raw_segment.speaker))

    if not cleaned:
        return []

    merged = [cleaned[0]]
    for segment in cleaned[1:]:
        prev = merged[-1]
        if segment.speaker == prev.speaker and (segment.start - prev.end) <= gap_threshold:
            merged[-1] = DiarizationSegment(
                start=prev.start,
                end=segment.end,
                speaker=prev.speaker,
            )
        else:
            merged.append(segment)
    return [_segment_to_tuple(segment) for segment in merged if (segment.end - segment.start) >= min_duration]


def split_long_speaker_segments(
    segments: list[LegacySegment],
    max_duration: float = DIARIZATION_MAX_SEGMENT_SEC,
    min_duration: float = DIARIZATION_MIN_SEGMENT_SEC,
) -> list[LegacySegment]:
    """
    Split long same-speaker spans into shorter ASR-friendly pieces.

    Args:
        segments: Cleaned diarization spans.
        max_duration: Maximum preferred chunk duration in seconds.
        min_duration: Minimum chunk duration to keep.

    Returns:
        Speaker segments capped to the requested maximum duration.
    """
    result: list[DiarizationSegment] = []
    for segment in map(_segment_from_tuple, segments):
        duration = segment.end - segment.start
        if duration <= max_duration:
            result.append(segment)
            continue

        parts = max(2, int(duration // max_duration) + (1 if duration % max_duration else 0))
        chunk_duration = duration / parts
        cursor = segment.start
        while cursor < segment.end:
            chunk_end = min(segment.end, cursor + chunk_duration)
            if (chunk_end - cursor) >= min_duration:
                result.append(
                    DiarizationSegment(
                        start=cursor,
                        end=chunk_end,
                        speaker=segment.speaker,
                    )
                )
            cursor = chunk_end
    return [_segment_to_tuple(segment) for segment in result]


def prepare_diarized_turns(
    segments: list[LegacySegment],
    *,
    gap_threshold: float = DIARIZATION_GAP_THRESHOLD_SEC,
    min_duration: float = DIARIZATION_MIN_SEGMENT_SEC,
    max_duration: float = DIARIZATION_MAX_SEGMENT_SEC,
) -> list[LegacySegment]:
    """
    Convert raw diarization output into stable speaker turns for transcription.

    Args:
        segments: Raw diarization spans.
        gap_threshold: Max gap for same-speaker merge.
        min_duration: Minimum segment duration to keep.
        max_duration: Maximum duration of a returned segment.

    Returns:
        Speaker turns ready for chunk extraction and ASR.
    """
    normalized = normalize_diarization_segments(
        segments,
        gap_threshold=gap_threshold,
        min_duration=min_duration,
    )
    return split_long_speaker_segments(
        normalized,
        max_duration=max_duration,
        min_duration=min_duration,
    )
