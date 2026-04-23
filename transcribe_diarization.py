from __future__ import annotations

import shutil
from functools import lru_cache

from config import settings
from helpers import HF_TOKEN, log
from tracing import start_observation
from transcribe_types import DiarizationSegment, LegacySegment

PYANNOTE_DEVICE = settings.pyannote_device
DIARIZATION_GAP_THRESHOLD_SEC = 1.0
DIARIZATION_MIN_SEGMENT_SEC = 0.5
DIARIZATION_MAX_SEGMENT_SEC = 20.0


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


def run_diarization(wav_path: str) -> list[LegacySegment]:
    """Run speaker diarization and return sorted legacy segments."""
    with start_observation(
        "pyannote.diarization",
        input={"audio_file": wav_path.rsplit("/", 1)[-1]},
        metadata={"model": settings.pyannote_model},
    ) as span:
        pipeline = get_diarizer()
        raw = pipeline(wav_path)
        annotation = getattr(raw, "speaker_diarization", None) or getattr(raw, "diarization", None) or raw
        segments = [
            (turn.start, turn.end, speaker)
            for turn, _, speaker in annotation.itertracks(yield_label=True)
        ]
        sorted_segments = sorted(segments, key=lambda x: x[0])
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
