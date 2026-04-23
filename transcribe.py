"""
Transcription stage for the web app plus a standalone CLI entry point.
"""

from __future__ import annotations

import asyncio
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from config import settings
from helpers import HF_TOKEN, log, tail_text

CANARY_MODEL = settings.canary_model
CANARY_DEVICE = settings.canary_device
PYANNOTE_DEVICE = settings.pyannote_device
CANARY_SEGMENT_BATCH_SIZE = settings.canary_segment_batch_size
DIARIZATION_GAP_THRESHOLD_SEC = 1.0
DIARIZATION_MIN_SEGMENT_SEC = 0.5
DIARIZATION_MAX_SEGMENT_SEC = 20.0
LegacySegment = tuple[float, float, str]


@dataclass(frozen=True)
class DiarizationSegment:
    """Speaker-attributed time span in seconds."""

    start: float
    end: float
    speaker: str


@dataclass(frozen=True)
class AudioChunk:
    """Extracted WAV chunk for a diarized speaker segment."""

    segment: DiarizationSegment
    path: str


@dataclass(frozen=True)
class WavConversionResult:
    """Metadata returned after converting media into a mono 16 kHz WAV file."""

    file_info: str
    format_name: str
    duration_display: str
    duration_sec: float
    codec: str
    has_video: bool
    output_path: str


@dataclass(frozen=True)
class PreparedAudio:
    """Temporary WAV file prepared for transcription."""

    path: str
    temp_dir: str

    def cleanup(self) -> None:
        """Remove the temporary directory that owns this prepared file."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe audio/video with NeMo Canary 1B v2.")
    parser.add_argument("audio", help="Path to audio/video file")
    parser.add_argument("--source-lang", default="ru", help="Source language code (default: ru)")
    parser.add_argument("--target-lang", default=None, help="Target language code (defaults to source)")
    return parser.parse_args()


def cuda_diagnostics(torch_module) -> str:
    return (
        f"torch={torch_module.__version__}, "
        f"torch.version.cuda={torch_module.version.cuda}, "
        f"torch.cuda.is_available()={torch_module.cuda.is_available()}, "
        f"nvidia-smi={'found' if shutil.which('nvidia-smi') else 'missing'}"
    )


def choose_device() -> str:
    import torch

    if CANARY_DEVICE not in {"cuda", "cpu", "auto"}:
        raise RuntimeError(
            f"Unsupported CANARY_DEVICE={CANARY_DEVICE!r}. Use one of: cuda, cpu, auto."
        )

    if CANARY_DEVICE == "cpu":
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"

    if CANARY_DEVICE == "auto":
        log.warning("CUDA is unavailable, falling back to CPU because CANARY_DEVICE=auto.")
        return "cpu"

    raise RuntimeError(
        "CUDA is required for Canary but is unavailable. "
        f"{cuda_diagnostics(torch)}. "
        "Run install.bat/install.sh to install a CUDA-enabled PyTorch build and verify NVIDIA drivers, "
        "or set CANARY_DEVICE=auto/cpu if you intentionally want CPU mode."
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
def get_canary_model():
    from nemo.collections.asr.models import ASRModel

    log.info("Loading NeMo Canary model...")
    device = choose_device()
    log.info("  [canary] device: %s", device)
    if not HF_TOKEN:
        log.warning(
            "No HuggingFace token found. If the model is gated set HF_TOKEN in .env."
        )

    model_ref = Path(CANARY_MODEL).expanduser()
    try:
        if model_ref.exists():
            model = ASRModel.restore_from(restore_path=str(model_ref.resolve()))
        else:
            model = ASRModel.from_pretrained(model_name=CANARY_MODEL)
    except OSError as exc:
        log.error(
            "Failed to download Canary model. Pass a local .nemo file via CANARY_MODEL "
            "env var or ensure network access. Error: %s",
            exc,
        )
        raise SystemExit(2) from exc

    if device == "cuda":
        model = model.cuda()
    else:
        model = model.cpu()

    log.info("Canary model ready (%s).", device.upper())
    return model


def convert_to_wav(input_path: str, output_path: str) -> WavConversionResult:
    """Convert media into mono 16 kHz WAV and return conversion metadata."""
    probe_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        input_path,
    ]
    probe_result = subprocess.run(
        probe_cmd,
        capture_output=True,
        check=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    probe_data = json.loads(probe_result.stdout)

    fmt = probe_data.get("format", {})
    format_name = fmt.get("format_long_name", fmt.get("format_name", "unknown"))
    duration_sec = float(fmt.get("duration", 0))
    duration_str = (
        f"{int(duration_sec // 3600):02d}:{int((duration_sec % 3600) // 60):02d}:"
        f"{int(duration_sec % 60):02d}"
    )
    file_size_mb = int(fmt.get("size", 0)) / (1024 * 1024)

    codec = "unknown"
    has_audio_stream = False
    has_video_stream = False
    for stream in probe_data.get("streams", []):
        if stream.get("codec_type") == "audio" and not has_audio_stream:
            has_audio_stream = True
            codec = stream.get("codec_long_name", stream.get("codec_name", "unknown"))
        elif stream.get("codec_type") == "video":
            has_video_stream = True

    if not has_audio_stream:
        raise ValueError(
            "В файле не найдена аудиодорожка. Загрузите видео или аудио с доступным звуком."
        )

    file_info = (
        f"File: {Path(input_path).name}\n"
        f"Format: {format_name}\n"
        f"Duration: {duration_str}\n"
        f"Audio codec: {codec}\n"
        f"Size: {file_size_mb:.1f} MB\n"
        f"Output: 16 kHz mono WAV"
    )

    ffmpeg_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        input_path,
        "-vn",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        output_path,
    ]
    subprocess.run(
        ffmpeg_cmd,
        capture_output=True,
        check=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    return WavConversionResult(
        file_info=file_info,
        format_name=format_name,
        duration_display=duration_str,
        duration_sec=duration_sec,
        codec=codec,
        has_video=has_video_stream,
        output_path=output_path,
    )


def transcribe_with_canary(
    wav_path: str,
    async_q: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    source_lang: str = "ru",
    target_lang: str | None = None,
) -> str:
    """Run single-pass transcription for one WAV file and return plain text."""
    model = get_canary_model()
    target_lang = target_lang or source_lang
    log.info("  [canary] starting inference (this may take several minutes)...")
    loop.call_soon_threadsafe(async_q.put_nowait, 1)
    try:
        outputs = model.transcribe(  # type: ignore[operator]
            audio=[wav_path],
            source_lang=source_lang,
            target_lang=target_lang,
        )
    except Exception:
        log.exception("  [canary] inference failed")
        raise
    finally:
        loop.call_soon_threadsafe(async_q.put_nowait, None)
    if outputs:
        return getattr(outputs[0], "text", str(outputs[0]))
    return ""


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
    pipeline = get_diarizer()
    raw = pipeline(wav_path)
    # pyannote >= 3.2 returns DiarizeOutput; older/legacy returns Annotation directly
    annotation = getattr(raw, 'speaker_diarization', None) or getattr(raw, 'diarization', None) or raw
    segments = [
        (turn.start, turn.end, speaker)
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]
    return sorted(segments, key=lambda x: x[0])


def _segment_from_tuple(segment: LegacySegment) -> DiarizationSegment:
    """Convert a legacy diarization tuple into a named segment."""
    start, end, speaker = segment
    return DiarizationSegment(start=float(start), end=float(end), speaker=str(speaker))


def _segment_to_tuple(segment: DiarizationSegment) -> LegacySegment:
    """Convert a named segment back to the legacy tuple shape used by callers."""
    return (segment.start, segment.end, segment.speaker)


def _normalize_output_texts(outputs, expected_count: int) -> list[str]:
    """Return exactly `expected_count` stripped transcription texts."""
    texts = [getattr(output, "text", str(output)).strip() for output in (outputs or [])]
    if len(texts) < expected_count:
        texts.extend([""] * (expected_count - len(texts)))
    return texts[:expected_count]


def _delete_chunk_files(chunks: list[AudioChunk]) -> None:
    """Best-effort removal of temporary chunk files."""
    for chunk in chunks:
        Path(chunk.path).unlink(missing_ok=True)


def _transcribe_chunk_batch(
    model,
    chunks: list[AudioChunk],
    *,
    source_lang: str,
    processed_before_batch: int,
) -> list[str]:
    """Transcribe one chunk batch and return one text per chunk in input order."""
    audio_paths = [chunk.path for chunk in chunks]
    try:
        outputs = model.transcribe(
            audio=audio_paths,
            source_lang=source_lang,
            target_lang=source_lang,
        )
    except Exception as exc:
        failed_range = f"{processed_before_batch + 1}-{processed_before_batch + len(chunks)}"
        log.warning("  [canary] chunk batch %s failed: %s", failed_range, exc)
        return [""] * len(chunks)
    return _normalize_output_texts(outputs, expected_count=len(chunks))


def _build_transcript_line(chunk: AudioChunk, text: str) -> str:
    """Format a non-empty chunk transcription for the final transcript."""
    return (
        f"{format_speaker_timestamp(chunk.segment.start)} "
        f"[{chunk.segment.speaker}]: {text}"
    )


def _report_chunk_progress(
    chunk: AudioChunk,
    *,
    processed: int,
    total: int,
    async_q: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Publish progress for one completed chunk and log its status."""
    progress = int(processed / total * 100) if total else 100
    loop.call_soon_threadsafe(async_q.put_nowait, progress)
    log.info(
        "  [canary] segment %d/%d (%s @ %.1fs) done",
        processed,
        total,
        chunk.segment.speaker,
        chunk.segment.start,
    )


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
            # When pyannote returns slight overlaps, prefer keeping chronology stable.
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


def format_speaker_timestamp(start: float) -> str:
    whole = int(start)
    h = whole // 3600
    m = (whole % 3600) // 60
    s = whole % 60
    return f"[{h:02d}:{m:02d}:{s:02d}]"


def extract_audio_chunk(wav_path: str, start: float, end: float, out_path: str) -> bool:
    """
    Extract one mono 16 kHz WAV chunk for a time range.

    Returns:
        `True` when the chunk file was created successfully, otherwise `False`.
    """
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", str(start), "-to", str(end),
        "-i", wav_path,
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0 and os.path.exists(out_path)


def prepare_audio_chunks(
    wav_path: str,
    segments: list[LegacySegment],
    tmp_dir: str,
) -> list[AudioChunk]:
    """
    Materialize WAV chunks for diarized speaker segments.

    Args:
        wav_path: Source WAV file.
        segments: Speaker segments to extract.
        tmp_dir: Directory where temporary chunk files will be created.

    Returns:
        Successfully extracted chunks in input order.
    """
    prepared_chunks: list[AudioChunk] = []
    for i, segment in enumerate(map(_segment_from_tuple, segments), 1):
        chunk_path = str(Path(tmp_dir) / f"chunk_{i:04d}.wav")
        if not extract_audio_chunk(wav_path, segment.start, segment.end, chunk_path):
            log.warning("  [canary] chunk %d: ffmpeg extraction failed", i)
            continue
        prepared_chunks.append(AudioChunk(segment=segment, path=chunk_path))
    return prepared_chunks


def batched(items: list[AudioChunk], batch_size: int) -> list[list[AudioChunk]]:
    """Split items into stable in-memory batches."""
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def transcribe_by_segments(
    wav_path: str,
    segments: list[LegacySegment],
    async_q: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    source_lang: str = "ru",
    tmp_dir: str | None = None,
) -> str:
    """
    Transcribe audio by diarized speaker turns.

    Args:
        wav_path: Source mono WAV file.
        segments: Raw or preprocessed diarization spans.
        async_q: Progress queue that receives integer percentages and a final `None`.
        loop: Event loop used to publish progress safely from this worker.
        source_lang: Source language passed to Canary.
        tmp_dir: Optional directory for chunk files. If omitted, a temp dir is created and removed.

    Returns:
        Timestamped and speaker-labeled transcript text.
    """
    model = get_canary_model()
    prepared_segments = prepare_diarized_turns(segments)
    transcript_lines: list[str] = []
    _tmp = tmp_dir or tempfile.mkdtemp(prefix="canary_chunks_")
    cleanup = tmp_dir is None
    try:
        prepared_chunks = prepare_audio_chunks(wav_path, prepared_segments, _tmp)
        total = len(prepared_chunks)

        processed = 0
        for batch in batched(prepared_chunks, CANARY_SEGMENT_BATCH_SIZE):
            try:
                texts = _transcribe_chunk_batch(
                    model,
                    batch,
                    source_lang=source_lang,
                    processed_before_batch=processed,
                )
            finally:
                _delete_chunk_files(batch)

            for chunk, text in zip(batch, texts):
                processed += 1
                if text:
                    transcript_lines.append(_build_transcript_line(chunk, text))
                _report_chunk_progress(
                    chunk,
                    processed=processed,
                    total=total,
                    async_q=async_q,
                    loop=loop,
                )
    finally:
        if cleanup:
            shutil.rmtree(_tmp, ignore_errors=True)
    loop.call_soon_threadsafe(async_q.put_nowait, None)
    return "\n".join(transcript_lines)


def prepare_audio(audio_path: str, target_sr: int = 16000) -> PreparedAudio:
    """
    Convert an input media file into a temporary mono WAV for transcription.

    Args:
        audio_path: Path to source media.
        target_sr: Target sample rate in Hz.

    Returns:
        Prepared audio metadata with the temp directory ownership.

    Raises:
        RuntimeError: If ffmpeg fails to create the WAV file.
    """
    source = Path(audio_path).expanduser().resolve()
    tmpdir = Path(tempfile.mkdtemp(prefix="canary_"))
    out_path = tmpdir / f"{source.stem}_mono_{target_sr}.wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-ar",
        str(target_sr),
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(tail_text(result.stderr.decode(errors="replace")))
    return PreparedAudio(path=str(out_path), temp_dir=str(tmpdir))


def main() -> int:
    args = parse_args()
    target_lang = args.target_lang or args.source_lang
    audio_path = str(Path(args.audio).expanduser().resolve())
    if not Path(audio_path).exists():
        print(f"File not found: {audio_path}", file=sys.stderr)
        return 1

    print(f"Device : {choose_device().upper()}", file=sys.stderr)
    print(f"Model  : {CANARY_MODEL}", file=sys.stderr)
    print(f"File   : {audio_path}", file=sys.stderr)
    print(f"Langs  : {args.source_lang} -> {target_lang}", file=sys.stderr)

    print("\nLoading model...", file=sys.stderr)
    get_canary_model()
    print("Model ready.", file=sys.stderr)

    print("Preparing audio...", file=sys.stderr)
    prepared = prepare_audio(audio_path)
    try:
        outputs = get_canary_model().transcribe(
            audio=[prepared.path],
            source_lang=args.source_lang,
            target_lang=target_lang,
        )
    except Exception as exc:
        print(f"Transcription failed: {exc}", file=sys.stderr)
        prepared.cleanup()
        return 1

    prepared.cleanup()
    if not outputs:
        print("No output from model.", file=sys.stderr)
        return 1

    result = outputs[0]
    text = getattr(result, "text", str(result))
    print("\n" + "─" * 60)
    print(text)
    print("─" * 60)

    out_path = audio_path + ".transcript.txt"
    Path(out_path).write_text(text, encoding="utf-8")
    print(f"\nSaved to: {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
