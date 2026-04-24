"""
Transcription stage for the web app plus a standalone CLI entry point.
"""

from __future__ import annotations

import asyncio
import argparse
import shutil
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

from config import settings
from helpers import HF_TOKEN, log
from tracing import start_observation
from transcribe_diarization import (
    DIARIZATION_GAP_THRESHOLD_SEC,
    DIARIZATION_MAX_SEGMENT_SEC,
    DIARIZATION_MIN_SEGMENT_SEC,
    get_diarizer,
    normalize_diarization_segments,
    prepare_diarized_turns,
    run_diarization,
    split_long_speaker_segments,
)
from transcribe_ffmpeg import (
    convert_to_wav,
    extract_audio_chunk,
    format_speaker_timestamp,
    prepare_audio,
)
from transcribe_types import (
    AudioChunk,
    DiarizationSegment,
    LegacySegment,
    PreparedAudio,
    WavConversionResult,
)

CANARY_MODEL = settings.canary_model
CANARY_DEVICE = settings.canary_device
CANARY_SEGMENT_BATCH_SIZE = settings.canary_segment_batch_size


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


def release_canary_model() -> None:
    import gc
    import torch

    if get_canary_model.cache_info().currsize == 0:
        return
    try:
        model = get_canary_model()
        get_canary_model.cache_clear()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        log.info("Canary model released from CUDA.")
    except Exception as exc:
        log.warning("Failed to release Canary model: %s", exc)


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


def transcribe_with_canary(
    wav_path: str,
    async_q: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    source_lang: str = "ru",
    target_lang: str | None = None,
) -> str:
    """Run single-pass transcription for one WAV file and return plain text."""
    with start_observation(
        "canary.transcribe-single",
        input={
            "audio_file": Path(wav_path).name,
            "source_lang": source_lang,
            "target_lang": target_lang or source_lang,
        },
        metadata={"model": CANARY_MODEL},
    ) as span:
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
        result = getattr(outputs[0], "text", str(outputs[0])) if outputs else ""
        if span is not None:
            span.update(output={"text_length": len(result)})
        return result

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
    for i, (start, end, speaker) in enumerate(segments, 1):
        chunk_path = str(Path(tmp_dir) / f"chunk_{i:04d}.wav")
        if not extract_audio_chunk(wav_path, start, end, chunk_path):
            log.warning("  [canary] chunk %d: ffmpeg extraction failed", i)
            continue
        segment = DiarizationSegment(start=float(start), end=float(end), speaker=str(speaker))
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
