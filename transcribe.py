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
from functools import lru_cache
from pathlib import Path

from helpers import HF_TOKEN, log, tail_text


CANARY_MODEL = os.environ.get("CANARY_MODEL", "nvidia/canary-1b-v2")
CANARY_DEVICE = os.environ.get("CANARY_DEVICE", "cuda").strip().lower()


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


def convert_to_wav(input_path: str, output_path: str) -> dict:
    """Convert any audio/video to 16 kHz mono WAV. Returns metadata dict."""
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
        f"File: {os.path.basename(input_path)}\n"
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

    return {
        "file_info": file_info,
        "format": format_name,
        "duration": duration_str,
        "duration_sec": duration_sec,
        "codec": codec,
        "has_video": has_video_stream,
    }


def transcribe_with_canary(
    wav_path: str,
    async_q: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    source_lang: str = "ru",
    target_lang: str | None = None,
) -> str:
    """Transcribe WAV using NeMo Canary (single-pass inference)."""
    model = get_canary_model()
    target_lang = target_lang or source_lang
    log.info("  [canary] starting inference (this may take several minutes)...")
    loop.call_soon_threadsafe(async_q.put_nowait, 1)
    try:
        outputs = model.transcribe(
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
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        use_auth_token=HF_TOKEN or True,
    )
    log.info("Diarizer ready.")
    return pipeline


def run_diarization(wav_path: str) -> list[tuple[float, float, str]]:
    """Return sorted list of (start_sec, end_sec, speaker_id) tuples."""
    pipeline = get_diarizer()
    diarization = pipeline(wav_path)
    segments = [
        (turn.start, turn.end, speaker)
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]
    return sorted(segments, key=lambda x: x[0])


def merge_speaker_segments(
    segments: list[tuple[float, float, str]],
    gap_threshold: float = 1.0,
    min_duration: float = 0.5,
) -> list[tuple[float, float, str]]:
    """Merge adjacent same-speaker segments with small gaps."""
    if not segments:
        return []
    merged = [list(segments[0])]
    for start, end, speaker in segments[1:]:
        prev = merged[-1]
        if speaker == prev[2] and (start - prev[1]) < gap_threshold:
            prev[1] = end
        else:
            merged.append([start, end, speaker])
    return [(s, e, sp) for s, e, sp in merged if (e - s) >= min_duration]


def extract_audio_chunk(wav_path: str, start: float, end: float, out_path: str) -> bool:
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", str(start), "-to", str(end),
        "-i", wav_path,
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0 and os.path.exists(out_path)


def transcribe_by_segments(
    wav_path: str,
    segments: list[tuple[float, float, str]],
    async_q: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    source_lang: str = "ru",
    tmp_dir: str | None = None,
) -> str:
    """Transcribe audio by speaker segments; return timestamped speaker-labeled transcript."""
    model = get_canary_model()
    merged = merge_speaker_segments(segments)
    results = []
    _tmp = tmp_dir or tempfile.mkdtemp(prefix="canary_chunks_")
    cleanup = tmp_dir is None
    try:
        total = len(merged)
        for i, (start, end, speaker) in enumerate(merged, 1):
            loop.call_soon_threadsafe(async_q.put_nowait, int(i / total * 100))
            chunk_path = os.path.join(_tmp, f"chunk_{i:04d}.wav")
            if not extract_audio_chunk(wav_path, start, end, chunk_path):
                log.warning("  [canary] chunk %d: ffmpeg extraction failed", i)
                continue
            try:
                outputs = model.transcribe(
                    audio=[chunk_path],
                    source_lang=source_lang,
                    target_lang=source_lang,
                )
                text = getattr(outputs[0], "text", str(outputs[0])) if outputs else ""
            except Exception as exc:
                log.warning("  [canary] chunk %d failed: %s", i, exc)
                text = ""
            finally:
                Path(chunk_path).unlink(missing_ok=True)
            if text.strip():
                h = int(start // 3600)
                m = int((start % 3600) // 60)
                s = int(start % 60)
                results.append(f"[{h:02d}:{m:02d}:{s:02d}] [{speaker}]: {text.strip()}")
            log.info("  [canary] segment %d/%d (%s @ %.1fs) done", i, total, speaker, start)
    finally:
        if cleanup:
            shutil.rmtree(_tmp, ignore_errors=True)
    loop.call_soon_threadsafe(async_q.put_nowait, None)
    return "\n".join(results)


def prepare_audio(audio_path: str, target_sr: int = 16000) -> str:
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
    return str(out_path)


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
            audio=[prepared],
            source_lang=args.source_lang,
            target_lang=target_lang,
        )
    except Exception as exc:
        print(f"Transcription failed: {exc}", file=sys.stderr)
        shutil.rmtree(str(Path(prepared).parent), ignore_errors=True)
        return 1

    shutil.rmtree(str(Path(prepared).parent), ignore_errors=True)
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
