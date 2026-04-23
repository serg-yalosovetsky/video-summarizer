from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

from helpers import tail_text
from tracing import start_observation
from transcribe_types import PreparedAudio, WavConversionResult


def _probe_media(input_path: str) -> dict:
    """Return raw ffprobe metadata for the input media file."""
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
    return json.loads(probe_result.stdout)


def _build_wav_conversion_result(
    *,
    input_path: str,
    output_path: str,
    probe_data: dict,
) -> WavConversionResult:
    """Build typed conversion metadata from ffprobe output."""
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

    return WavConversionResult(
        file_info=file_info,
        format_name=format_name,
        duration_display=duration_str,
        duration_sec=duration_sec,
        codec=codec,
        has_video=has_video_stream,
        output_path=output_path,
    )


def _run_ffmpeg_wav_conversion(input_path: str, output_path: str) -> None:
    """Convert input media into mono 16 kHz PCM WAV."""
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


def convert_to_wav(input_path: str, output_path: str) -> WavConversionResult:
    """Convert media into mono 16 kHz WAV and return conversion metadata."""
    with start_observation(
        "media.convert-to-wav",
        input={"input_file": Path(input_path).name},
        metadata={"output_file": Path(output_path).name},
    ) as span:
        probe_data = _probe_media(input_path)
        result = _build_wav_conversion_result(
            input_path=input_path,
            output_path=output_path,
            probe_data=probe_data,
        )
        _run_ffmpeg_wav_conversion(input_path, output_path)

        if span is not None:
            span.update(
                output={
                    "format_name": result.format_name,
                    "duration_sec": result.duration_sec,
                    "codec": result.codec,
                    "has_video": result.has_video,
                }
            )
        return result


def format_speaker_timestamp(start: float) -> str:
    """Render a segment start time in `[HH:MM:SS]` format."""
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
