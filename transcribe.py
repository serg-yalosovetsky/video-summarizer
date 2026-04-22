"""
Standalone transcription script — runs faster-whisper on a file and prints progress.

Usage:
    python transcribe.py <path/to/file>

The file is first converted to 16 kHz mono WAV via ffmpeg (same as the main app),
then transcribed with the same WhisperModel settings.
"""

import json
import os
import subprocess
import sys
import tempfile
import time

from faster_whisper import WhisperModel


def _tail_text(text: str, limit: int = 300) -> str:
    cleaned = (text or "").strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[-limit:]


def convert_to_wav(input_path: str, output_path: str) -> float:
    """Run ffprobe + ffmpeg. Returns duration in seconds."""
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-print_format", "json",
        "-show_streams", "-show_format",
        input_path,
    ]
    result = subprocess.run(
        probe_cmd,
        capture_output=True,
        check=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    probe_data = json.loads(result.stdout)
    fmt = probe_data.get("format", {})
    duration = float(fmt.get("duration", 0))

    has_audio_stream = any(
        stream.get("codec_type") == "audio"
        for stream in probe_data.get("streams", [])
    )
    if not has_audio_stream:
        raise ValueError("No audio stream found in the input file.")

    ffmpeg_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", input_path,
        "-vn",
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
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
    return duration


def transcribe(wav_path: str, model: WhisperModel, total_duration: float) -> str:
    segments, info = model.transcribe(wav_path, beam_size=5)
    total = total_duration or info.duration or 1.0
    texts = []
    last_pct = 0

    bar_width = 40

    for segment in segments:
        texts.append(segment.text.strip())
        pct = min(int(segment.end / total * 100), 99)

        if pct != last_pct:
            last_pct = pct
            filled = int(bar_width * pct / 100)
            bar = "█" * filled + "░" * (bar_width - filled)
            elapsed = segment.end
            remaining = (total - segment.end)
            print(f"\r  [{bar}] {pct:3d}%  {elapsed:.0f}s / {total:.0f}s  (~{remaining:.0f}s left)  ",
                  end="", flush=True)

    # Final 100%
    bar = "█" * bar_width
    print(f"\r  [{bar}] 100%  {total:.0f}s / {total:.0f}s                              ")
    return " ".join(texts)


def main():
    if len(sys.argv) < 2:
        print("Usage: python transcribe.py <path/to/file>")
        sys.exit(1)

    input_path = sys.argv[1]
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        sys.exit(1)

    print(f"\nFile : {input_path}")
    print(f"Size : {os.path.getsize(input_path) / 1024 / 1024:.1f} MB")

    # Load model — CUDA DLLs load lazily on first encode(), so we warm-up immediately
    print("\nLoading Whisper model...")
    t0 = time.monotonic()
    try:
        import numpy as np
        model = WhisperModel("base", device="cuda", compute_type="float16")
        mel = model.feature_extractor(np.zeros(16000, dtype=np.float32))
        model.encode(mel)  # forces cublas64_12.dll to load now
        print(f"  ready (CUDA)  ({time.monotonic() - t0:.2f}s)")
    except Exception as e:
        print(f"  CUDA unavailable ({e}), using CPU")
        t0 = time.monotonic()
        model = WhisperModel("base", device="cpu", compute_type="int8")
        print(f"  ready (CPU)  ({time.monotonic() - t0:.2f}s)")

    # Convert
    print("\n[ffmpeg] converting to 16kHz mono WAV...")
    t0 = time.monotonic()
    tmp_dir = tempfile.mkdtemp(dir=os.path.join(os.path.dirname(__file__), "tmp"))
    wav_path = os.path.join(tmp_dir, "audio.wav")
    try:
        duration = convert_to_wav(input_path, wav_path)
        print(f"  done  ({time.monotonic() - t0:.2f}s)  duration: {int(duration // 60)}:{int(duration % 60):02d}")
    except FileNotFoundError:
        print("  ERROR: ffprobe/ffmpeg not found — install FFmpeg and add to PATH")
        sys.exit(1)
    except ValueError as e:
        print(f"  ERROR: {e}")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr or ""
        stdout = e.stdout or ""
        print(f"  ERROR: {_tail_text(stderr or stdout or str(e))}")
        sys.exit(1)

    # Transcribe
    print("\n[whisper] transcribing...")
    t0 = time.monotonic()
    text = transcribe(wav_path, model, duration)
    elapsed = time.monotonic() - t0
    print(f"  done  ({elapsed:.2f}s)  {len(text)} chars")

    # Output
    print("\n" + "─" * 60)
    print(text)
    print("─" * 60)

    # Optionally save to file
    out_path = input_path + ".transcript.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\nSaved to: {out_path}")

    # Cleanup
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
