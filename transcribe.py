"""
Standalone transcription script — runs NeMo Canary 1B v2 on a file and prints the transcript.

Usage:
    python transcribe.py <path/to/file> [--source-lang en] [--target-lang en]

The file is first converted to 16 kHz mono WAV via torchaudio, then transcribed
with nvidia/canary-1b-v2.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import torch
from dotenv import load_dotenv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe audio/video with NeMo Canary 1B v2.")
    parser.add_argument("audio", help="Path to audio/video file")
    parser.add_argument("--source-lang", default="en", help="Source language code (default: en)")
    parser.add_argument("--target-lang", default=None, help="Target language code (defaults to source)")
    parser.add_argument(
        "--model",
        default="nvidia/canary-1b-v2",
        help="HuggingFace model name or local .nemo path",
    )
    return parser.parse_args()


def choose_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_hf_token() -> str | None:
    load_dotenv()
    token = (
        os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )
    if token:
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", token)
    return token


def prepare_audio(audio_path: str, target_sr: int = 16000) -> str:
    """Convert any audio/video to 16 kHz mono WAV using ffmpeg."""
    import subprocess

    source = Path(audio_path).expanduser().resolve()
    tmpdir = Path(tempfile.mkdtemp(prefix="canary_"))
    out_path = tmpdir / f"{source.stem}_mono_{target_sr}.wav"

    cmd = [
        "ffmpeg", "-y", "-i", str(source),
        "-ar", str(target_sr), "-ac", "1", "-c:a", "pcm_s16le",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode(errors="replace")[:500])

    return str(out_path)


def load_model(model_name: str, device: str):
    import requests
    from nemo.collections.asr.models import ASRModel

    model_ref = Path(model_name).expanduser()
    try:
        if model_ref.exists():
            model = ASRModel.restore_from(restore_path=str(model_ref.resolve()))
        else:
            model = ASRModel.from_pretrained(model_name=model_name)
    except requests.exceptions.RequestException as exc:
        print(f"Failed to download model: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    return model.cuda() if device == "cuda" else model.cpu()


def main() -> int:
    args = parse_args()
    target_lang = args.target_lang or args.source_lang

    audio_path = str(Path(args.audio).expanduser().resolve())
    if not Path(audio_path).exists():
        print(f"File not found: {audio_path}", file=sys.stderr)
        return 1

    token = load_hf_token()
    if not token:
        print("Warning: no HF token found — set HF_TOKEN in .env if the model is gated.", file=sys.stderr)

    device = choose_device()
    print(f"Device : {device.upper()}", file=sys.stderr)
    print(f"Model  : {args.model}", file=sys.stderr)
    print(f"File   : {audio_path}", file=sys.stderr)
    print(f"Langs  : {args.source_lang} → {target_lang}", file=sys.stderr)

    print("\nLoading model...", file=sys.stderr)
    model = load_model(args.model, device)
    print("Model ready.", file=sys.stderr)

    print("Preparing audio...", file=sys.stderr)
    prepared = prepare_audio(audio_path)

    print("Transcribing...", file=sys.stderr)
    import shutil
    try:
        outputs = model.transcribe(
            audio=[prepared],
            source_lang=args.source_lang,
            target_lang=target_lang,
        )
    finally:
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
