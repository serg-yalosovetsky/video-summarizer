import asyncio
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv()

# Pass HF token so NeMo/HuggingFace can download gated models without rate limits
_hf_token = (
    os.environ.get("HUGGING_FACE_HUB_TOKEN")
    or os.environ.get("HF_TOKEN")
    or os.environ.get("HUGGINGFACE_HUB_TOKEN")
)
if _hf_token:
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", _hf_token)
    os.environ.setdefault("HF_TOKEN", _hf_token)
    os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", _hf_token)

import httpx
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("summarizer")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _ollama_base_url() -> str:
    """Resolve Ollama host. Supports OLLAMA_BASE_URL or OLLAMA_HOST env override.
    Default: Windows host gateway so this works from WSL2."""
    host = os.environ.get("OLLAMA_BASE_URL") or os.environ.get("OLLAMA_HOST")
    if host:
        return host.rstrip("/")
    # In WSL2 the Windows host sits at the default route gateway
    try:
        import subprocess as _sp
        gw = _sp.check_output(
            ["ip", "route", "show", "default"], text=True
        ).split()[2]
        return f"http://{gw}:11434"
    except Exception:
        return "http://localhost:11434"


OLLAMA_URL = _ollama_base_url() + "/api/generate"
OLLAMA_MODEL = "gemma4:e4b"
MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500 MB

CANARY_MODEL = "nvidia/canary-1b-v2"

# ---------------------------------------------------------------------------
# NeMo Canary model — loaded once at startup
# ---------------------------------------------------------------------------

log.info("Loading NeMo Canary model...")


def _choose_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_canary():
    import requests as _requests
    from nemo.collections.asr.models import ASRModel

    device = _choose_device()
    log.info("  [canary] device: %s", device)

    if not _hf_token:
        log.warning(
            "No HuggingFace token found. If the model is gated set HF_TOKEN in .env."
        )

    model_ref = Path(CANARY_MODEL).expanduser()
    try:
        if model_ref.exists():
            model = ASRModel.restore_from(restore_path=str(model_ref.resolve()))
        else:
            model = ASRModel.from_pretrained(model_name=CANARY_MODEL)
    except _requests.exceptions.RequestException as exc:
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


canary_model = _load_canary()
_canary_device = _choose_device()

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

CLEAN_SYSTEM = (
    "You are a professional transcript editor. Your task is to clean and correct "
    "a raw audio transcription.\n"
    "Rules:\n"
    "- Fix obvious speech-to-text errors and wrong words\n"
    "- Add proper punctuation and capitalization\n"
    "- Break run-on sentences into readable paragraphs\n"
    "- Do NOT paraphrase, summarize, or change the meaning\n"
    "- Do NOT add any content that was not in the original\n"
    "- Preserve speaker intent and all factual content\n"
    "- Output ONLY the cleaned transcript, no commentary\n"
    "- Keep the same language as the input — do NOT translate"
)

CLEAN_PROMPT_TEMPLATE = "Clean and correct the following raw transcript:\n\n{transcript}"

SUMMARY_SYSTEM = (
    "You are an expert content summarizer. "
    "Create clear, structured summaries that capture the key information. "
    "IMPORTANT: Always respond in the same language as the input."
)

SUMMARY_PROMPT_TEMPLATE = (
    "Based on the following content, provide a comprehensive summary in the SAME LANGUAGE as the input. "
    "The summary must include:\n\n"
    "1. **Main Topic**: What this content is about in 1-2 sentences\n"
    "2. **Key Points**: The most important points covered (bullet list)\n"
    "3. **Details**: Relevant supporting information or context\n"
    "4. **Conclusion**: Main takeaway or outcome\n\n"
    "Content:\n{transcript}"
)

SHORT_SUMMARY_PROMPT_TEMPLATE = (
    "Write a structured short summary of the following content. "
    "The summary should be approximately 10% of the length of the full content.\n\n"
    "If the content is about a problem someone is trying to solve (e.g. a call, meeting, or discussion), structure the summary as:\n"
    "- **Problem**: what issue is being addressed\n"
    "- **Ways to solve**: approaches or actions taken/proposed\n"
    "- **Blockers**: obstacles preventing resolution\n"
    "- **Estimated resolution**: timeframe or next steps if mentioned\n\n"
    "If the content is not about solving a problem, write a plain structured summary covering the key points.\n\n"
    "Use the SAME LANGUAGE as the input. Output only the summary, no commentary.\n\n"
    "Content:\n{transcript}"
)

# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------


def convert_to_wav(input_path: str, output_path: str) -> dict:
    """Convert any audio/video to 16 kHz mono WAV. Returns metadata dict."""
    # Probe input file metadata
    probe_cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        input_path,
    ]
    probe_result = subprocess.run(probe_cmd, capture_output=True, check=True)
    probe_data = json.loads(probe_result.stdout)

    fmt = probe_data.get("format", {})
    format_name = fmt.get("format_long_name", fmt.get("format_name", "unknown"))
    duration_sec = float(fmt.get("duration", 0))
    duration_str = f"{int(duration_sec // 3600):02d}:{int((duration_sec % 3600) // 60):02d}:{int(duration_sec % 60):02d}"
    file_size_mb = int(fmt.get("size", 0)) / (1024 * 1024)

    # Find audio codec
    codec = "unknown"
    for stream in probe_data.get("streams", []):
        if stream.get("codec_type") == "audio":
            codec = stream.get("codec_long_name", stream.get("codec_name", "unknown"))
            break

    file_info = (
        f"File: {os.path.basename(input_path)}\n"
        f"Format: {format_name}\n"
        f"Duration: {duration_str}\n"
        f"Audio codec: {codec}\n"
        f"Size: {file_size_mb:.1f} MB\n"
        f"Output: 16 kHz mono WAV"
    )

    # Convert to WAV
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        output_path,
    ]
    subprocess.run(ffmpeg_cmd, capture_output=True, check=True)

    return {
        "file_info": file_info,
        "format": format_name,
        "duration": duration_str,
        "codec": codec,
    }


def transcribe_with_canary(
    wav_path: str,
    model,
    async_q: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    source_lang: str = "en",
    target_lang: str = "en",
) -> str:
    """Transcribe WAV using NeMo Canary (full file, single pass)."""
    log.info("  [canary] starting inference (this may take several minutes)...")
    loop.call_soon_threadsafe(async_q.put_nowait, 1)  # show bar immediately

    try:
        outputs = model.transcribe(
            audio=[wav_path],
            source_lang=source_lang,
            target_lang=target_lang,
        )
    except Exception:
        log.exception("  [canary] inference failed")
        raise

    loop.call_soon_threadsafe(async_q.put_nowait, None)  # sentinel

    if outputs:
        return getattr(outputs[0], "text", str(outputs[0]))
    return ""


def call_ollama(prompt: str, system: str = "") -> str:
    """Synchronous Ollama /api/generate call. Returns response text."""
    response = httpx.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "system": system,
            "stream": False,
        },
        timeout=900.0,
    )
    response.raise_for_status()
    return response.json()["response"]


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------


def sse(event: str, payload: dict) -> str:
    return f"data: {json.dumps({'event': event, 'payload': payload})}\n\n"


# ---------------------------------------------------------------------------
# SSE generator
# ---------------------------------------------------------------------------


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_TMP = os.path.join(BASE_DIR, "tmp")
os.makedirs(LOCAL_TMP, exist_ok=True)


def combine_sources(transcript: str, chat: str) -> str:
    """Merge video transcript and chat text into a single unified source."""
    parts = []
    if transcript.strip():
        parts.append(f"=== Video/Audio Transcript ===\n{transcript.strip()}")
    if chat.strip():
        parts.append(f"=== Chat Messages ===\n{chat.strip()}")
    return "\n\n".join(parts)


async def process_generator(file: UploadFile | None, chat_text: str, source_lang: str = "ru"):
    if not file and not chat_text.strip():
        yield sse("error", {"message": "Provide a video/audio file or chat text (or both).", "stage": "upload"})
        return

    tmp_dir = tempfile.mkdtemp(dir=LOCAL_TMP)
    total_start = time.monotonic()
    raw_transcript = ""

    try:
        loop = asyncio.get_event_loop()

        # Stage 1 + 2: video pipeline (skipped if no file uploaded)
        if file and file.filename:
            safe_filename = os.path.basename(file.filename)
            input_path = os.path.join(tmp_dir, safe_filename)
            wav_path = os.path.join(tmp_dir, "audio.wav")

            content = await file.read()
            size_mb = len(content) / (1024 * 1024)

            if len(content) > MAX_UPLOAD_BYTES:
                log.error("  [upload] ERROR: file exceeds 500 MB limit (%.1f MB)", size_mb)
                yield sse("error", {"message": "File exceeds 500 MB limit.", "stage": "upload"})
                return

            log.info("► [%s] Start — %.1f MB", safe_filename, size_mb)
            with open(input_path, "wb") as f:
                f.write(content)

            # Stage 1: FFmpeg conversion
            yield sse("status", {"message": "Converting media with FFmpeg..."})
            log.info("  [ffmpeg] converting...")
            t0 = time.monotonic()
            try:
                meta = await loop.run_in_executor(None, convert_to_wav, input_path, wav_path)
            except FileNotFoundError:
                log.error("  [ffmpeg] ERROR: ffprobe/ffmpeg not found in PATH")
                yield sse("error", {
                    "message": "ffprobe/ffmpeg not found. Install FFmpeg and add to PATH.",
                    "stage": "ffmpeg",
                })
                return
            except subprocess.CalledProcessError as exc:
                stderr = exc.stderr.decode(errors="replace") if exc.stderr else ""
                log.error("  [ffmpeg] ERROR: %s", stderr[:300])
                yield sse("error", {"message": f"FFmpeg error: {stderr[:500]}", "stage": "ffmpeg"})
                return
            log.info("  [ffmpeg] done — %s, %s  (%.2fs)", meta["format"], meta["duration"], time.monotonic() - t0)
            yield sse("ffmpeg_done", meta)

            # Stage 2: Canary transcription
            yield sse("status", {"message": "Transcribing audio with Canary (may take a few minutes)..."})
            log.info("  [canary] transcribing...")
            t0 = time.monotonic()
            try:
                async_q: asyncio.Queue = asyncio.Queue()
                future = loop.run_in_executor(
                    None, transcribe_with_canary, wav_path, canary_model, async_q, loop, source_lang, source_lang
                )
                while True:
                    pct = await async_q.get()
                    if pct is None:
                        break
                    yield sse("transcript_progress", {"pct": pct})
                raw_transcript = await future
            except Exception as exc:
                log.error("  [canary] ERROR: %s", exc)
                yield sse("error", {"message": f"Canary error: {exc}", "stage": "whisper"})
                return
            log.info("  [canary] done — %d chars  (%.2fs)", len(raw_transcript), time.monotonic() - t0)
            yield sse("transcript_progress", {"pct": 100})
            yield sse("transcript_done", {"text": raw_transcript})
        else:
            log.info("► No video file — chat-only mode")

        # Combine transcript + chat into one source for LLM stages
        combined = combine_sources(raw_transcript, chat_text)

        # Stage 3: Clean with Gemma
        yield sse("status", {"message": "Cleaning content with Gemma..."})
        log.info("  [gemma/clean] calling ollama...")
        t0 = time.monotonic()
        try:
            clean_prompt = CLEAN_PROMPT_TEMPLATE.format(transcript=combined)
            cleaned_text = await loop.run_in_executor(
                None, call_ollama, clean_prompt, CLEAN_SYSTEM
            )
        except Exception as exc:
            log.error("  [gemma/clean] ERROR: %s", exc)
            yield sse("error", {"message": f"Ollama (clean) error: {exc}", "stage": "clean"})
            return
        log.info("  [gemma/clean] done  (%.2fs)", time.monotonic() - t0)
        yield sse("cleaned_done", {"text": cleaned_text})

        # Stage 4: Summarize with Gemma
        yield sse("status", {"message": "Generating summary with Gemma..."})
        log.info("  [gemma/summary] calling ollama...")
        t0 = time.monotonic()
        try:
            summary_prompt = SUMMARY_PROMPT_TEMPLATE.format(transcript=cleaned_text)
            summary_text = await loop.run_in_executor(
                None, call_ollama, summary_prompt, SUMMARY_SYSTEM
            )
        except Exception as exc:
            log.error("  [gemma/summary] ERROR: %s", exc)
            yield sse("error", {"message": f"Ollama (summary) error: {exc}", "stage": "summary"})
            return
        log.info("  [gemma/summary] done  (%.2fs)", time.monotonic() - t0)
        yield sse("summary_done", {"text": summary_text})

        # Stage 5: Short TL;DR summary
        yield sse("status", {"message": "Generating short TL;DR with Gemma..."})
        log.info("  [gemma/tldr] calling ollama...")
        t0 = time.monotonic()
        try:
            tldr_prompt = SHORT_SUMMARY_PROMPT_TEMPLATE.format(transcript=cleaned_text)
            tldr_text = await loop.run_in_executor(
                None, call_ollama, tldr_prompt, SUMMARY_SYSTEM
            )
        except Exception as exc:
            log.error("  [gemma/tldr] ERROR: %s", exc)
            yield sse("error", {"message": f"Ollama (tldr) error: {exc}", "stage": "tldr"})
            return
        log.info("  [gemma/tldr] done  (%.2fs)", time.monotonic() - t0)
        yield sse("tldr_done", {"text": tldr_text})

        log.info("◄ Complete — total %.2fs", time.monotonic() - total_start)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Video Summarizer")

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.post("/process")
async def process_media(
    file: UploadFile | None = File(default=None),
    chat_text: str = Form(default=""),
    source_lang: str = Form(default="ru"),
):
    return StreamingResponse(
        process_generator(file, chat_text, source_lang),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
