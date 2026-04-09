import asyncio
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time

from dotenv import load_dotenv

load_dotenv()

# Pass HF token so faster-whisper can download gated models without rate limits
_hf_token = os.environ.get("HF_TOKEN")
if _hf_token:
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", _hf_token)

import httpx
from faster_whisper import WhisperModel
from fastapi import FastAPI, File, UploadFile
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

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma4:e4b"
MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500 MB

# Whisper model loaded once at startup (thread-safe for concurrent reads)
# CUDA DLLs are loaded lazily on first .transcribe() call, so we do a dummy
# encode to catch missing cublas/cudnn early and fall back to CPU if needed.
log.info("Loading Whisper model...")


CUDA_MODEL = "large-v3"
CPU_MODEL = "base"
HF_REPO_CUDA = f"Systran/faster-whisper-{CUDA_MODEL}"
HF_REPO_CPU = f"Systran/faster-whisper-{CPU_MODEL}"


def _download_model(repo_id: str) -> None:
    """Download all model files from HuggingFace with per-file progress logging."""
    from huggingface_hub import hf_hub_download, list_repo_files

    token = _hf_token or None
    log.info("  [hf] checking %s ...", repo_id)
    files = list(list_repo_files(repo_id, token=token))
    log.info("  [hf] %d file(s) to fetch:", len(files))
    for f in files:
        log.info("  [hf]   %s", f)

    for i, filename in enumerate(files, 1):
        log.info("  [hf] [%d/%d] %s ...", i, len(files), filename)
        path = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
        size_mb = os.path.getsize(path) / 1024 / 1024
        log.info("  [hf] [%d/%d] done — %.1f MB", i, len(files), size_mb)

    log.info("  [hf] all files ready.")


def _load_whisper() -> WhisperModel:
    import numpy as np

    try:
        _download_model(HF_REPO_CUDA)
        m = WhisperModel(CUDA_MODEL, device="cuda", compute_type="float16")
        # Warm-up: feature_extractor is CPU-only; encode() is what loads cublas
        mel = m.feature_extractor(np.zeros(16000, dtype=np.float32))
        m.encode(mel)  # forces cublas64_12.dll / cudnn to load right now
        log.info("Whisper model ready (CUDA).")
        return m
    except Exception as e:
        log.warning("CUDA unavailable (%s), falling back to CPU.", e)
        _download_model(HF_REPO_CPU)
        m = WhisperModel(CPU_MODEL, device="cpu", compute_type="int8")
        log.info("Whisper model ready (CPU).")
        return m


whisper_model = _load_whisper()

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
    "IMPORTANT: Always respond in the same language as the transcript."
)

SUMMARY_PROMPT_TEMPLATE = (
    "Based on the following transcript, provide a comprehensive summary in the SAME LANGUAGE as the transcript. "
    "The summary must include:\n\n"
    "1. **Main Topic**: What this content is about in 1-2 sentences\n"
    "2. **Key Points**: The most important points covered (bullet list)\n"
    "3. **Details**: Relevant supporting information or context\n"
    "4. **Conclusion**: Main takeaway or outcome\n\n"
    "Transcript:\n{transcript}"
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


def transcribe_with_progress(
    wav_path: str,
    model: WhisperModel,
    async_q: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
) -> str:
    """Transcribe WAV using faster-whisper, pushing progress % into an asyncio.Queue via call_soon_threadsafe."""
    segments, info = model.transcribe(wav_path, beam_size=5)
    total = info.duration or 1.0
    texts = []
    last_logged_pct = -1
    for segment in segments:
        texts.append(segment.text.strip())
        pct = min(int(segment.end / total * 100), 99)
        loop.call_soon_threadsafe(async_q.put_nowait, pct)
        # Log every 10%
        bucket = (pct // 10) * 10
        if bucket > last_logged_pct:
            last_logged_pct = bucket
            log.info("  [whisper] %d%%", bucket)
    loop.call_soon_threadsafe(async_q.put_nowait, None)  # sentinel
    return " ".join(texts)


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
        timeout=300.0,
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


async def process_generator(file: UploadFile):
    tmp_dir = tempfile.mkdtemp(dir=LOCAL_TMP)
    # Sanitize filename to avoid path traversal
    safe_filename = os.path.basename(file.filename or "upload")
    input_path = os.path.join(tmp_dir, safe_filename)
    wav_path = os.path.join(tmp_dir, "audio.wav")
    total_start = time.monotonic()

    try:
        # Save uploaded file
        content = await file.read()
        size_mb = len(content) / (1024 * 1024)

        if len(content) > MAX_UPLOAD_BYTES:
            log.error("  [upload] ERROR: file exceeds 500 MB limit (%.1f MB)", size_mb)
            yield sse("error", {"message": "File exceeds 500 MB limit.", "stage": "upload"})
            return

        log.info("► [%s] Start — %.1f MB", safe_filename, size_mb)

        with open(input_path, "wb") as f:
            f.write(content)

        loop = asyncio.get_event_loop()

        # Stage 1: FFmpeg conversion
        yield sse("status", {"message": "Converting media with FFmpeg..."})
        log.info("  [ffmpeg] converting...")
        t0 = time.monotonic()
        try:
            meta = await loop.run_in_executor(None, convert_to_wav, input_path, wav_path)
        except FileNotFoundError:
            log.error("  [ffmpeg] ERROR: ffprobe/ffmpeg not found in PATH")
            yield sse("error", {
                "message": "ffprobe/ffmpeg не найден. Установите FFmpeg и добавьте в системный PATH.",
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

        # Stage 2: Whisper transcription with progress
        yield sse("status", {"message": "Transcribing audio with Whisper..."})
        log.info("  [whisper] transcribing...")
        t0 = time.monotonic()
        try:
            async_q: asyncio.Queue = asyncio.Queue()
            future = loop.run_in_executor(
                None, transcribe_with_progress, wav_path, whisper_model, async_q, loop
            )
            # Drain progress events from the async queue until sentinel (None)
            while True:
                pct = await async_q.get()
                if pct is None:
                    break
                yield sse("transcript_progress", {"pct": pct})
            raw_text = await future
        except Exception as exc:
            log.error("  [whisper] ERROR: %s", exc)
            yield sse("error", {"message": f"Whisper error: {exc}", "stage": "whisper"})
            return
        log.info("  [whisper] done — %d chars  (%.2fs)", len(raw_text), time.monotonic() - t0)
        yield sse("transcript_progress", {"pct": 100})
        yield sse("transcript_done", {"text": raw_text})

        # Stage 3: Clean with Gemma
        yield sse("status", {"message": "Cleaning transcript with Gemma..."})
        log.info("  [gemma/clean] calling ollama...")
        t0 = time.monotonic()
        try:
            clean_prompt = CLEAN_PROMPT_TEMPLATE.format(transcript=raw_text)
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

        log.info("◄ [%s] Complete — total %.2fs", safe_filename, time.monotonic() - total_start)

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
async def process_media(file: UploadFile = File(...)):
    return StreamingResponse(
        process_generator(file),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
