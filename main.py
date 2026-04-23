from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from config import settings
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import sentry_sdk

from helpers import is_benign_nemo_transformer_log


def _message_from_sentry_payload(payload: dict | None) -> str:
    if not payload:
        return ""
    logentry = payload.get("logentry") or {}
    if isinstance(logentry, dict):
        formatted = logentry.get("formatted")
        message = logentry.get("message")
        if isinstance(formatted, str) and formatted:
            return formatted
        if isinstance(message, str) and message:
            return message
    message = payload.get("message")
    return message if isinstance(message, str) else ""


def _before_send(event, hint):
    if is_benign_nemo_transformer_log(_message_from_sentry_payload(event)):
        return None
    return event


def _before_breadcrumb(crumb, hint):
    if is_benign_nemo_transformer_log(_message_from_sentry_payload(crumb)):
        return None
    return crumb


sentry_sdk.init(
    dsn="https://cd075bbd8f1c11d4083f523e8eba375e@o4504272346480640.ingest.us.sentry.io/4511266802106368",
    # Add data like request headers and IP for users,
    # see https://docs.sentry.io/platforms/python/data-management/data-collected/ for more info
    send_default_pii=True,
    sample_rate=1.0,  # Adjust this value in production to control the volume of events sent to Sentry
    enable_tracing=True,  # Enable performance monitoring
    traces_sample_rate=1,  # Adjust this value in production to control the
    before_send=_before_send,
    before_breadcrumb=_before_breadcrumb,
)

from frames_analyze import (
    FRAME_MODEL,
    FRAME_TIMESTAMPS,
    MAX_FRAMES,
    analyze_frames_with_progress,
    analyze_speaker_frames,
    extract_frames,
    extract_frames_at,
    generate_frame_timestamps,
    is_context_sufficient,
)
from helpers import (
    combine_sources,
    ensure_ollama_ready,
    log,
    notify_done,
    remove_repetitions,
    sse,
    tail_text,
)
from summary import (
    classify_is_meeting,
    classify_text_language,
    clean_content,
    generate_personal_todo,
    generate_short_summary,
    generate_summary,
    local_preclean_content,
    looks_like_missing_content_response,
    prefer_meaningful_content,
    translate_summary_to_russian,
)
from contextlib import asynccontextmanager

OLLAMA_MODEL = settings.ollama_model
OLLAMA_CLEAN_MODEL = settings.ollama_clean_model
ARTIFACTS_DIR = settings.artifacts_dir
ARTIFACTS_DIR.mkdir(exist_ok=True)

from transcribe import (
    convert_to_wav,
    get_canary_model,
    run_diarization,
    transcribe_by_segments,
    transcribe_with_canary,
)


def _build_artifact_stem(label: str) -> str:
    stem = Path(label or "result").stem
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-._") or "result"
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{safe_stem}"


def _write_artifact(filename: str, content: str) -> Path:
    path = ARTIFACTS_DIR / filename
    path.write_text(content, encoding="utf-8")
    return path


async def process_generator(
    file: UploadFile | None,
    chat_text: str,
    source_lang: str = "ru",
):
    if not file and not chat_text.strip():
        yield sse(
            "error",
            {
                "message": "Provide a video/audio file or chat text (or both).",
                "stage": "upload",
            },
        )
        return

    tmp_dir = tempfile.mkdtemp(dir=str(settings.local_tmp))
    total_start = time.monotonic()
    raw_transcript = ""
    visual_context = ""
    request_label = "chat-only"
    artifact_stem = _build_artifact_stem(file.filename if file and file.filename else request_label)

    try:
        loop = asyncio.get_event_loop()

        if file and file.filename:
            safe_filename = os.path.basename(file.filename)
            request_label = safe_filename
            input_path = os.path.join(tmp_dir, safe_filename)
            wav_path = os.path.join(tmp_dir, "audio.wav")

            content = await file.read()
            size_mb = len(content) / (1024 * 1024)

            if len(content) > settings.max_upload_bytes:
                log.error("  [upload] ERROR: file exceeds 500 MB limit (%.1f MB)", size_mb)
                yield sse(
                    "error",
                    {"message": "File exceeds 500 MB limit.", "stage": "upload"},
                )
                return

            log.info("► [%s] Start - %.1f MB", safe_filename, size_mb)
            with open(input_path, "wb") as file_handle:
                file_handle.write(content)

            yield sse("status", {"message": "Converting media with FFmpeg..."})
            log.info("  [ffmpeg] converting...")
            t0 = time.monotonic()
            try:
                meta = await loop.run_in_executor(None, convert_to_wav, input_path, wav_path)
            except FileNotFoundError:
                log.error("  [ffmpeg] ERROR: ffprobe/ffmpeg not found in PATH")
                yield sse(
                    "error",
                    {
                        "message": "ffprobe/ffmpeg not found. Install FFmpeg and add to PATH.",
                        "stage": "ffmpeg",
                    },
                )
                return
            except ValueError as exc:
                log.error("  [ffmpeg] ERROR: %s", exc)
                yield sse("error", {"message": str(exc), "stage": "ffmpeg"})
                return
            except subprocess.CalledProcessError as exc:
                stderr = exc.stderr or ""
                stdout = exc.stdout or ""
                error_text = tail_text(stderr or stdout or str(exc))
                log.error("  [ffmpeg] ERROR: %s", error_text)
                yield sse(
                    "error",
                    {"message": f"FFmpeg error: {error_text}", "stage": "ffmpeg"},
                )
                return
            log.info(
                "  [ffmpeg] done - %s, %s  (%.2fs)",
                meta["format"],
                meta["duration"],
                time.monotonic() - t0,
            )
            yield sse("ffmpeg_done", meta)

            diarization_segments: list = []
            yield sse("status", {"message": "Running speaker diarization..."})
            log.info("  [pyannote] starting diarization...")
            t0 = time.monotonic()
            try:
                diarization_segments = await loop.run_in_executor(
                    None, run_diarization, wav_path
                )
                log.info(
                    "  [pyannote] done — %d segments (%.2fs)",
                    len(diarization_segments),
                    time.monotonic() - t0,
                )
                yield sse("diarization_done", {"segments_count": len(diarization_segments)})
            except Exception as exc:
                log.warning(
                    "  [pyannote] ERROR (non-fatal, falling back to single-pass): %s", exc
                )

            if meta.get("has_video"):
                yield sse("status", {"message": "Analyzing video frames for speaker context..."})
                log.info("  [frames] extracting frames...")
                t0 = time.monotonic()
                try:
                    total_analyzed = 0
                    if diarization_segments:
                        # Diarization tells us exactly when each speaker talks — only analyze those frames
                        unique_speakers = len({spk for _, _, spk in diarization_segments})
                        yield sse("status", {"message": f"Analyzing speaker frames ({unique_speakers} speakers)..."})
                        spk_q: asyncio.Queue = asyncio.Queue()
                        spk_t = loop.run_in_executor(
                            None,
                            analyze_speaker_frames,
                            input_path,
                            tmp_dir,
                            diarization_segments,
                            spk_q,
                            loop,
                            0,
                            unique_speakers,
                        )
                        while True:
                            progress = await spk_q.get()
                            if progress is None:
                                break
                            yield sse("frames_progress", progress)
                        visual_context = await spk_t
                        total_analyzed = unique_speakers
                        log.info(
                            "  [frames] speaker frames done (%d speakers, %.2fs)",
                            unique_speakers,
                            time.monotonic() - t0,
                        )
                    else:
                        # No diarization — fall back to generic adaptive sampling
                        frame_paths = await loop.run_in_executor(
                            None,
                            extract_frames,
                            input_path,
                            tmp_dir,
                            meta["duration_sec"],
                        )
                        if frame_paths:
                            done_ts = set(FRAME_TIMESTAMPS)
                            fq: asyncio.Queue = asyncio.Queue()
                            ft = loop.run_in_executor(
                                None,
                                analyze_frames_with_progress,
                                frame_paths,
                                fq,
                                loop,
                                0,
                                len(frame_paths),
                            )
                            while True:
                                progress = await fq.get()
                                if progress is None:
                                    break
                                yield sse("frames_progress", progress)
                            visual_context = await ft
                            total_analyzed = len(frame_paths)

                            while (
                                not is_context_sufficient(visual_context)
                                and total_analyzed < MAX_FRAMES
                            ):
                                remaining = MAX_FRAMES - total_analyzed
                                batch_count = min(4, remaining)
                                new_ts = generate_frame_timestamps(
                                    done_ts, meta["duration_sec"], batch_count
                                )
                                if not new_ts:
                                    break
                                done_ts.update(new_ts)
                                log.info(
                                    "  [frames] context insufficient, scanning %d more (total %d/%d)...",
                                    len(new_ts),
                                    total_analyzed + len(new_ts),
                                    MAX_FRAMES,
                                )
                                yield sse(
                                    "status",
                                    {
                                        "message": (
                                            f"Context insufficient - scanning more frames "
                                            f"({total_analyzed}/{MAX_FRAMES})..."
                                        )
                                    },
                                )
                                new_paths = await loop.run_in_executor(
                                    None,
                                    extract_frames_at,
                                    input_path,
                                    tmp_dir,
                                    new_ts,
                                )
                                if not new_paths:
                                    break
                                efq: asyncio.Queue = asyncio.Queue()
                                eft = loop.run_in_executor(
                                    None,
                                    analyze_frames_with_progress,
                                    new_paths,
                                    efq,
                                    loop,
                                    total_analyzed,
                                    total_analyzed + len(new_paths),
                                )
                                while True:
                                    progress = await efq.get()
                                    if progress is None:
                                        break
                                    yield sse("frames_progress", progress)
                                extra_ctx = await eft
                                visual_context = f"{visual_context}\n{extra_ctx}".strip()
                                total_analyzed += len(new_paths)

                            log.info(
                                "  [frames] done - %d frames  (%.2fs)",
                                total_analyzed,
                                time.monotonic() - t0,
                            )

                    yield sse(
                        "frames_done",
                        {
                            "context": visual_context,
                            "frames_count": total_analyzed,
                        },
                    )
                except Exception as exc:
                    log.warning("  [frames] ERROR (non-fatal): %s", exc)

            if diarization_segments:
                yield sse("status", {"message": "Transcribing audio by speaker segments with Canary..."})
            else:
                yield sse("status", {"message": "Transcribing audio with Canary (may take a few minutes)..."})
            log.info("  [canary] transcribing (diarized=%s)...", bool(diarization_segments))
            t0 = time.monotonic()
            try:
                async_q: asyncio.Queue = asyncio.Queue()
                if diarization_segments:
                    future = loop.run_in_executor(
                        None,
                        transcribe_by_segments,
                        wav_path,
                        diarization_segments,
                        async_q,
                        loop,
                        source_lang,
                        tmp_dir,
                    )
                else:
                    future = loop.run_in_executor(
                        None,
                        transcribe_with_canary,
                        wav_path,
                        async_q,
                        loop,
                        source_lang,
                    )
                while True:
                    pct = await async_q.get()
                    if pct is None:
                        break
                    yield sse("transcript_progress", {"pct": pct})
                raw_transcript = await future
            except Exception as exc:
                log.error("  [canary] ERROR: %s", exc)
                yield sse(
                    "error",
                    {"message": f"Canary error: {exc}", "stage": "whisper"},
                )
                return
            log.info(
                "  [canary] done - %d chars  (%.2fs)",
                len(raw_transcript),
                time.monotonic() - t0,
            )
            yield sse("transcript_progress", {"pct": 100})
            yield sse("transcript_done", {"text": raw_transcript})
        else:
            log.info("► No video file - chat-only mode")

        yield sse("status", {"message": "Cleaning content with Gemma..."})
        log.info("  [gemma/clean] calling ollama...")
        t0 = time.monotonic()
        try:
            cleaned_transcript = ""
            cleaned_chat = ""
            if raw_transcript.strip():
                cleaned_transcript = await loop.run_in_executor(
                    None,
                    clean_content,
                    raw_transcript,
                    visual_context,
                )
            if chat_text.strip():
                cleaned_chat = local_preclean_content(chat_text)
            cleaned_text = combine_sources(cleaned_transcript, cleaned_chat)
        except Exception as exc:
            log.error("  [gemma/clean] ERROR: %s", exc)
            yield sse(
                "error",
                {"message": f"Ollama (clean) error: {exc}", "stage": "clean"},
            )
            return
        log.info("  [gemma/clean] done  (%.2fs)", time.monotonic() - t0)
        cleaned_text = remove_repetitions(cleaned_text)
        combined = combine_sources(raw_transcript, chat_text)
        cleaned_text = prefer_meaningful_content(cleaned_text, combined)
        if looks_like_missing_content_response(cleaned_text):
            log.warning("  [gemma/clean] model returned missing-content placeholder; using combined source text")
        cleaned_artifact_path = _write_artifact(
            f"{artifact_stem}.cleaned.txt",
            cleaned_text,
        )
        yield sse(
            "cleaned_done",
            {
                "text": cleaned_text,
                "download_url": f"/artifacts/{cleaned_artifact_path.name}",
                "filename": cleaned_artifact_path.name,
            },
        )

        is_meeting = False
        try:
            is_meeting = await loop.run_in_executor(None, classify_is_meeting, cleaned_text)
            log.info("  [gemma/meeting-detect] is_meeting=%s", is_meeting)
        except Exception as exc:
            log.warning("  [gemma/meeting-detect] ERROR (non-fatal): %s", exc)

        if is_meeting:
            yield sse("status", {"message": "Meeting detected — generating meeting summary..."})
        else:
            yield sse("status", {"message": "Generating summary with Gemma..."})

        tldr_title = "Краткое саммари"
        tldr_stage = "tldr"
        tldr_message = "Generating short TL;DR with Gemma..."
        tldr_callable = None
        if is_meeting:
            tldr_title = "ToDo для меня"
            tldr_stage = "todo"
            tldr_message = "Generating personal ToDo for you..."

        yield sse("status", {"message": tldr_message})
        log.info("  [gemma/summary] calling ollama (meeting=%s)...", is_meeting)
        log.info("  [gemma/%s] calling ollama in parallel...", tldr_stage)
        summary_t0 = time.monotonic()
        tldr_t0 = time.monotonic()
        summary_input = prefer_meaningful_content(cleaned_text, combined)
        if summary_input != cleaned_text:
            log.warning("  [gemma/summary] cleaned text unusable; falling back to combined source")
        if is_meeting:
            tldr_callable = lambda: generate_personal_todo(summary_input)
        else:
            tldr_callable = lambda: generate_short_summary(summary_input)

        summary_future = loop.run_in_executor(
            None,
            lambda: generate_summary(summary_input, is_meeting=is_meeting),
        )
        tldr_future = loop.run_in_executor(None, tldr_callable)

        try:
            summary_text = await summary_future
        except Exception as exc:
            log.error("  [gemma/summary] ERROR: %s", exc)
            tldr_future.cancel()
            yield sse(
                "error",
                {"message": f"Ollama (summary) error: {exc}", "stage": "summary"},
            )
            return
        if looks_like_missing_content_response(summary_text):
            log.warning("  [gemma/summary] model returned missing-content placeholder; retrying with temperature=0.3 and truncated input")
            truncated = summary_input[:8000]
            summary_text = await loop.run_in_executor(
                None,
                lambda: generate_summary(truncated, is_meeting=is_meeting, options_override={"temperature": 0.3}),
            )
            if looks_like_missing_content_response(summary_text):
                log.error("  [gemma/summary] both attempts failed — model cannot process the content")
                yield sse(
                    "error",
                    {
                        "message": (
                            "Модель не змогла обробити транскрипцію. "
                            "Можливо, текст занадто довгий або модель не підтримує цей формат. "
                            "Очищена транскрипція доступна для перегляду."
                        ),
                        "stage": "summary",
                    },
                )
                return
        log.info("  [gemma/summary] done  (%.2fs)", time.monotonic() - summary_t0)

        russian_summary_text = None
        try:
            summary_language = await loop.run_in_executor(
                None,
                classify_text_language,
                summary_text,
            )
            log.info("  [gemma/summary-language] %s", summary_language)
        except Exception as exc:
            summary_language = "other"
            log.warning("  [gemma/summary-language] ERROR: %s", exc)

        if summary_language not in {"ru", "uk"}:
            yield sse("status", {"message": "Translating summary to Russian..."})
            log.info("  [gemma/summary-ru] calling ollama...")
            t0 = time.monotonic()
            try:
                russian_summary_text = await loop.run_in_executor(
                    None,
                    translate_summary_to_russian,
                    summary_text,
                )
                log.info("  [gemma/summary-ru] done  (%.2fs)", time.monotonic() - t0)
            except Exception as exc:
                log.warning("  [gemma/summary-ru] ERROR: %s", exc)

        yield sse(
            "summary_done",
            {
                "text": summary_text,
                "russian_text": russian_summary_text,
                "is_meeting": is_meeting,
            },
        )

        try:
            tldr_text = await tldr_future
        except Exception as exc:
            log.error("  [gemma/%s] ERROR: %s", tldr_stage, exc)
            yield sse(
                "error",
                {"message": f"Ollama ({tldr_stage}) error: {exc}", "stage": tldr_stage},
            )
            return
        if looks_like_missing_content_response(tldr_text):
            log.warning("  [gemma/%s] model returned missing-content placeholder; retrying with temperature=0.3", tldr_stage)
            truncated = summary_input[:8000]
            retry_callable = (
                (lambda: generate_personal_todo(truncated, options_override={"temperature": 0.3}))
                if is_meeting
                else (lambda: generate_short_summary(truncated, options_override={"temperature": 0.3}))
            )
            tldr_text = await loop.run_in_executor(None, retry_callable)
            if looks_like_missing_content_response(tldr_text):
                log.warning("  [gemma/%s] both attempts failed — skipping tldr", tldr_stage)
                tldr_text = "Не вдалося згенерувати короткий підсумок."
        log.info("  [gemma/%s] done  (%.2fs)", tldr_stage, time.monotonic() - tldr_t0)
        yield sse(
            "tldr_done",
            {
                "text": tldr_text,
                "title": tldr_title,
                "is_meeting": is_meeting,
            },
        )

        elapsed = time.monotonic() - total_start
        log.info("◄ [%s] Complete - total %.2fs", request_label, elapsed)
        await notify_done(
            title="Готово",
            message=f"{request_label} обработан за {elapsed:.0f}с",
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@asynccontextmanager
async def lifespan(app):
    loop = asyncio.get_event_loop()
    log.info(
        "Checking Ollama at startup (text=%s, clean=%s, frames=%s)...",
        settings.ollama_model,
        settings.ollama_clean_model,
        settings.frame_model,
    )
    await loop.run_in_executor(
        None,
        ensure_ollama_ready,
        settings.ollama_model,
        settings.ollama_clean_model,
        settings.frame_model,
    )
    log.info("Ollama ready — service accepting requests.")
    log.info("Pre-loading Canary model at startup...")
    await loop.run_in_executor(None, get_canary_model)
    log.info("Canary model ready — service accepting requests.")
    yield


app = FastAPI(title="Video Summarizer", lifespan=lifespan)
app.mount("/artifacts", StaticFiles(directory=str(ARTIFACTS_DIR)), name="artifacts")
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
