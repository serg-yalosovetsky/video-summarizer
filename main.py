from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import tempfile
import time

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from frames_analyze import (
    FRAME_TIMESTAMPS,
    MAX_FRAMES,
    analyze_frames_with_progress,
    extract_frames,
    extract_frames_at,
    generate_frame_timestamps,
    is_context_sufficient,
)
from helpers import LOCAL_TMP, MAX_UPLOAD_BYTES, combine_sources, log, remove_repetitions, sse, tail_text
from summary import (
    classify_is_meeting,
    classify_text_language,
    clean_content,
    generate_personal_todo,
    generate_short_summary,
    generate_summary,
    translate_summary_to_russian,
)
from contextlib import asynccontextmanager

from transcribe import convert_to_wav, get_canary_model, transcribe_with_canary


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

    tmp_dir = tempfile.mkdtemp(dir=LOCAL_TMP)
    total_start = time.monotonic()
    raw_transcript = ""
    visual_context = ""
    request_label = "chat-only"

    try:
        loop = asyncio.get_event_loop()

        if file and file.filename:
            safe_filename = os.path.basename(file.filename)
            request_label = safe_filename
            input_path = os.path.join(tmp_dir, safe_filename)
            wav_path = os.path.join(tmp_dir, "audio.wav")

            content = await file.read()
            size_mb = len(content) / (1024 * 1024)

            if len(content) > MAX_UPLOAD_BYTES:
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

            if meta.get("has_video"):
                yield sse("status", {"message": "Analyzing video frames for speaker context..."})
                log.info("  [frames] extracting frames...")
                t0 = time.monotonic()
                try:
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

            yield sse("status", {"message": "Transcribing audio with Canary (may take a few minutes)..."})
            log.info("  [canary] transcribing...")
            t0 = time.monotonic()
            try:
                async_q: asyncio.Queue = asyncio.Queue()
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

        combined = combine_sources(raw_transcript, chat_text)

        yield sse("status", {"message": "Cleaning content with Gemma..."})
        log.info("  [gemma/clean] calling ollama...")
        t0 = time.monotonic()
        try:
            cleaned_text = await loop.run_in_executor(
                None,
                clean_content,
                combined,
                visual_context,
            )
        except Exception as exc:
            log.error("  [gemma/clean] ERROR: %s", exc)
            yield sse(
                "error",
                {"message": f"Ollama (clean) error: {exc}", "stage": "clean"},
            )
            return
        log.info("  [gemma/clean] done  (%.2fs)", time.monotonic() - t0)
        cleaned_text = remove_repetitions(cleaned_text)
        yield sse("cleaned_done", {"text": cleaned_text})

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

        log.info("  [gemma/summary] calling ollama (meeting=%s)...", is_meeting)
        t0 = time.monotonic()
        try:
            summary_text = await loop.run_in_executor(
                None,
                lambda: generate_summary(cleaned_text, is_meeting=is_meeting),
            )
        except Exception as exc:
            log.error("  [gemma/summary] ERROR: %s", exc)
            yield sse(
                "error",
                {"message": f"Ollama (summary) error: {exc}", "stage": "summary"},
            )
            return
        log.info("  [gemma/summary] done  (%.2fs)", time.monotonic() - t0)

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

        tldr_title = "Краткое саммари"
        tldr_stage = "tldr"
        tldr_message = "Generating short TL;DR with Gemma..."
        tldr_callable = lambda: generate_short_summary(cleaned_text)
        if is_meeting:
            tldr_title = "ToDo для меня"
            tldr_stage = "todo"
            tldr_message = "Generating personal ToDo for you..."
            tldr_callable = lambda: generate_personal_todo(cleaned_text)

        yield sse("status", {"message": tldr_message})
        log.info("  [gemma/%s] calling ollama...", tldr_stage)
        t0 = time.monotonic()
        try:
            tldr_text = await loop.run_in_executor(None, tldr_callable)
        except Exception as exc:
            log.error("  [gemma/%s] ERROR: %s", tldr_stage, exc)
            yield sse(
                "error",
                {"message": f"Ollama ({tldr_stage}) error: {exc}", "stage": tldr_stage},
            )
            return
        log.info("  [gemma/%s] done  (%.2fs)", tldr_stage, time.monotonic() - t0)
        yield sse(
            "tldr_done",
            {
                "text": tldr_text,
                "title": tldr_title,
                "is_meeting": is_meeting,
            },
        )

        log.info("◄ [%s] Complete - total %.2fs", request_label, time.monotonic() - total_start)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@asynccontextmanager
async def lifespan(app):
    loop = asyncio.get_event_loop()
    log.info("Pre-loading Canary model at startup...")
    await loop.run_in_executor(None, get_canary_model)
    log.info("Canary model ready — service accepting requests.")
    yield


app = FastAPI(title="Video Summarizer", lifespan=lifespan)
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
