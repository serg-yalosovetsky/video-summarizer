from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path

from config import settings
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import sentry_sdk

from helpers import is_benign_nemo_transformer_log
from processing_pipeline import PipelineDeps, process_generator as process_generator_impl


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
    unload_ollama_models,
)
from summary import (
    classify_is_meeting,
    classify_text_language,
    clean_content,
    evaluate_speaker_context,
    build_quality_report,
    filter_reliable_context,
    substitute_speaker_names,
    generate_personal_todo,
    generate_short_summary,
    generate_summary,
    local_preclean_content,
    looks_like_missing_content_response,
    looks_truncated_response,
    prefer_meaningful_content,
    SUMMARY_RETRY_MIN_TOKENS,
    TLDR_RETRY_MIN_TOKENS,
    translate_summary_to_russian,
)
from contextlib import asynccontextmanager
from tracing import (
    check_langfuse_auth,
    current_trace_context,
    flush_langfuse,
    langfuse_is_enabled,
    start_observation,
    trace_sync_call,
)

OLLAMA_MODEL = settings.ollama_model
OLLAMA_CLEAN_MODEL = settings.ollama_clean_model
ARTIFACTS_DIR = settings.artifacts_dir
ARTIFACTS_DIR.mkdir(exist_ok=True)

from transcribe import (
    WavConversionResult,
    convert_to_wav,
    release_canary_model,
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


def _wav_meta_payload(meta: dict | WavConversionResult) -> dict:
    if isinstance(meta, WavConversionResult):
        return {
            "file_info": meta.file_info,
            "format": meta.format_name,
            "duration": meta.duration_display,
            "duration_sec": meta.duration_sec,
            "codec": meta.codec,
            "has_video": meta.has_video,
            "output_path": meta.output_path,
        }
    return meta


async def process_generator(
    file: UploadFile | None,
    chat_text: str,
    source_lang: str = "ru",
):
    deps = PipelineDeps(
        settings=settings,
        sse=sse,
        log=log,
        notify_done=notify_done,
        tail_text=tail_text,
        start_observation=start_observation,
        current_trace_context=current_trace_context,
        trace_sync_call=trace_sync_call,
        combine_sources=combine_sources,
        remove_repetitions=remove_repetitions,
        prefer_meaningful_content=prefer_meaningful_content,
        local_preclean_content=local_preclean_content,
        looks_like_missing_content_response=looks_like_missing_content_response,
        looks_truncated_response=looks_truncated_response,
        clean_content=clean_content,
        classify_is_meeting=classify_is_meeting,
        classify_text_language=classify_text_language,
        generate_summary=generate_summary,
        generate_short_summary=generate_short_summary,
        generate_personal_todo=generate_personal_todo,
        translate_summary_to_russian=translate_summary_to_russian,
        summary_retry_min_tokens=SUMMARY_RETRY_MIN_TOKENS,
        tldr_retry_min_tokens=TLDR_RETRY_MIN_TOKENS,
        convert_to_wav=convert_to_wav,
        run_diarization=run_diarization,
        transcribe_by_segments=transcribe_by_segments,
        transcribe_with_canary=transcribe_with_canary,
        extract_frames=extract_frames,
        extract_frames_at=extract_frames_at,
        analyze_frames_with_progress=analyze_frames_with_progress,
        analyze_speaker_frames=analyze_speaker_frames,
        generate_frame_timestamps=generate_frame_timestamps,
        is_context_sufficient=is_context_sufficient,
        max_frames=MAX_FRAMES,
        frame_timestamps=FRAME_TIMESTAMPS,
        build_artifact_stem=_build_artifact_stem,
        write_artifact=_write_artifact,
        wav_meta_payload=_wav_meta_payload,
        evaluate_speaker_context=evaluate_speaker_context,
        build_quality_report=build_quality_report,
        filter_reliable_context=filter_reliable_context,
        substitute_speaker_names=substitute_speaker_names,
        release_canary=release_canary_model,
        unload_ollama=lambda: unload_ollama_models(
            settings.ollama_model,
            settings.ollama_clean_model,
            settings.frame_model,
        ),
    )
    async for event in process_generator_impl(file, chat_text, source_lang, deps=deps):
        yield event


@asynccontextmanager
async def lifespan(app):
    loop = asyncio.get_event_loop()
    log.info(
        "Checking Ollama at startup (device=%s, text=%s, clean=%s, frames=%s)...",
        settings.ollama_device,
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
    if langfuse_is_enabled():
        log.info("Langfuse tracing enabled.")
    yield
    log.info("Shutting down — releasing GPU resources...")
    await loop.run_in_executor(None, release_canary_model)
    await loop.run_in_executor(
        None,
        unload_ollama_models,
        settings.ollama_model,
        settings.ollama_clean_model,
        settings.frame_model,
    )
    await loop.run_in_executor(None, flush_langfuse)


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
