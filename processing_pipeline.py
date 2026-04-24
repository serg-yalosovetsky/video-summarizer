from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from fastapi import UploadFile


class PipelineAbort(Exception):
    """Stop request processing after an error event has already been emitted."""


@dataclass(frozen=True)
class UploadTooLargeError(Exception):
    size_bytes: int
    limit_bytes: int


@dataclass
class ProcessState:
    tmp_dir: str
    chat_text: str
    source_lang: str
    request_label: str = "chat-only"
    artifact_stem: str = "chat-only"
    raw_transcript: str = ""
    visual_context: str = ""
    filtered_visual_context: str = ""
    cleaned_text: str = ""
    cleaned_artifact_path: Path | None = None
    summary_text: str = ""
    russian_summary_text: str | None = None
    tldr_text: str = ""
    tldr_title: str = "Краткое саммари"
    tldr_stage: str = "tldr"
    is_meeting: bool = False
    input_path: str | None = None
    wav_path: str | None = None
    media_meta: Any = None
    diarization_segments: list = field(default_factory=list)


@dataclass(frozen=True)
class PipelineDeps:
    settings: Any
    sse: Callable[[str, dict], str]
    log: Any
    notify_done: Callable[..., Any]
    tail_text: Callable[[str], str]
    start_observation: Callable[..., Any]
    current_trace_context: Callable[[], dict[str, str] | None]
    trace_sync_call: Callable[..., Any]
    combine_sources: Callable[[str, str], str]
    remove_repetitions: Callable[[str], str]
    prefer_meaningful_content: Callable[[str, str], str]
    local_preclean_content: Callable[[str], str]
    looks_like_missing_content_response: Callable[[str], bool]
    looks_truncated_response: Callable[[str], bool]
    clean_content: Callable[..., str]
    classify_is_meeting: Callable[[str], bool]
    classify_text_language: Callable[[str], str]
    generate_summary: Callable[..., str]
    generate_short_summary: Callable[..., str]
    generate_personal_todo: Callable[..., str]
    translate_summary_to_russian: Callable[[str], str]
    summary_retry_min_tokens: int
    tldr_retry_min_tokens: int
    convert_to_wav: Callable[..., Any]
    run_diarization: Callable[..., list]
    transcribe_by_segments: Callable[..., str]
    transcribe_with_canary: Callable[..., str]
    extract_frames: Callable[..., list[str]]
    extract_frames_at: Callable[..., list[str]]
    analyze_frames_with_progress: Callable[..., str]
    analyze_speaker_frames: Callable[..., str]
    generate_frame_timestamps: Callable[..., list]
    is_context_sufficient: Callable[[str], bool]
    max_frames: int
    frame_timestamps: list[int]
    build_artifact_stem: Callable[[str], str]
    write_artifact: Callable[[str, str], Path]
    wav_meta_payload: Callable[[Any], dict]
    evaluate_speaker_context: Callable[[str], dict]
    build_quality_report: Callable[[dict], str]
    filter_reliable_context: Callable[[str, dict], str]
    substitute_speaker_names: Callable[[str, dict], str]
    release_canary: Callable[[], None]
    release_diarizer: Callable[[], None]
    unload_ollama: Callable[[], None]
    unload_clean_model: Callable[[], None]


async def _run_in_executor_traced(
    deps: PipelineDeps,
    loop: asyncio.AbstractEventLoop,
    name: str,
    func,
    *args,
    input_payload: dict | None = None,
    metadata: dict | None = None,
    output_builder=None,
):
    trace_context = deps.current_trace_context()
    return await loop.run_in_executor(
        None,
        lambda: deps.trace_sync_call(
            name,
            func,
            *args,
            trace_context=trace_context,
            input=input_payload,
            metadata=metadata,
            output_builder=output_builder,
        ),
    )


def _run_traced_background(
    deps: PipelineDeps,
    loop: asyncio.AbstractEventLoop,
    name: str,
    func,
    *args,
    input_payload: dict | None = None,
    metadata: dict | None = None,
    output_builder=None,
):
    trace_context = deps.current_trace_context()
    return loop.run_in_executor(
        None,
        lambda: deps.trace_sync_call(
            name,
            func,
            *args,
            trace_context=trace_context,
            input=input_payload,
            metadata=metadata,
            output_builder=output_builder,
        ),
    )


def _set_root_error(root_span, status_message: str) -> None:
    if root_span is not None:
        root_span.update(level="ERROR", status_message=status_message)


async def _emit_progress_events(
    deps: PipelineDeps,
    queue: asyncio.Queue,
    event_name: str,
    *,
    payload_builder=lambda item: item,
):
    while True:
        item = await queue.get()
        if item is None:
            break
        yield deps.sse(event_name, payload_builder(item))


async def _sleep_between_stages(
    deps: PipelineDeps,
    next_stage_label: str,
):
    delay_seconds = max(0, int(getattr(deps.settings, "stage_delay_seconds", 0)))
    if delay_seconds <= 0:
        return
    seconds_label = "second" if delay_seconds == 1 else "seconds"
    deps.log.info("  [pause] waiting %d %s before %s...", delay_seconds, seconds_label, next_stage_label)
    yield deps.sse(
        "status",
        {"message": f"Waiting {delay_seconds} {seconds_label} before {next_stage_label}..."},
    )
    await asyncio.sleep(delay_seconds)


async def _save_upload_file(
    file: UploadFile,
    destination_path: str,
    *,
    max_bytes: int | None,
    chunk_size: int = 4 * 1024 * 1024,
) -> int:
    bytes_written = 0
    try:
        with open(destination_path, "wb") as file_handle:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if max_bytes is not None and bytes_written > max_bytes:
                    raise UploadTooLargeError(size_bytes=bytes_written, limit_bytes=max_bytes)
                file_handle.write(chunk)
    except Exception:
        Path(destination_path).unlink(missing_ok=True)
        raise
    finally:
        await file.close()
    return bytes_written


async def _process_upload_and_media(
    state: ProcessState,
    *,
    file: UploadFile,
    loop: asyncio.AbstractEventLoop,
    root_span,
    deps: PipelineDeps,
):
    safe_filename = os.path.basename(file.filename)
    state.request_label = safe_filename
    state.artifact_stem = deps.build_artifact_stem(safe_filename)
    state.input_path = os.path.join(state.tmp_dir, safe_filename)
    state.wav_path = os.path.join(state.tmp_dir, "audio.wav")

    try:
        size_bytes = await _save_upload_file(
            file,
            state.input_path,
            max_bytes=deps.settings.max_upload_bytes,
        )
    except UploadTooLargeError as exc:
        size_mb = exc.size_bytes / (1024 * 1024)
        limit_mb = exc.limit_bytes / (1024 * 1024)
        deps.log.error(
            "  [upload] ERROR: file exceeds configured %.1f MB limit (%.1f MB)",
            limit_mb,
            size_mb,
        )
        yield deps.sse(
            "error",
            {
                "message": f"File exceeds configured upload limit of {limit_mb:.0f} MB.",
                "stage": "upload",
            },
        )
        raise PipelineAbort

    size_mb = size_bytes / (1024 * 1024)
    deps.log.info("► [%s] Start - %.1f MB", safe_filename, size_mb)

    yield deps.sse("status", {"message": "Converting media with FFmpeg..."})
    deps.log.info("  [ffmpeg] converting...")
    t0 = time.monotonic()
    try:
        state.media_meta = await _run_in_executor_traced(
            deps,
            loop,
            "pipeline.ffmpeg",
            deps.convert_to_wav,
            state.input_path,
            state.wav_path,
            input_payload={"filename": safe_filename},
            metadata={"stage": "ffmpeg"},
            output_builder=lambda item: {
                "duration_sec": item.duration_sec,
                "has_video": item.has_video,
                "format_name": item.format_name,
            },
        )
    except FileNotFoundError:
        deps.log.error("  [ffmpeg] ERROR: ffprobe/ffmpeg not found in PATH")
        _set_root_error(root_span, "FFmpeg not available")
        yield deps.sse(
            "error",
            {
                "message": "ffprobe/ffmpeg not found. Install FFmpeg and add to PATH.",
                "stage": "ffmpeg",
            },
        )
        raise PipelineAbort
    except ValueError as exc:
        deps.log.error("  [ffmpeg] ERROR: %s", exc)
        _set_root_error(root_span, f"FFmpeg validation failed: {exc}")
        yield deps.sse("error", {"message": str(exc), "stage": "ffmpeg"})
        raise PipelineAbort
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr or ""
        stdout = exc.stdout or ""
        error_text = deps.tail_text(stderr or stdout or str(exc))
        deps.log.error("  [ffmpeg] ERROR: %s", error_text)
        _set_root_error(root_span, f"FFmpeg failed: {error_text}")
        yield deps.sse("error", {"message": f"FFmpeg error: {error_text}", "stage": "ffmpeg"})
        raise PipelineAbort

    deps.log.info(
        "  [ffmpeg] done - %s, %s  (%.2fs)",
        state.media_meta["format"],
        state.media_meta["duration"],
        time.monotonic() - t0,
    )
    yield deps.sse("ffmpeg_done", deps.wav_meta_payload(state.media_meta))

    yield deps.sse("status", {"message": "Running speaker diarization..."})
    deps.log.info("  [pyannote] starting diarization...")
    t0 = time.monotonic()
    try:
        state.diarization_segments = await _run_in_executor_traced(
            deps,
            loop,
            "pipeline.diarization",
            deps.run_diarization,
            state.wav_path,
            input_payload={"audio_file": os.path.basename(state.wav_path)},
            metadata={"stage": "diarization"},
            output_builder=lambda segments: {"segments_count": len(segments)},
        )
        deps.log.info(
            "  [pyannote] done — %d segments (%.2fs)",
            len(state.diarization_segments),
            time.monotonic() - t0,
        )
        yield deps.sse("diarization_done", {"segments_count": len(state.diarization_segments)})
    except Exception as exc:
        deps.log.warning("  [pyannote] ERROR (non-fatal, falling back to single-pass): %s", exc)


async def _analyze_frames_step(
    state: ProcessState,
    *,
    loop: asyncio.AbstractEventLoop,
    deps: PipelineDeps,
):
    if not state.media_meta or not state.media_meta.get("has_video"):
        return

    yield deps.sse("status", {"message": "Analyzing video frames for speaker context..."})
    deps.log.info("  [frames] extracting frames...")
    t0 = time.monotonic()
    try:
        total_analyzed = 0
        if state.diarization_segments:
            unique_speakers = len({spk for _, _, spk in state.diarization_segments})
            yield deps.sse("status", {"message": f"Analyzing speaker frames ({unique_speakers} speakers)..."})
            progress_q: asyncio.Queue = asyncio.Queue()
            future = _run_traced_background(
                deps,
                loop,
                "pipeline.speaker-frame-analysis",
                deps.analyze_speaker_frames,
                state.input_path,
                state.tmp_dir,
                state.diarization_segments,
                progress_q,
                loop,
                0,
                unique_speakers,
                input_payload={"speaker_count": unique_speakers},
                metadata={"stage": "frames"},
                output_builder=lambda text: {"context_chars": len(text)},
            )
            async for event in _emit_progress_events(deps, progress_q, "frames_progress"):
                yield event
            state.visual_context = await future
            total_analyzed = unique_speakers
            deps.log.info(
                "  [frames] speaker frames done (%d speakers, %.2fs)",
                unique_speakers,
                time.monotonic() - t0,
            )
        else:
            frame_paths = await _run_in_executor_traced(
                deps,
                loop,
                "pipeline.extract-frames",
                deps.extract_frames,
                state.input_path,
                state.tmp_dir,
                state.media_meta["duration_sec"],
                input_payload={"duration_sec": state.media_meta["duration_sec"]},
                metadata={"stage": "frames"},
                output_builder=lambda paths: {"frames_count": len(paths)},
            )
            if frame_paths:
                done_ts = set(deps.frame_timestamps)
                progress_q: asyncio.Queue = asyncio.Queue()
                future = _run_traced_background(
                    deps,
                    loop,
                    "pipeline.frame-analysis",
                    deps.analyze_frames_with_progress,
                    frame_paths,
                    progress_q,
                    loop,
                    0,
                    len(frame_paths),
                    input_payload={"frames_count": len(frame_paths)},
                    metadata={"stage": "frames"},
                    output_builder=lambda text: {"context_chars": len(text)},
                )
                async for event in _emit_progress_events(deps, progress_q, "frames_progress"):
                    yield event
                state.visual_context = await future
                total_analyzed = len(frame_paths)

                while not deps.is_context_sufficient(state.visual_context) and total_analyzed < deps.max_frames:
                    remaining = deps.max_frames - total_analyzed
                    batch_count = min(4, remaining)
                    new_ts = deps.generate_frame_timestamps(
                        done_ts,
                        state.media_meta["duration_sec"],
                        batch_count,
                    )
                    if not new_ts:
                        break
                    done_ts.update(new_ts)
                    deps.log.info(
                        "  [frames] context insufficient, scanning %d more (total %d/%d)...",
                        len(new_ts),
                        total_analyzed + len(new_ts),
                        deps.max_frames,
                    )
                    yield deps.sse(
                        "status",
                        {
                            "message": (
                                f"Context insufficient - scanning more frames "
                                f"({total_analyzed}/{deps.max_frames})..."
                            )
                        },
                    )
                    new_paths = await _run_in_executor_traced(
                        deps,
                        loop,
                        "pipeline.extract-extra-frames",
                        deps.extract_frames_at,
                        state.input_path,
                        state.tmp_dir,
                        new_ts,
                        input_payload={"timestamps": new_ts},
                        metadata={"stage": "frames"},
                        output_builder=lambda paths: {"frames_count": len(paths)},
                    )
                    if not new_paths:
                        break

                    extra_progress_q: asyncio.Queue = asyncio.Queue()
                    extra_future = _run_traced_background(
                        deps,
                        loop,
                        "pipeline.extra-frame-analysis",
                        deps.analyze_frames_with_progress,
                        new_paths,
                        extra_progress_q,
                        loop,
                        total_analyzed,
                        total_analyzed + len(new_paths),
                        input_payload={"frames_count": len(new_paths)},
                        metadata={"stage": "frames"},
                        output_builder=lambda text: {"context_chars": len(text)},
                    )
                    async for event in _emit_progress_events(deps, extra_progress_q, "frames_progress"):
                        yield event
                    extra_ctx = await extra_future
                    state.visual_context = f"{state.visual_context}\n{extra_ctx}".strip()
                    total_analyzed += len(new_paths)

                deps.log.info(
                    "  [frames] done - %d frames  (%.2fs)",
                    total_analyzed,
                    time.monotonic() - t0,
                )

        quality_eval = deps.evaluate_speaker_context(state.visual_context)
        quality_report = deps.build_quality_report(quality_eval)
        state.filtered_visual_context = deps.filter_reliable_context(state.visual_context, quality_eval)
        deps.log.info(
            "  [frames] quality=%s reliable=%d suspicious=%d unidentified=%d",
            quality_eval["quality_label"],
            len(quality_eval["reliable"]),
            len(quality_eval["suspicious_same_appearance"]),
            len(quality_eval["unidentified"]),
        )
        display_context = state.visual_context
        if quality_eval["quality_label"] == "high" and quality_eval.get("reliable"):
            display_context = deps.substitute_speaker_names(state.visual_context, quality_eval)
        yield deps.sse(
            "frames_done",
            {
                "context": display_context,
                "quality_report": quality_report,
                "quality_label": quality_eval["quality_label"],
                "frames_count": total_analyzed,
            },
        )
    except Exception as exc:
        deps.log.warning("  [frames] ERROR (non-fatal): %s", exc)


async def _transcribe_step(
    state: ProcessState,
    *,
    loop: asyncio.AbstractEventLoop,
    deps: PipelineDeps,
):
    if state.diarization_segments:
        yield deps.sse("status", {"message": "Transcribing audio by speaker segments with Canary..."})
    else:
        yield deps.sse("status", {"message": "Transcribing audio with Canary (may take a few minutes)..."})

    deps.log.info("  [canary] transcribing (diarized=%s)...", bool(state.diarization_segments))
    t0 = time.monotonic()
    try:
        async_q: asyncio.Queue = asyncio.Queue()
        if state.diarization_segments:
            future = _run_traced_background(
                deps,
                loop,
                "pipeline.transcribe-by-segments",
                deps.transcribe_by_segments,
                state.wav_path,
                state.diarization_segments,
                async_q,
                loop,
                state.source_lang,
                state.tmp_dir,
                input_payload={
                    "audio_file": os.path.basename(state.wav_path),
                    "segments_count": len(state.diarization_segments),
                    "source_lang": state.source_lang,
                },
                metadata={"stage": "transcription"},
                output_builder=lambda text: {"text_length": len(text)},
            )
        else:
            future = _run_traced_background(
                deps,
                loop,
                "pipeline.transcribe-single",
                deps.transcribe_with_canary,
                state.wav_path,
                async_q,
                loop,
                state.source_lang,
                input_payload={
                    "audio_file": os.path.basename(state.wav_path),
                    "source_lang": state.source_lang,
                },
                metadata={"stage": "transcription"},
                output_builder=lambda text: {"text_length": len(text)},
            )

        async for event in _emit_progress_events(
            deps,
            async_q,
            "transcript_progress",
            payload_builder=lambda pct: {"pct": pct},
        ):
            yield event
        state.raw_transcript = await future
    except Exception as exc:
        deps.log.error("  [canary] ERROR: %s", exc)
        yield deps.sse("error", {"message": f"Canary error: {exc}", "stage": "whisper"})
        raise PipelineAbort

    deps.log.info(
        "  [canary] done - %d chars  (%.2fs)",
        len(state.raw_transcript),
        time.monotonic() - t0,
    )
    yield deps.sse("transcript_progress", {"pct": 100})
    yield deps.sse("transcript_done", {"text": state.raw_transcript})


async def _clean_content_step(
    state: ProcessState,
    *,
    loop: asyncio.AbstractEventLoop,
    root_span,
    deps: PipelineDeps,
):
    yield deps.sse("status", {"message": "Cleaning content with Gemma..."})
    deps.log.info("  [gemma/clean] calling ollama...")
    t0 = time.monotonic()
    try:
        cleaned_transcript = ""
        cleaned_chat = ""
        if state.raw_transcript.strip():
            clean_ctx = state.filtered_visual_context or state.visual_context
            cleaned_transcript = await _run_in_executor_traced(
                deps,
                loop,
                "pipeline.clean-content",
                deps.clean_content,
                state.raw_transcript,
                clean_ctx,
                input_payload={
                    "transcript_chars": len(state.raw_transcript),
                    "visual_context_chars": len(clean_ctx),
                },
                metadata={"stage": "clean"},
                output_builder=lambda text: {"text_length": len(text)},
            )
        if state.chat_text.strip():
            cleaned_chat = deps.local_preclean_content(state.chat_text)
        state.cleaned_text = deps.combine_sources(cleaned_transcript, cleaned_chat)
    except Exception as exc:
        deps.log.error("  [gemma/clean] ERROR: %s", exc)
        _set_root_error(root_span, f"Cleaning failed: {exc}")
        yield deps.sse("error", {"message": f"Ollama (clean) error: {exc}", "stage": "clean"})
        raise PipelineAbort

    deps.log.info("  [gemma/clean] done  (%.2fs)", time.monotonic() - t0)
    state.cleaned_text = deps.remove_repetitions(state.cleaned_text)
    combined = deps.combine_sources(state.raw_transcript, state.chat_text)
    state.cleaned_text = deps.prefer_meaningful_content(state.cleaned_text, combined)
    if deps.looks_like_missing_content_response(state.cleaned_text):
        deps.log.warning("  [gemma/clean] model returned missing-content placeholder; using combined source text")
    state.cleaned_artifact_path = deps.write_artifact(
        f"{state.artifact_stem}.cleaned.txt",
        state.cleaned_text,
    )
    yield deps.sse(
        "cleaned_done",
        {
            "text": state.cleaned_text,
            "download_url": f"/artifacts/{state.cleaned_artifact_path.name}",
            "filename": state.cleaned_artifact_path.name,
        },
    )


def _substitute_speaker_names_step(state: ProcessState, *, deps: PipelineDeps) -> str | None:
    if not state.visual_context or not state.cleaned_text:
        return None
    eval_result = deps.evaluate_speaker_context(state.visual_context)
    reliable = eval_result.get("reliable", [])
    if not reliable:
        return None
    state.cleaned_text = deps.substitute_speaker_names(state.cleaned_text, eval_result)
    if state.cleaned_artifact_path is not None:
        deps.write_artifact(state.cleaned_artifact_path.name, state.cleaned_text)
    deps.log.info(
        "  [speaker-names] substituted %d reliable speakers: %s",
        len(reliable),
        ", ".join(reliable),
    )
    if state.cleaned_artifact_path is None:
        return None
    return deps.sse(
        "cleaned_done",
        {
            "text": state.cleaned_text,
            "download_url": f"/artifacts/{state.cleaned_artifact_path.name}",
            "filename": state.cleaned_artifact_path.name,
        },
    )


async def _classify_meeting_step(state: ProcessState, *, loop: asyncio.AbstractEventLoop, deps: PipelineDeps):
    try:
        state.is_meeting = await _run_in_executor_traced(
            deps,
            loop,
            "pipeline.classify-meeting",
            deps.classify_is_meeting,
            state.cleaned_text,
            input_payload={"text_chars": len(state.cleaned_text)},
            metadata={"stage": "classify"},
            output_builder=lambda value: {"is_meeting": bool(value)},
        )
        deps.log.info("  [gemma/meeting-detect] is_meeting=%s", state.is_meeting)
    except Exception as exc:
        deps.log.warning("  [gemma/meeting-detect] ERROR (non-fatal): %s", exc)


async def _generate_summary_text(
    loop: asyncio.AbstractEventLoop,
    summary_input: str,
    *,
    is_meeting: bool,
    root_span,
    deps: PipelineDeps,
) -> str:
    summary_future = _run_traced_background(
        deps,
        loop,
        "pipeline.generate-summary",
        lambda: deps.generate_summary(summary_input, is_meeting=is_meeting),
        input_payload={"text_chars": len(summary_input), "is_meeting": is_meeting},
        metadata={"stage": "summary"},
        output_builder=lambda text: {"text_length": len(text)},
    )
    try:
        summary_text = await summary_future
    except Exception as exc:
        deps.log.error("  [gemma/summary] ERROR: %s", exc)
        _set_root_error(root_span, f"Summary failed: {exc}")
        raise

    if deps.looks_like_missing_content_response(summary_text):
        deps.log.warning("  [gemma/summary] model returned missing-content placeholder; retrying with temperature=0.3 and truncated input")
        truncated = summary_input[:8000]
        summary_text = await _run_traced_background(
            deps,
            loop,
            "pipeline.generate-summary-retry",
            lambda: deps.generate_summary(
                truncated,
                is_meeting=is_meeting,
                options_override={"temperature": 0.3},
            ),
            input_payload={"text_chars": len(truncated), "is_meeting": is_meeting},
            metadata={"stage": "summary-retry"},
            output_builder=lambda text: {"text_length": len(text)},
        )
        if deps.looks_like_missing_content_response(summary_text):
            _set_root_error(root_span, "Summary retries exhausted")
            raise RuntimeError(
                "Модель не змогла обробити транскрипцію. Можливо, текст занадто довгий або модель не підтримує цей формат."
            )
    elif deps.looks_truncated_response(summary_text):
        deps.log.warning("  [gemma/summary] response looks truncated; retrying with higher token limit")
        truncated = summary_input[:12000]
        summary_text = await _run_traced_background(
            deps,
            loop,
            "pipeline.generate-summary-retry-truncated",
            lambda: deps.generate_summary(
                truncated,
                is_meeting=is_meeting,
                options_override={
                    "temperature": 0.2,
                    "num_predict": max(deps.settings.ollama_summary_max_tokens, deps.summary_retry_min_tokens),
                },
            ),
            input_payload={"text_chars": len(truncated), "is_meeting": is_meeting},
            metadata={"stage": "summary-retry-truncated"},
            output_builder=lambda text: {"text_length": len(text)},
        )
    return summary_text


async def _generate_tldr_text(
    loop: asyncio.AbstractEventLoop,
    summary_input: str,
    *,
    is_meeting: bool,
    tldr_stage: str,
    deps: PipelineDeps,
) -> str:
    failure_message = (
        "Не вдалося згенерувати персональний ToDo."
        if is_meeting
        else "Не вдалося згенерувати короткий підсумок."
    )
    base_callable = (
        (lambda: deps.generate_personal_todo(summary_input))
        if is_meeting
        else (lambda: deps.generate_short_summary(summary_input))
    )
    tldr_text: str | None = None
    try:
        tldr_text = await _run_traced_background(
            deps,
            loop,
            f"pipeline.generate-{tldr_stage}",
            base_callable,
            input_payload={"text_chars": len(summary_input), "is_meeting": is_meeting},
            metadata={"stage": tldr_stage},
            output_builder=lambda text: {"text_length": len(text)},
        )
    except Exception as exc:
        deps.log.warning(
            "  [gemma/%s] invalid structured output or generation error; retrying with temperature=0.3: %s",
            tldr_stage,
            exc,
        )

    if tldr_text is None or deps.looks_like_missing_content_response(tldr_text):
        if tldr_text is not None:
            deps.log.warning(
                "  [gemma/%s] model returned missing-content placeholder; retrying with temperature=0.3",
                tldr_stage,
            )
        truncated = summary_input[:8000]
        retry_callable = (
            (lambda: deps.generate_personal_todo(truncated, options_override={"temperature": 0.3}))
            if is_meeting
            else (lambda: deps.generate_short_summary(truncated, options_override={"temperature": 0.3}))
        )
        try:
            tldr_text = await _run_traced_background(
                deps,
                loop,
                f"pipeline.retry-{tldr_stage}",
                retry_callable,
                input_payload={"text_chars": len(truncated), "is_meeting": is_meeting},
                metadata={"stage": f"{tldr_stage}-retry"},
                output_builder=lambda text: {"text_length": len(text)},
            )
        except Exception as exc:
            deps.log.warning("  [gemma/%s] retry failed — skipping output: %s", tldr_stage, exc)
            return failure_message
        if deps.looks_like_missing_content_response(tldr_text):
            deps.log.warning("  [gemma/%s] both attempts failed — skipping output", tldr_stage)
            return failure_message
    elif deps.looks_truncated_response(tldr_text):
        deps.log.warning("  [gemma/%s] response looks truncated; retrying with higher token limit", tldr_stage)
        truncated = summary_input[:12000]
        retry_callable = (
            (
                lambda: deps.generate_personal_todo(
                    truncated,
                    options_override={
                        "temperature": 0.2,
                        "num_predict": max(deps.settings.ollama_summary_max_tokens, deps.tldr_retry_min_tokens),
                    },
                )
            )
            if is_meeting
            else (
                lambda: deps.generate_short_summary(
                    truncated,
                    options_override={
                        "temperature": 0.2,
                        "num_predict": max(deps.settings.ollama_summary_max_tokens, deps.tldr_retry_min_tokens),
                    },
                )
            )
        )
        tldr_text = await _run_traced_background(
            deps,
            loop,
            f"pipeline.retry-{tldr_stage}-truncated",
            retry_callable,
            input_payload={"text_chars": len(truncated), "is_meeting": is_meeting},
            metadata={"stage": f"{tldr_stage}-retry-truncated"},
            output_builder=lambda text: {"text_length": len(text)},
        )
    return tldr_text


async def _summary_and_tldr_step(
    state: ProcessState,
    *,
    loop: asyncio.AbstractEventLoop,
    root_span,
    deps: PipelineDeps,
):
    if state.is_meeting:
        yield deps.sse("status", {"message": "Meeting detected — generating meeting summary..."})
        state.tldr_title = "ToDo для меня"
        state.tldr_stage = "todo"
        tldr_message = "Generating personal ToDo for you..."
    else:
        yield deps.sse("status", {"message": "Generating summary with Gemma..."})
        state.tldr_title = "Краткое саммари"
        state.tldr_stage = "tldr"
        tldr_message = "Generating short TL;DR with Gemma..."

    yield deps.sse("status", {"message": tldr_message})
    deps.log.info("  [gemma/summary] calling ollama (meeting=%s)...", state.is_meeting)
    summary_t0 = time.monotonic()

    combined = deps.combine_sources(state.raw_transcript, state.chat_text)
    summary_input = deps.prefer_meaningful_content(state.cleaned_text, combined)
    if summary_input != state.cleaned_text:
        deps.log.warning("  [gemma/summary] cleaned text unusable; falling back to combined source")

    try:
        state.summary_text = await _generate_summary_text(loop, summary_input, is_meeting=state.is_meeting, root_span=root_span, deps=deps)
    except Exception as exc:
        message = str(exc)
        if "Модель не змогла обробити транскрипцію" in message:
            yield deps.sse("error", {"message": message + " Очищена транскрипція доступна для перегляду.", "stage": "summary"})
        else:
            yield deps.sse("error", {"message": f"Ollama (summary) error: {exc}", "stage": "summary"})
        raise PipelineAbort
    deps.log.info("  [gemma/summary] done  (%.2fs)", time.monotonic() - summary_t0)

    summary_language = "other"
    try:
        summary_language = await _run_in_executor_traced(
            deps,
            loop,
            "pipeline.classify-summary-language",
            deps.classify_text_language,
            state.summary_text,
            input_payload={"text_chars": len(state.summary_text)},
            metadata={"stage": "language"},
            output_builder=lambda value: {"language": value},
        )
        deps.log.info("  [gemma/summary-language] %s", summary_language)
    except Exception as exc:
        deps.log.warning("  [gemma/summary-language] ERROR: %s", exc)

    if summary_language not in {"ru", "uk"}:
        yield deps.sse("status", {"message": "Translating summary to Russian..."})
        deps.log.info("  [gemma/summary-ru] calling ollama...")
        t0 = time.monotonic()
        try:
            state.russian_summary_text = await _run_in_executor_traced(
                deps,
                loop,
                "pipeline.translate-summary-ru",
                deps.translate_summary_to_russian,
                state.summary_text,
                input_payload={"text_chars": len(state.summary_text)},
                metadata={"stage": "translation"},
                output_builder=lambda text: {"text_length": len(text)},
            )
            deps.log.info("  [gemma/summary-ru] done  (%.2fs)", time.monotonic() - t0)
        except Exception as exc:
            deps.log.warning("  [gemma/summary-ru] ERROR: %s", exc)

    yield deps.sse(
        "summary_done",
        {
            "text": state.summary_text,
            "russian_text": state.russian_summary_text,
            "is_meeting": state.is_meeting,
        },
    )

    tldr_t0 = time.monotonic()
    deps.log.info("  [gemma/%s] calling ollama...", state.tldr_stage)
    try:
        state.tldr_text = await _generate_tldr_text(loop, summary_input, is_meeting=state.is_meeting, tldr_stage=state.tldr_stage, deps=deps)
    except Exception as exc:
        deps.log.error("  [gemma/%s] ERROR: %s", state.tldr_stage, exc)
        _set_root_error(root_span, f"{state.tldr_stage} generation failed: {exc}")
        yield deps.sse(
            "error",
            {"message": f"Ollama ({state.tldr_stage}) error: {exc}", "stage": state.tldr_stage},
        )
        raise PipelineAbort

    deps.log.info("  [gemma/%s] done  (%.2fs)", state.tldr_stage, time.monotonic() - tldr_t0)
    yield deps.sse(
        "tldr_done",
        {
            "text": state.tldr_text,
            "title": state.tldr_title,
            "is_meeting": state.is_meeting,
        },
    )


async def process_generator(
    file: UploadFile | None,
    chat_text: str,
    source_lang: str,
    *,
    deps: PipelineDeps,
):
    if not file and not chat_text.strip():
        yield deps.sse(
            "error",
            {
                "message": "Provide a video/audio file or chat text (or both).",
                "stage": "upload",
            },
        )
        return

    state = ProcessState(
        tmp_dir=tempfile.mkdtemp(dir=str(deps.settings.local_tmp)),
        chat_text=chat_text,
        source_lang=source_lang,
    )
    state.artifact_stem = deps.build_artifact_stem(
        file.filename if file and file.filename else state.request_label
    )
    total_start = time.monotonic()

    with deps.start_observation(
        "video-summarizer.process",
        input={
            "filename": file.filename if file and file.filename else None,
            "has_file": bool(file and file.filename),
            "chat_chars": len(chat_text.strip()),
            "source_lang": source_lang,
        },
        metadata={
            "feature": "video-summarization",
            "source_type": "media+chat" if file and chat_text.strip() else ("media" if file else "chat"),
        },
    ) as root_span:
        loop = asyncio.get_event_loop()
        try:
            if file and file.filename:
                async for event in _process_upload_and_media(
                    state,
                    file=file,
                    loop=loop,
                    root_span=root_span,
                    deps=deps,
                ):
                    yield event
                await loop.run_in_executor(None, deps.release_diarizer)
                async for event in _sleep_between_stages(deps, "frame analysis"):
                    yield event
                async for event in _analyze_frames_step(state, loop=loop, deps=deps):
                    yield event
                await loop.run_in_executor(None, deps.unload_ollama)
                async for event in _sleep_between_stages(deps, "transcription"):
                    yield event
                async for event in _transcribe_step(state, loop=loop, deps=deps):
                    yield event
                await loop.run_in_executor(None, deps.release_canary)
                async for event in _sleep_between_stages(deps, "content cleaning"):
                    yield event
            else:
                deps.log.info("► No video file - chat-only mode")

            async for event in _clean_content_step(state, loop=loop, root_span=root_span, deps=deps):
                yield event
            await loop.run_in_executor(None, deps.unload_clean_model)
            sse_event = _substitute_speaker_names_step(state, deps=deps)
            if sse_event:
                yield sse_event
            async for event in _sleep_between_stages(deps, "meeting classification"):
                yield event
            await _classify_meeting_step(state, loop=loop, deps=deps)
            async for event in _sleep_between_stages(deps, "summary generation"):
                yield event
            async for event in _summary_and_tldr_step(state, loop=loop, root_span=root_span, deps=deps):
                yield event
            await loop.run_in_executor(None, deps.unload_ollama)
        except PipelineAbort:
            return
        finally:
            shutil.rmtree(state.tmp_dir, ignore_errors=True)

        elapsed = time.monotonic() - total_start
        deps.log.info("◄ [%s] Complete - total %.2fs", state.request_label, elapsed)
        if root_span is not None and state.cleaned_artifact_path is not None:
            root_span.update(
                output={
                    "request_label": state.request_label,
                    "elapsed_sec": round(elapsed, 2),
                    "artifact": state.cleaned_artifact_path.name,
                    "is_meeting": state.is_meeting,
                    "transcript_chars": len(state.raw_transcript),
                    "visual_context_chars": len(state.visual_context),
                }
            )
        await deps.notify_done(
            title="Готово",
            message=f"{state.request_label} обработан за {elapsed:.0f}с",
        )
