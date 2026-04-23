from __future__ import annotations

import os
import re

from config import settings
from helpers import call_ollama, load_user_profile
from prompts import (
    CLEAN_PROMPT_TEMPLATE,
    CLEAN_SYSTEM,
    LANGUAGE_CHECK_PROMPT_TEMPLATE,
    LANGUAGE_CHECK_SYSTEM,
    MEETING_DETECTION_PROMPT,
    MEETING_DETECTION_SYSTEM,
    MEETING_SUMMARY_PROMPT_TEMPLATE,
    MEETING_SUMMARY_SYSTEM,
    PERSONAL_TODO_PROMPT_TEMPLATE,
    PERSONAL_TODO_SYSTEM,
    RUSSIAN_TRANSLATION_PROMPT_TEMPLATE,
    RUSSIAN_TRANSLATION_SYSTEM,
    SHORT_SUMMARY_PROMPT_TEMPLATE,
    SUMMARY_PROMPT_TEMPLATE,
    SUMMARY_SYSTEM,
)

OLLAMA_FAST_OPTIONS = {
    "temperature": 0,
    "top_p": 0.9,
}
OLLAMA_CLASSIFIER_OPTIONS = {
    **OLLAMA_FAST_OPTIONS,
    "num_predict": 8,
}
OLLAMA_SUMMARY_OPTIONS = {
    **OLLAMA_FAST_OPTIONS,
    "num_predict": settings.ollama_summary_max_tokens,
}
OLLAMA_CLEAN_OPTIONS = {
    **OLLAMA_FAST_OPTIONS,
    "num_predict": settings.ollama_clean_max_tokens,
}
MAX_VISUAL_CONTEXT_CHARS = settings.max_visual_context_chars
OLLAMA_CLEAN_MODEL = settings.ollama_clean_model
MISSING_CONTENT_RESPONSE_PATTERNS = (
    "please provide the content you would like me to summarize",
    "i am ready to create the comprehensive",
    "пожалуйста, предоставьте контент",
    "как только текст будет доступен",
    "i'm ready to create the comprehensive",
    "once the text is available",
)
TIMESTAMP_RE = re.compile(r"\[\d{2}:\d{2}:\d{2}\]")
SUMMARY_DIRECT_MAX_CHARS = 12000
SUMMARY_CHUNK_TARGET_CHARS = 6000


def trim_visual_context(visual_context: str) -> str:
    text = visual_context.strip()
    if len(text) <= MAX_VISUAL_CONTEXT_CHARS:
        return text
    return text[:MAX_VISUAL_CONTEXT_CHARS].rstrip() + "\n...[truncated]"


def build_context_block(visual_context: str) -> str:
    if not visual_context:
        return ""
    return f"Visual context from video frames:\n{trim_visual_context(visual_context)}\n\n"


def local_preclean_content(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\t", " ")
    normalized = re.sub(r"[ ]{2,}", " ", normalized)
    normalized = re.sub(r" *([,.;:!?])", r"\1", normalized)
    normalized = re.sub(r"([,.;:!?])([^\s\n])", r"\1 \2", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def has_meaningful_content(text: str, min_chars: int = 24) -> bool:
    normalized = local_preclean_content(text)
    if len(normalized) < min_chars:
        return False
    return bool(re.search(r"[A-Za-zА-Яа-яІіЇїЄєҐґ]", normalized))


def looks_like_missing_content_response(text: str) -> bool:
    normalized = local_preclean_content(text).lower()
    if not normalized:
        return True
    return any(pattern in normalized for pattern in MISSING_CONTENT_RESPONSE_PATTERNS)


def prefer_meaningful_content(primary: str, fallback: str) -> str:
    if has_meaningful_content(primary) and not looks_like_missing_content_response(primary):
        return primary
    return local_preclean_content(fallback)


def count_timestamps(text: str) -> int:
    return len(TIMESTAMP_RE.findall(text or ""))


def preserves_timestamp_structure(source: str, candidate: str) -> bool:
    source_count = count_timestamps(source)
    if source_count == 0:
        return True
    candidate_count = count_timestamps(candidate)
    if candidate_count == 0:
        return False
    if candidate_count < max(1, int(source_count * 0.8)):
        return False
    source_ts = TIMESTAMP_RE.findall(source)
    candidate_ts = TIMESTAMP_RE.findall(candidate)
    return source_ts[:3] == candidate_ts[:3] and source_ts[-3:] == candidate_ts[-3:]


def split_for_summary(text: str, target_chars: int = SUMMARY_CHUNK_TARGET_CHARS) -> list[str]:
    text = local_preclean_content(text)
    if len(text) <= target_chars:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        block_len = len(block) + 2
        if current and current_len + block_len > target_chars:
            chunks.append("\n\n".join(current))
            current = [block]
            current_len = block_len
            continue
        if len(block) > target_chars:
            lines = [line for line in block.splitlines() if line.strip()]
            for line in lines:
                if current and current_len + len(line) + 1 > target_chars:
                    chunks.append("\n\n".join(current))
                    current = []
                    current_len = 0
                current.append(line)
                current_len += len(line) + 1
            continue
        current.append(block)
        current_len += block_len
    if current:
        chunks.append("\n\n".join(current))
    return chunks or [text]


def clean_content(transcript: str, visual_context: str = "") -> str:
    normalized_transcript = local_preclean_content(transcript)
    prompt = CLEAN_PROMPT_TEMPLATE.format(
        context_block=build_context_block(visual_context),
        transcript=normalized_transcript,
    )
    cleaned = call_ollama(
        prompt,
        CLEAN_SYSTEM,
        model=OLLAMA_CLEAN_MODEL,
        options=OLLAMA_CLEAN_OPTIONS,
    )
    if preserves_timestamp_structure(normalized_transcript, cleaned):
        return cleaned
    return normalized_transcript


def classify_is_meeting(text: str) -> bool:
    prompt = MEETING_DETECTION_PROMPT.format(text=text[:2000])
    raw = call_ollama(
        prompt,
        MEETING_DETECTION_SYSTEM,
        options=OLLAMA_CLASSIFIER_OPTIONS,
    ).strip().lower()
    return raw.startswith("yes")


def generate_summary(
    transcript: str,
    *,
    is_meeting: bool = False,
    options_override: dict | None = None,
) -> str:
    transcript = local_preclean_content(transcript)
    user_profile = load_user_profile()
    user_name = user_profile["primary_name"]
    user_aliases = ", ".join(user_profile["aliases"])
    opts = {**OLLAMA_SUMMARY_OPTIONS, **(options_override or {})}

    if len(transcript) > SUMMARY_DIRECT_MAX_CHARS:
        chunk_summaries = []
        for index, chunk in enumerate(split_for_summary(transcript), start=1):
            chunk_prompt = (
                "Summarize this transcript chunk for later merging into one final summary.\n"
                "Preserve names, decisions, tasks, blockers, numbers, and technical terms.\n"
                "Be concise but do not omit concrete responsibilities.\n\n"
                f"Chunk {index}:\n---\n{chunk}\n---"
            )
            chunk_summaries.append(call_ollama(chunk_prompt, SUMMARY_SYSTEM, options=opts))
        transcript = "\n\n".join(
            f"Chunk {index} summary:\n{item}"
            for index, item in enumerate(chunk_summaries, start=1)
        )

    if is_meeting:
        prompt = MEETING_SUMMARY_PROMPT_TEMPLATE.format(
            transcript=transcript,
            user_name=user_name,
            user_aliases=user_aliases,
        )
        system = MEETING_SUMMARY_SYSTEM
    else:
        prompt = SUMMARY_PROMPT_TEMPLATE.format(
            transcript=transcript,
            user_name=user_name,
            user_aliases=user_aliases,
        )
        system = SUMMARY_SYSTEM
    return call_ollama(prompt, system, options=opts)


def generate_short_summary(
    transcript: str,
    *,
    options_override: dict | None = None,
) -> str:
    prompt = SHORT_SUMMARY_PROMPT_TEMPLATE.format(transcript=transcript)
    opts = {**OLLAMA_SUMMARY_OPTIONS, **(options_override or {})}
    return call_ollama(prompt, SUMMARY_SYSTEM, options=opts)


def generate_personal_todo(
    transcript: str,
    *,
    options_override: dict | None = None,
) -> str:
    user_profile = load_user_profile()
    prompt = PERSONAL_TODO_PROMPT_TEMPLATE.format(
        transcript=transcript,
        user_name=user_profile["primary_name"],
        user_aliases=", ".join(user_profile["aliases"]),
    )
    opts = {**OLLAMA_SUMMARY_OPTIONS, **(options_override or {})}
    return call_ollama(prompt, PERSONAL_TODO_SYSTEM, options=opts)


def detect_language_heuristically(text: str) -> str:
    lowered = text.lower()
    letters = re.findall(r"[a-zа-яіїєґ]+", lowered)
    if not letters:
        return "other"
    joined = " ".join(letters)
    if any(ch in joined for ch in "іїєґ"):
        return "uk"
    if "ы" in joined or "э" in joined or "ъ" in joined:
        return "ru"
    cyrillic_chars = re.findall(r"[а-яіїєґ]", joined)
    if not cyrillic_chars:
        return "other"
    cyrillic_ratio = len(cyrillic_chars) / max(len(re.findall(r"[a-zа-яіїєґ]", joined)), 1)
    return "ru" if cyrillic_ratio >= 0.6 else "other"


def classify_text_language(text: str) -> str:
    heuristic = detect_language_heuristically(text)
    if heuristic in {"ru", "uk"}:
        return heuristic
    prompt = LANGUAGE_CHECK_PROMPT_TEMPLATE.format(text=text)
    raw = call_ollama(
        prompt,
        LANGUAGE_CHECK_SYSTEM,
        options=OLLAMA_CLASSIFIER_OPTIONS,
    ).strip().lower()
    token = raw.split()[0] if raw else "other"
    return token if token in {"ru", "uk", "other"} else "other"


def translate_summary_to_russian(text: str) -> str:
    prompt = RUSSIAN_TRANSLATION_PROMPT_TEMPLATE.format(text=text)
    return call_ollama(prompt, RUSSIAN_TRANSLATION_SYSTEM, options=OLLAMA_SUMMARY_OPTIONS)
