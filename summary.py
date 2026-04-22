from __future__ import annotations

from helpers import call_ollama
from prompts import (
    CLEAN_PROMPT_TEMPLATE,
    CLEAN_SYSTEM,
    LANGUAGE_CHECK_PROMPT_TEMPLATE,
    LANGUAGE_CHECK_SYSTEM,
    MEETING_DETECTION_PROMPT,
    MEETING_DETECTION_SYSTEM,
    MEETING_SUMMARY_PROMPT_TEMPLATE,
    MEETING_SUMMARY_SYSTEM,
    RUSSIAN_TRANSLATION_PROMPT_TEMPLATE,
    RUSSIAN_TRANSLATION_SYSTEM,
    SHORT_SUMMARY_PROMPT_TEMPLATE,
    SUMMARY_PROMPT_TEMPLATE,
    SUMMARY_SYSTEM,
)


def build_context_block(visual_context: str) -> str:
    if not visual_context:
        return ""
    return f"Visual context from video frames:\n{visual_context}\n\n"


def clean_content(transcript: str, visual_context: str = "") -> str:
    prompt = CLEAN_PROMPT_TEMPLATE.format(
        context_block=build_context_block(visual_context),
        transcript=transcript,
    )
    return call_ollama(prompt, CLEAN_SYSTEM)


def classify_is_meeting(text: str) -> bool:
    prompt = MEETING_DETECTION_PROMPT.format(text=text[:2000])
    raw = call_ollama(prompt, MEETING_DETECTION_SYSTEM).strip().lower()
    return raw.startswith("yes")


def generate_summary(transcript: str, *, is_meeting: bool = False) -> str:
    if is_meeting:
        prompt = MEETING_SUMMARY_PROMPT_TEMPLATE.format(transcript=transcript)
        system = MEETING_SUMMARY_SYSTEM
    else:
        prompt = SUMMARY_PROMPT_TEMPLATE.format(transcript=transcript)
        system = SUMMARY_SYSTEM
    return call_ollama(prompt, system)


def generate_short_summary(transcript: str) -> str:
    prompt = SHORT_SUMMARY_PROMPT_TEMPLATE.format(transcript=transcript)
    return call_ollama(prompt, SUMMARY_SYSTEM)


def classify_text_language(text: str) -> str:
    prompt = LANGUAGE_CHECK_PROMPT_TEMPLATE.format(text=text)
    raw = call_ollama(prompt, LANGUAGE_CHECK_SYSTEM).strip().lower()
    token = raw.split()[0] if raw else "other"
    return token if token in {"ru", "uk", "other"} else "other"


def translate_summary_to_russian(text: str) -> str:
    prompt = RUSSIAN_TRANSLATION_PROMPT_TEMPLATE.format(text=text)
    return call_ollama(prompt, RUSSIAN_TRANSLATION_SYSTEM)
