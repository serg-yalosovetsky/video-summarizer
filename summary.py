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
    "num_predict": 256,
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
LIKELY_COMPLETE_ENDINGS = (
    ".",
    "!",
    "?",
    "…",
    ":",
    ";",
    ")",
    "]",
    '"',
)
TIMESTAMP_RE = re.compile(r"\[\d{2}:\s*\d{2}:\s*\d{2}\]")
SUMMARY_DIRECT_MAX_CHARS = 12000
SUMMARY_CHUNK_TARGET_CHARS = 6000
SUMMARY_RETRY_MIN_TOKENS = 2048
TLDR_RETRY_MIN_TOKENS = 1536


def trim_visual_context(visual_context: str) -> str:
    text = visual_context.strip()
    if len(text) <= MAX_VISUAL_CONTEXT_CHARS:
        return text
    return text[:MAX_VISUAL_CONTEXT_CHARS].rstrip() + "\n...[truncated]"


def build_context_block(visual_context: str) -> str:
    if not visual_context:
        return ""
    return f"Visual context from video frames:\n{trim_visual_context(visual_context)}\n\n"


_SPEAKER_LINE_RE = re.compile(r"^\[(\w+)\s*@\s*\d+s\]\s*(.*)", re.IGNORECASE)
_NAME_RE = re.compile(r"(?i)\bname:\s*([^,\n]+)")
_NO_CONTEXT_NAME_MARKERS = frozenset([
    "no visible", "no name", "unknown", "not visible",
    "no label", "no tag", "cannot see", "not identified",
])
_GENDER_TOKENS = frozenset(["male", "female", "man", "woman", "boy", "girl"])
_APPEARANCE_KEYWORDS = re.compile(
    r"\b(dark|light|blue|grey|gray|black|white|brown|red|green|blazer|shirt|"
    r"jacket|glasses|hoodie|suit|tie|beard|hair|bald|coat|sweater)\b",
    re.IGNORECASE,
)


def _normalise_name(name: str) -> str:
    return " ".join(name.strip().upper().split())


def _is_no_context_name(name: str | None) -> bool:
    if not name:
        return True
    lower = name.strip().lower()
    return any(marker in lower for marker in _NO_CONTEXT_NAME_MARKERS)


def _appearance_features(appearance: str) -> frozenset[str]:
    lower = appearance.lower()
    gender = next((t for t in _GENDER_TOKENS if t in lower), "")
    keywords = frozenset(m.group(0).lower() for m in _APPEARANCE_KEYWORDS.finditer(appearance))
    return frozenset({gender} | keywords) - {""}


def _appearances_similar(a: str, b: str) -> bool:
    fa = _appearance_features(a)
    fb = _appearance_features(b)
    if not fa or not fb:
        return False
    shared = fa & fb
    # Must share gender and at least one other feature
    has_gender = bool(shared & _GENDER_TOKENS)
    has_other = bool(shared - _GENDER_TOKENS)
    return has_gender and has_other


def evaluate_speaker_context(visual_context: str) -> dict:
    """Parse visual_context and assess speaker identification quality.

    Returns a dict with keys:
      speaker_names, speaker_appearances, reliable,
      suspicious_same_appearance, suspicious_diff_appearance,
      unidentified, quality_score, quality_label
    """
    speaker_names: dict[str, str | None] = {}
    speaker_appearances: dict[str, str] = {}

    for line in visual_context.strip().splitlines():
        m = _SPEAKER_LINE_RE.match(line.strip())
        if not m:
            continue
        speaker_id = m.group(1)
        rest = m.group(2)

        name_m = _NAME_RE.search(rest)
        raw_name = name_m.group(1).strip() if name_m else None
        name = None if _is_no_context_name(raw_name) else (raw_name or None)

        appearance = rest
        if name_m:
            appearance = rest[name_m.end():].lstrip(", ")

        speaker_names[speaker_id] = name
        speaker_appearances[speaker_id] = appearance

    # Group speakers by normalised name
    name_to_speakers: dict[str, list[str]] = {}
    for spk, name in speaker_names.items():
        if name:
            key = _normalise_name(name)
            name_to_speakers.setdefault(key, []).append(spk)

    suspicious_same: list[tuple[str, list[str]]] = []
    suspicious_diff: list[tuple[str, list[str]]] = []

    for norm_name, speakers in name_to_speakers.items():
        if len(speakers) < 2:
            continue
        # Compare appearances pairwise
        appearances = [speaker_appearances.get(s, "") for s in speakers]
        all_similar = all(
            _appearances_similar(appearances[i], appearances[j])
            for i in range(len(appearances))
            for j in range(i + 1, len(appearances))
        )
        display_name = speaker_names[speakers[0]] or norm_name
        if all_similar:
            suspicious_same.append((display_name, speakers))
        else:
            suspicious_diff.append((display_name, speakers))

    skip_speakers: set[str] = {s for _, group in suspicious_same for s in group}
    identified = [s for s, n in speaker_names.items() if n and s not in skip_speakers]
    unidentified = [s for s, n in speaker_names.items() if not n]

    total = len(speaker_names)
    quality_score = len(identified) / total if total else 0.0
    if quality_score >= 0.7:
        quality_label = "high"
    elif quality_score >= 0.4:
        quality_label = "medium"
    elif quality_score > 0:
        quality_label = "low"
    else:
        quality_label = "none"

    return {
        "speaker_names": speaker_names,
        "speaker_appearances": speaker_appearances,
        "reliable": identified,
        "suspicious_same_appearance": suspicious_same,
        "suspicious_diff_appearance": suspicious_diff,
        "unidentified": unidentified,
        "quality_score": quality_score,
        "quality_label": quality_label,
    }


def build_quality_report(eval_result: dict) -> str:
    reliable = eval_result["reliable"]
    suspicious_same = eval_result["suspicious_same_appearance"]
    suspicious_diff = eval_result["suspicious_diff_appearance"]
    unidentified = eval_result["unidentified"]
    total = len(eval_result["speaker_names"])
    label = eval_result["quality_label"].upper()
    lines = [f"Speaker identification quality: {label} ({len(reliable)}/{total} reliable)"]

    if reliable:
        names_str = ", ".join(
            f"{s} → {eval_result['speaker_names'][s]}"
            for s in sorted(reliable)
        )
        lines.append(f"✓ Reliable: {names_str}")

    if unidentified:
        lines.append(f"✗ Unidentified (no name in any frame): {', '.join(sorted(unidentified))}")

    for name, speakers in suspicious_same:
        spk_str = ", ".join(sorted(speakers))
        lines.append(
            f"⚠ Duplicate name + similar appearance → skipped: "
            f"{spk_str} both show \"{name}\" (same person assigned to multiple speakers)"
        )

    for name, speakers in suspicious_diff:
        spk_str = ", ".join(sorted(speakers))
        lines.append(
            f"? Same name, different appearance (kept): "
            f"{spk_str} both show \"{name}\" — may be two different people"
        )

    return "\n".join(lines)


def filter_reliable_context(visual_context: str, eval_result: dict) -> str:
    """Replace name field for suspicious-same-appearance speakers to prevent wrong substitution."""
    skip_speakers = {s for _, group in eval_result["suspicious_same_appearance"] for s in group}
    if not skip_speakers:
        return visual_context

    filtered: list[str] = []
    for line in visual_context.splitlines():
        m = _SPEAKER_LINE_RE.match(line.strip())
        if m and m.group(1) in skip_speakers:
            line = _NAME_RE.sub("name: [ambiguous — skipped]", line, count=1)
        filtered.append(line)
    return "\n".join(filtered)


_SPEAKER_TAG_IN_TRANSCRIPT_RE = re.compile(r"\[SPEAKER_(\w+)\]")


def substitute_speaker_names(text: str, eval_result: dict) -> str:
    """Replace [SPEAKER_XX] tags with identified names for reliable speakers only."""
    reliable: list[str] = eval_result.get("reliable", [])
    speaker_names: dict[str, str | None] = eval_result.get("speaker_names", {})
    if not reliable:
        return text
    for speaker_id in reliable:
        name = speaker_names.get(speaker_id)
        if name and not _is_no_context_name(name):
            text = text.replace(f"[{speaker_id}]", f"[{name}]")
    return text


def local_preclean_content(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\t", " ")
    normalized = re.sub(r"[ ]{2,}", " ", normalized)
    # Exclude ':' to preserve [HH:MM:SS] timestamp format
    normalized = re.sub(r" *([,.;!?])", r"\1", normalized)
    normalized = re.sub(r"([,.;!?])([^\s\n])", r"\1 \2", normalized)
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


def looks_truncated_response(text: str) -> bool:
    normalized = local_preclean_content(text)
    if len(normalized) < 80:
        return False
    if normalized.endswith(LIKELY_COMPLETE_ENDINGS):
        return False
    if normalized.endswith("**"):
        return False
    last_line = normalized.splitlines()[-1].strip()
    if re.match(r"^[-*]\s+[^\s].{0,80}$", last_line):
        return False
    return bool(re.search(r"[A-Za-zА-Яа-яІіЇїЄєҐґ0-9]$", normalized))


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


_SPEAKER_TAG_RE = re.compile(r"\[SPEAKER_\w+\]")


def _has_multiple_speakers(text: str) -> bool:
    speakers = set(_SPEAKER_TAG_RE.findall(text[:4000]))
    return len(speakers) >= 2


def classify_is_meeting(text: str) -> bool:
    if _has_multiple_speakers(text):
        return True
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
