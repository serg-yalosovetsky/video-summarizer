from __future__ import annotations


FRAME_ANALYSIS_SYSTEM = (
    "You are a video frame analyst. Describe what you see concisely: "
    "the setting, all visible people and their names (from name tags, slides, "
    "lower thirds, or context), and any on-screen text. Focus on identifying speakers."
)

FRAME_ANALYSIS_PROMPT = (
    "Analyze this video frame. Identify:\n"
    "1) The setting/context (meeting, lecture, interview, etc.)\n"
    "2) All visible people - list their names if shown anywhere on screen\n"
    "3) Any text, titles, logos visible\n"
    "Be concise. If names are not visible, describe people briefly (e.g. 'man in blue shirt')."
)

CLEAN_SYSTEM = (
    "You are a professional transcript editor. Your task is to clean and correct "
    "user-provided content.\n"
    "Rules:\n"
    "- Fix obvious speech-to-text errors and wrong words\n"
    "- Add proper punctuation and capitalization\n"
    "- Break run-on sentences into readable paragraphs\n"
    "- Do NOT paraphrase, summarize, or change the meaning\n"
    "- Do NOT add any content that was not in the original\n"
    "- Preserve speaker intent and all factual content\n"
    "- Preserve every existing [HH:MM:SS] timestamp exactly as written\n"
    "- Preserve the original order of lines\n"
    "- Never drop the ending of the transcript, even if the text is noisy or repetitive\n"
    "- For timestamped transcript input, keep the same number of timestamped entries whenever possible\n"
    "- The transcript may contain [SPEAKER_XX] labels from diarization. "
    "If visual context identifies who SPEAKER_00, SPEAKER_01 etc. are, "
    "replace [SPEAKER_XX] with the actual person's name. "
    "If a speaker cannot be identified, keep [SPEAKER_XX] as-is.\n"
    "- If visual context identifies speakers and there are NO diarization labels, "
    "prepend [Speaker Name]: only to matching transcript lines\n"
    "- Do NOT invent timestamps for chat messages or untimestamped text\n"
    "- Output ONLY the cleaned content, no commentary\n"
    "- Keep the same language as the input - do NOT translate"
)

CLEAN_PROMPT_TEMPLATE = (
    "{context_block}"
    "Clean and correct the following transcript.\n\n"
    "Rules:\n"
    "- Preserve every [HH:MM:SS] timestamp exactly as written\n"
    "- If [SPEAKER_XX] labels are present and the visual context above identifies who they are, "
    "replace [SPEAKER_XX] with the person's real name. Otherwise keep [SPEAKER_XX] as-is.\n"
    "- If there are no [SPEAKER_XX] labels but the visual context identifies speakers, "
    "prepend [Name]: before the spoken text on matching lines.\n"
    "- Do not add new timestamps to chat messages\n"
    "- Keep the same order and number of lines\n"
    "- Do not omit the end of the transcript\n\n"
    "---\n"
    "{transcript}\n"
    "---"
)

SUMMARY_SYSTEM = (
    "You are an expert content summarizer. "
    "Create clear, structured summaries that capture the key information. "
    "IMPORTANT: Always respond in the same language as the input."
)

SUMMARY_PROMPT_TEMPLATE = (
    "Read the transcript below and provide a comprehensive summary in the SAME LANGUAGE as the transcript.\n\n"
    "The summary must include:\n\n"
    "1. **Main Topic**: What this content is about in 1-2 sentences\n"
    "2. **Key Points**: The most important points covered (bullet list)\n"
    "3. **Details**: Relevant supporting information or context\n"
    "4. **What concerns {user_name}**: highlight all mentions, tasks, requests, blockers, decisions, risks, or follow-ups "
    "that concern {user_name} or these aliases: {user_aliases}. If there is nothing relevant, say so explicitly.\n"
    "5. **Conclusion**: Main takeaway or outcome\n\n"
    "Transcript:\n---\n{transcript}\n---"
)

SHORT_SUMMARY_PROMPT_TEMPLATE = (
    "Read the transcript below and write a structured short summary (approximately 10% of the original length).\n\n"
    "If the transcript is about a problem or discussion, structure as:\n"
    "- **Problem**: what issue is being addressed\n"
    "- **Ways to solve**: approaches or actions taken/proposed\n"
    "- **Blockers**: obstacles preventing resolution\n"
    "- **Estimated resolution**: timeframe or next steps if mentioned\n\n"
    "Otherwise write a plain structured summary of the key points.\n\n"
    "Do not stop mid-sentence. Make sure the final line is complete.\n\n"
    "Use the SAME LANGUAGE as the transcript. Output only the summary.\n\n"
    "Transcript:\n---\n{transcript}\n---"
)

PERSONAL_TODO_SYSTEM = (
    "You are a precise meeting task extractor. "
    "Output only a numbered list. "
    "IMPORTANT: Always respond in the same language as the transcript."
)

PERSONAL_TODO_PROMPT_TEMPLATE = (
    "Read the meeting transcript below and extract action items assigned to {user_name}. "
    "{user_name} may also appear as: {user_aliases}.\n\n"
    "Rules:\n"
    "- Include ONLY tasks explicitly assigned to or requested of {user_name}\n"
    "- Ignore tasks assigned to other participants\n"
    "- Each task = exactly one sentence\n"
    "- Do not cut off the final item; finish the sentence completely\n"
    "- Format each line: [HH:MM:SS] [Assigner] → <concrete action>\n"
    "  (use timestamp and speaker name exactly as they appear in the transcript)\n"
    "- If no tasks are assigned to {user_name}, output one line: "
    "'Задач для {user_name} не найдено.'\n\n"
    "Transcript:\n---\n{transcript}\n---"
)

MEETING_DETECTION_SYSTEM = (
    "You are a classifier. Answer only 'yes' or 'no', nothing else."
)

MEETING_DETECTION_PROMPT = (
    "Is the following transcript a recording of a work meeting or team discussion "
    "(e.g. stand-up, planning, retrospective, sync, interview, negotiation)? "
    "Answer only 'yes' or 'no'.\n\n"
    "Transcript (first 2000 chars):\n{text}"
)

MEETING_SUMMARY_SYSTEM = (
    "You are an expert meeting analyst. "
    "Create clear, structured meeting summaries. "
    "IMPORTANT: Always respond in the same language as the transcript."
)

MEETING_SUMMARY_PROMPT_TEMPLATE = (
    "Read the meeting transcript below and produce a structured summary "
    "in the SAME LANGUAGE as the transcript.\n\n"
    "The summary must include exactly these sections:\n\n"
    "1. **Тема встречи**: What the meeting was about (1-2 sentences)\n"
    "2. **Участники и роли**: List each participant with their role or position if identifiable\n"
    "3. **Проблема / задача**: The main problem or goal the meeting addressed\n"
    "4. **Варианты решения**: Approaches, options, or solutions discussed\n"
    "5. **Договорённости**: Concrete action items — who does what and by when (if mentioned)\n"
    "6. **Что касается {user_name}**: separately highlight all mentions, tasks, responsibilities, blockers, approvals, "
    "deadlines, decisions, and follow-ups related to {user_name} or these aliases: {user_aliases}. "
    "If nothing concerns {user_name}, say so explicitly.\n"
    "7. **Краткое описание**: 3-5 sentence overall description of the meeting\n\n"
    "Transcript:\n---\n{transcript}\n---"
)

SPEAKER_FRAME_SYSTEM = (
    "You are identifying a meeting participant from a single video frame. "
    "Be concise and factual."
)

SPEAKER_FRAME_PROMPT_TEMPLATE = (
    "This frame was captured at the moment {speaker_id} is speaking (at {ts}s into the video).\n"
    "Describe this person:\n"
    "1) Any visible name tag, lower-third, or on-screen label — state the exact name if present\n"
    "2) Their appearance (gender, clothing, hair) so they can be distinguished from others\n"
    "3) Their position in the frame (left, center, right)\n"
    "If you cannot see any person, say so briefly."
)

LANGUAGE_CHECK_SYSTEM = (
    "You are a strict language classifier. "
    "Return only one token: ru, uk, or other."
)

LANGUAGE_CHECK_PROMPT_TEMPLATE = (
    "Classify the main language of the following text. "
    "Return only one token: ru, uk, or other.\n\n"
    "Text:\n{text}"
)

RUSSIAN_TRANSLATION_SYSTEM = (
    "You are a professional translator. "
    "Translate the text into natural Russian. "
    "Preserve structure, formatting, bullet lists, and meaning. "
    "Output ONLY the translated text."
)

RUSSIAN_TRANSLATION_PROMPT_TEMPLATE = (
    "Translate the following summary into Russian.\n\n"
    "Summary:\n{text}"
)
