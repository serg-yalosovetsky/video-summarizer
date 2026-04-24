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
    "Create detailed, thorough summaries that cover all important information. "
    "Be comprehensive — do not omit details, examples, or context. "
    "IMPORTANT: Always respond in the same language as the input."
)

SHORT_SUMMARY_SYSTEM = (
    "You are a precise TLDR generator. "
    "Always respond in the same language as the transcript. "
    "Output ONLY valid JSON that matches the requested schema."
)

SUMMARY_PROMPT_TEMPLATE = (
    "Read the transcript below and provide a DETAILED and COMPREHENSIVE summary in the SAME LANGUAGE as the transcript.\n\n"
    "Be thorough — cover every topic, argument, and point discussed. "
    "Do not skip examples, numbers, names, or context. The summary should be long and detailed, not brief.\n\n"
    "The summary must include:\n\n"
    "1. **Main Topic**: What this content is about in 1-2 sentences\n"
    "2. **Key Points**: All important points covered (bullet list — include every significant point)\n"
    "3. **Details**: Thorough supporting information — examples, numbers, names, quotes, reasoning, context, "
    "and any other relevant details from the transcript\n"
    "4. **What concerns {user_name}**: highlight all mentions, tasks, requests, blockers, decisions, risks, or follow-ups "
    "that concern {user_name} or these aliases: {user_aliases}. If there is nothing relevant, say so explicitly.\n"
    "5. **Conclusion**: Main takeaway or outcome\n\n"
    "Transcript:\n---\n{transcript}\n---"
)

SHORT_SUMMARY_PROMPT_TEMPLATE = (
    "Read the transcript below and return ONLY valid JSON.\n\n"
    "Use exactly this JSON object shape:\n"
    "{{\n"
    '  "summary": "short 1-2 sentence TLDR",\n'
    '  "problem": "main issue being addressed, or empty string",\n'
    '  "ways_to_solve": ["approach 1", "approach 2"],\n'
    '  "blockers": ["blocker 1", "blocker 2"],\n'
    '  "estimated_resolution": "timeframe or next step, or empty string",\n'
    '  "key_points": ["important point 1", "important point 2"]\n'
    "}}\n\n"
    "Rules:\n"
    "- Use the SAME LANGUAGE as the transcript\n"
    "- If this is a problem/discussion, fill problem, ways_to_solve, blockers, and estimated_resolution when relevant\n"
    "- If this is not a problem/discussion, leave those fields as empty string or empty arrays and use summary/key_points\n"
    "- Every string must be complete; do not cut off the last sentence\n"
    "- Keep it concise and concrete\n"
    "- Do not wrap the JSON in markdown fences\n\n"
    "Transcript:\n---\n{transcript}\n---"
)

PERSONAL_TODO_SYSTEM = (
    "You are a precise meeting task extractor. "
    "Always respond in the same language as the transcript. "
    "Output ONLY valid JSON that matches the requested schema."
)

PERSONAL_TODO_PROMPT_TEMPLATE = (
    "Read the meeting transcript below and extract action items assigned to {user_name}. "
    "{user_name} may also appear as any of these aliases or forms: {user_aliases}. "
    "Treat all of them as the same person.\n\n"

    "Your task is to find ONLY action items that clearly belong to {user_name}.\n\n"

    "Include an item when at least one of these is true:\n"
    "- another participant directly asks, instructs, or assigns {user_name} to do something\n"
    "- a task is addressed to one of {user_name}'s aliases, initials, transliterations, or speaker labels\n"
    "- {user_name} explicitly accepts, volunteers, or commits to doing something\n"
    "- the wording is indirect but still clearly assigns responsibility to {user_name}, for example: "
    "\"please check\", \"you need to\", \"can you fix\", \"let's ask {user_name} to\", "
    "\"{user_name} will do\", \"{user_name} should\", \"{user_name} can take this\"\n\n"

    "Do NOT include an item when any of these is true:\n"
    "- the transcript only mentions {user_name} without giving them a concrete action\n"
    "- the action belongs to someone else\n"
    "- it is only a discussion, suggestion, background detail, status update, or summary of a problem\n"
    "- the task is ambiguous and it is not clear that {user_name} is responsible\n\n"

    "Extraction rules:\n"
    "- Use ONLY information present in the transcript\n"
    "- Do NOT infer tasks from general context if the responsibility is unclear\n"
    "- Prefer precision over recall: if unsure, omit the item\n"
    "- Each action must be one concrete, complete sentence\n"
    "- Preserve the same language as the transcript\n"
    "- Keep the action concise and specific\n"
    "- Deduplicate overlapping items; keep the clearest version\n"
    "- Use the timestamp from the line where the task is stated most explicitly\n"
    "- Use the assigner exactly as it appears in the transcript when possible\n"
    "- If {user_name} assigns a task to themselves or explicitly commits to it, use {user_name} as assigner\n"
    "- If no clear assigner is visible but the responsible line clearly belongs to a speaker label or alias, use that exact visible label/name\n\n"

    "Return ONLY valid JSON in exactly this shape:\n"
    '{{"items": [{{"timestamp": "HH:MM:SS", "assigner": "Name", "action": "Concrete action sentence."}}]}}\n\n'

    "Strict JSON rules:\n"
    '- Use timestamp without brackets, for example "00:12:34"\n'
    '- Use assigner without brackets\n'
    '- Do not add any fields\n'
    '- Do not output markdown fences\n'
    '- Do not output explanations or commentary\n'
    '- If there are no action items for {user_name}, return exactly: {{"items": []}}\n\n'

    "Transcript:\n---\n{transcript}\n---"
)


PERSONAL_TODO_PROMPT_TEMPLATE_FEW_SHOT = (
    "Read the meeting transcript below and extract action items assigned to {user_name}. "
    "{user_name} may also appear as any of these aliases or forms: {user_aliases}. "
    "Treat all of them as the same person.\n\n"

    "Include ONLY tasks that clearly belong to {user_name}. "
    "A task counts if it is directly assigned, clearly requested, or explicitly accepted by {user_name}. "
    "If responsibility is unclear, omit it.\n\n"

    "Positive examples:\n"
    'Transcript line: [00:12:34] [Agnis]: Sergey, please check why Deep-Agent crashes on startup.\n'
    'Output item: {{"timestamp":"00:12:34","assigner":"Agnis","action":"Check why Deep-Agent crashes on startup."}}\n\n'
    'Transcript line: [00:18:02] [SY]: Okay, I will create the test database today.\n'
    'Output item: {{"timestamp":"00:18:02","assigner":"SY","action":"Create the test database today."}}\n\n'
    'Transcript line: [00:22:10] [Agnis]: Can you also get VPN access?\n'
    'Output item: {{"timestamp":"00:22:10","assigner":"Agnis","action":"Get VPN access."}}\n\n'

    "Negative examples:\n"
    'Transcript line: [00:09:11] [Agnis]: Sergey had this issue before.\n'
    'Output: no item\n\n'
    'Transcript line: [00:14:50] [Agnis]: We need to fix this bug.\n'
    'Output: no item if it is not clear who is responsible\n\n'

    "Rules:\n"
    "- Use ONLY information present in the transcript\n"
    "- Ignore tasks assigned to other participants\n"
    "- Ignore mentions of {user_name} that do not contain a concrete action\n"
    "- Each action must be exactly one complete sentence\n"
    "- Preserve the same language as the transcript\n"
    "- Keep actions concrete, specific, and short\n"
    "- Deduplicate overlapping items\n"
    "- Use the timestamp where the task is stated most explicitly\n"
    "- Use the assigner exactly as written in the transcript\n"
    "- If {user_name} self-assigns or explicitly commits, use that visible name/alias as assigner\n\n"

    "Return ONLY valid JSON in exactly this shape:\n"
    '{{"items": [{{"timestamp": "HH:MM:SS", "assigner": "Name", "action": "Concrete action sentence."}}]}}\n'
    '- If no tasks are assigned to {user_name}, return exactly: {{"items": []}}\n'
    "- Do not wrap JSON in markdown fences\n"
    "- Do not output any text except JSON\n\n"

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
    "Create detailed, thorough meeting summaries that cover all decisions, discussions, and details. "
    "IMPORTANT: Always respond in the same language as the transcript."
)

MEETING_SUMMARY_PROMPT_TEMPLATE = (
    "Read the meeting transcript below and produce a DETAILED and COMPREHENSIVE structured summary "
    "in the SAME LANGUAGE as the transcript.\n\n"
    "Be thorough — cover all discussion points, decisions, arguments, and context. Do not omit details.\n\n"
    "The summary must include exactly these sections:\n\n"
    "1. **Тема встречи**: What the meeting was about (1-2 sentences)\n"
    "2. **Участники и роли**: List each participant with their role or position if identifiable\n"
    "3. **Проблема / задача**: The main problem or goal the meeting addressed — include all context and background discussed\n"
    "4. **Варианты решения**: All approaches, options, or solutions discussed — include arguments for/against each\n"
    "5. **Договорённости**: Concrete action items — who does what and by when (if mentioned); include all details\n"
    "6. **Что касается {user_name}**: separately highlight all mentions, tasks, responsibilities, blockers, approvals, "
    "deadlines, decisions, and follow-ups related to {user_name} or these aliases: {user_aliases}. "
    "If nothing concerns {user_name}, say so explicitly.\n"
    "7. **Краткое описание**: 3-5 sentence overall description of the meeting\n\n"
    "Transcript:\n---\n{transcript}\n---"
)

ACTIVE_SPEAKER_DETECT_SYSTEM = "Find the active speaker in a video conference frame."

ACTIVE_SPEAKER_DETECT_PROMPT = (
    "Find the participant with a glowing, highlighted, or bright border. "
    "This can be a video panel border or a glowing ring around a circular avatar.\n\n"
    "Return:\n"
    "- has_active_speaker: true if any highlighted participant is visible\n"
    "- speaker_position: the participant position in a 3x3 grid: top-left, top-center, top-right, "
    "middle-left, middle-center, middle-right, bottom-left, bottom-center, or bottom-right\n"
    "Use null for speaker_position if no highlighted participant is visible."
)

CAPTION_EXTRACT_SYSTEM = "Extract the most recent speaker name from a caption overlay."

CAPTION_EXTRACT_PROMPT = (
    "Look only at the caption or transcription window near the bottom-center of the frame.\n"
    "It is usually a dark box with white text and may contain multiple entries like "
    "\"NAME: spoken text\".\n\n"
    "Return:\n"
    "- has_caption: true if this caption window is visible\n"
    "- last_speaker_name: only the name from the last visible entry\n"
    "Use null for last_speaker_name if the caption window is missing or unreadable."
)

SPEAKER_APPEARANCE_SYSTEM = "Describe a video conference participant at a known position."

SPEAKER_APPEARANCE_PROMPT_TEMPLATE = (
    "Look only at the participant located at {position} in this frame.\n"
    "Describe the visible person briefly using only appearance details such as gender, clothing color, "
    "and hair color. If unclear, leave the description empty."
)

SPEAKER_NAME_SYSTEM = "Read the participant name label at a known position."

SPEAKER_NAME_PROMPT_TEMPLATE = (
    "Look only at the participant located at {position} in this frame.\n"
    "Read the name label shown on that participant panel or avatar. "
    "Return null if no readable name label is visible."
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
