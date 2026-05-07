from __future__ import annotations

from dataclasses import dataclass
import re


MAX_COMPACT_PROMPT_CHARS = 700

LOGO_PROMPT_HINTS = (
    "logo",
    "wordmark",
    "brand-ready",
    "branding",
    "vector logo",
    "transparent background",
    "typography",
    "merchandise",
    "helmet branding",
    "car livery",
)

STRUCTURED_SECTION_RE = re.compile(r"^([A-Za-z][A-Za-z0-9 /&-]{1,36}):\s*(.+)$")
PLACEHOLDER_RE = re.compile(r"\[[^\[\]]+\]")

POSITIVE_OUTPUT_PREFIXES = ("clean", "transparent", "high contrast", "professional", "vector")
NEGATIVE_PHRASE_PREFIXES = ("avoid ", "no ", "not ", "without ")


@dataclass(frozen=True)
class FacePrompt:
    label: str
    role_prompt: str
    position: str


def _normalize_prompt_lines(prompt: str) -> list[str]:
    normalized = prompt.replace("\r\n", "\n").replace("\r", "\n")
    lines: list[str] = []
    for raw_line in normalized.split("\n"):
        line = re.sub(r"^\s*(?:[-*]+|\d+[.)])\s*", "", raw_line).strip()
        line = re.sub(r"\s+", " ", line)
        if line:
            lines.append(line)
    return lines


def _strip_terminal_punctuation(text: str) -> str:
    return text.strip().strip(" .;")


def _has_terminal_punctuation(text: str) -> bool:
    return bool(text) and text[-1] in ".!?"


def _is_logo_prompt(prompt: str) -> bool:
    lowered = prompt.lower()
    return any(hint in lowered for hint in LOGO_PROMPT_HINTS)


def _split_section(line: str) -> tuple[str, str] | None:
    match = STRUCTURED_SECTION_RE.match(line)
    if not match:
        return None
    return match.group(1).strip().lower(), match.group(2).strip()


def _split_phrases(text: str) -> list[str]:
    return [_strip_terminal_punctuation(part) for part in re.split(r",|;", text) if part.strip()]


def _is_negative_phrase(phrase: str) -> bool:
    lowered = phrase.lower()
    return lowered.startswith(NEGATIVE_PHRASE_PREFIXES)


def _positive_output_phrases(text: str) -> str:
    phrases = []
    for phrase in _split_phrases(text):
        lowered = phrase.lower()
        if _is_negative_phrase(phrase):
            continue
        if lowered.startswith(POSITIVE_OUTPUT_PREFIXES) or "background" in lowered:
            phrases.append(phrase)
    return ", ".join(phrases)


def _clean_logo_intro(line: str) -> str:
    replacements = (
        (r"^create\s+(?:a|an)\s+", ""),
        (r"^the logo should feel\s+", "logo feels "),
        (r"^the design should communicate\s+", "communicates "),
    )
    cleaned = _strip_terminal_punctuation(line)
    for pattern, replacement in replacements:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    return cleaned


def _split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def _append_fit(parts: list[str], value: str, max_chars: int) -> None:
    cleaned = _strip_terminal_punctuation(value)
    if not cleaned:
        return

    candidate = ". ".join(parts + [cleaned])
    if len(candidate) <= max_chars:
        parts.append(cleaned)


def _compact_logo_prompt(lines: list[str]) -> str:
    intro_core: list[str] = []
    intro_details: list[str] = []
    sections: dict[str, list[str]] = {}
    output_parts: list[str] = []

    for line in lines:
        section = _split_section(line)
        if section:
            label, value = section
            if label in {"avoid", "negative prompt"}:
                continue
            if label == "output":
                positive_output = _positive_output_phrases(value)
                if positive_output:
                    output_parts.append(positive_output)
                continue
            sections.setdefault(label, []).append(_strip_terminal_punctuation(value))
            continue

        if line.lower().startswith("avoid "):
            continue
        for index, sentence in enumerate(_split_sentences(line)):
            cleaned = _clean_logo_intro(sentence)
            if index == 0 and "logo" in cleaned.lower():
                intro_core.append(cleaned)
            else:
                intro_details.append(cleaned)

    parts: list[str] = []
    for line in intro_core:
        _append_fit(parts, line, MAX_COMPACT_PROMPT_CHARS)

    for label in ("style", "colors"):
        for value in sections.get(label, []):
            _append_fit(parts, value, MAX_COMPACT_PROMPT_CHARS)

    for value in output_parts:
        _append_fit(parts, value, MAX_COMPACT_PROMPT_CHARS)

    for label in ("typography", "icon idea", "icon"):
        for value in sections.get(label, []):
            _append_fit(parts, value, MAX_COMPACT_PROMPT_CHARS)

    for line in intro_details:
        _append_fit(parts, line, MAX_COMPACT_PROMPT_CHARS)

    if not parts:
        return ". ".join(lines)

    return ". ".join(parts)


def _normalize_scene_prompt(scene_prompt: str) -> str:
    lines = _normalize_prompt_lines(scene_prompt)
    if not lines:
        return ""
    if _is_logo_prompt(scene_prompt):
        return _compact_logo_prompt(lines)
    return ". ".join(_strip_terminal_punctuation(line) for line in lines if line)


def compose_generation_prompt(
    scene_prompt: str,
    face_prompts: list[FacePrompt],
    interaction_prompt: str | None = None,
) -> str:
    cleaned_scene = _normalize_scene_prompt(scene_prompt)
    parts: list[str] = [cleaned_scene] if cleaned_scene else []

    if len(face_prompts) == 1:
        subject_prompt = face_prompts[0].role_prompt.strip().rstrip(".")
        if subject_prompt:
            parts.append(f"Main subject: {subject_prompt}")
    elif len(face_prompts) >= 2:
        parts.append("Two distinct people in the same scene")
        for face_prompt in face_prompts[:2]:
            role_text = face_prompt.role_prompt.strip().rstrip(".")
            if role_text:
                parts.append(f"{face_prompt.label} on the {face_prompt.position} side: {role_text}")
            else:
                parts.append(f"{face_prompt.label} is on the {face_prompt.position} side")

        if interaction_prompt and interaction_prompt.strip():
            parts.append(
                "Interaction between Person 1 and Person 2: "
                + interaction_prompt.strip().rstrip(".")
            )

        parts.append(
            "Keep Person 1 and Person 2 as separate people with the correct identity, pose, and action assigned to each"
        )

    resolved = ". ".join(part for part in parts if part).strip()
    if not resolved:
        return ""
    return resolved if _has_terminal_punctuation(resolved) else resolved + "."


def derive_negative_prompt_terms(scene_prompt: str) -> list[str]:
    terms: list[str] = []
    lines = _normalize_prompt_lines(scene_prompt)
    is_logo = _is_logo_prompt(scene_prompt)

    if is_logo:
        terms.extend(
            [
                "photo mockup",
                "3d mockup",
                "busy background",
                "watermark",
                "stray letters",
                "misspelled text",
            ]
        )

    for line in lines:
        section = _split_section(line)
        value = section[1] if section else line
        lowered = value.lower()

        if lowered.startswith("avoid "):
            avoid_sentences = _split_sentences(value[6:])
            terms.append(_strip_terminal_punctuation(avoid_sentences[0] if avoid_sentences else value[6:]))

        for phrase in _split_phrases(value):
            lowered_phrase = phrase.lower()
            if lowered_phrase.startswith("no "):
                terms.append(_strip_terminal_punctuation(phrase[3:]))
            elif lowered_phrase.startswith("without "):
                terms.append(_strip_terminal_punctuation(phrase[8:]))
            elif lowered_phrase.startswith("not "):
                terms.append(_strip_terminal_punctuation(phrase[4:]))

        if "official formula 1 logo" in lowered or "official f1 logo" in lowered:
            terms.extend(
                [
                    "official Formula 1 logo",
                    "copied F1 logo",
                    "trademarked logo imitation",
                ]
            )
        if "extra text" in lowered:
            terms.extend(["extra text", "unrelated words"])

    deduped: list[str] = []
    seen: set[str] = set()
    for term in terms:
        cleaned = _strip_terminal_punctuation(term)
        key = cleaned.lower()
        if cleaned and key not in seen:
            deduped.append(cleaned)
            seen.add(key)
    return deduped


def compose_prompt_warning(scene_prompt: str) -> str | None:
    warnings: list[str] = []
    placeholders = PLACEHOLDER_RE.findall(scene_prompt)
    if placeholders:
        examples = ", ".join(placeholders[:3])
        warnings.append(
            f"Prompt still contains placeholder text ({examples}). Replace placeholders with exact names when you want specific text in the image."
        )

    if _is_logo_prompt(scene_prompt) and re.search(
        r"\b(wordmark|typography|font|named)\b",
        scene_prompt,
        flags=re.IGNORECASE,
    ):
        warnings.append(
            "Logo prompts with text work best with short, exact brand names; Stable Diffusion 1.5 can distort lettering."
        )

    return " ".join(warnings) if warnings else None


def compose_negative_prompt(
    base_negative_prompt: str,
    num_faces: int,
    scene_prompt: str | None = None,
) -> str:
    additions: list[str] = []
    if scene_prompt:
        additions.extend(derive_negative_prompt_terms(scene_prompt))

    if num_faces >= 2:
        additions.extend(
            [
                "merged faces",
                "blended identity",
                "duplicate person",
                "fused bodies",
                "extra arms",
            ]
        )

    parts = [base_negative_prompt.strip().strip(", ")] if base_negative_prompt.strip() else []
    parts.extend(additions)
    return ", ".join(part for part in parts if part)
