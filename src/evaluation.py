"""Decode reference text and measure literal word matches without model calls."""

from html import unescape
from pathlib import Path
import re
import unicodedata

from jiwer import process_words

MAX_REFERENCE_BYTES = 1024 * 1024
_TIME = r"(?:\d{1,3}:)?\d{1,2}:\d{2}(?:[.,]\d{1,3})?"
_CUE_TIME = re.compile(rf"^{_TIME}\s*-->\s*{_TIME}(?:[ \t]+[^\r\n]*)?$")
_TRANSCRIPT_TIME = re.compile(rf"^\[\s*{_TIME}(?:\s*-->\s*{_TIME})?\s*\]\s*")
_EXPLICIT_SPEAKER = re.compile(r"^(?:SPK[ _-]*\d+|speaker(?:[ _-]*\d+)?)\b[^:：]*[:：]\s*", re.I)
_NAMED_SPEAKER = re.compile(r"^[^\W\d_][\w .'’()\-]{0,70}[:：]\s+")
_ANNOTATION = re.compile(r"^(?:\((?:overlap|backchannel)\)|\[(?:overlap|backchannel)\])\s*", re.I)
_NON_TARGET = re.compile(r"^\[(?:BG|FX)(?:\s|\])", re.I)
_COMMENT = re.compile(r"^#(?:\s|$)")
_CAPTION_TAG = re.compile(
    r"</?(?:b|i|u|font|ruby|rt|v|lang|c(?:\.[\w-]+)*)(?:\s+[^<>]*)?>",
    re.I,
)


def _normalize_words(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(
        " " if unicodedata.category(character).startswith("P") else character
        for character in text
    )
    return " ".join(text.split())


def decode_reference(data: bytes) -> str:
    """Read UTF-8 (optional BOM) or BOM-marked UTF-16, of at most 1 MiB.

    Preserve the original text except its BOM and surrounding whitespace.
    Reject invalid encoding, binary control characters (including NUL), and
    references without letters or digits. Tabs and line endings are allowed.
    Uploaded content stays in memory and is never sent to a model.
    """
    if len(data) > MAX_REFERENCE_BYTES:
        raise ValueError("The reference TXT file must be 1 MiB or smaller.")
    try:
        encoding = "utf-16" if data.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
        text = data.decode(encoding)
    except UnicodeDecodeError as exc:
        raise ValueError("The reference file must contain UTF-8 or BOM-marked UTF-16 text.") from exc
    if any(
        unicodedata.category(character) == "Cc" and character not in "\t\r\n"
        for character in text
    ):
        raise ValueError("The reference TXT file contains binary or unsupported control characters.")
    if not any(character.isalnum() for character in _normalize_words(text)):
        raise ValueError("The reference TXT file must contain words, not only whitespace or punctuation.")
    return text.strip()


def _is_han(character: str) -> bool:
    # Unicode names include supplementary-plane and compatibility ideographs.
    return character == "〇" or unicodedata.name(character, "").startswith(
        ("CJK UNIFIED IDEOGRAPH-", "CJK COMPATIBILITY IDEOGRAPH-")
    )


def _strip_annotations(text: str) -> str:
    while (match := _ANNOTATION.match(text)) is not None:
        text = text[match.end():].lstrip()
    return text


def _transcript_segments(lines: list[str], *, named_speakers: bool) -> tuple[list[str], int]:
    segments = []
    current = None
    removed = 0
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            current = None
            continue
        if _COMMENT.match(line) or _NON_TARGET.match(line):
            removed += 1
            current = None
            continue
        timed = _TRANSCRIPT_TIME.match(line)
        if timed:
            line = line[timed.end():]
        line = _strip_annotations(line)
        speaker = _EXPLICIT_SPEAKER.match(line)
        if speaker is None and (timed or named_speakers):
            speaker = _NAMED_SPEAKER.match(line)
        if speaker:
            line = line[speaker.end():]
        line = _strip_annotations(line).strip()
        if not line:
            removed += 1
            continue
        if timed or speaker or current is None:
            current = []
            segments.append(current)
        current.append(line)
    return [" ".join(segment) for segment in segments], removed


def _caption_segments(lines: list[str]) -> tuple[list[str], int]:
    segments = []
    current = None
    removed = 0
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            current = None
            index += 1
            continue
        if current is None and (
            re.match(r"^(?:WEBVTT|NOTE)(?:\s|$)", line) or line in {"STYLE", "REGION"}
        ):
            while index < len(lines) and lines[index].strip():
                removed += 1
                index += 1
            continue
        if _CUE_TIME.fullmatch(line):
            removed += 1
            current = []
            segments.append(current)
            index += 1
            continue
        # Only a line immediately before a cue time, at a block boundary, is an ID.
        if current is None and index + 1 < len(lines) and _CUE_TIME.fullmatch(lines[index + 1].strip()):
            removed += 1
            index += 1
            continue
        if _COMMENT.match(line) or _NON_TARGET.match(line):
            removed += 1
            index += 1
            continue
        if current is None:
            raise ValueError("The caption file contains text outside a timed cue.")
        line = _CAPTION_TAG.sub("", line)
        line = re.sub(rf"<{_TIME}>", "", line)
        line = re.sub(r"\{\\an[1-9]\}", "", line)
        line = _strip_annotations(unescape(line)).strip()
        line = _strip_annotations(_EXPLICIT_SPEAKER.sub("", line, count=1)).strip()
        if line:
            current.append(line)
        else:
            removed += 1
        index += 1
    return [" ".join(segment) for segment in segments if segment], removed


def parse_reference(data: bytes, filename: str = "reference.txt", *, format: str = "auto") -> dict:
    """Extract spoken text from plain TXT, annotated transcripts, SRT, or VTT.

    Auto recognizes timed cues, bracketed transcript times, explicit SPK/speaker
    labels, and # / BG / FX metadata. ``transcript`` forces this cleanup for TXT;
    ``plain`` preserves all literal content. Only leading known overlap/backchannel
    tags are removed; dialogue colons, parentheses, numbers, and order are retained.

    The result contains text, original_text, a human-readable format, segment_count,
    removed_lines, and has_cjk (Han ideographs). Segments are turns, caption cues,
    or nonempty plain-text lines. Removed lines count nonempty physical lines fully
    discarded as metadata, not blank lines or prefixes removed from spoken lines.
    No uploaded content is written to disk or sent to a model.
    """
    if format not in {"auto", "plain", "transcript"}:
        raise ValueError("Reference format must be auto, plain, or transcript.")
    original = decode_reference(data)
    lines = original.splitlines()
    suffix = Path(filename).suffix.lower()
    is_vtt = bool(re.match(r"^WEBVTT(?:\s|$)", original)) or suffix == ".vtt"
    is_caption = is_vtt or suffix == ".srt" or any(_CUE_TIME.fullmatch(line.strip()) for line in lines)
    is_transcript = format == "transcript" or any(
        pattern.match(line.strip())
        for line in lines
        for pattern in (_TRANSCRIPT_TIME, _EXPLICIT_SPEAKER, _NON_TARGET, _COMMENT, _ANNOTATION)
    )
    if format == "plain" or not (is_caption or is_transcript):
        text = original
        segments = [line for line in lines if line.strip()]
        label, removed = "Plain text", 0
    elif is_caption:
        segments, removed = _caption_segments(lines)
        text = "\n".join(segments)
        label = "WebVTT captions" if is_vtt else "SubRip captions"
    else:
        segments, removed = _transcript_segments(lines, named_speakers=format == "transcript")
        text = "\n".join(segments)
        label = "Timestamped transcript" if any(_TRANSCRIPT_TIME.match(line.strip()) for line in lines) else "Annotated transcript"
    if not any(character.isalnum() for character in _normalize_words(text)):
        raise ValueError("The reference contains no spoken text after removing metadata.")
    return {
        "text": text,
        "original_text": original,
        "format": label,
        "segment_count": len(segments),
        "removed_lines": removed,
        "has_cjk": any(_is_han(character) for character in text),
    }


def word_match_score(reference: str, hypothesis: str) -> dict[str, float | int]:
    """Return 1 - word MER and alignment counts for two complete text strings.

    NFKC normalization, case folding, punctuation-to-space replacement, and
    whitespace splitting define the words. Apostrophes and hyphens split words;
    accents and digits remain. No language-specific segmentation is performed.
    This measures literal word overlap, not meaning or translation quality.

    wMER = (substitutions + deletions + insertions) / (hits + substitutions +
    deletions + insertions), so both wMER and score stay between zero and one.
    Two empty normalized texts score one; only one empty text scores zero.
    Join completed segments with spaces before calling; missing or failed
    hypotheses must be withheld by the caller, not converted to empty strings.
    """
    return _match_score(_normalize_words(reference), _normalize_words(hypothesis))


def mixed_match_score(reference: str, hypothesis: str) -> dict[str, float | int]:
    """Score individual Han characters and whitespace-separated non-Han words.

    Apply the same Unicode, case, and punctuation normalization as word_match_score;
    split Han even when adjacent to English or numbers. The score is 1 minus token
    match error rate, with denominator H + S + D + I, not conventional mixed error
    rate's reference-length denominator. For API compatibility, ``wmer`` holds this
    match error rate and the ``*_words`` counts hold mixed token counts.
    """
    def tokenize(text):
        return " ".join("".join(
            f" {character} " if _is_han(character) else character
            for character in _normalize_words(text)
        ).split())

    return _match_score(tokenize(reference), tokenize(hypothesis))


def _match_score(reference: str, hypothesis: str) -> dict[str, float | int]:
    result = process_words(reference, hypothesis)
    return {
        "score": 1.0 - result.mer,
        "wmer": result.mer,
        "hits": result.hits,
        "substitutions": result.substitutions,
        "deletions": result.deletions,
        "insertions": result.insertions,
        "reference_words": result.hits + result.substitutions + result.deletions,
        "hypothesis_words": result.hits + result.substitutions + result.insertions,
    }
