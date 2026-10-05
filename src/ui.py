"""Presentation helpers for the original transcript and selected English model."""

from html import escape

from src.subtitle_exports import iter_conversation_records


_PLACEHOLDERS = {
    "pending": ("translation-pending", "Translating…"),
    "failed": ("translation-error", "Translation unavailable"),
    "filtered": ("translation-filtered", "Background speech filtered"),
}


def render_transcript(
    texts: list[str],
    translations: list[str | None] | None = None,
    errors: list[str | None] | None = None,
    *,
    filtered: list[bool] | None = None,
    timings: list[dict | None] | None = None,
    profile_label: str = "Fast English",
    model: str | None = None,
    reference_text: str | None = None,
    reference_segments: list[str] | None = None,
    reference_title: str = "Reference transcript",
) -> str:
    """Pair numbered source segments with the selected model's English output.

    All supplied text is escaped. Pending, failed, or filtered rows keep their
    original source and an English status placeholder. References remain in a
    separate, independently numbered pane; they are never aligned by row count.
    """
    label = escape(profile_label)
    headings = (
        '<div class="conversation-columns" aria-hidden="true"><span>#</span>'
        '<span>Original transcript</span>'
        f'<span>English <small>{label}</small></span></div>'
    )
    rows = []
    for row in iter_conversation_records(texts, translations, errors, filtered=filtered,
                                         timings=timings, profile_label=profile_label, model=model):
        if row["status"] == "translated":
            content = f'<span class="translation-text" lang="en">{escape(row["translation"])}</span>'
        else:
            state_class, placeholder = _PLACEHOLDERS[row["status"]]
            content = f'<span class="{state_class}">{placeholder}</span>'
        rows.append(
            f'<li class="transcript-row" data-segment-id="{row["segment_id"]}" data-status="{row["status"]}">'
            f'<span class="line-number" aria-hidden="true">{row["segment_id"]:02d}</span>'
            '<div class="source-cell"><span class="cell-label">Original</span>'
            f'<span class="segment-text">{escape(row["source_text"])}</span></div>'
            f'<div class="translation-cell"><span class="cell-label">English · {label}</span>'
            f'{content}</div></li>'
        )
    if rows:
        conversation = (
            headings + '<ol class="transcript-lines" role="list" '
            f'aria-label="Transcript segments with English translations">{"".join(rows)}</ol>'
        )
    else:
        conversation = headings + (
            '<div class="transcript-empty">'
            '<div class="empty-icon" aria-hidden="true">'
            '<svg viewBox="0 0 32 32" fill="none" stroke="currentColor" '
            'stroke-width="1.6"><path d="M7 5h18v22H7zM11 11h10M11 16h10M11 21h6"/>'
            '</svg></div><h3>Your conversation starts here</h3>'
            '<p>Record your voice or upload a WAV file.<br>'
            'Read the original alongside its English translation.</p></div>'
        )
    references = reference_segments
    if references is None and reference_text is not None:
        references = reference_text.splitlines()
    return _with_reference(conversation, references, reference_title)


def _with_reference(conversation, lines, title):
    if lines is None:
        return conversation
    rows = [
        '<li class="reference-row">'
        f'<span class="reference-number" aria-hidden="true">R{index:02d}</span>'
        f'<span class="reference-text">{escape(line)}</span></li>'
        for index, line in enumerate(lines, start=1)
    ]
    return (
        '<div class="conversation-reference">'
        f'<div class="model-conversation">{conversation}</div>'
        f'<section class="reference-pane" aria-label="{escape(title)}">'
        f'<div class="reference-heading">{escape(title)}</div>'
        '<ol class="reference-lines" role="list" aria-label="Reference file lines">'
        f'{"".join(rows)}</ol></section></div>'
        '<p class="reference-note">Reference lines follow the file; '
        'speech segments follow pauses.</p>'
    )


def export_conversation(
    texts, translations=None, errors=None, *, filtered=None, timings=None,
    profile_label="Fast English", model=None,
) -> str:
    """Export sources and selected English with explicit unfinished row states."""
    placeholders = {
        "pending": "[Translation pending]",
        "failed": "[Translation unavailable]",
        "filtered": "[Background speech filtered]",
    }
    rows = []
    for row in iter_conversation_records(texts, translations, errors, filtered=filtered,
                                         timings=timings, profile_label=profile_label, model=model):
        english = row["translation"] if row["status"] == "translated" else placeholders[row["status"]]
        lines = [f'{row["segment_id"]:02d}', f'Original: {row["source_text"]}',
                 f'English ({profile_label}): {english}', f'Status: {row["status"].capitalize()}']
        if model:
            lines.append(f"Model: {model}")
        if row["start_s"] is not None:
            lines.append(f'Time: {row["start_s"]:.3f}–{row["end_s"]:.3f} s')
        rows.append("\n".join(lines))
    return "\n\n".join(rows)
