"""Small, presentation-only helpers for the Streamlit conversation view."""

from html import escape


def render_transcript(
    texts: list[str],
    translations: list[str | None] | None = None,
    errors: list[str | None] | None = None,
    tencent_translations: list[str | None] | None = None,
    tencent_errors: list[str | None] | None = None,
) -> str:
    """Pair each numbered source segment with both English translations.

    Numbers belong to segments, so wrapping a long sentence never changes them.
    Model output is escaped before being included in the HTML; spoken markup is
    displayed as text. Missing translations remain visibly pending, and failed
    translations leave the source intact. All columns use the same row index,
    even when one translation provider finishes before the other.
    """
    headings = (
        '<div class="conversation-columns" aria-hidden="true"><span>#</span>'
        '<span>Original transcript</span><span>English <small>OpenAI</small></span>'
        '<span>English <small>Tencent · Local</small></span></div>'
    )
    if not texts:
        return headings + (
            '<div class="transcript-empty">'
            '<div class="empty-icon" aria-hidden="true">'
            '<svg viewBox="0 0 32 32" fill="none" stroke="currentColor" '
            'stroke-width="1.6"><path d="M7 5h18v22H7zM11 11h10M11 16h10M11 21h6"/>'
            '</svg></div>'
            '<h3>Your conversation starts here</h3>'
            '<p>Record your voice or upload a WAV file.<br>'
            'Compare the original with English from OpenAI and local Tencent.</p></div>'
        )

    providers = (
        ("openai", "OpenAI · English", translations or [], errors or []),
        ("tencent", "Tencent · English", tencent_translations or [], tencent_errors or []),
    )
    rows = []
    for index, text in enumerate(texts):
        cells = []
        for provider, label, values, failures in providers:
            english = values[index] if index < len(values) else None
            error = failures[index] if index < len(failures) else None
            if english is not None:
                translated = (
                    f'<span class="translation-text {provider}-text" lang="en">'
                    f'{escape(english)}</span>'
                )
            elif error:
                translated = '<span class="translation-error">Translation unavailable</span>'
            else:
                translated = '<span class="translation-pending">Translating…</span>'
            cells.append(
                f'<div class="translation-cell {provider}-cell">'
                f'<span class="cell-label">{label}</span>{translated}</div>'
            )
        rows.append(
            '<li class="transcript-row">'
            f'<span class="line-number" aria-hidden="true">{index + 1:02d}</span>'
            '<div class="source-cell"><span class="cell-label">Original</span>'
            f'<span class="segment-text">{escape(text)}</span></div>'
            f'{"".join(cells)}</li>'
        )
    return (
        headings + '<ol class="transcript-lines" role="list" '
        f'aria-label="Transcript segments with English translations">{"".join(rows)}</ol>'
    )


def export_conversation(
    texts, translations, errors, tencent_translations=None, tencent_errors=None,
) -> str:
    """Export the source and both providers, including unfinished translations."""
    providers = (
        ("OpenAI", translations or [], errors or []),
        ("Tencent local", tencent_translations or [], tencent_errors or []),
    )
    rows = []
    for index, text in enumerate(texts):
        lines = [f"{index + 1:02d}", f"Original: {text}"]
        for provider, values, failures in providers:
            english = values[index] if index < len(values) else None
            error = failures[index] if index < len(failures) else None
            if english is None:
                english = "[Translation unavailable]" if error else "[Translation pending]"
            lines.append(f"English ({provider}): {english}")
        rows.append("\n".join(lines))
    return "\n\n".join(rows)
