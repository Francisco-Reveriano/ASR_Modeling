"""Small, presentation-only helpers for the Streamlit conversation view."""

from html import escape


def render_transcript(
    texts: list[str],
    translations: list[str | None] | None = None,
    errors: list[str | None] | None = None,
    tencent_translations: list[str | None] | None = None,
    tencent_errors: list[str | None] | None = None,
    *,
    reference_text: str | None = None,
    reference_title: str = "Reference transcript",
    slow_lane: dict | None = None,
    speakers: list[str | None] | None = None,
) -> str:
    """Pair each numbered source segment with both English translations.

    Numbers belong to segments, so wrapping a long sentence never changes them.
    Model output is escaped before being included in the HTML; spoken markup is
    displayed as text. Missing translations remain visibly pending, and failed
    translations leave the source intact. All columns use the same row index,
    even when one translation provider finishes before the other.

    An optional reference has its own numbered lines in a separate pane. File
    lines and detected speech segments need not correspond, so they are never
    paired by position or truncated to the shorter list.

    The optional Astra column selects either the latest visible translation or
    the sealed authoritative text. Status badges distinguish drafts, reviews,
    and fallbacks; an unsealed authoritative row never displays a draft.
    """
    extra_class = " has-slow-lane" if slow_lane is not None else ""
    headings = (
        f'<div class="conversation-columns{extra_class}" aria-hidden="true"><span>#</span>'
        '<span>Original transcript</span><span>English <small>OpenAI</small></span>'
        '<span>English <small class="tencent-heading">Tencent · Local</small></span>'
        + ('<span>English <small class="astra-heading">Astra · Correction</small></span>'
           if slow_lane is not None else "")
        + '</div>'
    )
    if not texts:
        conversation = headings + (
            '<div class="transcript-empty">'
            '<div class="empty-icon" aria-hidden="true">'
            '<svg viewBox="0 0 32 32" fill="none" stroke="currentColor" '
            'stroke-width="1.6"><path d="M7 5h18v22H7zM11 11h10M11 16h10M11 21h6"/>'
            '</svg></div>'
            '<h3>Your conversation starts here</h3>'
            '<p>Record your voice or upload a WAV file.<br>'
            'Compare the original with English from OpenAI and local Tencent.</p></div>'
        )
        return _with_reference(conversation, reference_text, reference_title, slow_lane is not None)

    providers = (
        ("openai", "OpenAI · English", translations or [], errors or []),
        ("tencent", "Tencent · English", tencent_translations or [], tencent_errors or []),
    )
    rows = []
    for index, text in enumerate(texts):
        cells = []
        segment_attribute = ""
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
        if slow_lane is not None:
            fast_text = translations[index] if translations and index < len(translations) else None
            entry = _slow_lane_entry(slow_lane, index, fast_text)
            if entry["segment_id"] is not None:
                segment_attribute = f' data-segment-id="{escape(str(entry["segment_id"]))}"'
            if entry["displayed"] is None:
                content = f'<span class="translation-pending">{entry["placeholder"]}</span>'
            else:
                language = '' if entry["source_fallback"] else ' lang="en"'
                content = f'<span class="translation-text astra-text"{language}>{escape(entry["displayed"])}</span>'
            cells.append(
                '<div class="translation-cell astra-cell">'
                '<span class="cell-label">Astra · English</span>'
                f'<span class="slow-badge {entry["tone"]}">{entry["label"]}</span>'
                f'{content}</div>'
            )
        speaker_label = ""
        if speakers is not None:
            speaker = speakers[index] if index < len(speakers) else None
            speaker_label = f'<span class="speaker-label">{escape(speaker or "Speaker unknown")}</span>'
        rows.append(
            f'<li class="transcript-row"{segment_attribute}>'
            f'<span class="line-number" aria-hidden="true">{index + 1:02d}</span>'
            '<div class="source-cell"><span class="cell-label">Original</span>'
            f'{speaker_label}<span class="segment-text">{escape(text)}</span></div>'
            f'{"".join(cells)}</li>'
        )
    conversation = (
        headings + f'<ol class="transcript-lines{extra_class}" role="list" '
        f'aria-label="Transcript segments with English translations">{"".join(rows)}</ol>'
    )
    return _with_reference(conversation, reference_text, reference_title, slow_lane is not None)


def _slow_lane_entry(snapshot: dict, index: int, fast_text: str | None = None) -> dict:
    """Resolve one display/export row without altering the worker's snapshot."""
    def value(key):
        values = snapshot.get(key) or []
        return values[index] if index < len(values) else None

    status = value("statuses") or "waiting"
    live = value("translations")
    if live is None and fast_text is not None:
        live = fast_text
        if status in {"corrected", "confirmed"}:
            status = "draft"
    authoritative = value("authoritative")
    segment = value("segments") or {}
    screen_version, stored_version = segment.get("screen_version"), segment.get("version")
    newer_final = (
        isinstance(screen_version, int) and isinstance(stored_version, int)
        and screen_version < stored_version and authoritative is not None
    )
    final_view = snapshot.get("view") == "authoritative"
    displayed = authoritative if final_view else live
    labels = {
        "waiting": ("Draft", "draft"),
        "draft": ("Draft", "draft"),
        "corrected": ("Corrected", "reviewed"),
        "confirmed": ("Confirmed", "reviewed"),
        "timeout": ("Draft · Timed out", "fallback"),
        "paused": ("Draft · Paused", "fallback"),
        "failed": ("Draft · Review failed", "fallback"),
    }
    label, tone = labels.get(status, ("Draft", "draft"))
    if final_view:
        if authoritative is None:
            label, tone = "Pending final", "waiting"
        elif live is not None and authoritative != live:
            label, tone = ("Corrected · Final", "reviewed") if newer_final else ("Final · Stored version", "fallback")
        elif status in {"corrected", "confirmed"}:
            label = f"{label} · Final"
        else:
            label, tone = {
                "timeout": ("Final fallback · Timed out", "fallback"),
                "paused": ("Final fallback · Paused", "fallback"),
                "failed": ("Final fallback · Review failed", "fallback"),
            }.get(status, ("Final fallback", "fallback"))
    elif newer_final and live != authoritative:
        label, tone = "Draft · Final correction available", "draft"
    elif live is None:
        label, tone = {
            "timeout": ("Timed out", "fallback"),
            "paused": ("Paused", "fallback"),
            "failed": ("Review failed", "fallback"),
        }.get(status, ("Waiting", "waiting"))
    fallback_flag = segment.get("source_fallback") if final_view else segment.get(
        "screen_source_fallback", segment.get("source_fallback"),
    )
    source_fallback = bool(fallback_flag) and displayed is not None and (
        displayed == segment.get("source_text", displayed)
    )
    if source_fallback:
        label, tone = ("Final · Source fallback" if final_view else "Source fallback"), "fallback"
    return {
        "displayed": displayed, "live": live, "authoritative": authoritative,
        "label": label, "tone": tone, "segment_id": segment.get("segment_id"),
        "source_fallback": source_fallback,
        "placeholder": "Awaiting final text…" if final_view else "Waiting for fast translation…",
    }


def _with_reference(conversation: str, text: str | None, title: str, slow_lane: bool = False) -> str:
    """Place the complete reference beside the model output without row pairing."""
    if text is None:
        return conversation
    rows = []
    for index, line in enumerate(text.splitlines(), start=1):
        rows.append(
            '<li class="reference-row">'
            f'<span class="reference-number" aria-hidden="true">R{index:02d}</span>'
            f'<span class="reference-text">{escape(line)}</span></li>'
        )
    return (
        f'<div class="conversation-reference{" has-slow-lane" if slow_lane else ""}">'
        f'<div class="model-conversation">{conversation}</div>'
        f'<section class="reference-pane" aria-label="{escape(title)}">'
        f'<div class="reference-heading">{escape(title)}</div>'
        '<ol class="reference-lines" role="list" aria-label="Reference file lines">'
        f'{"".join(rows)}</ol></section></div>'
        '<p class="reference-note">Reference lines follow the file; '
        'speech segments follow pauses.</p>'
    )


def export_conversation(
    texts, translations, errors, tencent_translations=None, tencent_errors=None,
    *, slow_lane=None, speakers=None,
) -> str:
    """Export the source and both providers, including unfinished translations."""
    providers = (
        ("OpenAI", translations or [], errors or []),
        ("Tencent local", tencent_translations or [], tencent_errors or []),
    )
    rows = []
    for index, text in enumerate(texts):
        lines = [f"{index + 1:02d}", f"Original: {text}"]
        if speakers is not None:
            speaker = speakers[index] if index < len(speakers) else None
            lines.append(f"Speaker: {speaker or 'Unknown'}")
        for provider, values, failures in providers:
            english = values[index] if index < len(values) else None
            error = failures[index] if index < len(failures) else None
            if english is None:
                english = "[Translation unavailable]" if error else "[Translation pending]"
            lines.append(f"English ({provider}): {english}")
        if slow_lane is not None:
            fast_text = translations[index] if translations and index < len(translations) else None
            entry = _slow_lane_entry(slow_lane, index, fast_text)
            if entry["segment_id"] is not None:
                lines.append(f"Segment ID: {entry['segment_id']}")
            displayed = entry["displayed"]
            lines.extend([
                f"English (Astra correction): {displayed if displayed is not None else '[' + entry['placeholder'] + ']'}",
                f"Astra status: {entry['label']}",
            ])
            authoritative = entry["authoritative"]
            if authoritative != displayed or authoritative is None:
                lines.append(f"Astra authoritative: {authoritative if authoritative is not None else '[Not sealed]'}")
            if entry["live"] != displayed and entry["live"] is not None:
                lines.append(f"Astra latest visible version: {entry['live']}")
        rows.append("\n".join(lines))
    return "\n\n".join(rows)
