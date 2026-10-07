"""Small, presentation-only helpers for the Streamlit conversation view."""

from html import escape

from src.translation_validation import contains_cjk

_PROVIDERS = ("openai", "tencent", "astra")
_CONVERSATION_REVIEW_STATUSES = {"waiting", "reviewing", "corrected", "confirmed", "failed", "blocked", "disabled"}


def _selected_providers(providers, slow_lane):
    """Keep the usual provider order while selecting only available columns."""
    if (not isinstance(providers, (tuple, list))
        or any(not isinstance(provider, str) or provider not in _PROVIDERS for provider in providers)
        or len(set(providers)) != len(providers)):
        raise ValueError("providers must be unique selections from openai, tencent, and astra.")
    return tuple(provider for provider in _PROVIDERS
                 if provider in providers and (provider != "astra" or slow_lane is not None))


def _translation_unavailable(text: str | None, *, source_fallback: bool = False) -> bool:
    """Reject untranslated targets, including snapshots created before validation."""
    return text is not None and (source_fallback or not text.strip() or contains_cjk(text))


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
    providers: tuple[str, ...] = _PROVIDERS,
    transcription_label: str | None = None,
) -> str:
    """Pair each numbered source segment with the selected English translations.

    Numbers belong to segments, so wrapping a long sentence never changes them.
    Model output is escaped before being included in the HTML; spoken markup is
    displayed as text. Missing translations remain visibly pending, and failed
    translations leave the source intact. All columns use the same row index,
    even when one translation provider finishes before the other.

    An optional reference has its own numbered lines in a separate pane. File
    lines and detected speech segments need not correspond, so they are never
    paired by position or truncated to the shorter list.

    The optional Astra column replaces drafts with accepted reviewed text.
    The final view shows only sealed results. Status badges distinguish drafts,
    reviews, and fallbacks; an unsealed final row never displays a draft.
    """
    selected = _selected_providers(providers, slow_lane)
    has_astra = "astra" in selected
    extra_class = (" has-slow-lane" if has_astra else "") + f" provider-columns-{len(selected)}"
    source_badge = f' <small>{escape(transcription_label)}</small>' if transcription_label else ""
    source_label = f"Original · {escape(transcription_label)}" if transcription_label else "Original"
    headings = (
        f'<div class="conversation-columns{extra_class}" aria-hidden="true"><span>#</span>'
        f'<span>Original transcript{source_badge}</span>'
        + ('<span>English <small>OpenAI</small></span>' if "openai" in selected else "")
        + ('<span>English <small class="tencent-heading">Tencent · Local</small></span>'
           if "tencent" in selected else "")
        + ('<span>English <small class="astra-heading">Astra · Correction</small></span>'
           if has_astra else "")
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
            'Read the original alongside the selected English translations.</p></div>'
        )
        return _with_reference(conversation, reference_text, reference_title, len(selected) == 3)

    provider_values = (
        ("openai", "OpenAI · English", translations or [], errors or []),
        ("tencent", "Tencent · English", tencent_translations or [], tencent_errors or []),
    )
    rows = []
    for index, text in enumerate(texts):
        cells = []
        segment_attribute = ""
        for provider, label, values, failures in provider_values:
            if provider not in selected:
                continue
            english = values[index] if index < len(values) else None
            error = failures[index] if index < len(failures) else None
            if english is not None and not _translation_unavailable(english):
                translated = (
                    f'<span class="translation-text {provider}-text" lang="en">'
                    f'{escape(english)}</span>'
                )
            elif error or english is not None:
                translated = '<span class="translation-error">Translation unavailable</span>'
            else:
                translated = '<span class="translation-pending">Translating…</span>'
            cells.append(
                f'<div class="translation-cell {provider}-cell">'
                f'<span class="cell-label">{label}</span>{translated}</div>'
            )
        if has_astra:
            fast_text = translations[index] if translations and index < len(translations) else None
            entry = _slow_lane_entry(slow_lane, index, fast_text)
            if entry["segment_id"] is not None:
                segment_attribute = f' data-segment-id="{escape(str(entry["segment_id"]))}"'
            if entry["displayed"] is None:
                state_class = "translation-error" if entry["unavailable"] else "translation-pending"
                content = f'<span class="{state_class}">{entry["placeholder"]}</span>'
            else:
                content = f'<span class="translation-text astra-text" lang="en">{escape(entry["displayed"])}</span>'
            corrected_class = " astra-corrected" if entry["corrected"] else ""
            review_note = ""
            if entry["review_note"]:
                review_note = (
                    f'<span class="context-review-note {entry["review_tone"]}" '
                    f'title="{escape(entry["review_title"])}">{entry["review_note"]}</span>'
                )
            cells.append(
                f'<div class="translation-cell astra-cell{corrected_class}">'
                '<span class="cell-label">Astra · English</span>'
                f'<span class="slow-badge {entry["tone"]}">{entry["label"]}</span>'
                f'{review_note}'
                f'{content}</div>'
            )
        speaker_label = ""
        if speakers is not None:
            speaker = speakers[index] if index < len(speakers) else None
            speaker_label = f'<span class="speaker-label">{escape(speaker or "Speaker unknown")}</span>'
        rows.append(
            f'<li class="transcript-row"{segment_attribute}>'
            f'<span class="line-number" aria-hidden="true">{index + 1:02d}</span>'
            f'<div class="source-cell"><span class="cell-label">{source_label}</span>'
            f'{speaker_label}<span class="segment-text">{escape(text)}</span></div>'
            f'{"".join(cells)}</li>'
        )
    conversation = (
        headings + f'<ol class="transcript-lines{extra_class}" role="list" '
        f'aria-label="Transcript segments with English translations">{"".join(rows)}</ol>'
    )
    return _with_reference(conversation, reference_text, reference_title, len(selected) == 3)


def _slow_lane_entry(snapshot: dict, index: int, fast_text: str | None = None) -> dict:
    """Resolve one display/export row without altering the worker's snapshot."""
    def value(key):
        values = snapshot.get(key) or []
        return values[index] if index < len(values) else None

    status = value("statuses") or "waiting"
    authoritative = value("authoritative")
    live = value("translations")
    segment = value("segments") or {}
    review_stage = segment.get("conversation_review_status")
    if not isinstance(review_stage, str) or review_stage not in _CONVERSATION_REVIEW_STATUSES:
        review_stage = None
    context_unfinished = review_stage in {"waiting", "reviewing", "failed", "blocked"}
    live_fallback = bool(segment.get("screen_source_fallback", segment.get("source_fallback")))
    if live is None and fast_text is not None:
        live = fast_text
        live_fallback = False
        if status in {"corrected", "confirmed"} and authoritative is None:
            status = "draft"
    reviewed = status in {"corrected", "confirmed"} and authoritative is not None
    final_view = snapshot.get("view") == "authoritative"
    failed_final = (
        final_view and authoritative is None and status == "failed"
        and bool(segment.get("error", True))
    )
    # Older snapshots can still have a frozen screen pointer. Publish the
    # accepted review in both views instead of requiring a manual view switch.
    displayed = authoritative if final_view or reviewed else live
    labels = {
        "waiting": ("Draft", "draft"),
        "draft": ("Draft", "draft"),
        "corrected": ("Corrected", "reviewed"),
        "confirmed": ("Confirmed", "reviewed"),
        "filtered": ("Background filtered", "draft"),
        "timeout": ("Draft · Timed out", "fallback"),
        "paused": ("Draft · Paused", "fallback"),
        "failed": ("Draft · Review failed", "fallback"),
    }
    label, tone = labels.get(status, ("Draft", "draft"))
    if final_view:
        if authoritative is None:
            label, tone = ("Review failed", "fallback") if failed_final else ("Pending final", "waiting")
        elif status in {"corrected", "confirmed", "filtered"}:
            if not context_unfinished:
                label = f"{label} · Final"
        elif live is not None and authoritative != live:
            label, tone = "Final · Stored version", "fallback"
        else:
            label, tone = {
                "timeout": ("Final fallback · Timed out", "fallback"),
                "paused": ("Final fallback · Paused", "fallback"),
                "failed": ("Final fallback · Review failed", "fallback"),
            }.get(status, ("Final fallback", "fallback"))
    elif displayed is None:
        label, tone = {
            "timeout": ("Timed out", "fallback"),
            "paused": ("Paused", "fallback"),
            "failed": ("Review failed", "fallback"),
        }.get(status, ("Waiting", "waiting"))
    source_fallback = bool(segment.get("source_fallback")) if final_view or reviewed else live_fallback
    unavailable = _translation_unavailable(displayed, source_fallback=source_fallback)
    placeholder = "Awaiting final text…" if final_view else "Waiting for fast translation…"
    if failed_final:
        placeholder = "Retry correction to finish this translation."
    if unavailable:
        displayed = None
        placeholder = "Translation unavailable"
        label = "Translation unavailable · Final" if final_view and not context_unfinished else "Translation unavailable"
        tone = "fallback"
    review_note, review_tone, review_title = "", "waiting", ""
    if review_stage in {"corrected", "confirmed"}:
        if reviewed and not unavailable:
            review_note, review_tone = "Reviewed in context", "complete"
            review_title = ("The conversation review corrected this translation."
                            if review_stage == "corrected"
                            else "The conversation review confirmed this translation without changing it.")
    elif context_unfinished:
        review_note, review_tone, review_title = {
            "waiting": ("Context review pending", "waiting", "The conversation review has not started yet."),
            "reviewing": ("Reviewing in context…", "waiting", "The accepted first review remains visible during conversation review."),
            "failed": ("Context review failed", "fallback", "The accepted first review is retained. Retry the conversation review."),
            "blocked": ("Waiting for earlier review", "waiting", "Conversation review proceeds in order. Earlier pending or failed reviews must finish before this row can be reviewed."),
        }[review_stage]
    # Supplementary TXT versions are English fields too. Keep the raw source
    # and review history in the worker snapshot, but never export them as English.
    if _translation_unavailable(live, source_fallback=live_fallback):
        live = "[Translation unavailable]"
    if _translation_unavailable(authoritative, source_fallback=bool(segment.get("source_fallback"))):
        authoritative = "[Translation unavailable]"
    first_review_status = segment.get("first_pass_status", status)
    if not isinstance(first_review_status, str) or first_review_status not in labels:
        first_review_status = status
    return {
        "displayed": displayed, "live": live, "authoritative": authoritative,
        "label": label, "tone": tone, "segment_id": segment.get("segment_id"),
        "unavailable": unavailable or failed_final, "placeholder": placeholder,
        "corrected": status == "corrected" and reviewed and not unavailable,
        "first_review_status": first_review_status,
        "conversation_review_status": review_stage,
        "review_note": review_note, "review_tone": review_tone, "review_title": review_title,
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
    *, slow_lane=None, speakers=None, providers=_PROVIDERS, transcription_label=None,
) -> str:
    """Export the source and selected providers, including unfinished reviews."""
    selected = _selected_providers(providers, slow_lane)
    provider_values = (
        ("openai", "OpenAI", translations or [], errors or []),
        ("tencent", "Tencent local", tencent_translations or [], tencent_errors or []),
    )
    rows = []
    for index, text in enumerate(texts):
        lines = [f"{index + 1:02d}", f"Original: {text}"]
        if speakers is not None:
            speaker = speakers[index] if index < len(speakers) else None
            lines.append(f"Speaker: {speaker or 'Unknown'}")
        for key, provider, values, failures in provider_values:
            if key not in selected:
                continue
            english = values[index] if index < len(values) else None
            error = failures[index] if index < len(failures) else None
            if _translation_unavailable(english):
                english = "[Translation unavailable]"
            elif english is None:
                english = "[Translation unavailable]" if error else "[Translation pending]"
            lines.append(f"English ({provider}): {english}")
        if "astra" in selected:
            fast_text = translations[index] if translations and index < len(translations) else None
            entry = _slow_lane_entry(slow_lane, index, fast_text)
            if entry["segment_id"] is not None:
                lines.append(f"Segment ID: {entry['segment_id']}")
            displayed = entry["displayed"]
            lines.extend([
                f"English (Astra correction): {displayed if displayed is not None else '[' + entry['placeholder'] + ']'}",
                f"Astra status: {entry['label']}",
            ])
            if entry["conversation_review_status"] is not None:
                lines.extend([
                    f"Astra first review: {entry['first_review_status'].capitalize()}",
                    f"Astra conversation review: {entry['conversation_review_status'].capitalize()}",
                ])
            authoritative = entry["authoritative"]
            if authoritative != displayed or authoritative is None:
                lines.append(f"Astra authoritative: {authoritative if authoritative is not None else '[Not sealed]'}")
            if entry["live"] != displayed and entry["live"] is not None:
                lines.append(f"Astra latest visible version: {entry['live']}")
        rows.append("\n".join(lines))
    source = f"Original transcript model: {transcription_label}\n\n" if rows and transcription_label else ""
    return source + "\n\n".join(rows)
