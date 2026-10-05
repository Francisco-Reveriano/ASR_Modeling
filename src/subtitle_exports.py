"""Export selected English translations using measured source audio timings."""

import csv
from io import StringIO
import math

from src.translation_validation import contains_cjk, contains_reasoning_markup


_RECORD_FIELDS = ("segment_id", "start_s", "end_s", "source_text", "translation",
                  "status", "filtered", "profile_label", "model")


def _at(values, index):
    return values[index] if values is not None and index < len(values) else None


def _valid_timing(timing):
    if not isinstance(timing, dict):
        return None, None
    start, end = timing.get("start_s"), timing.get("end_s")
    for value in (start, end):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None, None
        try:
            if not math.isfinite(value):
                return None, None
        except OverflowError:
            return None, None
    return (start, end) if 0 <= start < end else (None, None)


def iter_conversation_records(
    texts, translations=None, errors=None, *, filtered=None, timings=None,
    profile_label="Fast English", model=None,
):
    """Yield safe public rows in source order without exposing provider errors.

    Only successful, nonempty English belongs in the translation field. Explicit
    filtered flags identify a deliberate omission; the displayed marker alone
    does not. Missing array entries are pending, so rows stay aligned while work
    completes. References and private endpoint settings are never accepted here.
    """
    for index, source in enumerate(texts):
        english = _at(translations, index)
        error = _at(errors, index)
        if error:
            status = "failed"
        elif _at(filtered, index) is True:
            status = "filtered"
        elif english is None:
            status = "pending"
        elif (not isinstance(english, str) or not english.strip()
              or contains_cjk(english) or contains_reasoning_markup(english)):
            status = "failed"
        else:
            status = "translated"
        start, end = _valid_timing(_at(timings, index))
        yield {
            "segment_id": index + 1,
            "start_s": start,
            "end_s": end,
            "source_text": source,
            "translation": english if status == "translated" else None,
            "status": status,
            "filtered": status == "filtered",
            "profile_label": profile_label,
            "model": model,
        }


def conversation_records(texts, translations=None, errors=None, **kwargs):
    """Return JSON-ready public rows with the same state rules as the screen."""
    return list(iter_conversation_records(texts, translations, errors, **kwargs))


def _timestamp(milliseconds, decimal):
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    seconds, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{decimal}{milliseconds:03d}"


def export_captions(
    texts, translations=None, errors=None, *, format="srt", filtered=None,
    timings=None, profile_label="Fast English", model=None,
):
    """Return only completed English cues with valid, measured audio offsets."""
    if format not in {"srt", "vtt"}:
        raise ValueError("Caption format must be srt or vtt.")
    cues = []
    for row in iter_conversation_records(texts, translations, errors, filtered=filtered,
                                         timings=timings, profile_label=profile_label, model=model):
        if row["status"] != "translated" or row["start_s"] is None:
            continue
        start, end = round(row["start_s"] * 1000), round(row["end_s"] * 1000)
        if end <= start:
            continue
        # Keep spoken markup and newlines from injecting markup or extra cues.
        clean = " ".join(row["translation"].split()).replace("-->", "→").replace("<", "‹").replace(">", "›")
        decimal = "," if format == "srt" else "."
        cues.append(f"{len(cues) + 1}\n{_timestamp(start, decimal)} --> {_timestamp(end, decimal)}\n{clean}")
    if not cues:
        return ""
    return ("WEBVTT\n\n" if format == "vtt" else "") + "\n\n".join(cues) + "\n"


def export_bilingual_csv(
    texts, translations=None, errors=None, *, filtered=None, timings=None,
    profile_label="Fast English", model=None,
):
    """Include every source row, its translation state, timing, and public model."""
    output = StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(_RECORD_FIELDS)
    for row in iter_conversation_records(texts, translations, errors, filtered=filtered,
                                         timings=timings, profile_label=profile_label, model=model):
        values = [row[field] for field in _RECORD_FIELDS]
        writer.writerow([
            "'" + value if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")) else value
            for value in values
        ])
    return output.getvalue()
