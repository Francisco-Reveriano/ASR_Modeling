"""Export sealed subtitles using measured audio offsets, never invented timings."""

import csv
from io import StringIO
import math


def _sealed_rows(snapshot, speakers):
    final = snapshot.get("authoritative", [])
    for index, record in enumerate(snapshot.get("segments", [])):
        text = final[index] if index < len(final) else None
        if text is None:
            continue
        timing = record.get("timing") or {}
        speaker = speakers[index] if speakers and index < len(speakers) else timing.get("speaker_id", "")
        yield record, text, timing.get("start_s"), timing.get("end_s"), speaker or ""


def _timestamp(seconds, decimal):
    milliseconds = round(seconds * 1000)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    seconds, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{decimal}{milliseconds:03d}"


def export_captions(snapshot, *, format="srt", speakers=None):
    """Return SRT/VTT for sealed rows with valid source audio start/end offsets."""
    if format not in {"srt", "vtt"}:
        raise ValueError("Caption format must be srt or vtt.")
    rows = []
    for record, text, start, end, speaker in _sealed_rows(snapshot, speakers):
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in (start, end)):
            continue
        if start < 0 or end <= start:
            continue
        decimal = "," if format == "srt" else "."
        # Keep generated output from injecting a second caption cue. Visible
        # angle brackets remain ordinary words when read in subtitle players.
        clean = " ".join(text.split()).replace("-->", "→").replace("<", "‹").replace(">", "›")
        if record.get("source_fallback"):
            clean = "[Source fallback] " + clean
        if speaker:
            clean = f"{speaker}: {clean}"
        rows.append(f"{len(rows) + 1}\n{_timestamp(start, decimal)} --> {_timestamp(end, decimal)}\n{clean}")
    if not rows:
        return ""
    return ("WEBVTT\n\n" if format == "vtt" else "") + "\n\n".join(rows) + "\n"


def export_bilingual_csv(snapshot, *, speakers=None):
    """Include only sealed rows, plus source, IDs, speaker, and review status."""
    output = StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["segment_id", "start_s", "end_s", "speaker", "source_text", "final_text", "seal_reason", "source_fallback"])
    for record, text, start, end, speaker in _sealed_rows(snapshot, speakers):
        values = [record["segment_id"], start, end, speaker, record["source_text"], text,
                  record.get("seal_reason", ""), record.get("source_fallback", False)]
        writer.writerow(["'" + value if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")) else value
                         for value in values])
    return output.getvalue()
