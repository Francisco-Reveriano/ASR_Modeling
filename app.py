"""Run with: python -m streamlit run app.py"""

from html import escape
import json
import os
from pathlib import Path
from time import sleep

from dotenv import load_dotenv
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from src.evaluation import mixed_match_score, parse_reference, word_match_score
from src.glossary import load_glossary
from src.pipeline import LiveTranscriber, load_transcriber, load_vad
from src.reasoning import DEFAULT_REASONING_MODEL, PROMPT_VERSION, correct_translations
from src.reference_tables import (
    parsed_table_reference, read_reference_tables, recommend_columns,
    suggest_header_row, suggest_table, table_columns,
)
from src.tencent import TENCENT_ERROR_MESSAGE, translate_with_tencent
from src.translation import ENV_FILE, TranslationSession
from src.slow_lane import SlowLaneConfig, SlowLaneSession
from src.diarization import DiarizationSession, speaker_labels
from src.subtitle_exports import export_bilingual_csv, export_captions
from src.ui import export_conversation, render_transcript
from src.uploads import decode_wav, speech_segments, transcribe_in_background

st.set_page_config(page_title="Breeze Voice", page_icon="🎙️", layout="wide")
st.html(Path(__file__).parent / "assets" / "style.css")
st.markdown(
    '<div class="brand-bar"><div class="brand">'
    '<svg viewBox="0 0 32 32" fill="none" stroke="currentColor" '
    'stroke-width="3" stroke-linecap="round" aria-hidden="true">'
    '<path d="M4 13v6M10 8v16M16 4v24M22 10v12M28 14v4"/></svg>'
    'breeze <span>/ voice</span></div>'
    '<div class="local-badge">Breeze + Tencent on this Mac · OpenAI via API</div></div>',
    unsafe_allow_html=True,
)
st.title("Voice transcription & translation")
st.markdown(
    '<p class="page-intro">Speak live, upload a WAV file, or evaluate transcription against a reference.</p>',
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_transcriber():
    return load_transcriber()


def slow_config():
    """Freeze the controls into the new session's reproducible configuration."""
    return SlowLaneConfig(
        model=DEFAULT_REASONING_MODEL, reasoning_effort="medium",
        window_n=st.session_state.get("slow_window_n", 4),
        revision_horizon_s=st.session_state.get("slow_horizon", 20),
        seal_timeout_s=st.session_state.get("slow_seal_timeout", 30),
        request_timeout_s=st.session_state.get("slow_request_timeout", 20),
        max_output_tokens=st.session_state.get("slow_max_tokens", 4096),
        confidence_threshold=st.session_state.get("slow_confidence", 0.6),
        source_lang="zh-TW+en", target_lang="en",
        enabled=st.session_state.get("enable_corrections", True),
    )


def terminology():
    if "glossary" not in st.session_state:
        st.session_state.glossary = load_glossary(
            st.session_state.get("glossary_path") or None,
            st.session_state.get("dnt_path") or None,
        )
    return st.session_state.glossary


def reset_translation(*, diarization_max_pending_seconds=600):
    """End old work and allocate independent fast, slow, and speaker workers."""
    # Load configured files before closing a usable previous session. File I/O
    # never runs on the audio, translator, or correction worker's critical path.
    glossary = load_glossary(
        st.session_state.get("glossary_path") or None,
        st.session_state.get("dnt_path") or None,
    )
    config = slow_config()
    for key in ("translation", "tencent_translation", "slow_lane", "diarization"):
        previous = st.session_state.get(key)
        if previous is not None:
            previous.close()
    st.session_state.glossary = glossary
    st.session_state.translation = TranslationSession(glossary=glossary)
    st.session_state.tencent_translation = TranslationSession(
        translate_with_tencent, failure_message=TENCENT_ERROR_MESSAGE,
    )
    st.session_state.slow_lane = SlowLaneSession(correct_translations, config=config, glossary=glossary)
    st.session_state.diarization = DiarizationSession(
        enabled=st.session_state.get("enable_diarization", True),
        max_pending_seconds=diarization_max_pending_seconds,
    )


def conversation_extras(state, texts, translated):
    """Publish fast drafts before consulting either asynchronous worker."""
    if texts and "slow_lane" not in st.session_state:
        st.session_state.slow_lane = SlowLaneSession(
            correct_translations, config=slow_config(), glossary=terminology(),
        )
    lane = st.session_state.get("slow_lane")
    original_timings = state.get("timings", []) if state else []
    timings = [dict(original_timings[index]) if index < len(original_timings) and original_timings[index] else {}
               for index in range(len(texts))]
    diarization = st.session_state.get("diarization")
    speakers = None
    if diarization is not None:
        # Legacy sessions have no audio offsets. Keep their speaker unknown
        # rather than inventing a timeline from transcript or reference rows.
        speakers = speaker_labels(
            [timing if "start_s" in timing and "end_s" in timing else None for timing in timings],
            diarization.snapshot()["segments"],
        )
        for timing, speaker in zip(timings, speakers):
            if speaker is not None:
                timing["speaker_id"] = speaker
    if lane is not None:
        lane.submit(
            texts, translated["openai"]["translations"],
            draft_errors=translated["openai"]["errors"], timings=timings,
        )
        slow = lane.snapshot()
    else:
        slow = {"translations": [], "statuses": [], "authoritative": [],
                "segments": [], "pending": 0, "status": "idle", "metrics": {}}
    slow["view"] = "authoritative" if st.session_state.get("subtitle_view") == "Final record" else "speculative"
    return {"slow_lane": slow, "speakers": speakers}


def translation_snapshot(texts):
    """Send the same original text to independent cloud and local workers.

    Neither translator consumes the other's output or waits for it. Each queue
    deduplicates segment positions across UI polls. Creating a missing provider
    independently also preserves existing results after an app update.
    """
    if texts and "translation" not in st.session_state:
        st.session_state.translation = TranslationSession()
    if texts and "tencent_translation" not in st.session_state:
        st.session_state.tencent_translation = TranslationSession(
            translate_with_tencent, failure_message=TENCENT_ERROR_MESSAGE,
        )
    results = {}
    for provider, key in (("openai", "translation"), ("tencent", "tencent_translation")):
        translation = st.session_state.get(key)
        if translation is None:
            results[provider] = {"translations": [], "errors": [], "pending": 0}
        else:
            translation.submit(texts)
            results[provider] = translation.snapshot()
    return results


def conversation_args(texts, translated):
    """Keep rendering and downloads on the same source/provider ordering."""
    return (
        texts, translated["openai"]["translations"], translated["openai"]["errors"],
        translated["tencent"]["translations"], translated["tencent"]["errors"],
    )


def reference_view_args(state):
    """Show the submitted reference independently of the ASR pause boundaries."""
    evaluation = state.get("evaluation", {}) if state else {}
    reference = evaluation.get("reference_view") or evaluation.get("reference")
    if reference is None:
        return {}
    return {
        "reference_text": reference["text"],
        "reference_title": reference.get("title", "Reference transcript"),
    }


def reference_description(reference):
    """Identify the exact submitted file and, for tables, its selected column."""
    parts = [reference["name"]]
    if "sheet" in reference:
        parts.extend([reference["sheet"], reference["column"]])
    return " · ".join(parts)


def evaluation_references(evaluation):
    """Return the source reference used to score Breeze transcription."""
    primary = evaluation.get("reference") or {
        "text": evaluation["reference_text"], "name": evaluation["reference_name"],
        "has_cjk": False, "format": "Plain text",
    }
    if evaluation.get("reference_kind", "english") == "source":
        return {"breeze": primary}
    # A previously submitted translation reference cannot grade source speech.
    return {"breeze": None}


def evaluation_results(state):
    """Score the complete Breeze transcript; translators are never scored.

    The reference belongs to the submitted job and is never passed to a model.
    Cache completed alignments in this session so polling does not recalculate
    the same full-file edit distance. New evaluations get a fresh cache.
    """
    evaluation = state["evaluation"]
    cached = evaluation.setdefault("scores", {})
    results = {}
    for provider, reference in evaluation_references(evaluation).items():
        mixed = reference is not None and reference["has_cjk"]
        result = {
            "label": "Breeze",
            "metric": "Mixed match" if mixed else "1-wMER", "metrics": None,
        }
        if reference is None:
            result["status"] = "Source transcript needed"
        elif not state["finished"]:
            result["status"] = "Waiting for transcription"
        elif state["error"]:
            result["status"] = "Transcription failed · no final score"
        else:
            hypothesis = " ".join(state["texts"])
            if provider not in cached or cached[provider]["hypothesis"] != hypothesis:
                score = mixed_match_score if mixed else word_match_score
                cached[provider] = {
                    "hypothesis": hypothesis,
                    "metrics": score(reference["text"], hypothesis),
                }
            result.update(status="Final score", metrics=cached[provider]["metrics"])
        results[provider] = result
    return results


def evaluation_panel(state, results):
    """Show Breeze's whole-file transcription score."""
    st.markdown('<h3 class="evaluation-heading">Transcription score</h3>', unsafe_allow_html=True)
    st.caption("Reference match · Higher is better")
    columns = st.columns(len(results), gap="medium")
    for column, (provider, result) in zip(columns, results.items()):
        metrics = result["metrics"]
        with column, st.container(key=f"evaluation_{provider}", border=True):
            st.metric(
                f"{result['label']} · {result['metric']}",
                f"{metrics['score']:.1%}" if metrics else "—",
                help="Matches divided by matches, substitutions, deletions, and insertions.",
            )
            st.caption(result["status"])
            if metrics:
                st.caption(
                    f"Matches {metrics['hits']} · Substitutions {metrics['substitutions']} · "
                    f"Deletions {metrics['deletions']} · Insertions {metrics['insertions']}"
                )
    with st.expander("Reference & scoring details"):
        references = evaluation_references(state["evaluation"])
        for provider in ("breeze",):
            reference = references.get(provider)
            if reference is not None:
                st.caption(f"Source transcript: {reference_description(reference)} · {reference['format']}")
                with st.container(height=180):
                    st.text(reference["text"])
        view = state["evaluation"].get("reference_view")
        if view is not None and view["title"] == "English reference":
            st.caption(f"English reference: {reference_description(view)}")
            st.caption("Compare this reference with the English model results. Translation is not scored.")
        st.markdown(
            "**Match score = matches / (matches + substitutions + deletions + insertions).** "
            "An English transcript uses words (1-wMER). A Chinese/English transcript uses each Chinese character "
            "and each English word (Mixed match). Case and punctuation are ignored; Unicode is normalized."
        )
        st.caption(
            "The full file is aligned in transcript order, without speaker or time alignment. "
            "Overlapping speech can affect the score. These scores measure text matching; "
            "valid paraphrases can score lower. References stay local and are never given to a model."
        )


def evaluation_report(state, results):
    """Include final scores or explicit pending/failure states in the TXT export."""
    lines = [
        "Evaluation: Breeze transcription score",
        f"Audio: {state['name']}",
        "Scoring: whole-file alignment; Unicode normalized; case/punctuation ignored.",
        "Formula: matches / (matches + substitutions + deletions + insertions)",
        "1-wMER uses words. Mixed match uses Chinese characters and English words.",
        "Higher is better. Text matching does not measure semantic quality.",
    ]
    for provider, result in results.items():
        label = f"{result['label']} ({result['metric']})"
        metrics = result["metrics"]
        if metrics:
            lines.append(
                f"{label}: {metrics['score']:.1%} (score={metrics['score']:.6f}; "
                f"H={metrics['hits']}, S={metrics['substitutions']}, "
                f"D={metrics['deletions']}, I={metrics['insertions']})"
            )
        else:
            lines.append(f"{label}: {result['status']}")
    for provider, reference in evaluation_references(state["evaluation"]).items():
        if reference is None:
            continue
        lines.extend([
            "", f"Source transcript reference: {reference_description(reference)} ({reference['format']})",
            "Spoken reference used for scoring:", reference["text"],
        ])
        if reference.get("original_text", reference["text"]) != reference["text"]:
            lines.extend(["", "Original uploaded reference:", reference["original_text"]])
    view = state["evaluation"].get("reference_view")
    if view is not None and view["title"] == "English reference":
        lines.extend([
            "", f"English reference: {reference_description(view)}",
            "For comparison only; not scored or aligned to speech segments.", view["text"],
        ])
    return "\n".join(lines)


def transcript_panel(state=None, context=None):
    """Draw the current snapshot without changing the recording session."""
    texts = state["texts"] if state else []
    translated = translation_snapshot(texts)
    extras = conversation_extras(state, texts, translated)
    translation_pending = any(result["pending"] for result in translated.values())
    translation_failed = any(any(result["errors"]) for result in translated.values())
    is_upload = state is not None and "name" in state
    is_evaluation = state is not None and "evaluation" in state
    status, tone = "Ready", ""
    if state:
        if state["error"]:
            status = "Needs attention"
        elif state["pending"]:
            status, tone = "Transcribing…", "working"
        elif state["accepting"]:
            status, tone = (
                ("Listening…", "active") if context.state.playing
                else ("Connecting microphone…", "working")
            )
        elif state["finished"]:
            status = "Evaluation complete" if is_evaluation else (
                "File transcribed" if is_upload else "Recording stopped"
            )
        else:
            status, tone = "Finishing…", "working"
    if translation_pending and state and state["finished"] and not state["error"]:
        status, tone = "Translating…", "working"
    elif translation_failed and state and state["finished"]:
        status, tone = "Needs attention", ""

    heading = "Evaluation results" if is_evaluation else "Your conversation"
    st.markdown(
        f'<div class="transcript-heading"><h2>{heading}</h2>'
        f'<span class="status-badge {tone}" role="status">{status}</span></div>',
        unsafe_allow_html=True,
    )
    if state and state["error"]:
        st.error(state["error"])
    if is_upload:
        st.markdown(
            f'<p class="source-file">File: {escape(state["name"])}</p>',
            unsafe_allow_html=True,
        )
    if state and state["pending"]:
        action = "Listening and transcribing" if state["accepting"] else "Finishing transcription"
        st.caption(f"{action} · {state['pending']} segment(s) pending")
    elif state and state["accepting"] and not context.state.playing:
        st.caption("If the connection stalls, open Microphone settings to check browser access.")
    if state and state["finished"] and not texts and not state["error"]:
        st.info(
            "No speech detected in this file. Try another WAV file."
            if is_upload else "No speech detected. Start a new recording to try again."
        )

    for provider, label, key, retry_key in (
        ("openai", "OpenAI", "translation", "retry_translation"),
        ("tencent", "Tencent · Local", "tencent_translation", "retry_tencent_translation"),
    ):
        result = translated[provider]
        if result["pending"]:
            st.caption(f"{label} · Translating {result['pending']} segment(s) into English")
        errors = list(dict.fromkeys(error for error in result["errors"] if error))
        if errors:
            st.warning(f"{label}: {' '.join(errors)}")
            if st.button(f"Retry {label}", key=retry_key, icon=":material/refresh:"):
                st.session_state[key].retry_failed()
                st.rerun()
    if is_evaluation:
        results = evaluation_results(state)
        evaluation_panel(state, results)
    slow = extras["slow_lane"]
    slow_status = slow["status"]
    st.caption({
        "active": "Astra · Reviewing recent segments · Medium reasoning",
        "paused": "Astra · Corrections paused · Fast translations continue",
        "degraded": "Astra · Corrections unavailable · Fast translations continue",
        "closed": "Astra · Session ended",
        "idle": "Astra · Ready · Medium reasoning",
    }.get(slow_status, "Astra · Medium reasoning"))
    if slow_status == "degraded" and st.button("Retry open corrections", key="retry_corrections"):
        st.session_state.slow_lane.retry_failed()
        st.rerun()
    diarization = st.session_state.get("diarization")
    if diarization is not None:
        speaker_state = diarization.snapshot()
        st.caption(f"Nemotron · Speaker detection {speaker_state['status']}")
        if speaker_state.get("error"):
            st.caption(speaker_state["error"])
    st.markdown(
        render_transcript(*conversation_args(texts, translated), **extras, **reference_view_args(state)),
        unsafe_allow_html=True,
    )
    download_text = export_conversation(*conversation_args(texts, translated), **extras)
    if is_evaluation:
        download_text += "\n\n" + evaluation_report(state, results)
    if slow.get("segments"):
        with st.expander("Correction history & session export"):
            st.caption(
                "Live subtitles keep their current text after the revision window. "
                "Final record shows sealed results, including accepted corrections that arrived later. "
                "A sealed result never changes."
            )
            st.json({"status": slow_status, "metrics": slow.get("metrics", {}),
                     "config": slow.get("config", {}), "learned_terms": slow.get("learned_terms", [])})
            record = dict(slow, prompt_version=PROMPT_VERSION)
            if diarization is not None:
                record["diarization"] = speaker_state
            st.download_button(
                "Download session JSON", json.dumps(record, ensure_ascii=False, indent=2),
                "breeze-session.json", "application/json", key="download_session", on_click="ignore",
            )
            st.download_button(
                "Download final bilingual CSV", export_bilingual_csv(slow, speakers=extras["speakers"]),
                "breeze-final.csv", "text/csv", key="download_final_csv", on_click="ignore",
            )
            for caption_format in ("srt", "vtt"):
                captions = export_captions(slow, format=caption_format, speakers=extras["speakers"])
                st.download_button(
                    f"Download final {caption_format.upper()}", captions, f"breeze-final.{caption_format}",
                    "text/vtt" if caption_format == "vtt" else "text/plain",
                    disabled=not captions, key=f"download_final_{caption_format}", on_click="ignore",
                )
    with st.container(key="transcript_footer"):
        count_column, download_column = st.columns([1, 1], vertical_alignment="center")
        count_column.markdown(
            f'<span class="segment-count">{len(texts)} '
            f'{"segment" if len(texts) == 1 else "segments"} · Numbered by pause</span>',
            unsafe_allow_html=True,
        )
        download_column.download_button(
            "Download evaluation" if is_evaluation else "Download conversation",
            data=download_text,
            file_name="breeze-evaluation.txt" if is_evaluation else "breeze-conversation.txt",
            mime="text/plain",
            icon=":material/download:",
            disabled=not texts and not is_evaluation,
            on_click="ignore",
            width="stretch",
            wrap=True,
            key="download_transcript",
        )


@st.fragment(run_every=0.5)
def show_transcript(session=None, context=None, control_state=None):
    """Poll ASR and translation for either input, including after audio stops."""
    state = session.snapshot() if session else st.session_state.upload_state
    if session and (state["accepting"], state["finished"]) != control_state:
        # Refresh controls only when capture ends or the worker finishes.
        st.rerun()
    transcript_panel(state, context)


def recording_panel():
    session = st.session_state.get("recording")
    context = st.session_state.get("microphone_context")
    state = session.snapshot() if session else None
    browser_active = context and (context.state.playing or context.state.signalling)
    busy = bool(
        browser_active or (state and not state["finished"])
        or st.session_state.get("upload_job") is not None
    )

    start_column, stop_column = st.columns(2)
    start = start_column.button(
        "Start recording", disabled=busy, type="primary", icon=":material/mic:",
        width="stretch", wrap=True, key="start_recording",
    )
    stop = stop_column.button(
        "Stop recording", disabled=not state or not state["accepting"],
        icon=":material/stop:", width="stretch", wrap=True, key="stop_recording",
    )
    if start:
        st.session_state.model_error = None
        try:
            with st.spinner("Loading local speech models…"):
                transcribe = get_transcriber()
                vad = load_vad()
            reset_translation()
            session = LiveTranscriber(transcribe, vad, diarization=st.session_state.diarization)
        except Exception as exc:
            st.session_state.model_error = f"Could not load the local speech models: {exc}"
        else:
            st.session_state.recording = session
            st.session_state.recording_number = st.session_state.get("recording_number", 0) + 1
            st.session_state.microphone_connected = False
            st.session_state.transcript_source = "microphone"
            st.session_state.pop("upload_state", None)
    if st.session_state.get("model_error"):
        st.error(st.session_state.model_error)
    if stop:
        session.finish()
        # Controls above were rendered from the pre-stop state. An empty
        # recording can finish synchronously, so no later fragment transition
        # would otherwise unlock Start and the next-session settings.
        st.rerun()

    st.caption(
        "Start recording and allow microphone access in your browser. "
        "A new recording clears the previous transcript."
    )
    # Keep the component mounted outside the polling fragment. Its device picker
    # remains accessible without duplicating recording controls in the main view.
    with st.expander("Microphone settings", icon=":material/tune:"):
        if session is None:
            st.caption("Microphone devices will be available after you start recording.")
        else:
            context = webrtc_streamer(
                key=f"microphone-{st.session_state.recording_number}",
                mode=WebRtcMode.SENDONLY,
                desired_playing_state=session.snapshot()["accepting"],
                media_stream_constraints={"video": False, "audio": True},
                rtc_configuration={"iceServers": []},
                audio_frame_callback=session.push,
                on_audio_ended=session.finish,
                async_processing=False,
            )
            st.session_state.microphone_context = context
            if context.state.playing:
                st.session_state.microphone_connected = True
            elif st.session_state.microphone_connected and not context.state.signalling:
                session.finish()

    return session, context


@st.cache_data(show_spinner=False, max_entries=4)
def reference_tables(data, filename):
    """Reuse a bounded in-memory parse while the user adjusts column choices."""
    return read_reference_tables(data, filename)


def clear_reference_settings(*keys):
    """Reset dependent selectors before their widgets are created on the rerun."""
    for key in ("evaluation_error", *keys):
        st.session_state.pop(key, None)


def reference_inputs(busy):
    """Preview the source and English columns; freeze them only on Run.

    These controls live in the left input pane. Source text is the only scoring
    input. The separately selected English text is a display/export reference;
    neither selection is ever passed to the speech or translation models.
    """
    column_keys = ("eval_text_column", "eval_display_column")
    uploaded = st.file_uploader(
        "Reference transcript", type=["txt", "srt", "vtt", "xlsx", "csv", "tsv"],
        key="eval_reference_file", disabled=busy, on_change=clear_reference_settings,
        args=("eval_reference_sheet", "eval_header_row", *column_keys), max_upload_size=1,
    )
    settings = st.expander("Reference settings")
    with settings:
        format_choice = st.selectbox(
            "Reference format", ["Auto-detect", "Plain text", "Transcript / captions"],
            key="eval_reference_format", disabled=busy, on_change=clear_reference_settings,
            help="Transcript mode removes recognized timestamps, speaker labels, and non-speech notes. "
                 "Plain text keeps the selected content as written.",
        )
    st.caption("TXT, SRT, VTT, XLSX, CSV or TSV · Up to 1 MiB. Text files: UTF-8 or UTF-16 with BOM.")
    if uploaded is None:
        return None, False, None
    parse_format = {
        "Auto-detect": "auto", "Plain text": "plain", "Transcript / captions": "transcript",
    }[format_choice]
    try:
        view = None
        if Path(uploaded.name).suffix.lower() in {".xlsx", ".csv", ".tsv"}:
            tables = reference_tables(uploaded.getvalue(), uploaded.name)
            sheets = list(tables)
            sheet = st.selectbox(
                "Worksheet", sheets, index=sheets.index(suggest_table(tables)),
                key="eval_reference_sheet", disabled=busy, on_change=clear_reference_settings,
                args=("eval_header_row", *column_keys),
            )
            rows = tables[sheet]
            with settings:
                header_number = st.number_input(
                    "Header row (0 = no header)", min_value=0, max_value=len(rows),
                    value=suggest_header_row(rows) + 1, step=1,
                    key="eval_header_row", disabled=busy, on_change=clear_reference_settings,
                    args=column_keys,
                    help="Rows through the header are excluded. Choose 0 to include every row.",
                )
            header_row = header_number - 1 if header_number else None
            labels = table_columns(rows, header_row)
            choices = list(range(len(labels)))
            recommended = recommend_columns(rows, header_row)
            source_column = st.selectbox(
                "Transcription reference column", choices, index=recommended["source"],
                format_func=lambda index: labels[index], key="eval_text_column", disabled=busy,
                placeholder="Choose the original-language transcript",
                on_change=clear_reference_settings,
                help="Breeze is scored against this column. For your sample, choose text_zh_TW.",
            )
            display_choices = [None, *choices]
            display_column = st.selectbox(
                "English reference column", display_choices,
                index=display_choices.index(recommended["display"]),
                format_func=lambda index: "Use transcription reference" if index is None else labels[index],
                key="eval_display_column", disabled=busy, on_change=clear_reference_settings,
                help="Show this column in the reference pane below the model results. For your sample, choose translation_en. It is not scored.",
            )
            if source_column is None:
                raise ValueError("Choose a transcription reference column containing the words spoken in the audio.")

            def parse_column(column):
                parsed = parsed_table_reference(
                    rows, header_row=header_row, text_column=column, format=parse_format,
                )
                parsed.update(
                    name=uploaded.name, sheet=sheet, column=labels[column], header_row=header_number,
                )
                return parsed

            reference = parse_column(source_column)
            if display_column is not None and display_column != source_column:
                view = dict(parse_column(display_column), title="English reference")
            data_rows = len(rows) - header_number
            st.caption(
                f"Source: {len(reference['rows'])} text row(s); "
                f"{data_rows - len(reference['rows'])} blank cell(s) skipped."
            )
            if view is not None:
                st.caption(
                    f"English: {len(view['rows'])} text row(s); "
                    f"{data_rows - len(view['rows'])} blank cell(s) skipped."
                )
        else:
            reference = parse_reference(uploaded.getvalue(), uploaded.name, format=parse_format)
            reference["name"] = uploaded.name
        if view is None:
            view = dict(reference, title="Reference transcript")
        st.caption(f"{reference['format']} · {reference['segment_count']} reference segment(s)")
        with st.expander("Preview spoken reference"):
            st.caption(f"Excluded {reference['removed_lines']} metadata / non-target line(s).")
            with st.container(height=180):
                st.text(reference["text"])
        if view["title"] == "English reference":
            with st.expander("Preview English reference"):
                with st.container(height=180):
                    st.text(view["text"])
        st.caption("Breeze is scored against the spoken reference. A separate reference pane shows your selected text for comparison.")
        return {
            "reference_text": reference["text"], "reference_name": uploaded.name,
            "reference_kind": "source", "reference": reference, "reference_view": view,
        }, True, None
    except ValueError as exc:
        return None, True, str(exc)


def upload_panel(session, context, *, evaluate=False):
    """Validate a chosen file, then rerun with controls disabled before ASR."""
    state = session.snapshot() if session else None
    live_busy = bool(
        (context and (context.state.playing or context.state.signalling))
        or (state and not state["finished"])
    )
    busy = live_busy or st.session_state.get("upload_job") is not None
    error_key = "evaluation_error" if evaluate else "upload_error"
    uploaded = st.file_uploader(
        "Evaluation audio (.wav)" if evaluate else "Choose a WAV file",
        type=["wav"], key="eval_wav_file" if evaluate else "wav_file", disabled=busy,
        on_change=lambda: st.session_state.pop(error_key, None),
    )
    if uploaded is not None:
        st.audio(uploaded.getvalue(), format="audio/wav")
    evaluation, has_reference, reference_error = None, False, None
    if evaluate:
        evaluation, has_reference, reference_error = reference_inputs(busy)
    if st.button(
        "Run evaluation" if evaluate else "Transcribe file",
        key="evaluate_file" if evaluate else "transcribe_file", type="primary",
        icon=":material/analytics:" if evaluate else ":material/audio_file:",
        width="stretch", wrap=True,
        disabled=busy or uploaded is None or (evaluate and not has_reference),
    ):
        st.session_state[error_key] = None
        try:
            if reference_error:
                raise ValueError(reference_error)
            with st.spinner("Reading WAV audio…"):
                audio = decode_wav(uploaded.getvalue())
            reset_translation(diarization_max_pending_seconds=max(600, len(audio) / 16000 + 1))
        except (ValueError, OSError) as exc:
            st.session_state[error_key] = str(exc)
        else:
            st.session_state.upload_job = {
                "audio": audio, "name": uploaded.name, "next_segment": 0,
            }
            st.session_state.upload_state = {
                "texts": [], "pending": 0, "accepting": False,
                "finished": False, "error": None, "name": uploaded.name,
                "timings": [], "retained_audio": audio,
            }
            st.session_state.diarization.push(audio)
            st.session_state.diarization.finish()
            if evaluate:
                st.session_state.upload_state["evaluation"] = evaluation
            st.session_state.transcript_source = "upload"
            st.session_state.model_error = None
            st.rerun()
    error = reference_error or st.session_state.get(error_key)
    if error:
        st.error(error)
    if live_busy:
        st.caption("Stop recording and wait for transcription to finish before uploading a file.")
    elif evaluate:
        st.caption(
            "Transcribe and translate the audio, then score the Breeze transcript against your reference. "
            "Starting a new evaluation replaces the current results."
        )
    else:
        st.caption("Preview your audio, then transcribe it. A new transcription replaces the current text.")


def process_upload():
    """Transcribe file segments in order, without filling the live audio queue.

    The job was stored on the previous run so input controls are already
    disabled. A checkpoint after each segment lets an interrupted Streamlit
    run resume without dropping or repeating completed text. Completed jobs
    are removed, so ordinary reruns and downloads never repeat inference.
    No file or transcript is written to disk.
    """
    job = st.session_state.upload_job
    state = st.session_state.upload_state
    st.markdown(
        '<div class="transcript-heading"><h2>Your conversation</h2>'
        '<span class="status-badge working" role="status">Processing file…</span></div>',
        unsafe_allow_html=True,
    )
    try:
        if "segments" not in job:
            with st.spinner("Finding speech in your WAV file…"):
                job["segments"] = speech_segments(job["audio"], load_vad(), with_timestamps=True)
        segments = job["segments"]
        state["pending"] = len(segments) - job["next_segment"]
        # Silence should not require loading the large speech model.
        if segments:
            with st.spinner("Loading local speech model…"):
                transcribe = get_transcriber()
            progress = st.progress(
                job["next_segment"] / len(segments),
                text=f"Transcribing {len(segments)} speech segment(s)…",
            )
            output = st.empty()
            for index in range(job["next_segment"], len(segments)):
                segment = segments[index]
                # Compatibility for older in-flight jobs that stored bare arrays.
                timing = {key: value for key, value in segment.items() if key != "audio"} if isinstance(segment, dict) else {}
                if "asr_task" not in job:
                    job["asr_task"] = transcribe_in_background(
                        transcribe, segment["audio"] if isinstance(segment, dict) else segment,
                    )
                task = job["asr_task"]
                while not task.done():
                    # Keep completed fast drafts and corrections flowing while
                    # the next local decode waits for GPU access or inference.
                    translated = translation_snapshot(state["texts"])
                    output.markdown(
                        render_transcript(
                            *conversation_args(state["texts"], translated),
                            **conversation_extras(state, state["texts"], translated),
                            **reference_view_args(state),
                        ),
                        unsafe_allow_html=True,
                    )
                    sleep(0.1)
                text, timing["asr_start_ms"], timing["asr_final_ms"] = task.result()
                if text:
                    state["texts"].append(text)
                    state.setdefault("timings", []).append(timing)
                # Save progress before UI calls, which may interrupt this run.
                job["next_segment"] = index + 1
                job.pop("asr_task")
                state["pending"] -= 1
                translated = translation_snapshot(state["texts"])
                output.markdown(
                    render_transcript(
                        *conversation_args(state["texts"], translated),
                        **conversation_extras(state, state["texts"], translated), **reference_view_args(state),
                    ),
                    unsafe_allow_html=True,
                )
                progress.progress(
                    (index + 1) / len(segments),
                    text=f"Transcribed {index + 1} of {len(segments)} speech segments",
                )
    except Exception as exc:
        action = "Run evaluation" if "evaluation" in state else "Transcribe file"
        state["error"] = f"Could not transcribe this WAV file: {exc}. You can retry {action}."
    # Streamlit reruns use a BaseException. Let those keep the job/checkpoint;
    # only success or an ordinary processing error completes this attempt.
    state["finished"] = True
    state["pending"] = 0
    st.session_state.pop("upload_job", None)
    st.rerun()


def translation_controls():
    """Configure the next session and pause correction work independently."""
    load_dotenv(ENV_FILE, override=False)
    recording = st.session_state.get("recording")
    busy = st.session_state.get("upload_job") is not None or bool(
        recording and not recording.snapshot()["finished"]
    )

    def toggle_corrections():
        lane = st.session_state.get("slow_lane")
        if lane is not None:
            lane.set_enabled(st.session_state.enable_corrections)

    with st.expander("Translation & speaker settings"):
        st.toggle("Astra corrections", value=True, key="enable_corrections", on_change=toggle_corrections)
        st.caption("GPT-6 Astra · Medium reasoning · Reviews fast OpenAI drafts asynchronously.")
        st.selectbox("Astra view", ["Live subtitles", "Final record"], key="subtitle_view")
        st.toggle("Nemotron speaker detection", value=True, key="enable_diarization", disabled=busy)
        st.caption("Speaker detection runs locally. Settings below apply to the next recording or file.")
        st.number_input("Context segments", min_value=1, max_value=8, value=4,
                        key="slow_window_n", disabled=busy)
        st.number_input("Revision window (seconds)", min_value=0, max_value=120, value=20,
                        key="slow_horizon", disabled=busy)
        st.number_input("Seal timeout (seconds)", min_value=5, max_value=120, value=30,
                        key="slow_seal_timeout", disabled=busy)
        st.number_input("Reasoning timeout (seconds)", min_value=1, max_value=60, value=20,
                        key="slow_request_timeout", disabled=busy)
        st.number_input("Reasoning output cap", min_value=512, max_value=16384, value=4096, step=512,
                        key="slow_max_tokens", disabled=busy)
        st.slider("Minimum correction confidence", min_value=0.0, max_value=1.0, value=0.6, step=0.05,
                  key="slow_confidence", disabled=busy)
        st.text_input("Glossary file path", value=os.getenv("GLOSSARY_FILE", ""), key="glossary_path")
        st.text_input("Do-not-translate file path", value=os.getenv("DNT_FILE", ""), key="dnt_path")
        st.caption("Optional CSV, TSV, JSON or XLSX glossary. A DNT list may also be TXT. Evaluation references are never glossary inputs.")
        if st.button("Reload terminology", key="reload_glossary"):
            try:
                updated = load_glossary(st.session_state.glossary_path or None, st.session_state.dnt_path or None)
                if "glossary" in st.session_state:
                    st.session_state.glossary.replace_master(updated)
                else:
                    st.session_state.glossary = updated
                st.success("Terminology reloaded for current and future segments.")
            except (ValueError, OSError) as exc:
                st.error(f"Could not reload terminology: {exc}")
        if "glossary" in st.session_state:
            st.caption(f"{len(st.session_state.glossary.entries)} master terminology entries loaded.")
        elif not st.session_state.glossary_path and not st.session_state.dnt_path:
            st.caption("No master glossary configured. Tool-like identifiers are still protected.")


with st.container(key="workspace"):
    input_column, transcript_column = st.columns([1, 3], gap="medium")
with input_column, st.container(key="input_card"):
    st.markdown(
        '<div class="panel-heading"><div><div class="eyebrow">Speaking language</div>'
        '<div class="language-name">Mandarin / Chinglish</div></div>'
        '<span class="language-tag">中文 + EN</span></div>',
        unsafe_allow_html=True,
    )
    translation_controls()
    microphone_tab, upload_tab, evaluation_tab = st.tabs(["Microphone", "Upload WAV", "Evaluate"])
    with microphone_tab:
        st.markdown(
            '<div class="mic-stage"><div class="mic-symbol" aria-hidden="true">'
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" '
            'stroke-linecap="round"><rect x="9" y="2" width="6" height="12" rx="3"/>'
            '<path d="M5 10v2a7 7 0 0 0 14 0v-2M12 19v3M8 22h8"/></svg></div>'
            '<h3>Your voice, in English.</h3>'
            '<p>Speak naturally. Compare fast translations<br>with contextual Astra corrections.</p></div>',
            unsafe_allow_html=True,
        )
        session, context = recording_panel()
    with upload_tab:
        upload_panel(session, context)
    with evaluation_tab:
        upload_panel(session, context, evaluate=True)
    st.markdown(
        '<div class="privacy-note">Audio and Tencent translation stay on this Mac. '
        'Transcript text and configured glossary context go to OpenAI for fast translation and Astra corrections.</div>',
        unsafe_allow_html=True,
    )

with transcript_column, st.container(key="transcript_card"):
    if st.session_state.get("upload_job") is not None:
        process_upload()
    elif st.session_state.get("transcript_source") == "upload":
        show_transcript()
    elif session is None:
        transcript_panel()
    else:
        state = session.snapshot()
        show_transcript(session, context, (state["accepting"], state["finished"]))

st.markdown(
    '<p class="workspace-note">One line per speech segment. '
    'Astra reviews recent source text and fast drafts. '
    'Speaker labels update independently; reference lines follow the uploaded file.</p>',
    unsafe_allow_html=True,
)
