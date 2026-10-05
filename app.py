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
from src.pipeline import LiveTranscriber, load_vad
from src.reference_tables import (
    parsed_table_reference, read_reference_tables, recommend_columns,
    suggest_header_row, suggest_table, table_columns,
)
from src.translation import TranslationSession
from src.subtitle_exports import conversation_records, export_bilingual_csv, export_captions
from src.ui import export_conversation, render_transcript
from src.uploads import decode_wav, speech_segments, transcribe_in_background
from src.vllm import (
    VllmConfigError, create_asr, create_translator, default_fast_profile,
    load_asr_config, load_fast_profiles,
)

ENV_FILE = Path(__file__).resolve().parent / ".env"

st.set_page_config(page_title="vLLM Voice", page_icon="🎙️", layout="wide")
st.html(Path(__file__).parent / "assets" / "style.css")
st.markdown(
    '<div class="brand-bar"><div class="brand">vLLM <span>/ voice</span></div>'
    '<div class="local-badge">Local speech detection · vLLM transcription & translation</div></div>',
    unsafe_allow_html=True,
)
st.title("Voice transcription & translation")
st.markdown(
    '<p class="page-intro">Speak live, upload a WAV file, or evaluate transcription against a reference.</p>',
    unsafe_allow_html=True,
)


def close_conversation():
    """Close captured workers before replacing their conversation state."""
    recording = st.session_state.pop("recording", None)
    if recording is not None:
        recording.close()
    job = st.session_state.pop("upload_job", None)
    if job is not None and job.get("transcribe") is not None:
        job["transcribe"].close()
    translation = st.session_state.pop("translation", None)
    if translation is not None:
        translation.close()
    for key in ("upload_state", "transcript_source", "conversation_config", "microphone_context",
                "model_error", "upload_error", "evaluation_error", "recording_ice"):
        st.session_state.pop(key, None)


def terminology_path(value):
    """Resolve optional local terminology files independently of launch cwd."""
    if not value or not value.strip():
        return None
    path = Path(value.strip())
    return path if path.is_absolute() else ENV_FILE.parent / path


def prepare_conversation():
    """Validate and create a selected translator before replacing usable results."""
    load_dotenv(ENV_FILE, override=False)
    asr = load_asr_config()
    profiles = load_fast_profiles()
    alias = st.session_state.get("next_fast_profile", default_fast_profile(profiles))
    if alias not in profiles:
        raise VllmConfigError("Choose a configured translation model, then retry.")
    profile = profiles[alias]
    glossary = load_glossary(
        terminology_path(st.session_state.get("glossary_path")),
        terminology_path(st.session_state.get("dnt_path")),
    )
    translator = create_translator(profile)
    try:
        translation = TranslationSession(
            translator, contextual=True, owned_client=True, glossary=glossary,
        )
    except Exception:
        translator.close()
        raise
    return {"asr": asr, "fast": profile, "alias": alias}, translation


def activate_conversation(config, translation):
    close_conversation()
    st.session_state.conversation_config = config
    st.session_state.translation = translation


def translation_callback(translation):
    """Capture a worker, never Streamlit state, for immediate ASR publication."""
    def on_result(texts, _snapshot):
        translation.submit(texts)
    return on_result


def upload_translation_callback(translation, texts):
    prefix = tuple(texts)

    def on_result(text, _started, _finished):
        if text:
            translation.submit([*prefix, text])
    return on_result


def translation_snapshot(texts):
    translation = st.session_state.get("translation")
    if translation is None:
        return {"translations": [], "errors": [], "filtered": [], "pending": 0}
    translation.submit(texts, allow_prefix=True)
    return translation.snapshot()


def conversation_args(texts, translated):
    return texts, translated["translations"], translated["errors"]


def conversation_extras(state, translated):
    config = st.session_state.get("conversation_config")
    profile = config["fast"] if config else None
    return {
        "filtered": translated.get("filtered", []),
        "timings": state.get("timings", []) if state else [],
        "profile_label": profile.label if profile else "Fast English",
        "model": profile.model if profile else None,
    }


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
    """Return the source reference used to score ASR transcription."""
    primary = evaluation.get("reference") or {
        "text": evaluation["reference_text"], "name": evaluation["reference_name"],
        "has_cjk": False, "format": "Plain text",
    }
    if evaluation.get("reference_kind", "english") == "source":
        return {"asr": primary}
    # A previously submitted translation reference cannot grade source speech.
    return {"asr": None}


def evaluation_results(state):
    """Score the complete ASR transcript; translators are never scored.

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
            "label": "ASR",
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
    """Show ASR's whole-file transcription score."""
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
        for provider in ("asr",):
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
        "Evaluation: ASR transcription score",
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
    """Render source and the one frozen English profile, without model calls."""
    texts = state["texts"] if state else []
    translated = translation_snapshot(texts)
    extras = conversation_extras(state, translated)
    is_upload = state is not None and "name" in state
    is_evaluation = state is not None and "evaluation" in state
    status, tone = "Ready", ""
    playing = bool(context and context.state.playing)
    if state:
        if state["error"]:
            status = "Needs attention"
        elif state["pending"]:
            status, tone = "Transcribing…", "working"
        elif state["accepting"]:
            status, tone = ("Listening…", "active") if playing else ("Connecting microphone…", "working")
        elif not state["finished"]:
            status, tone = "Finishing…", "working"
        elif translated["pending"]:
            status, tone = "Translating…", "working"
        elif any(translated["errors"]):
            status = "Needs attention"
        else:
            status = "Evaluation complete" if is_evaluation else "File transcribed" if is_upload else "Recording stopped"
    heading = "Evaluation results" if is_evaluation else "Your conversation"
    st.markdown(
        f'<div class="transcript-heading"><h2>{heading}</h2>'
        f'<span class="status-badge {tone}" role="status">{status}</span></div>',
        unsafe_allow_html=True,
    )
    if state and state["error"]:
        st.error(state["error"])
    if is_upload:
        st.markdown(f'<p class="source-file">File: {escape(state["name"])}</p>', unsafe_allow_html=True)
    if state and state["pending"]:
        action = "Listening and transcribing" if state["accepting"] else "Finishing transcription"
        st.caption(f"{action} · {state['pending']} segment(s) pending")
    elif state and state["accepting"] and not playing:
        st.caption("If the connection stalls, open Microphone settings to check browser access.")
    if state and state["finished"] and not texts and not state["error"]:
        st.info("No speech detected in this file. Try another WAV file." if is_upload
                else "No speech detected. Start a new recording to try again.")
    if translated["pending"]:
        st.caption(f"{extras['profile_label']} · Translating {translated['pending']} segment(s) into English")
    errors = list(dict.fromkeys(error for error in translated["errors"] if error))
    if errors:
        st.warning(" ".join(errors))
        if st.button("Retry translation", key="retry_translation", icon=":material/refresh:"):
            st.session_state.translation.retry_failed()
            st.rerun()
    if is_evaluation:
        results = evaluation_results(state)
        evaluation_panel(state, results)
    args = conversation_args(texts, translated)
    st.markdown(render_transcript(*args, **extras, **reference_view_args(state)), unsafe_allow_html=True)
    download_text = export_conversation(*args, **extras)
    if is_evaluation:
        download_text += "\n\n" + evaluation_report(state, results)
    if texts:
        with st.expander("Session exports"):
            config = st.session_state["conversation_config"]
            record = {
                "config": {"asr": config["asr"].public(), "translation": config["fast"].public(),
                           "profile": config["alias"]},
                "segments": conversation_records(*args, **extras),
                "asr_finished": state["finished"], "asr_failed": bool(state["error"]),
            }
            st.download_button(
                "Download session JSON", json.dumps(record, ensure_ascii=False, indent=2),
                "vllm-session.json", "application/json", key="download_session", on_click="ignore",
            )
            st.download_button(
                "Download bilingual CSV", export_bilingual_csv(*args, **extras),
                "vllm-transcript.csv", "text/csv", key="download_csv", on_click="ignore",
            )
            for caption_format in ("srt", "vtt"):
                captions = export_captions(*args, format=caption_format, **extras)
                st.download_button(
                    f"Download {caption_format.upper()}", captions, f"vllm-transcript.{caption_format}",
                    "text/vtt" if caption_format == "vtt" else "text/plain", disabled=not captions,
                    key=f"download_{caption_format}", on_click="ignore",
                )
    with st.container(key="transcript_footer"):
        count_column, download_column = st.columns([1, 1], vertical_alignment="center")
        count_column.markdown(
            f'<span class="segment-count">{len(texts)} '
            f'{"segment" if len(texts) == 1 else "segments"} · Numbered by pause</span>',
            unsafe_allow_html=True,
        )
        download_column.download_button(
            "Download evaluation" if is_evaluation else "Download conversation", data=download_text,
            file_name="vllm-evaluation.txt" if is_evaluation else "vllm-conversation.txt", mime="text/plain",
            icon=":material/download:", disabled=not texts and not is_evaluation, on_click="ignore",
            width="stretch", wrap=True, key="download_transcript",
        )


@st.fragment(run_every=0.5)
def show_transcript(session=None, context=None, control_state=None):
    state = session.snapshot() if session else st.session_state.upload_state
    if session and (state["accepting"], state["finished"]) != control_state:
        st.rerun()
    transcript_panel(state, context)


def webrtc_ice_servers():
    """Read explicit browser ICE settings without exposing TURN credentials."""
    message = "WEBRTC_ICE_SERVERS_JSON must be a JSON array of ICE servers with urls and optional username/credential."
    try:
        servers = json.loads(os.getenv("WEBRTC_ICE_SERVERS_JSON", "").strip() or "[]")
        if not isinstance(servers, list):
            raise ValueError()
        for server in servers:
            if not isinstance(server, dict) or set(server) - {"urls", "username", "credential"}:
                raise ValueError()
            urls = server.get("urls")
            urls = [urls] if isinstance(urls, str) else urls
            if not isinstance(urls, list) or not urls:
                raise ValueError()
            if any(not isinstance(url, str) or ":" not in url
                   or url.split(":", 1)[0] not in {"stun", "stuns", "turn", "turns"}
                   or not url.split(":", 1)[1] or any(character.isspace() for character in url)
                   for url in urls):
                raise ValueError()
            if any(not isinstance(server[key], str) for key in ("username", "credential") if key in server):
                raise ValueError()
        return servers
    except (TypeError, ValueError):
        raise ValueError(message) from None


def recording_panel(configuration_ready):
    session = st.session_state.get("recording")
    context = st.session_state.get("microphone_context")
    state = session.snapshot() if session else None
    browser_active = context and (context.state.playing or context.state.signalling)
    busy = bool(browser_active or (state and not state["finished"]) or st.session_state.get("upload_job") is not None)
    try:
        ice_servers = webrtc_ice_servers()
    except ValueError as exc:
        st.error(str(exc))
        configuration_ready = False
        ice_servers = []
    start_column, stop_column = st.columns(2)
    start = start_column.button(
        "Start recording", disabled=busy or not configuration_ready, type="primary", icon=":material/mic:",
        width="stretch", wrap=True, key="start_recording",
    )
    stop = stop_column.button(
        "Stop recording", disabled=not state or not state["accepting"], icon=":material/stop:",
        width="stretch", wrap=True, key="stop_recording",
    )
    if start:
        st.session_state.model_error = None
        translation = transcribe = None
        try:
            config, translation = prepare_conversation()
            with st.spinner("Preparing speech detection…"):
                vad = load_vad()
            transcribe = create_asr(config["asr"])
            new_session = LiveTranscriber(transcribe, vad, on_result=translation_callback(translation))
        except Exception as exc:
            if translation is not None:
                translation.close()
            if transcribe is not None:
                transcribe.close()
            st.session_state.model_error = str(exc) if isinstance(exc, VllmConfigError) else (
                "Could not prepare speech processing. Check the vLLM configuration and local VAD, then retry."
            )
        else:
            activate_conversation(config, translation)
            session = new_session
            st.session_state.recording = session
            st.session_state.recording_ice = ice_servers
            st.session_state.recording_number = st.session_state.get("recording_number", 0) + 1
            st.session_state.microphone_connected = False
            st.session_state.transcript_source = "microphone"
    if st.session_state.get("model_error"):
        st.error(st.session_state.model_error)
    if stop:
        session.finish()
        st.rerun()
    st.caption("Start recording and allow microphone access in your browser. A new recording clears the previous transcript.")
    with st.expander("Microphone settings", icon=":material/tune:"):
        if session is None:
            st.caption("Microphone devices will be available after you start recording.")
        else:
            context = webrtc_streamer(
                key=f"microphone-{st.session_state.recording_number}", mode=WebRtcMode.SENDONLY,
                desired_playing_state=session.snapshot()["accepting"],
                media_stream_constraints={"video": False, "audio": True},
                rtc_configuration={"iceServers": st.session_state.get("recording_ice", [])},
                audio_frame_callback=session.push, on_audio_ended=session.finish, async_processing=False,
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
                help="ASR is scored against this column. For your sample, choose text_zh_TW.",
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
        st.caption("ASR is scored against the spoken reference. A separate reference pane shows your selected text for comparison.")
        return {
            "reference_text": reference["text"], "reference_name": uploaded.name,
            "reference_kind": "source", "reference": reference, "reference_view": view,
        }, True, None
    except ValueError as exc:
        return None, True, str(exc)


def upload_panel(session, context, configuration_ready, *, evaluate=False):
    """Validate local inputs and freeze the shared profile only on submission."""
    state = session.snapshot() if session else None
    live_busy = bool((context and (context.state.playing or context.state.signalling)) or (state and not state["finished"]))
    busy = live_busy or st.session_state.get("upload_job") is not None
    error_key = "evaluation_error" if evaluate else "upload_error"
    uploaded = st.file_uploader(
        "Evaluation audio (.wav)" if evaluate else "Choose a WAV file", type=["wav"],
        key="eval_wav_file" if evaluate else "wav_file", disabled=busy,
        on_change=lambda: st.session_state.pop(error_key, None),
    )
    if uploaded is not None:
        st.audio(uploaded.getvalue(), format="audio/wav")
    evaluation, has_reference, reference_error = None, False, None
    if evaluate:
        evaluation, has_reference, reference_error = reference_inputs(busy)
    if st.button(
        "Run evaluation" if evaluate else "Transcribe file", key="evaluate_file" if evaluate else "transcribe_file",
        type="primary", icon=":material/analytics:" if evaluate else ":material/audio_file:", width="stretch", wrap=True,
        disabled=busy or not configuration_ready or uploaded is None or (evaluate and not has_reference),
    ):
        st.session_state[error_key] = None
        try:
            if reference_error:
                raise ValueError(reference_error)
            with st.spinner("Reading WAV audio…"):
                audio = decode_wav(uploaded.getvalue())
            config, translation = prepare_conversation()
        except (ValueError, OSError) as exc:
            st.session_state[error_key] = str(exc)
        except Exception:
            st.session_state[error_key] = "Could not prepare this file. Check the vLLM configuration and retry."
        else:
            activate_conversation(config, translation)
            st.session_state.upload_job = {"audio": audio, "next_segment": 0, "config": config}
            st.session_state.upload_state = {
                "texts": [], "pending": 0, "accepting": False, "finished": False,
                "error": None, "name": uploaded.name, "timings": [],
            }
            if evaluate:
                st.session_state.upload_state["evaluation"] = evaluation
            st.session_state.transcript_source = "upload"
            st.rerun()
    error = reference_error or st.session_state.get(error_key)
    if error:
        st.error(error)
    if live_busy:
        st.caption("Stop recording and wait for transcription to finish before uploading a file.")
    elif evaluate:
        st.caption("Transcribe and translate the audio, then score the source transcript against your local reference.")
    else:
        st.caption("Preview your audio, then transcribe it. A new transcription replaces the current text.")


def process_upload():
    """Keep exactly one remote ASR task per local checkpoint across UI reruns."""
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
        if segments:
            if "transcribe" not in job:
                job["transcribe"] = create_asr(job["config"]["asr"])
            progress = st.progress(job["next_segment"] / len(segments), text=f"Transcribing {len(segments)} speech segment(s)…")
            output = st.empty()
            for index in range(job["next_segment"], len(segments)):
                segment = segments[index]
                timing = {key: value for key, value in segment.items() if key != "audio"} if isinstance(segment, dict) else {}
                if "asr_task" not in job:
                    job["asr_task"] = transcribe_in_background(
                        job["transcribe"], segment["audio"] if isinstance(segment, dict) else segment,
                        on_result=upload_translation_callback(st.session_state.translation, state["texts"]),
                    )
                task = job["asr_task"]
                while not task.done():
                    translated = translation_snapshot(state["texts"])
                    output.markdown(
                        render_transcript(*conversation_args(state["texts"], translated),
                                          **conversation_extras(state, translated), **reference_view_args(state)),
                        unsafe_allow_html=True,
                    )
                    sleep(0.1)
                text, timing["asr_start_ms"], timing["asr_final_ms"] = task.result()
                if text:
                    state["texts"].append(text)
                    state["timings"].append(timing)
                job["next_segment"] = index + 1
                job.pop("asr_task")
                state["pending"] -= 1
                translated = translation_snapshot(state["texts"])
                output.markdown(
                    render_transcript(*conversation_args(state["texts"], translated),
                                      **conversation_extras(state, translated), **reference_view_args(state)),
                    unsafe_allow_html=True,
                )
                progress.progress((index + 1) / len(segments), text=f"Transcribed {index + 1} of {len(segments)} speech segments")
    except Exception:
        action = "Run evaluation" if "evaluation" in state else "Transcribe file"
        state["error"] = f"Could not transcribe this WAV file. Check the ASR endpoint and local audio, then retry {action}."
    # Streamlit reruns use BaseException: retain the task and its owned client.
    state["finished"] = True
    state["pending"] = 0
    if job.get("transcribe") is not None:
        job["transcribe"].close()
    st.session_state.pop("upload_job", None)
    st.rerun()


def translation_controls():
    """Choose the next conversation's profile for all three input modes."""
    load_dotenv(ENV_FILE, override=False)
    recording = st.session_state.get("recording")
    context = st.session_state.get("microphone_context")
    busy = st.session_state.get("upload_job") is not None or bool(
        recording and not recording.snapshot()["finished"]
    ) or bool(context and (context.state.playing or context.state.signalling))
    ready = False
    try:
        profiles = load_fast_profiles()
        default = default_fast_profile(profiles)
        load_asr_config()
    except VllmConfigError as exc:
        st.error(str(exc))
        st.caption("Configure the ASR endpoint and at least one named fast profile in .env, then restart Streamlit.")
    else:
        if st.session_state.get("next_fast_profile") not in profiles:
            st.session_state.next_fast_profile = default
        st.selectbox(
            "Translation model", list(profiles), format_func=lambda alias: profiles[alias].label,
            key="next_fast_profile", disabled=busy,
            help="Applies to the next recording, WAV upload, or evaluation. Running work keeps its selected model.",
        )
        ready = True
    with st.expander("Terminology settings"):
        st.text_input("Glossary file path", value=os.getenv("GLOSSARY_FILE", ""), key="glossary_path", disabled=busy)
        st.text_input("Do-not-translate file path", value=os.getenv("DNT_FILE", ""), key="dnt_path", disabled=busy)
        st.caption("Optional local terminology files, frozen when a conversation starts. Evaluation references are never model inputs.")
    if st.button("Clear conversation", key="clear_conversation", disabled=not st.session_state.get("conversation_config")):
        close_conversation()
        st.rerun()
    return ready


with st.container(key="workspace"):
    input_column, transcript_column = st.columns([1, 3], gap="medium")
with input_column, st.container(key="input_card"):
    st.markdown(
        '<div class="panel-heading"><div><div class="eyebrow">Speaking language</div>'
        '<div class="language-name">Taiwanese Hokkien / Mandarin / English</div></div></div>',
        unsafe_allow_html=True,
    )
    configuration_ready = translation_controls()
    microphone_tab, upload_tab, evaluation_tab = st.tabs(["Microphone", "Upload WAV", "Evaluate"])
    with microphone_tab:
        st.markdown(
            '<div class="mic-stage"><h3>Your voice, in English.</h3>'
            '<p>Speech detection stays local. Your selected vLLM endpoints transcribe and translate.</p></div>',
            unsafe_allow_html=True,
        )
        session, context = recording_panel(configuration_ready)
    with upload_tab:
        upload_panel(session, context, configuration_ready)
    with evaluation_tab:
        upload_panel(session, context, configuration_ready, evaluate=True)
    st.markdown(
        '<div class="privacy-note">Speech segments are sent to the configured vLLM ASR endpoint. '
        'Transcript text and terminology context go to the selected vLLM translation endpoint. '
        'Evaluation references stay on the app server.</div>',
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
    '<p class="workspace-note">One line per speech segment. English translations appear as they finish; '
    'reference lines follow the uploaded file.</p>', unsafe_allow_html=True,
)
