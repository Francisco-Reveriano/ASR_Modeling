"""Run with: python -m streamlit run app.py"""

from html import escape
from pathlib import Path

import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from src.evaluation import mixed_match_score, parse_reference, word_match_score
from src.pipeline import LiveTranscriber, load_transcriber, load_vad
from src.tencent import TENCENT_ERROR_MESSAGE, translate_with_tencent
from src.translation import TranslationSession
from src.ui import export_conversation, render_transcript
from src.uploads import decode_wav, speech_segments

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


def reset_translation():
    """Give each provider a fresh queue when a recording or file replaces text."""
    for key in ("translation", "tencent_translation"):
        previous = st.session_state.get(key)
        if previous is not None:
            previous.close()
    st.session_state.translation = TranslationSession()
    st.session_state.tencent_translation = TranslationSession(
        translate_with_tencent, failure_message=TENCENT_ERROR_MESSAGE,
    )


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
                st.caption(f"Source transcript: {reference['name']} · {reference['format']}")
                with st.container(height=180):
                    st.text(reference["text"])
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
            "", f"Source transcript reference: {reference['name']} ({reference['format']})",
            "Spoken reference used for scoring:", reference["text"],
        ])
        if reference.get("original_text", reference["text"]) != reference["text"]:
            lines.extend(["", "Original uploaded reference:", reference["original_text"]])
    return "\n".join(lines)


def transcript_panel(state=None, context=None):
    """Draw the current snapshot without changing the recording session."""
    texts = state["texts"] if state else []
    translated = translation_snapshot(texts)
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
    st.markdown(
        render_transcript(*conversation_args(texts, translated)),
        unsafe_allow_html=True,
    )
    download_text = export_conversation(*conversation_args(texts, translated))
    if is_evaluation:
        download_text += "\n\n" + evaluation_report(state, results)
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
            session = LiveTranscriber(transcribe, vad)
        except Exception as exc:
            st.session_state.model_error = f"Could not load the local speech models: {exc}"
        else:
            reset_translation()
            st.session_state.recording = session
            st.session_state.recording_number = st.session_state.get("recording_number", 0) + 1
            st.session_state.microphone_connected = False
            st.session_state.transcript_source = "microphone"
            st.session_state.pop("upload_state", None)
    if st.session_state.get("model_error"):
        st.error(st.session_state.model_error)
    if stop:
        session.finish()

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


def reference_inputs(busy):
    """Preview parsed references and freeze them only when Run is pressed."""
    def clear_error():
        st.session_state.pop("evaluation_error", None)

    uploaded = st.file_uploader(
        "Reference transcript", type=["txt", "srt", "vtt"],
        key="eval_reference_file", disabled=busy, on_change=clear_error, max_upload_size=1,
    )
    with st.expander("Reference settings"):
        format_choice = st.selectbox(
            "Reference format", ["Auto-detect", "Plain text", "Transcript / captions"],
            key="eval_reference_format", disabled=busy, on_change=clear_error,
            help="Transcript mode removes recognized timestamps, speaker labels, and non-speech notes. "
                 "Plain text keeps the file content as written.",
        )
    st.caption("TXT, SRT or VTT · UTF-8 or UTF-16 with BOM · Up to 1 MiB per reference.")
    if uploaded is None:
        return None, False, None
    try:
        reference = parse_reference(
            uploaded.getvalue(), uploaded.name,
            format={"Auto-detect": "auto", "Plain text": "plain", "Transcript / captions": "transcript"}[format_choice],
        )
        reference["name"] = uploaded.name
        st.caption(f"{reference['format']} · {reference['segment_count']} reference segment(s)")
        with st.expander("Preview spoken reference"):
            st.caption(f"Excluded {reference['removed_lines']} metadata / non-target line(s).")
            with st.container(height=180):
                st.text(reference["text"])
        st.caption("Use the words spoken in the audio, in their original language. Breeze is scored against this transcript.")
        return {
            "reference_text": reference["text"], "reference_name": uploaded.name,
            "reference_kind": "source", "reference": reference,
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
        except ValueError as exc:
            st.session_state[error_key] = str(exc)
        else:
            reset_translation()
            st.session_state.upload_job = {
                "audio": audio, "name": uploaded.name, "next_segment": 0,
            }
            st.session_state.upload_state = {
                "texts": [], "pending": 0, "accepting": False,
                "finished": False, "error": None, "name": uploaded.name,
            }
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
                job["segments"] = speech_segments(job["audio"], load_vad())
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
                text = transcribe(segments[index])
                if text:
                    state["texts"].append(text)
                # Save progress before UI calls, which may interrupt this run.
                job["next_segment"] = index + 1
                state["pending"] -= 1
                translated = translation_snapshot(state["texts"])
                output.markdown(
                    render_transcript(*conversation_args(state["texts"], translated)),
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


with st.container(key="workspace"):
    input_column, transcript_column = st.columns([1, 3], gap="medium")
with input_column, st.container(key="input_card"):
    st.markdown(
        '<div class="panel-heading"><div><div class="eyebrow">Speaking language</div>'
        '<div class="language-name">Taiwanese Hokkien</div></div>'
        '<span class="language-tag">台語</span></div>',
        unsafe_allow_html=True,
    )
    microphone_tab, upload_tab, evaluation_tab = st.tabs(["Microphone", "Upload WAV", "Evaluate"])
    with microphone_tab:
        st.markdown(
            '<div class="mic-stage"><div class="mic-symbol" aria-hidden="true">'
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" '
            'stroke-linecap="round"><rect x="9" y="2" width="6" height="12" rx="3"/>'
            '<path d="M5 10v2a7 7 0 0 0 14 0v-2M12 19v3M8 22h8"/></svg></div>'
            '<h3>Your voice, in English.</h3>'
            '<p>Speak naturally. Compare two<br>English translations of your words.</p></div>',
            unsafe_allow_html=True,
        )
        session, context = recording_panel()
    with upload_tab:
        upload_panel(session, context)
    with evaluation_tab:
        upload_panel(session, context, evaluate=True)
    st.markdown(
        '<div class="privacy-note">Audio and Tencent translation stay on this Mac. '
        'Transcript text is also sent to OpenAI for its translation.</div>',
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
    'Both models translate the same original into English. '
    'Tencent loads locally on its first translation.</p>',
    unsafe_allow_html=True,
)
