"""Run with: python -m streamlit run app.py"""

from dataclasses import replace
from html import escape
import json
import os
from pathlib import Path

from dotenv import load_dotenv
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from src.evaluation import mixed_match_score, parse_reference, word_match_score
from src.glossary import load_glossary
from src.pipeline import LiveTranscriber, load_transcriber, load_vad
from src.openai_transcription import TRANSCRIPTION_MODEL, load_openai_transcriber
from src.realtime_translation import REALTIME_TRANSLATION_MODEL, RealtimeTranslationSession
from src.speech import ASTRA_SPEECH_MODES, SPEECH_MODELS, SpeechSession, astra_speech_result
from src.speech_player import render_speech_player
from src.reasoning import (
    PROMPT_VERSION, correct_translations, correction_output_token_limit, correction_settings,
)
from src.reference_tables import (
    parsed_table_reference, read_reference_tables, recommend_columns,
    suggest_header_row, suggest_table, table_columns,
)
from src.tencent import TENCENT_ERROR_MESSAGE, local_model_status, translate_with_tencent
from src.translation import ENV_FILE, TranslationSession, background_filter_enabled
from src.slow_lane import SlowLaneConfig, SlowLaneSession
from src.conversation_review import ConversationReviewSession
from src.diarization import DiarizationSession, speaker_labels, split_speaker_turns
from src.subtitle_exports import export_bilingual_csv, export_captions
from src.ui import export_conversation, render_transcript
from src.uploads import decode_wav, prepare_speaker_turns_in_background, speech_segments, transcribe_in_background

TRANSLATION_TYPES = {
    "Compare all translations": (("openai", "tencent", "astra"), True),
    "Fast English": (("openai",), False),
    "Corrected English": (("openai", "astra"), False),
    "Fully reviewed English": (("openai", "astra"), True),
}
DEFAULT_MODEL_PAIR = "gpt-live-transcribe + gpt-6-luna"
LEGACY_MODEL_PAIR = "Breeze + OpenAI"
MODEL_PAIRS = (DEFAULT_MODEL_PAIR, LEGACY_MODEL_PAIR, REALTIME_TRANSLATION_MODEL)

st.set_page_config(page_title="Voice transcription", page_icon="🎙️", layout="wide")
st.html(Path(__file__).parent / "assets" / "style.css")
st.markdown(
    '<div class="brand-bar"><div class="brand">'
    '<svg viewBox="0 0 32 32" fill="none" stroke="currentColor" '
    'stroke-width="3" stroke-linecap="round" aria-hidden="true">'
    '<path d="M4 13v6M10 8v16M16 4v24M22 10v12M28 14v4"/></svg>'
    'speech <span>/ voice</span></div>'
    '<div class="local-badge">gpt-live-transcribe + gpt-6-luna · OpenAI</div></div>',
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


def selected_transcriber(model_pair):
    if model_pair == DEFAULT_MODEL_PAIR:
        return load_openai_transcriber()
    if model_pair == LEGACY_MODEL_PAIR:
        return get_transcriber()
    raise ValueError("Choose a supported transcription model.")


def model_selector(key, *, disabled):
    selected = st.selectbox("Transcription + translation model", MODEL_PAIRS, key=key, disabled=disabled)
    descriptions = {
        DEFAULT_MODEL_PAIR: "Audio is sent to OpenAI for transcription; gpt-6-luna translates the transcript.",
        LEGACY_MODEL_PAIR: "Breeze transcribes on this Mac; transcript text is sent to your configured OpenAI translator.",
        REALTIME_TRANSLATION_MODEL: "Audio streams to OpenAI for direct English captions. "
            "Source captions appear separately. This mode uses no Luna, Tencent, Astra, glossary, or speaker processing. "
            "Uploaded audio streams at playback speed.",
    }
    st.caption(descriptions[selected])
    return selected


def speech_controls(prefix, *, disabled, model_pair):
    enabled = st.toggle("Read English aloud", key=f"{prefix}_speech_enabled", disabled=disabled)
    model = st.selectbox("Text-to-speech model", SPEECH_MODELS,
                         key=f"{prefix}_speech_model", disabled=disabled)
    mode = ASTRA_SPEECH_MODES[0]
    if enabled and model_pair != REALTIME_TRANSLATION_MODEL:
        mode = st.selectbox("Astra speech mode", ASTRA_SPEECH_MODES,
                            key=f"{prefix}_astra_speech_mode", disabled=disabled)
        if mode == "Live corrections":
            st.caption("Speaks the first accepted Astra correction. gpt-6-astra uses Low reasoning for speed; "
                       "background review keeps your configured reasoning. Later text refinements are not replayed.")
        else:
            st.caption("Waits for all enabled Astra reviews at your configured reasoning before speaking.")
    if enabled:
        st.caption("English speech starts automatically after a 1-minute lead-in, "
                   "counted from the first accepted translation. "
                   "A small audio buffer smooths uneven delivery. "
                   "Astra modes wait for accepted corrections. Use headphones while recording.")
        if model_pair != REALTIME_TRANSLATION_MODEL:
            st.toggle("Different voice per speaker", value=True, key=f"{prefix}_speaker_voices", disabled=disabled)
            st.caption("Uses detected speaker labels. Mixed segments use the dominant speaker; "
                       "unknown speakers use the default voice.")
        else:
            st.caption("Realtime translation uses one voice because it has no speaker labels.")
    return (model if enabled else None), mode


def slow_config():
    """Freeze the controls into the new session's reproducible configuration."""
    model, reasoning_effort = correction_settings()
    return SlowLaneConfig(
        model=model, reasoning_effort=reasoning_effort,
        filter_background_speech=background_filter_enabled(),
        max_output_tokens=st.session_state.get("slow_max_tokens", correction_output_token_limit()),
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


def translation_configuration(name="Compare all translations", model_pair=LEGACY_MODEL_PAIR):
    """Return the provider choices frozen when a conversation starts."""
    if model_pair == REALTIME_TRANSLATION_MODEL:
        return {"type": "Realtime English", "providers": ("openai",), "second_review": False,
                "model_pair": model_pair, "translation_model": REALTIME_TRANSLATION_MODEL}
    providers, second_review = TRANSLATION_TYPES[name]
    return {"type": name, "providers": providers, "second_review": second_review,
            "model_pair": model_pair,
            "translation_model": "gpt-6-luna" if model_pair == DEFAULT_MODEL_PAIR else None}


def current_translation_configuration():
    # Preserve the provenance of sessions already in memory after an app update.
    fallback = LEGACY_MODEL_PAIR if st.session_state.get("transcript_source") else DEFAULT_MODEL_PAIR
    return st.session_state.get("translation_config", translation_configuration(model_pair=fallback))


def original_transcript_label():
    """Identify the ASR used for these results, independently of translation."""
    return {
        DEFAULT_MODEL_PAIR: f"OpenAI ASR · {TRANSCRIPTION_MODEL}",
        LEGACY_MODEL_PAIR: "Breeze ASR · Local",
        REALTIME_TRANSLATION_MODEL: "OpenAI ASR · gpt-realtime-whisper",
    }.get(current_translation_configuration().get("model_pair"))


def speech_source_label():
    selected = current_translation_configuration()
    if "astra" in selected["providers"]:
        return f"Astra Correction · {selected.get('astra_speech_mode', 'Full review')}"
    return "Realtime English" if selected.get("model_pair") == REALTIME_TRANSLATION_MODEL else "OpenAI English"


def new_review_session(config, glossary, selected):
    conversation_effort = config.reasoning_effort
    if (selected.get("speech_model") and selected.get("astra_speech_mode") == "Live corrections"
            and config.model == "gpt-6-astra"):
        config = replace(config, reasoning_effort="low")
    if selected["second_review"]:
        return ConversationReviewSession(correct_translations, config=config, glossary=glossary,
                                         conversation_reasoning_effort=conversation_effort)
    return SlowLaneSession(correct_translations, config=config, glossary=glossary)


def ensure_review_session(texts):
    selected = current_translation_configuration()
    if "astra" not in selected["providers"]:
        return None
    if texts and "slow_lane" not in st.session_state:
        st.session_state.slow_lane = new_review_session(slow_config(), terminology(), selected)
    return st.session_state.get("slow_lane")


def correction_callback(lane):
    """Bind fast results to their own conversation without worker UI access."""
    def on_result(texts, result):
        lane.submit(
            texts, result["translations"], draft_errors=result.get("errors", []),
            draft_filtered=result.get("filtered", []), allow_prefix=True,
        )
    return on_result


def reset_translation(*, translation_type="Compare all translations", model_pair=LEGACY_MODEL_PAIR,
                      diarization_max_pending_seconds=600, speech_model=None,
                      astra_speech_mode="Live corrections", speaker_voices=True):
    """End old work and allocate independent fast, slow, and speaker workers."""
    # Load configured files before closing a usable previous session. File I/O
    # never runs on the audio, translator, or correction worker's critical path.
    glossary = None if model_pair == REALTIME_TRANSLATION_MODEL else load_glossary(
        st.session_state.get("glossary_path") or None,
        st.session_state.get("dnt_path") or None,
    )
    selected = translation_configuration(translation_type, model_pair)
    config = slow_config() if "astra" in selected["providers"] else None
    for key in ("translation", "tencent_translation", "slow_lane", "diarization", "realtime_translation", "speech"):
        previous = st.session_state.pop(key, None)
        if previous is not None:
            previous.close()
    st.session_state.translation_config = selected
    selected["speech_model"] = speech_model
    selected["astra_speech_mode"] = astra_speech_mode
    selected["speaker_voices"] = speaker_voices and model_pair != REALTIME_TRANSLATION_MODEL
    if speech_model is not None:
        st.session_state.speech = SpeechSession(speech_model, speaker_voices=selected["speaker_voices"])
    if model_pair == REALTIME_TRANSLATION_MODEL:
        return
    st.session_state.glossary = glossary
    lane = new_review_session(config, glossary, selected) if config is not None else None
    if lane is not None:
        st.session_state.slow_lane = lane
    st.session_state.translation = TranslationSession(
        glossary=glossary, on_result=correction_callback(lane) if lane is not None else None,
        model=selected["translation_model"],
    )
    if "tencent" in selected["providers"]:
        st.session_state.tencent_translation = TranslationSession(
            translate_with_tencent, failure_message=TENCENT_ERROR_MESSAGE,
            cancellable=True,
        )
    st.session_state.diarization = DiarizationSession(
        enabled=st.session_state.get("enable_diarization", True),
        max_pending_seconds=diarization_max_pending_seconds,
    )


def conversation_metadata(state, texts):
    """Copy available audio offsets and speaker labels on the Streamlit thread."""
    original_timings = state.get("timings", []) if state else []
    timings = [dict(original_timings[index]) if index < len(original_timings) and original_timings[index] else {}
               for index in range(len(texts))]
    diarization = st.session_state.get("diarization")
    speakers = None
    if diarization is not None:
        # Legacy sessions have no audio offsets. Keep their speaker unknown
        # rather than inventing a timeline from transcript or reference rows.
        speaker_state = diarization.snapshot()
        covered = speaker_state.get("processed_seconds", float("inf"))
        speakers = speaker_labels(
            [timing if "start_s" in timing and "end_s" in timing and
             (speaker_state["status"] == "complete" or timing["end_s"] <= covered + 1e-6)
             else None for timing in timings], speaker_state["segments"],
        )
        for timing, speaker in zip(timings, speakers):
            if speaker is not None:
                timing["speaker_id"] = speaker
    return timings, speakers


def conversation_extras(state, texts, translated):
    """Refresh metadata and retain polling recovery for fast-result observers."""
    lane = ensure_review_session(texts)
    timings, speakers = conversation_metadata(state, texts)
    if lane is not None:
        lane.submit(
            texts, translated["openai"]["translations"],
            draft_errors=translated["openai"]["errors"], timings=timings,
            draft_filtered=translated["openai"].get("filtered", []),
        )
        slow = lane.snapshot()
    else:
        slow = {"translations": [], "statuses": [], "authoritative": [],
                "segments": [], "pending": 0, "status": "idle", "metrics": {}}
    slow["view"] = "authoritative" if st.session_state.get("subtitle_view") == "Final record" else "speculative"
    return {"slow_lane": slow, "speakers": speakers,
            "providers": current_translation_configuration()["providers"],
            "transcription_label": original_transcript_label()}


def translation_snapshot(texts, state=None):
    """Send the same original text to independent cloud and local workers.

    Neither translator consumes the other's output or waits for it. Each queue
    deduplicates segment positions across UI polls. Creating a missing provider
    independently also preserves existing results after an app update.
    """
    selected = current_translation_configuration()
    if selected.get("model_pair") == REALTIME_TRANSLATION_MODEL:
        return {
            "openai": (state or {}).get("direct_translation", {"translations": [], "errors": [], "pending": 0}),
            "tencent": {"translations": [], "errors": [], "pending": 0},
        }
    providers = selected["providers"]
    lane = ensure_review_session(texts)
    if texts and "translation" not in st.session_state:
        st.session_state.translation = TranslationSession(
            glossary=terminology(), on_result=correction_callback(lane) if lane is not None else None,
            model=current_translation_configuration().get("translation_model"),
        )
    if texts and "tencent" in providers and "tencent_translation" not in st.session_state:
        st.session_state.tencent_translation = TranslationSession(
            translate_with_tencent, failure_message=TENCENT_ERROR_MESSAGE,
            cancellable=True,
        )
    if texts and lane is not None:
        # A fast result can arrive before the next UI refresh. Seed source and
        # available metadata before starting its provider, then let the captured
        # lane callback publish completed drafts without waiting for a poll.
        timings, _ = conversation_metadata(state, texts)
        lane.submit(texts, [], timings=timings)
    results = {}
    for provider, key in (("openai", "translation"), ("tencent", "tencent_translation")):
        translation = st.session_state.get(key) if provider in providers else None
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


def correction_status(slow):
    """Show review progress and recovery only for a selected correction lane."""
    slow_status = slow["status"]
    active_reviews = slow.get("active_reviews", slow.get("pending", 0))
    queued_reviews = max(0, slow.get("pending", 0) - active_reviews)
    review_progress = f"{active_reviews} segment(s) under review"
    if queued_reviews:
        review_progress += f" · {queued_reviews} awaiting review"
    status_caption = {
        "active": f"Astra · {review_progress}",
        "paused": "Astra · Corrections paused · Fast translations continue",
        "degraded": "Astra · Some reviews need retry · Fast translations continue",
        "closed": "Astra · Session ended",
        "idle": "Astra · Ready",
    }.get(slow_status, "Astra")
    if slow_status == "degraded" and slow.get("pending"):
        status_caption += f" · {review_progress}"
    if slow.get("config"):
        config = slow["config"]
        status_caption += f" · {config['model']} · {config['reasoning_effort'].capitalize()} reasoning"
    st.caption(status_caption)
    conversation_review = slow.get("conversation_review", {})
    if conversation_review.get("enabled"):
        reviewed, total = conversation_review.get("reviewed", 0), conversation_review.get("total", 0)
        progress = f"Conversation review {reviewed}/{total}"
        if conversation_review.get("status") == "degraded":
            progress += " · Needs retry"
        elif conversation_review.get("status") == "paused":
            progress += " · Paused"
        elif conversation_review.get("active_reviews"):
            progress += " · Reviewing"
        elif conversation_review.get("pending"):
            progress += " · Waiting for earlier reviews"
        elif total and reviewed == total:
            progress += " · Complete"
        st.caption(progress)
    if slow_status == "degraded" and st.button("Retry open corrections", key="retry_corrections"):
        st.session_state.slow_lane.retry_failed()
        st.rerun()
    review_errors = list(dict.fromkeys(
        segment["error"] for segment in slow.get("segments", []) if segment.get("error")
    ))
    if review_errors:
        st.warning("Astra: " + " ".join(review_errors))
    conversation_errors = list(dict.fromkeys(
        segment["conversation_review_error"] for segment in slow.get("segments", [])
        if segment.get("conversation_review_error")
    ))
    if conversation_errors:
        st.warning("Conversation review: " + " ".join(conversation_errors))


def local_worker_status():
    """Show local worker readiness and backlog during and after ASR."""
    translation = st.session_state.get("tencent_translation")
    if translation is not None:
        activity = translation.progress()
        rows = ", ".join(str(row) for row in activity["active_rows"])
        st.caption(f"Tencent · Local · {local_model_status()} · "
                   f"active row {rows or '—'} · {activity['queued']} queued · "
                   f"{activity['elapsed_s']:.0f}s on current row")
    speaker_state = None
    diarization = st.session_state.get("diarization")
    if diarization is not None:
        speaker_state = diarization.snapshot()
        st.caption(f"Nemotron · Speaker detection {speaker_state['status']}")
        if speaker_state.get("received_seconds"):
            st.caption(f"Speaker timing: {speaker_state['processed_seconds']:.1f}s / "
                       f"{speaker_state['received_seconds']:.1f}s processed")
        if speaker_state.get("error"):
            st.caption(speaker_state["error"])
    return speaker_state


def transcript_panel(state=None, context=None):
    """Draw the current snapshot without changing the recording session."""
    texts = state["texts"] if state else []
    translated = translation_snapshot(texts, state)
    extras = conversation_extras(state, texts, translated)
    translation_pending = any(result["pending"] for result in translated.values())
    translation_failed = any(any(result["errors"]) for result in translated.values())
    is_upload = state is not None and "name" in state
    is_evaluation = state is not None and "evaluation" in state
    status, tone = "Ready", ""
    if state:
        if state["error"]:
            status = "Needs attention"
        elif state.get("realtime") and is_upload and not state["finished"]:
            status, tone = "Streaming translation…", "working"
        elif state["pending"]:
            status, tone = "Transcribing…", "working"
        elif state["accepting"]:
            status, tone = (
                ("Listening…", "active") if context.state.playing
                else ("Connecting microphone…", "working")
            )
        elif state["finished"]:
            status = "Evaluation complete" if is_evaluation else (
                ("File translated" if state.get("realtime") else "File transcribed") if is_upload else "Recording stopped"
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
    model_pair = current_translation_configuration().get("model_pair", LEGACY_MODEL_PAIR)
    realtime = model_pair == REALTIME_TRANSLATION_MODEL
    st.caption(f"Transcription + translation: {model_pair}")
    if realtime and state:
        st.caption("Continuous English captions · " + (
            "Incomplete" if state["error"] else "Complete" if state["finished"] else "Streaming"
        ))
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
        if provider not in extras["providers"]:
            continue
        result = translated[provider]
        if result["pending"] and not realtime:
            st.caption(f"{label} · Translating {result['pending']} segment(s) into English")
        errors = list(dict.fromkeys(error for error in result["errors"] if error))
        if errors:
            st.warning(f"{label}: {' '.join(errors)}")
            if not realtime and st.button(f"Retry {label}", key=retry_key, icon=":material/refresh:"):
                st.session_state[key].retry_failed()
                st.rerun()
    if is_evaluation:
        results = evaluation_results(state)
        evaluation_panel(state, results)
    slow = extras["slow_lane"]
    slow_status = slow["status"]
    if "astra" in extras["providers"]:
        correction_status(slow)
    speaker_state = local_worker_status()
    st.markdown(
        render_transcript(*conversation_args(texts, translated), **extras, **reference_view_args(state)),
        unsafe_allow_html=True,
    )
    download_text = (
        f"Transcription + translation: {model_pair}\n\n"
        + export_conversation(*conversation_args(texts, translated), **extras)
    )
    speech_model = current_translation_configuration().get("speech_model")
    if speech_model:
        download_text = f"Text-to-speech: {speech_model} ({speech_source_label()})\n" + download_text
    if realtime and state:
        completion = "Incomplete" if state["error"] else "Complete" if state["finished"] else "Streaming — incomplete"
        download_text = f"English captions: {completion}\n" + download_text
    if is_evaluation:
        download_text += "\n\n" + evaluation_report(state, results)
    if slow.get("segments"):
        with st.expander("Correction history & session export"):
            st.caption(
                "Accepted corrections replace the Astra draft and show Corrected when review finishes. "
                "Unchanged reviews show Confirmed. The first review is retained in the audit; "
                "the latest accepted review appears in the transcript and exports."
            )
            st.json({"status": slow_status, "metrics": slow.get("metrics", {}),
                     "config": slow.get("config", {}), "learned_terms": slow.get("learned_terms", [])})
            record = dict(slow, prompt_version=PROMPT_VERSION,
                          translation_config=current_translation_configuration())
            if speaker_state is not None:
                record["diarization"] = speaker_state
            if st.session_state.get("speech") is not None:
                record["speech_voices"] = st.session_state.speech.snapshot()["voices"]
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
            '<span class="segment-count">Continuous source and English captions</span>' if realtime else (
                f'<span class="segment-count">{len(texts)} '
                f'{"segment" if len(texts) == 1 else "segments"} · Numbered by pause</span>'
            ),
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


@st.fragment(run_every=0.25)
def show_speech():
    """Deliver PCM independently of caption rendering, ASR and correction work."""
    speech = st.session_state.get("speech")
    if speech is None:
        render_speech_player(armed=bool(st.session_state.get("microphone_speech_enabled") or
                                       st.session_state.get("upload_speech_enabled")))
        return
    # Keep the iframe in the same first slot as standby so Start/Transcribe's
    # gesture-unlocked AudioContext survives the transition into a session.
    render_speech_player(speech)
    if st.session_state.get("transcript_source") == "upload":
        state = st.session_state.get("upload_state", {})
        job = st.session_state.get("upload_job", {})
        if job.get("realtime_session") is not None:
            state = job["realtime_session"].snapshot()
    else:
        recording = st.session_state.get("recording")
        state = recording.snapshot() if recording else {}
    texts = state.get("texts", [])
    translated = translation_snapshot(texts, state)
    selected = current_translation_configuration()
    _, speakers = conversation_metadata(state, texts)
    if "astra" in selected["providers"]:
        slow = conversation_extras(state, texts, translated)["slow_lane"]
        result = astra_speech_result(slow, segment_count=len(texts),
                                    conversation_review=selected["second_review"] and
                                    selected.get("astra_speech_mode", "Full review") == "Full review")
    else:
        result = translated["openai"]
    speech.submit(result, speakers=speakers, realtime=bool(state.get("realtime")),
                  final=bool(state.get("finished") and not result["pending"]))
    st.caption(f"Spoken English · {speech.model} · {speech_source_label()} · 1-minute lead-in")
    if "astra" in selected["providers"] and result["pending"]:
        st.caption("Voice waits for Astra's accepted corrections in conversation order. "
                   "Pending, paused or failed reviews are not spoken. Retry failed corrections to continue.")
    status = speech.snapshot()
    if status["voices"] and status["speaker_voices"]:
        st.caption("Voices · " + " · ".join(f"{speaker}: {voice.capitalize()}"
                                          for speaker, voice in status["voices"].items()))
    if status["error"]:
        st.warning(status["error"])
    if status["skipped"]:
        st.caption(f"Voice skipped {status['skipped']} unavailable English segment(s).")


def recording_panel():
    session = st.session_state.get("recording")
    context = st.session_state.get("microphone_context")
    state = session.snapshot() if session else None
    browser_active = context and (context.state.playing or context.state.signalling)
    busy = bool(
        browser_active or (state and not state["finished"])
        or st.session_state.get("upload_job") is not None
    )

    model_pair = model_selector("microphone_model_pair", disabled=busy)
    speech_model, astra_speech_mode = speech_controls("microphone", disabled=busy, model_pair=model_pair)
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
            with st.spinner("Preparing speech recognition…"):
                if model_pair == REALTIME_TRANSLATION_MODEL:
                    prepared_realtime = RealtimeTranslationSession()
                else:
                    transcribe = selected_transcriber(model_pair)
                    vad = load_vad()
            reset_translation(translation_type=st.session_state.get(
                "microphone_translation_type", "Compare all translations",
            ), model_pair=model_pair, speech_model=speech_model, astra_speech_mode=astra_speech_mode,
                speaker_voices=st.session_state.get("microphone_speaker_voices", True))
            if model_pair == REALTIME_TRANSLATION_MODEL:
                session = prepared_realtime
                st.session_state.realtime_translation = session
            else:
                session = LiveTranscriber(transcribe, vad, diarization=st.session_state.diarization)
        except Exception as exc:
            st.session_state.model_error = f"Could not prepare speech recognition: {exc}"
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
        st.selectbox(
            "Translation type", list(TRANSLATION_TYPES), key="microphone_translation_type",
            disabled=busy or model_pair == REALTIME_TRANSLATION_MODEL or bool(session and not session.snapshot()["finished"]),
            help="Applies to the next recording. Fast uses OpenAI; Corrected adds Astra; "
                 "Fully reviewed adds a conversation review. Compare all also runs Tencent locally.",
        )
        if session is None:
            st.caption("Microphone devices will be available after you start recording.")
        else:
            context = webrtc_streamer(
                key=f"microphone-{st.session_state.recording_number}",
                mode=WebRtcMode.SENDONLY,
                desired_playing_state=session.snapshot()["accepting"],
                media_stream_constraints={"video": False, "audio": {
                    "echoCancellation": True, "noiseSuppression": True,
                }},
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
    # Evaluation retains its existing Breeze baseline and local-audio behavior.
    model_pair = LEGACY_MODEL_PAIR if evaluate else model_selector("upload_model_pair", disabled=busy)
    speech_model, astra_speech_mode = (None, ASTRA_SPEECH_MODES[0]) if evaluate else speech_controls(
        "upload", disabled=busy, model_pair=model_pair,
    )
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
            realtime_session = RealtimeTranslationSession() if model_pair == REALTIME_TRANSLATION_MODEL else None
            reset_translation(model_pair=model_pair,
                              diarization_max_pending_seconds=max(600, len(audio) / 16000 + 1),
                              speech_model=speech_model, astra_speech_mode=astra_speech_mode,
                              speaker_voices=st.session_state.get("upload_speaker_voices", True))
        except (ValueError, OSError) as exc:
            st.session_state[error_key] = str(exc)
        else:
            st.session_state.upload_job = {
                "audio": audio, "name": uploaded.name, "next_segment": 0,
                "model_pair": model_pair,
            }
            st.session_state.upload_state = {
                "texts": [], "pending": 0, "accepting": False,
                "finished": False, "error": None, "name": uploaded.name,
                "timings": [],
            }
            if realtime_session is not None:
                st.session_state.realtime_translation = realtime_session
                st.session_state.upload_job["realtime_session"] = realtime_session
                realtime_session.start_file(st.session_state.upload_job.pop("audio"))
            else:
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


@st.fragment(run_every=0.5)
def show_realtime_upload():
    job = st.session_state.upload_job
    state = st.session_state.upload_state
    state.update(job["realtime_session"].snapshot())
    if state["finished"]:
        st.session_state.pop("upload_job")
        st.rerun()
    st.progress(min(1.0, state["sent_seconds"] / max(state["duration"], 0.001)),
                text="Streaming file audio into English…")
    if st.button("Stop file translation", key="stop_realtime_upload"):
        job["realtime_session"].close()
        state.update(job["realtime_session"].snapshot())
        state.update(finished=True, accepting=False, error="File translation stopped. Captions are incomplete.")
        state["direct_translation"]["pending"] = 0
        st.session_state.pop("upload_job")
        st.rerun()
    transcript_panel(state)


@st.fragment(run_every=0.5)
def process_upload():
    """Transcribe file segments in order, without filling the live audio queue.

    The job was stored on the previous run so input controls are already
    disabled. A checkpoint after each segment lets an interrupted Streamlit
    run resume without dropping or repeating completed text. Completed jobs
    are removed along with their audio and segment views; results retain only
    text and metadata, and ordinary reruns and downloads never repeat inference.
    No file or transcript is written to disk.
    """
    job = st.session_state.upload_job
    if job.get("model_pair") == REALTIME_TRANSLATION_MODEL:
        show_realtime_upload()
        return
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
            model_pair = job.get("model_pair", LEGACY_MODEL_PAIR)

            def transcribe(audio):
                # Model loading and inference both belong to the background task.
                return selected_transcriber(model_pair)(audio)

            progress = st.progress(
                job["next_segment"] / len(segments),
                text=f"Transcribing {len(segments)} speech segment(s)…",
            )
            translated = translation_snapshot(state["texts"], state)
            local_worker_status()
            output = st.empty()
            output.markdown(
                render_transcript(
                    *conversation_args(state["texts"], translated),
                    **conversation_extras(state, state["texts"], translated), **reference_view_args(state),
                ), unsafe_allow_html=True,
            )
            for index in range(job["next_segment"], len(segments)):
                segment = segments[index]
                if isinstance(segment, dict) and not segment.get("_turns_prepared"):
                    diarization = st.session_state.diarization
                    speaker_state = diarization.snapshot()
                    if "speaker_task" in job:
                        if not job["speaker_task"].done():
                            st.caption("Preparing speaker turns locally… Translations and audio continue independently.")
                            return
                        parts = job.pop("speaker_task").result()
                    elif ("end_s" in segment and speaker_state["status"] == "active" and
                          speaker_state.get("processed_seconds", 0) < segment["end_s"]):
                        job["speaker_task"] = prepare_speaker_turns_in_background(segment, diarization)
                        st.caption("Preparing speaker turns locally… Translations and audio continue independently.")
                        return
                    else:
                        timing = {key: value for key, value in segment.items() if key != "audio"}
                        parts = split_speaker_turns(segment["audio"], timing, speaker_state["segments"])
                    for part in parts:
                        part["_turns_prepared"] = True
                    segments[index:index + 1] = parts
                    if len(parts) > 1:
                        # Recompute the loop bound/progress after adding speaker turns.
                        state["pending"] = len(segments) - job["next_segment"]
                        return
                    segment = parts[0]
                # Compatibility for older in-flight jobs that stored bare arrays.
                timing = {key: value for key, value in segment.items() if key not in {"audio", "_turns_prepared"}} if isinstance(segment, dict) else {}
                if "asr_task" not in job:
                    job["asr_task"] = transcribe_in_background(
                        transcribe, segment["audio"] if isinstance(segment, dict) else segment,
                    )
                task = job["asr_task"]
                if not task.done():
                    # Keep completed fast drafts and corrections flowing while
                    # the next local decode waits for GPU access or inference.
                    translated = translation_snapshot(state["texts"], state)
                    output.markdown(
                        render_transcript(
                            *conversation_args(state["texts"], translated),
                            **conversation_extras(state, state["texts"], translated),
                            **reference_view_args(state),
                        ),
                        unsafe_allow_html=True,
                    )
                    # Never hold Streamlit's render thread waiting for ASR.
                    # The fragment polls this same Future without repeating it.
                    return
                text, timing["asr_start_ms"], timing["asr_final_ms"] = task.result()
                if text:
                    state["texts"].append(text)
                    state.setdefault("timings", []).append(timing)
                # Save progress before UI calls, which may interrupt this run.
                job["next_segment"] = index + 1
                job.pop("asr_task")
                state["pending"] -= 1
                translated = translation_snapshot(state["texts"], state)
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
                if st.session_state.get("speech") is not None and index + 1 < len(segments):
                    return
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
        if lane is not None and "astra" in current_translation_configuration()["providers"]:
            lane.set_enabled(st.session_state.enable_corrections)

    with st.expander("Translation & speaker settings"):
        st.toggle("Astra corrections", value=True, key="enable_corrections", on_change=toggle_corrections)
        try:
            model, reasoning_effort = correction_settings()
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.caption(f"{model} · {reasoning_effort.capitalize()} reasoning · Configured review setting. "
                       "Live corrections speech uses Low reasoning for the first gpt-6-astra pass.")
        st.selectbox("Astra view", ["Live subtitles", "Final record"], key="subtitle_view")
        st.toggle("Nemotron speaker detection", value=True, key="enable_diarization", disabled=busy)
        st.caption("Speaker detection runs locally. Settings below apply to the next recording or file.")
        st.caption(
            "Astra reviews each completed fast translation independently. Reviews run in parallel; "
            "each accepted correction updates its text and turns the cell subtly green as soon as it finishes. "
            "Starting a new conversation cancels pending reviews."
        )
        try:
            output_token_limit = correction_output_token_limit()
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.number_input(
                "Reasoning + output token cap", min_value=512, max_value=16384,
                value=output_token_limit, step=512, key="slow_max_tokens", disabled=busy,
                help="Combined limit for model reasoning and the translation response. "
                     "Defaults to OPENAI_CORRECTION_MAX_OUTPUT_TOKENS (16384 when unset). "
                     "A smaller cap can prevent high-reasoning reviews from finishing.",
            )
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
        '<div class="privacy-note">OpenAI transcription sends audio to OpenAI. '
        'Breeze keeps audio on this Mac. Transcript text and glossary context go to OpenAI '
        'for translation and Astra corrections. Evaluation references and Tencent processing stay local.</div>',
        unsafe_allow_html=True,
    )

with transcript_column, st.container(key="transcript_card"):
    show_speech()
    if st.session_state.get("upload_job") is not None:
        if st.session_state.upload_job.get("model_pair") == REALTIME_TRANSLATION_MODEL:
            show_realtime_upload()
        else:
            process_upload()
    elif st.session_state.get("transcript_source") == "upload":
        show_transcript()
    elif session is None:
        transcript_panel()
    else:
        state = session.snapshot()
        show_transcript(session, context, (state["accepting"], state["finished"]))

st.markdown(
    '<p class="workspace-note">Realtime translation streams continuous English captions alongside source text.</p>'
    if current_translation_configuration().get("model_pair") == REALTIME_TRANSLATION_MODEL else
    '<p class="workspace-note">One line per speech segment. '
    'Astra reviews source text and fast drafts. '
    'Speaker labels update independently; reference lines follow the uploaded file.</p>',
    unsafe_allow_html=True,
)
