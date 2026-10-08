"""Framework-independent conversation ownership for the local API server.

The coordinator advances providers and speech without a browser poll. Audio
capture, upload inference, translator queues and correction workers remain
independent; the only shared GPU serialization is in the existing model adapters.
"""

from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from functools import lru_cache
import json
from pathlib import Path
from threading import Event, Lock, RLock, Thread, current_thread
from time import monotonic
from uuid import uuid4

import av
import numpy as np

from src.conversation_review import ConversationReviewSession
from src.diarization import DiarizationSession, prepare_speaker_turns, speaker_labels
from src.evaluation import mixed_match_score, word_match_score
from src.glossary import load_glossary
from src.openai_transcription import load_openai_transcriber
from src.pipeline import LiveTranscriber, load_transcriber, load_vad
from src.realtime_translation import RealtimeTranslationSession
from src.reasoning import correct_translations
from src.slow_lane import SlowLaneConfig, SlowLaneSession
from src.speech import ASTRA_SPEECH_MODES, SPEECH_MODELS, SpeechSession, astra_speech_result
from src.subtitle_exports import export_bilingual_csv, export_captions
from src.tencent import TENCENT_ERROR_MESSAGE, translate_with_tencent
from src.translation import TranslationSession, background_filter_enabled
from src.translation_validation import contains_cjk
from src.ui import export_conversation
from src.uploads import speech_segments


DEFAULT_PAIR = "gpt-live-transcribe + gpt-6-luna"
LEGACY_PAIR = "Breeze + OpenAI"
REALTIME_PAIR = "gpt-realtime-translate"
TRANSLATION_TYPES = {
    "Compare all translations": (("openai", "tencent", "astra"), True),
    "Fast English": (("openai",), False),
    "Corrected English": (("openai", "astra"), False),
    "Fully reviewed English": (("openai", "astra"), True),
}
PREPARE_ERROR = "Could not prepare speech recognition. Check the selected model assets and OpenAI configuration."
UPLOAD_ERROR = "Could not transcribe this WAV file. Check the selected model and retry the upload."
_BREEZE_LOAD_LOCK = Lock()


@lru_cache(maxsize=1)
def _breeze():
    return load_transcriber()


def _select_transcriber(pair):
    if pair == DEFAULT_PAIR:
        return load_openai_transcriber()
    # lru_cache alone permits duplicate concurrent cache misses. Serialize the
    # cached call so only one set of local Breeze weights can be constructed.
    with _BREEZE_LOAD_LOCK:
        return _breeze()


def _empty_translation():
    return {"translations": [], "errors": [], "filtered": [], "pending": 0}


def _empty_correction():
    return {"translations": [], "statuses": [], "authoritative": [],
            "segments": [], "pending": 0, "status": "disabled", "metrics": {}}


def _at(values, index, default=None):
    return values[index] if index < len(values) else default


def _english(text, *, source_fallback=False):
    return text if (isinstance(text, str) and text.strip() and not source_fallback
                    and not contains_cjk(text)) else None


def _review_work_pending(snapshot):
    """Distinguish runnable reviews from rows blocked behind a failed review.

    A degraded second pass can count later rows as pending forever. They need
    an explicit retry of an earlier failure, so they must not keep the whole
    conversation busy. Work queued for a global request slot remains pending.
    """
    if not snapshot.get("pending"):
        return False
    if snapshot.get("status") != "degraded" or snapshot.get("active_reviews"):
        return True

    def ready(record):
        return record.get("target_text") is not None and not record.get("sealed") and not record.get("error")

    first = snapshot.get("first_pass", snapshot)
    if any(ready(record) for record in first.get("segments", [])):
        return True
    # Conversation reviews run in order. Only the earliest unsealed record
    # can be runnable; all later accepted first passes must wait for it.
    second = snapshot.get("second_pass") or {}
    for record in second.get("segments", []):
        if not record.get("sealed"):
            return ready(record)
    return False


class Conversation:
    """Own one immutable input configuration and its cancellable workers.

    Dependencies are factories/callables injected by tests. Public snapshots
    never expose PCM or count as a speech-player heartbeat. `speech_snapshot`
    belongs exclusively to a connected consumer.
    """

    def __init__(self, settings, *, kind="microphone", name=None, audio=None,
                 evaluation=None, dependencies=None):
        if kind not in {"microphone", "upload", "evaluation"}:
            raise ValueError("Choose microphone, upload, or evaluation.")
        self.id = uuid4().hex
        self.kind = kind
        self.name = Path(name).name if name else ("Microphone" if kind == "microphone" else "Audio upload")
        self._settings = deepcopy(settings.model_dump() if hasattr(settings, "model_dump") else dict(settings))
        if kind != "microphone":
            self._settings["translation_type"] = "Compare all translations"
        if kind == "evaluation":
            self._settings.update(model_pair=LEGACY_PAIR, speech_enabled=False)
        pair = self._settings["model_pair"]
        if pair not in {DEFAULT_PAIR, LEGACY_PAIR, REALTIME_PAIR}:
            raise ValueError("Choose a supported transcription model.")
        self._realtime = pair == REALTIME_PAIR
        self._providers, self._second_review = (("openai",), False) if self._realtime else (
            TRANSLATION_TYPES[self._settings["translation_type"]])
        self._deps = dict(dependencies or {})
        self._lock = RLock()
        self._operation = RLock()
        self._stop = Event()
        self._wake = Event()
        self._cancelled = False
        self._closed = False
        self._prepared = False
        self._finish_requested = False
        self._input_finished = kind != "microphone"
        self._input_state = {"texts": [], "timings": [], "pending": 0, "finished": False,
                             "accepting": False, "error": None}
        self._pipeline = None
        self._diarization = None
        self._translation = None
        self._tencent = None
        self._lane = None
        self._speech = None
        self._glossary = None
        self._audio = audio
        self._received_seconds = len(audio) / 16000 if audio is not None else 0.0
        self._capture_rate = None
        self._capture_samples = 0
        self._total = 0
        self._completed = 0
        self._error = None
        self._evaluation = deepcopy(evaluation)
        self._score = None
        self._speech_start_index = 0
        self._speech_baseline = ""
        self._snapshot = {}
        self._last_body = None
        self._revision = 0
        self._refresh()
        self._coordinator = Thread(target=self._coordinate, daemon=True, name=f"conversation-{self.id[:8]}")
        self._input_worker = Thread(target=self._prepare, daemon=True, name=f"input-{self.id[:8]}")
        self._coordinator.start()
        self._input_worker.start()

    def _dependency(self, name, default):
        return self._deps.get(name, default)

    def _attach(self, name, value):
        """A slow factory may finish after deletion; never attach its late worker."""
        with self._operation:
            if self._cancelled or self._closed:
                if hasattr(value, "close"):
                    value.close()
                return False
            setattr(self, name, value)
            return True

    def _prepare(self):
        try:
            if self._realtime:
                pipeline = self._dependency("realtime_factory", RealtimeTranslationSession)()
                if not self._attach("_pipeline", pipeline):
                    return
            else:
                glossary = self._dependency("load_glossary", load_glossary)(
                    self._settings.get("glossary_path") or None, self._settings.get("dnt_path") or None)
                if not self._attach("_glossary", glossary):
                    return
                lane = self._make_lane(glossary)
                if lane is not None and not self._attach("_lane", lane):
                    return
                factory = self._dependency("translation_factory", TranslationSession)
                translation = factory(glossary=glossary, on_result=self._on_translation if lane else None,
                                      model="gpt-6-luna" if self._settings["model_pair"] == DEFAULT_PAIR else None)
                if not self._attach("_translation", translation):
                    return
                if "tencent" in self._providers:
                    tencent = factory(translate_with_tencent, failure_message=TENCENT_ERROR_MESSAGE, cancellable=True)
                    if not self._attach("_tencent", tencent):
                        return
                diarization = self._dependency("diarization_factory", DiarizationSession)(
                    enabled=self._settings.get("diarization", True),
                    max_pending_seconds=max(600, self._received_seconds + 1))
                if not self._attach("_diarization", diarization):
                    return
                if self.kind == "microphone":
                    transcribe = self._dependency("select_transcriber", _select_transcriber)(self._settings["model_pair"])
                    if self._stop.is_set():
                        return
                    pipeline = LiveTranscriber(transcribe, self._dependency("load_vad", load_vad)(),
                                               diarization=diarization)
                    if not self._attach("_pipeline", pipeline):
                        return
            if self.kind == "microphone" or self._realtime:
                if not self._mark_prepared():
                    return
            self._wake.set()
            if self.kind != "microphone":
                if self._realtime:
                    with self._operation:
                        if not self._stop.is_set() and not self._finish_requested:
                            self._pipeline.start_file(self._audio)
                            self._audio = None
                else:
                    self._run_upload()
        except Exception:
            with self._operation:
                if not self._stop.is_set():
                    self._error = UPLOAD_ERROR if self._prepared else PREPARE_ERROR
                    self._input_state.update(finished=True, accepting=False, pending=0)
                    self._input_finished = True
        finally:
            with self._lock:
                self._audio = None
            self._wake.set()

    def _mark_prepared(self):
        with self._operation:
            if self._stop.is_set():
                return False
            if self._settings.get("speech_enabled"):
                self._enable_speech(initial=True)
            self._prepared = True
            if self.kind == "microphone" and self._finish_requested:
                self._pipeline.finish()
            self._wake.set()
            return True

    def _make_lane(self, glossary):
        if "astra" not in self._providers:
            return None
        config = SlowLaneConfig(
            model=self._settings.get("correction_model", "gpt-6-astra"),
            reasoning_effort=self._settings.get("reasoning_effort", "medium"),
            max_output_tokens=self._settings.get("max_output_tokens", 4096),
            confidence_threshold=self._settings.get("confidence_threshold", 0.6),
            enabled=self._settings.get("corrections_enabled", True),
            filter_background_speech=background_filter_enabled(),
        )
        conversation_effort = config.reasoning_effort
        if (self._settings.get("speech_enabled") and self._settings.get("astra_speech_mode") == "Live corrections"
                and config.model == "gpt-6-astra"):
            config = replace(config, reasoning_effort="low")
        correct = self._dependency("correct", correct_translations)
        if self._second_review:
            return ConversationReviewSession(correct, config=config, glossary=glossary,
                                             conversation_reasoning_effort=conversation_effort)
        return SlowLaneSession(correct, config=config, glossary=glossary)

    def _on_translation(self, texts, result):
        with self._operation:
            if self._stop.is_set() or self._lane is None:
                return
            self._lane.submit(texts, result["translations"], draft_errors=result.get("errors"),
                              draft_filtered=result.get("filtered"), allow_prefix=True)
        self._wake.set()

    def _run_upload(self):
        # This worker owns the one decoded array and its views. No frontend
        # polling, event socket, or local translation can stop its progression.
        audio = self._audio
        if audio is None or self._stop.is_set():
            return
        if self._finish_requested:
            with self._lock:
                self._input_state.update(finished=True, pending=0)
                self._error = "File transcription stopped. Results are incomplete."
            return
        self._diarization.push(audio)
        self._diarization.finish()
        segments = self._dependency("speech_segments", speech_segments)(
            audio, self._dependency("load_vad", load_vad)(), with_timestamps=True)
        with self._lock:
            self._total = len(segments)
            self._input_state["pending"] = len(segments)
        transcribe = (self._dependency("select_transcriber", _select_transcriber)(self._settings["model_pair"])
                      if segments and not self._stop.is_set() and not self._finish_requested else None)
        # Expose readiness only after VAD and the selected ASR callable are
        # usable. The browser retains previous results during this phase.
        if not self._mark_prepared():
            return
        for segment in segments:
            if self._stop.is_set() or self._finish_requested:
                break
            timing = {key: value for key, value in segment.items() if key != "audio"}
            parts = prepare_speaker_turns(segment["audio"], timing, self._diarization, timeout=30)
            with self._lock:
                self._total += len(parts) - 1
                self._input_state["pending"] += len(parts) - 1
            for part in parts:
                if self._stop.is_set() or self._finish_requested:
                    break
                timing = {key: value for key, value in part.items() if key != "audio"}
                timing["asr_start_ms"] = monotonic() * 1000
                text = transcribe(part["audio"])
                timing["asr_final_ms"] = monotonic() * 1000
                with self._lock:
                    if self._stop.is_set():
                        return
                    if text:
                        self._input_state["texts"].append(text)
                        self._input_state["timings"].append(timing)
                    self._completed += 1
                    self._input_state["pending"] -= 1
                self._wake.set()
        with self._lock:
            if not self._stop.is_set():
                self._input_state.update(finished=True, pending=0)
                if self._finish_requested and self._completed < self._total:
                    self._error = "File transcription stopped. Results are incomplete."

    def _coordinate(self):
        while not self._stop.is_set():
            try:
                with self._operation:
                    if not self._stop.is_set():
                        self._refresh()
            except Exception:
                with self._lock:
                    self._error = "Conversation processing failed. End this session and start again."
                self.cancel()
                return
            self._wake.wait(0.05)
            self._wake.clear()

    def _capture_state(self):
        with self._lock:
            state = deepcopy(self._input_state)
        if self._pipeline is not None:
            state = self._pipeline.snapshot()
        if self._prepared and self.kind == "microphone" and not state.get("accepting", False):
            self._input_finished = True
        return state

    def _metadata(self, state):
        timings = deepcopy(state.get("timings", []))
        diarization = self._diarization.snapshot() if self._diarization else {
            "status": "disabled", "segments": [], "pending": 0, "error": None,
            "received_seconds": 0, "processed_seconds": 0}
        covered = diarization.get("processed_seconds", 0)
        valid = [timing if "end_s" in timing and "start_s" in timing and
                 (diarization["status"] == "complete" or timing["end_s"] <= covered + 1e-6)
                 else None for timing in timings]
        speakers = speaker_labels(valid, diarization.get("segments", []))
        for index, timing in enumerate(timings):
            # A label frozen at a speaker boundary remains available even if
            # optional diarization later fails or is canceled.
            speakers[index] = speakers[index] or timing.get("speaker_id")
            if speakers[index]:
                timing["speaker_id"] = speakers[index]
        return timings, speakers, diarization

    def _refresh(self):
        if self._closed or self._cancelled:
            return
        state = self._capture_state()
        texts = state.get("texts", [])
        timings, speakers, diarization = self._metadata(state)
        translated = {"openai": _empty_translation(), "tencent": _empty_translation()}
        if self._realtime:
            translated["openai"] = state.get("direct_translation", _empty_translation())
        else:
            if self._lane:
                self._lane.submit(texts, [], timings=timings)
            for key, provider in (("openai", self._translation), ("tencent", self._tencent)):
                if provider is not None:
                    provider.submit(texts)
                    translated[key] = provider.snapshot()
            if self._lane:
                result = translated["openai"]
                self._lane.submit(texts, result["translations"], draft_errors=result["errors"],
                                  draft_filtered=result.get("filtered"), timings=timings)
        slow = self._lane.snapshot() if self._lane else _empty_correction()
        if self._speech:
            self._submit_speech(state, translated["openai"], slow, speakers)
        error = self._error or state.get("error")
        pending = sum(translated[key].get("pending", 0) for key in ("openai", "tencent"))
        pending += int(_review_work_pending(slow))
        speaker_pending = bool(diarization.get("pending") or diarization.get("status") == "active")
        finished = bool(state.get("finished") and not pending and not speaker_pending)
        accepting = bool(self._prepared and not self._finish_requested and state.get("accepting"))
        if error:
            status = "failed" if finished or not self._prepared else "processing"
        elif not self._prepared:
            status = "preparing"
        elif finished:
            status = "complete"
        elif self.kind == "microphone" and accepting:
            status = "recording" if self._received_seconds else "ready"
        else:
            status = "processing"
        rows = []
        if not self._realtime:
            for index, source in enumerate(texts):
                values = {}
                for provider in ("openai", "tencent"):
                    if provider not in self._providers:
                        continue
                    result = translated[provider]
                    original = _at(result["translations"], index)
                    text, failure = _english(original), _at(result["errors"], index)
                    if original is not None and text is None and not failure:
                        failure = "English translation is unavailable."
                    filtered = _at(result.get("filtered", []), index, False)
                    values[provider] = {"text": None if filtered else text,
                                        "status": "filtered" if filtered else "unavailable" if failure else
                                        "complete" if text else "pending", "error": failure}
                if "astra" in self._providers:
                    record = _at(slow.get("segments", []), index, {})
                    values["astra"] = {
                        "text": _english(_at(slow.get("translations", []), index),
                                         source_fallback=bool(record.get("source_fallback"))),
                        "status": _at(slow.get("statuses", []), index, "waiting"),
                        "error": record.get("error") or record.get("conversation_review_error"),
                    }
                preferred = values.get("astra", {})
                fast = values.get("openai", {})
                english = preferred.get("text") or fast.get("text")
                if preferred.get("status") == "filtered" or fast.get("status") == "filtered":
                    english, row_status = None, "filtered"
                elif preferred.get("text") and preferred.get("status") in {"corrected", "confirmed"}:
                    row_status = preferred["status"]
                else:
                    row_status = "draft" if english else fast.get("status", "pending")
                timing = _at(timings, index, {})
                rows.append({"id": f"{self.id}:{index}", "index": index, "source": source,
                             "start_s": timing.get("start_s"), "end_s": timing.get("end_s"),
                             "speaker": _at(speakers, index), "english": english, "status": row_status,
                             "translations": values})
        direct = translated["openai"]
        realtime = {"source": texts[0] if texts else "", "english": _at(direct["translations"], 0) or "",
                    "incomplete": bool(not state.get("finished") or error or self._cancelled)} if self._realtime else None
        self._publish({
            "id": self.id, "kind": self.kind, "name": self.name, "status": status,
            "accepting": accepting, "input_finished": self._input_finished,
            "stop_requested": self._finish_requested, "finished": finished,
            "error": error, "settings": deepcopy(self._settings), "providers": list(self._providers),
            "transcription_label": {DEFAULT_PAIR: "OpenAI ASR · gpt-live-transcribe",
                                    LEGACY_PAIR: "Breeze ASR · Local",
                                    REALTIME_PAIR: "OpenAI ASR · gpt-realtime-whisper"}[self._settings["model_pair"]],
            "processing_disclosure": self._disclosure(), "segments": rows, "realtime": realtime,
            "progress": {"completed": self._completed if self.kind != "microphone" else len(texts),
                         "total": self._total, "received_seconds": round(self._received_seconds, 3),
                         "processed_seconds": state.get("sent_seconds", max([t.get("end_s", 0) for t in timings] or [0]))},
            "correction": slow, "diarization": diarization, "evaluation": self._evaluation_state(state),
            "workers": {key: provider.progress() for key, provider in
                        (("openai", self._translation), ("tencent", self._tencent)) if provider is not None},
            "speech": self._speech_metadata(),
        })

    def _publish(self, body):
        with self._lock:
            if body != self._last_body:
                self._revision += 1
                self._last_body = deepcopy(body)
                self._snapshot = {**body, "revision": self._revision}

    def _disclosure(self):
        if self._realtime:
            text = "Audio is sent to OpenAI for realtime English translation and separate source captions."
        elif self._settings["model_pair"] == DEFAULT_PAIR:
            text = "Audio is sent to OpenAI for transcription. Transcript text and terminology are sent to OpenAI for English translation and enabled corrections."
        else:
            text = "Breeze transcribes audio locally. Transcript text and terminology are sent to OpenAI for English translation and enabled corrections."
        if not self._realtime:
            text += " Enabled Silero, Tencent and Nemotron processing runs on this computer."
        if self._settings.get("speech_enabled"):
            text += " Accepted English text is sent to OpenAI for AI-generated speech."
        return text + " Evaluation references stay local. Session results are held in memory until cleared or disconnected for 30 minutes."

    def _evaluation_state(self, state):
        if self._evaluation is None:
            return None
        reference = self._evaluation.get("reference")
        result = {**deepcopy(self._evaluation), "metric": "Mixed match" if reference and reference.get("has_cjk") else "1-wMER",
                  "metrics": None, "status": "Waiting for transcription"}
        if not reference or self._evaluation.get("reference_kind") != "source":
            result["status"] = "Source transcript needed"
        elif state.get("finished"):
            if state.get("error") or self._error:
                result["status"] = "Transcription failed · no final score"
            else:
                if self._score is None:
                    score = mixed_match_score if reference.get("has_cjk") else word_match_score
                    self._score = score(reference["text"], " ".join(state.get("texts", [])))
                result.update(status="Final score", metrics=self._score)
        return result

    def _submit_speech(self, state, fast, slow, speakers):
        if "astra" in self._providers:
            result = astra_speech_result(slow, segment_count=len(state.get("texts", [])),
                                        conversation_review=self._second_review and
                                        self._settings.get("astra_speech_mode") == "Full review")
        else:
            result = deepcopy(fast)
        if self._realtime:
            english = _at(result.get("translations", []), 0) or ""
            if not english.startswith(self._speech_baseline):
                # Do not replay old or revised captions after opting in.
                return
            result["translations"] = [english[len(self._speech_baseline):]]
        else:
            for key in ("translations", "errors", "filtered"):
                result[key] = result.get(key, [])[self._speech_start_index:]
            result["pending"] = sum(text is None and not _at(result.get("errors", []), index)
                                    and not _at(result.get("filtered", []), index, False)
                                    for index, text in enumerate(result["translations"]))
            speakers = speakers[self._speech_start_index:]
        self._speech.submit(result, speakers=speakers, realtime=self._realtime,
                            final=bool(state.get("finished") and not result.get("pending", 0)))

    def _enable_speech(self, *, initial=False):
        if self._speech:
            self._speech.close()
        state = self._capture_state()
        self._speech_start_index = 0 if initial else len(state.get("texts", []))
        english = _at(state.get("direct_translation", {}).get("translations", []), 0) or ""
        self._speech_baseline = "" if initial else english
        self._speech = self._dependency("speech_factory", SpeechSession)(
            self._settings.get("speech_model", SPEECH_MODELS[0]),
            speaker_voices=self._settings.get("speaker_voices", True) and not self._realtime)

    def _speech_metadata(self):
        if not self._speech:
            return None
        state = self._speech.snapshot(consumer=False)
        state.pop("chunks", None)
        # Countdown is carried by the audio stream; whole seconds avoid
        # publishing revisions merely because another millisecond elapsed.
        delay = state.get("playback_delay_ms")
        if delay is not None:
            state["playback_delay_ms"] = ((delay + 999) // 1000) * 1000
        return state

    def snapshot(self):
        with self._lock:
            return deepcopy(self._snapshot)

    def speech_snapshot(self):
        with self._operation:
            return self._speech.snapshot() if self._speech and not self._cancelled else None

    def acknowledge(self, speech_session_id, played):
        with self._operation:
            if self._speech and not self._cancelled:
                self._speech.acknowledge(speech_session_id, played)

    def push_pcm(self, pcm, sample_rate):
        if not isinstance(pcm, bytes) or not pcm or len(pcm) % 2:
            raise ValueError("Audio batches must contain complete PCM16 samples.")
        if type(sample_rate) is not int or not 8000 <= sample_rate <= 192000:
            raise ValueError("Unsupported capture sample rate.")
        if len(pcm) > sample_rate * 2:
            raise ValueError("Send microphone audio in batches of at most one second.")
        with self._operation:
            if self.kind != "microphone" or not self._prepared or self._stop.is_set() or self._finish_requested:
                raise ValueError("This session is not accepting microphone audio.")
            if self._capture_rate is not None and self._capture_rate != sample_rate:
                raise ValueError("Capture sample rate cannot change during a session.")
            if not self._pipeline.snapshot().get("accepting"):
                raise ValueError("This session stopped accepting microphone audio.")
            self._capture_rate = sample_rate
            samples = np.frombuffer(pcm, dtype="<i2").copy().reshape(1, -1)
            frame = av.AudioFrame.from_ndarray(samples, format="s16", layout="mono")
            frame.sample_rate, frame.pts = sample_rate, self._capture_samples
            frame.time_base = Fraction(1, sample_rate)
            self._capture_samples += samples.shape[1]
            self._received_seconds = self._capture_samples / sample_rate
            self._pipeline.push(frame)
        self._wake.set()

    def finish(self):
        with self._operation:
            if self._stop.is_set():
                return
            self._finish_requested = self._input_finished = True
            if self._pipeline:
                if self.kind == "microphone":
                    self._pipeline.finish()
                elif self._realtime:
                    self._pipeline.close()
                    self._error = "File translation stopped. Captions are incomplete."
            self._refresh()
        self._wake.set()

    def cancel(self):
        """Preserve already-published results; discard every late worker result."""
        with self._operation:
            if self._cancelled or self._closed:
                return
            self._cancelled = True
            self._stop.set()
            self._wake.set()
            state = self.snapshot()
            for worker in (self._pipeline, self._translation, self._tencent, self._lane,
                           self._diarization, self._speech):
                if worker is not None:
                    worker.close()
            with self._lock:
                self._audio = None
            state.pop("revision", None)
            state.update(status="cancelled", accepting=False, input_finished=True, finished=True,
                         speech=None, error=self._error or state.get("error"))
            if state.get("realtime"):
                state["realtime"]["incomplete"] = True
            self._publish(state)

    def close(self):
        """Delete retained results and release workers without unbounded joins."""
        self.cancel()
        with self._operation:
            self._closed = True
            self._evaluation = self._score = self._glossary = None
            self._input_state = {"texts": [], "timings": []}
            # Closed adapters retain transcripts for stopped-session exports.
            # Deletion has a different retention contract: drop those stores,
            # even while an already-canceled provider call finishes elsewhere.
            self._pipeline = self._translation = self._tencent = None
            self._lane = self._diarization = self._speech = None
            state = self.snapshot()
            state.pop("revision", None)
            state.update(segments=[], realtime=None, correction=_empty_correction(), evaluation=None,
                         diarization={"status": "disabled", "segments": [], "pending": 0}, speech=None)
            self._publish(state)
        for thread in (self._coordinator, self._input_worker):
            if thread is not current_thread():
                thread.join(timeout=0.2)

    def command(self, command):
        kind = command.get("type")
        if kind == "stop":
            self.finish()
            return self.snapshot()
        if kind == "cancel":
            self.cancel()
            return self.snapshot()
        with self._operation:
            if self._stop.is_set():
                raise ValueError("This session has ended.")
            if kind == "retry":
                provider = {"openai": self._translation, "tencent": self._tencent, "astra": self._lane}.get(command.get("provider"))
                if provider is None:
                    raise ValueError("That provider is not active for this session.")
                provider.retry_failed()
            elif kind == "corrections":
                if type(command.get("enabled")) is not bool:
                    raise ValueError("enabled must be a boolean.")
                self._settings["corrections_enabled"] = command["enabled"]
                if self._lane:
                    self._lane.set_enabled(command["enabled"])
            elif kind == "speech":
                if type(command.get("enabled")) is not bool:
                    raise ValueError("enabled must be a boolean.")
                if self.kind == "evaluation" and command["enabled"]:
                    raise ValueError("Spoken English is unavailable for evaluation sessions.")
                model = command.get("model") or self._settings.get("speech_model", SPEECH_MODELS[0])
                mode = command.get("mode") or self._settings.get("astra_speech_mode", ASTRA_SPEECH_MODES[0])
                voices = command.get("speaker_voices")
                if voices is None:
                    voices = self._settings.get("speaker_voices", True)
                if model not in SPEECH_MODELS or mode not in ASTRA_SPEECH_MODES or type(voices) is not bool:
                    raise ValueError("Choose supported speech settings.")
                changed = (not self._settings.get("speech_enabled") or model != self._settings.get("speech_model")
                           or mode != self._settings.get("astra_speech_mode") or voices != self._settings.get("speaker_voices"))
                self._settings.update(speech_enabled=command["enabled"], speech_model=model,
                                      astra_speech_mode=mode, speaker_voices=voices)
                if not command["enabled"]:
                    if self._speech:
                        self._speech.close()
                    self._speech = None
                elif self._prepared and changed:
                    self._enable_speech()
            elif kind == "terminology":
                if self._realtime:
                    raise ValueError("Realtime translation does not use terminology files.")
                glossary_path = command.get("glossary_path", self._settings.get("glossary_path")) or None
                dnt_path = command.get("dnt_path", self._settings.get("dnt_path")) or None
                try:
                    updated = self._dependency("load_glossary", load_glossary)(glossary_path, dnt_path)
                except Exception:
                    raise ValueError("Could not reload terminology. Check the selected local files.") from None
                if self._glossary is None:
                    raise ValueError("Wait for session preparation before reloading terminology.")
                self._glossary.replace_master(updated)
                self._settings.update(glossary_path=glossary_path, dnt_path=dnt_path)
            else:
                raise ValueError("Unknown session command.")
            self._refresh()
        self._wake.set()
        return self.snapshot()

    def export(self, fmt):
        state = self.snapshot()
        rows = state["segments"]
        basename = f"conversation-{self.id[:8]}"
        if fmt == "json":
            return json.dumps(state, ensure_ascii=False, indent=2), "application/json", basename + ".json"
        if fmt == "txt":
            if state["realtime"] is not None:
                result = state["realtime"]
                text = ("Incomplete realtime translation\n\n" if result["incomplete"] else "")
                text += f"Original transcript model: {state['transcription_label']}\n\nOriginal: {result['source']}\n\nEnglish: {result['english'] or '[Translation unavailable]'}"
            else:
                text = export_conversation(
                    [r["source"] for r in rows], [r["translations"].get("openai", {}).get("text") for r in rows],
                    [r["translations"].get("openai", {}).get("error") for r in rows],
                    [r["translations"].get("tencent", {}).get("text") for r in rows],
                    [r["translations"].get("tencent", {}).get("error") for r in rows],
                    slow_lane=state["correction"], speakers=[r["speaker"] for r in rows],
                    providers=tuple(self._providers), transcription_label=state["transcription_label"])
            settings = state["settings"]
            speech = (f"{settings.get('speech_model')} · {settings.get('astra_speech_mode')} · 60-second lead-in"
                      if settings.get("speech_enabled") else "Disabled")
            metadata = (f"Model pair: {settings['model_pair']}\n"
                        f"Translation workflow: {settings['translation_type']}\n"
                        f"Spoken English: {speech}\n")
            text = metadata + "\n" + text
            if state["evaluation"] is not None:
                result = state["evaluation"]
                metric = result.get("metrics")
                report = ["Evaluation: Breeze transcription score", f"Audio: {state['name']}",
                          "Scoring: whole-file alignment; Unicode normalized; case/punctuation ignored.",
                          "Formula: matches / (matches + substitutions + deletions + insertions)",
                          "Higher is better. Text matching does not measure semantic quality."]
                if metric:
                    report.append(f"Breeze ({result['metric']}): {metric['score']:.1%} "
                                  f"(score={metric['score']:.6f}; H={metric['hits']}, "
                                  f"S={metric['substitutions']}, D={metric['deletions']}, I={metric['insertions']})")
                else:
                    report.append(f"Breeze ({result['metric']}): {result['status']}")
                for key, label in (("reference", "Source transcript reference used for scoring"),
                                   ("reference_view", "Reference for visual comparison; not scored or aligned to segments")):
                    reference = result.get(key)
                    if reference is None or (key == "reference_view" and reference == result.get("reference")):
                        continue
                    description = " · ".join(str(reference[field]) for field in ("name", "sheet", "column", "format")
                                             if reference.get(field) is not None)
                    report.extend(["", f"{label}: {description}", reference["text"]])
                    if reference.get("original_text", reference["text"]) != reference["text"]:
                        report.extend(["", "Original uploaded reference:", reference["original_text"]])
                text += "\n\n" + "\n".join(report)
            return text, "text/plain; charset=utf-8", basename + ".txt"
        if fmt not in {"csv", "srt", "vtt"}:
            raise ValueError("Choose txt, json, csv, srt, or vtt.")
        if self._realtime:
            raise ValueError("Realtime captions have no aligned timings. Use TXT or JSON.")
        captions = deepcopy(state["correction"]) if "astra" in self._providers else {
            "authoritative": [r["english"] for r in rows],
            "segments": [{"segment_id": r["id"], "source_text": r["source"], "status": r["status"],
                          "seal_reason": "filtered" if r["status"] == "filtered" else "complete",
                          "timing": {"start_s": r["start_s"], "end_s": r["end_s"]}} for r in rows],
        }
        for record, row in zip(captions.get("segments", []), rows):
            record["segment_id"] = row["id"]
        speakers = [r["speaker"] for r in rows]
        if fmt == "csv":
            return export_bilingual_csv(captions, speakers=speakers), "text/csv; charset=utf-8", basename + ".csv"
        return (export_captions(captions, format=fmt, speakers=speakers),
                "text/vtt; charset=utf-8" if fmt == "vtt" else "application/x-subrip", basename + "." + fmt)
