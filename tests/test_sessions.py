"""Exercise autonomous API conversation ownership without models or network."""

from copy import deepcopy
import gc
import json
from threading import Event
from time import monotonic, sleep
import unittest
from unittest.mock import patch
import weakref

import numpy as np

from server.models import SessionSettings
from server.sessions import Conversation, DEFAULT_PAIR, LEGACY_PAIR, REALTIME_PAIR
from src.diarization import DiarizationSession
from src.glossary import Glossary
from src.speech import SpeechSession
from src.translation import TranslationSession


def confirmed(request):
    return {"corrections": [], "no_change": [
        {"segment_id": item["segment_id"], "base_version": item["base_version"]}
        for item in request["segments"]]}


class VAD:
    def __init__(self):
        self.calls = 0

    def __call__(self, chunk):
        self.calls += 1
        if self.calls == 1:
            return {"start": 0}


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.sessions = []
        self.releases = []
        self.transcribed = []
        self.translated = []
        self.reviewed = []
        self.spoken = []

    def tearDown(self):
        for session in self.sessions:
            session.cancel()
        for release in self.releases:
            release.set()
        for session in self.sessions:
            session.close()

    def wait_for(self, predicate, timeout=3):
        deadline = monotonic() + timeout
        while monotonic() < deadline:
            if predicate():
                return
            sleep(0.005)
        self.fail("Conversation did not reach the expected state: " +
                  repr([s.snapshot() for s in self.sessions]))

    def release(self):
        event = Event()
        self.releases.append(event)
        return event

    def transcribe(self, audio):
        self.transcribed.append(audio.copy())
        return f"source {len(self.transcribed)}"

    def translation_factory(self, *args, **kwargs):
        provider = "tencent" if args else "openai"

        def translate(text):
            self.translated.append((provider, text))
            return "English " + text

        return TranslationSession(translate, on_result=kwargs.get("on_result"))

    def review(self, request):
        self.reviewed.append(deepcopy(request))
        return confirmed(request)

    def synthesize(self, text, *, model, voice):
        self.spoken.append((text, model, voice))
        yield b"\x01\x00" * 4800

    def make(self, *, settings=None, dependencies=None, **kwargs):
        deps = {
            "select_transcriber": lambda pair: self.transcribe,
            "load_vad": VAD,
            "speech_segments": lambda audio, vad, **options: [
                {"audio": audio[start:start + 1600], "start_s": start / 16000,
                 "end_s": min(start + 1600, len(audio)) / 16000}
                for start in range(0, len(audio), 1600)],
            "diarization_factory": lambda **options: DiarizationSession(enabled=False),
            "translation_factory": self.translation_factory,
            "correct": self.review,
            "load_glossary": lambda *args: Glossary(),
            "speech_factory": lambda model, **options: SpeechSession(model, synthesize=self.synthesize, **options),
        }
        deps.update(dependencies or {})
        settings = settings or SessionSettings(translation_type="Fast English", diarization=False)
        session = Conversation(settings, dependencies=deps, **kwargs)
        self.sessions.append(session)
        return session

    def ready(self, **kwargs):
        session = self.make(**kwargs)
        self.wait_for(lambda: session.snapshot()["status"] == "ready")
        return session

    def test_upload_drives_all_providers_without_snapshot_polling(self):
        reviewed = Event()

        def review(request):
            response = self.review(request)
            if request.get("review_stage") == "conversation" and len(self.reviewed) == 6:
                reviewed.set()
            return response

        session = self.make(kind="upload", audio=np.ones(4800, dtype=np.float32),
                            dependencies={"correct": review})
        # Wait on provider-side evidence. No caller polls to advance processing.
        self.assertTrue(reviewed.wait(3))
        self.wait_for(lambda: session.snapshot()["finished"])
        state = session.snapshot()
        self.assertEqual(state["status"], "complete")
        self.assertEqual(state["providers"], ["openai", "tencent", "astra"])
        self.assertEqual(state["progress"]["completed"], 3)
        self.assertEqual([row["id"] for row in state["segments"]], [f"{session.id}:{i}" for i in range(3)])
        self.assertEqual([row["source"] for row in state["segments"]], ["source 1", "source 2", "source 3"])
        self.assertTrue(all(row["status"] == "confirmed" for row in state["segments"]))
        self.assertEqual([text for provider, text in self.translated if provider == "openai"],
                         [text for provider, text in self.translated if provider == "tencent"])
        self.assertIsNone(session._audio)

    def test_speaker_completion_enriches_metadata_before_conversation_finishes(self):
        speakers = ControlledSpeakers()
        session = self.make(kind="upload", audio=np.ones(1600, dtype=np.float32),
                            dependencies={"diarization_factory": lambda **options: speakers})
        self.wait_for(lambda: session.snapshot()["segments"] and
                      session.snapshot()["segments"][0]["status"] == "confirmed")
        pending = session.snapshot()
        self.assertEqual(pending["status"], "processing")
        self.assertFalse(pending["finished"])
        self.assertIsNone(pending["segments"][0]["speaker"])
        speakers.state.update(status="complete", pending=0, processed_seconds=0.1, segments=[
            {"start_s": 0, "end_s": 0.1, "speaker_id": "Speaker 2"},
        ])
        self.wait_for(lambda: session.snapshot()["finished"])
        state = session.snapshot()
        self.assertEqual(state["segments"][0]["speaker"], "Speaker 2")
        self.assertEqual(state["correction"]["segments"][0]["timing"]["speaker_id"], "Speaker 2")
        self.assertEqual(state["segments"][0]["id"], pending["segments"][0]["id"])
        self.assertGreater(state["revision"], pending["revision"])

    def test_first_translation_and_review_continue_while_later_asr_is_blocked(self):
        later_started, reviewed = Event(), Event()
        release = self.release()
        calls = []

        def transcribe(audio):
            calls.append(len(audio))
            if len(calls) == 2:
                later_started.set()
                release.wait(3)
            return f"source {len(calls)}"

        def correct(request):
            reviewed.set()
            return self.review(request)

        session = self.make(kind="upload", audio=np.ones(3200, dtype=np.float32), dependencies={
            "select_transcriber": lambda pair: transcribe, "correct": correct,
        })
        self.assertTrue(later_started.wait(1))
        self.assertTrue(reviewed.wait(1))
        self.wait_for(lambda: session.snapshot()["segments"][0]["status"] == "confirmed")
        self.assertEqual(len(session.snapshot()["segments"]), 1)
        self.assertFalse(session.snapshot()["finished"])
        release.set()
        self.wait_for(lambda: session.snapshot()["finished"])
        self.assertEqual(calls, [1600, 1600])

    def test_upload_stop_is_independent_from_fully_received_file_input(self):
        later_started, release = Event(), self.release()
        calls = []

        def transcribe(audio):
            calls.append(len(audio))
            if len(calls) == 2:
                later_started.set()
                release.wait(3)
            return f"source {len(calls)}"

        session = self.make(kind="upload", audio=np.ones(4800, dtype=np.float32),
                            dependencies={"select_transcriber": lambda pair: transcribe})
        self.assertTrue(later_started.wait(1))
        self.wait_for(lambda: bool(session.snapshot()["segments"]))
        active = session.snapshot()
        self.assertTrue(active["input_finished"])
        self.assertFalse(active["stop_requested"])
        self.assertFalse(active["finished"])
        session.finish()
        self.assertTrue(session.snapshot()["stop_requested"])
        self.assertEqual(session.snapshot()["segments"][0]["source"], "source 1")
        release.set()
        self.wait_for(lambda: session.snapshot()["finished"])
        stopped = session.snapshot()
        self.assertEqual(len(stopped["segments"]), 2)
        self.assertIn("incomplete", stopped["error"])
        self.assertEqual(calls, [1600, 1600])

    def test_pending_source_fallback_is_never_english_and_astra_can_recover_it(self):
        entered, release = Event(), self.release()
        source = "今天的天氣很好。"

        def unavailable(text):
            raise RuntimeError("SECRET upstream details")

        def translation(*args, **kwargs):
            return (self.translation_factory(*args, **kwargs) if args else
                    TranslationSession(unavailable, on_result=kwargs.get("on_result")))

        def correct(request):
            row = request["segments"][0]
            if request.get("review_stage") == "conversation":
                return confirmed(request)
            self.assertTrue(row["source_fallback"])
            entered.set()
            release.wait(3)
            return {"no_change": [], "corrections": [{
                "segment_id": row["segment_id"], "base_version": row["base_version"],
                "target_text": "The weather is nice today.", "change_type": ["omission"],
                "confidence": 0.99, "rationale": "Translated missing English.", "term_pairs": [],
            }]}

        session = self.make(kind="upload", audio=np.ones(1600, dtype=np.float32), dependencies={
            "select_transcriber": lambda pair: lambda audio: source,
            "translation_factory": translation, "correct": correct,
        })
        self.assertTrue(entered.wait(1))
        self.wait_for(lambda: bool(session.snapshot()["segments"]))
        row = session.snapshot()["segments"][0]
        self.assertEqual(row["source"], source)
        self.assertIsNone(row["english"])
        self.assertIsNone(row["translations"]["astra"]["text"])
        self.assertEqual(session.export("txt")[0].count(source), 1)
        release.set()
        self.wait_for(lambda: session.snapshot()["finished"])
        self.assertEqual(session.snapshot()["segments"][0]["english"], "The weather is nice today.")
        self.assertNotIn("SECRET", json.dumps(session.snapshot()))

    def test_upload_audio_and_segment_views_are_released_on_partial_asr_failure(self):
        audio = np.ones(3200, dtype=np.float32)
        references = [weakref.ref(audio)]
        calls = []

        def segments(audio, vad, **options):
            parts = [{"audio": audio[start:start + 1600], "start_s": start / 16000,
                      "end_s": (start + 1600) / 16000} for start in (0, 1600)]
            references.extend(weakref.ref(part["audio"]) for part in parts)
            return parts

        def transcribe(audio):
            calls.append(len(audio))
            if len(calls) == 2:
                raise RuntimeError("SECRET failing ASR details")
            return "first source"

        session = self.make(kind="evaluation", audio=audio, evaluation={
            "reference_kind": "source", "reference": {"text": "first source second", "has_cjk": False},
        }, dependencies={"speech_segments": segments, "select_transcriber": lambda pair: transcribe})
        del audio
        self.wait_for(lambda: session.snapshot()["finished"])
        session._input_worker.join(1)
        gc.collect()
        self.assertTrue(all(reference() is None for reference in references))
        state = session.snapshot()
        self.assertEqual([row["source"] for row in state["segments"]], ["first source"])
        self.assertEqual(state["status"], "failed")
        self.assertIsNone(state["evaluation"]["metrics"])
        self.assertIn("no final score", state["evaluation"]["status"])
        self.assertNotIn("SECRET", json.dumps(state))

    def test_failed_new_preparation_and_terminology_reload_preserve_previous_results(self):
        original = self.make(kind="upload", audio=np.ones(1600, dtype=np.float32))
        self.wait_for(lambda: original.snapshot()["finished"])
        before = original.snapshot()

        def broken(*args):
            raise RuntimeError("SECRET local credentials")

        replacement = self.make(dependencies={"load_glossary": broken})
        self.wait_for(lambda: replacement.snapshot()["finished"])
        self.assertEqual(replacement.snapshot()["status"], "failed")
        self.assertEqual(original.snapshot(), before)
        original._deps["load_glossary"] = broken
        with self.assertRaisesRegex(ValueError, "Could not reload terminology"):
            original.command({"type": "terminology", "glossary_path": "missing.csv"})
        self.assertEqual(original.snapshot(), before)

    def test_upload_remains_preparing_until_asr_load_succeeds_or_fails(self):
        for fails in (False, True):
            with self.subTest(fails=fails):
                entered, release = Event(), self.release()

                def load(pair):
                    entered.set()
                    release.wait(3)
                    if fails:
                        raise RuntimeError("SECRET invalid local weights")
                    return self.transcribe

                session = self.make(kind="upload", audio=np.ones(1600, dtype=np.float32),
                                    dependencies={"select_transcriber": load})
                self.assertTrue(entered.wait(1))
                sleep(0.06)
                state = session.snapshot()
                self.assertEqual(state["status"], "preparing")
                self.assertFalse(state["finished"])
                self.assertFalse(state["accepting"])
                release.set()
                self.wait_for(lambda: session.snapshot()["finished"])
                self.assertEqual(session.snapshot()["status"], "failed" if fails else "complete")
                self.assertNotIn("SECRET", json.dumps(session.snapshot()))

    def test_failed_second_review_stops_busy_status_and_retry_reopens_processing(self):
        retry_entered, retry_release = Event(), self.release()
        requests = []
        first_attempts = []

        def correct(request):
            requests.append(deepcopy(request))
            if request.get("review_stage") == "conversation" and request["segments"][0]["source_text"] == "source 1":
                first_attempts.append(True)
                if len(first_attempts) == 1:
                    raise RuntimeError("SECRET review failure")
                retry_entered.set()
                retry_release.wait(3)
            return confirmed(request)

        session = self.make(kind="upload", audio=np.ones(4800, dtype=np.float32), dependencies={"correct": correct})
        self.wait_for(lambda: session.snapshot()["finished"])
        failed = session.snapshot()
        self.assertEqual(failed["status"], "complete")
        self.assertEqual(failed["correction"]["status"], "degraded")
        self.assertGreater(failed["correction"]["pending"], 0)
        self.assertEqual(failed["correction"]["active_reviews"], 0)
        self.assertTrue(failed["segments"][0]["translations"]["astra"]["error"])
        self.assertEqual([row["english"] for row in failed["segments"]],
                         ["English source 1", "English source 2", "English source 3"])
        session.command({"type": "retry", "provider": "astra"})
        self.assertTrue(retry_entered.wait(1))
        self.assertFalse(session.snapshot()["finished"])
        retry_release.set()
        self.wait_for(lambda: session.snapshot()["finished"])
        final = session.snapshot()
        self.assertEqual(final["correction"]["pending"], 0)
        self.assertEqual(final["correction"]["conversation_review"]["reviewed"], 3)
        self.assertEqual(len([request for request in requests if request.get("review_stage") != "conversation"]), 3)
        self.assertNotIn("SECRET", json.dumps(final))

    def test_degraded_session_with_globally_queued_runnable_review_stays_busy(self):
        self.enterContext(patch("src.slow_lane._MAX_ACTIVE_REVIEWS", 1))
        second_asr, second_release = Event(), self.release()
        holder_entered, holder_release = Event(), self.release()
        calls = []

        def transcribe(audio):
            calls.append(True)
            if len(calls) == 2:
                second_asr.set()
                second_release.wait(3)
            return f"source {len(calls)}"

        def correct(request):
            if request["segments"][0]["source_text"] == "source 1":
                raise RuntimeError("A failed first row")
            return confirmed(request)

        target = self.make(kind="upload", audio=np.ones(3200, dtype=np.float32), dependencies={
            "select_transcriber": lambda pair: transcribe, "correct": correct,
        })
        self.assertTrue(second_asr.wait(1))
        self.wait_for(lambda: target.snapshot()["correction"].get("status") == "degraded")

        def hold(request):
            holder_entered.set()
            holder_release.wait(3)
            return confirmed(request)

        self.make(kind="upload", audio=np.ones(1600, dtype=np.float32), dependencies={
            "select_transcriber": lambda pair: lambda audio: "holder", "correct": hold,
        })
        self.assertTrue(holder_entered.wait(1))
        second_release.set()
        self.wait_for(lambda: len(target.snapshot()["segments"]) == 2 and
                      target.snapshot()["correction"].get("pending", 0) > 0)
        queued = target.snapshot()
        self.assertEqual(queued["correction"]["status"], "degraded")
        self.assertEqual(queued["correction"]["active_reviews"], 0)
        self.assertFalse(queued["finished"])
        holder_release.set()
        self.wait_for(lambda: target.snapshot()["finished"])

    def test_microphone_stop_flushes_last_samples_and_preserves_result(self):
        session = self.ready()
        samples = np.arange(1800, dtype=np.int16)
        session.push_pcm(samples[:900].astype("<i2").tobytes(), 48000)
        session.push_pcm(samples[900:].astype("<i2").tobytes(), 48000)
        session.finish()
        self.wait_for(lambda: session.snapshot()["finished"])
        state = session.snapshot()
        self.assertEqual(state["status"], "complete")
        self.assertFalse(state["accepting"])
        self.assertEqual(len(state["segments"]), 1)
        self.assertEqual(sum(len(part) for part in self.transcribed), 600)
        self.assertEqual(state["progress"]["received_seconds"], 0.037)
        self.assertIn("OpenAI ASR", state["transcription_label"])
        with self.assertRaises(ValueError):
            session.push_pcm(b"\x00\x00", 48000)

    def test_declared_pcm_rate_cannot_change_and_batches_are_bounded(self):
        session = self.ready()
        session.push_pcm(b"\x00\x00" * 100, 16000)
        for pcm, rate in ((b"x", 16000), (b"\x00\x00", 48000),
                          (b"\x00\x00" * 16001, 16000), (b"\x00\x00", 1)):
            with self.assertRaises(ValueError):
                session.push_pcm(pcm, rate)

    def test_settings_are_frozen_and_snapshot_revisions_only_change_with_state(self):
        settings = SessionSettings(translation_type="Fast English", diarization=False)
        session = self.ready(settings=settings)
        settings.model_pair = LEGACY_PAIR
        initial = session.snapshot()
        initial["settings"]["model_pair"] = "mutated"
        initial["segments"].append({"source": "mutated"})
        sleep(0.15)
        state = session.snapshot()
        self.assertEqual(state["settings"]["model_pair"], DEFAULT_PAIR)
        self.assertEqual(state["segments"], [])
        self.assertEqual(state["revision"], initial["revision"])
        session.finish()
        self.wait_for(lambda: session.snapshot()["finished"])
        self.assertGreater(session.snapshot()["revision"], initial["revision"])

    def test_cancel_suppresses_late_upload_and_translation_results(self):
        entered, release = Event(), self.release()

        def transcribe(audio):
            entered.set()
            release.wait(3)
            return "late transcription"

        session = self.make(kind="upload", audio=np.ones(3200, dtype=np.float32),
                            dependencies={"select_transcriber": lambda pair: transcribe})
        self.assertTrue(entered.wait(1))
        session.cancel()
        state = session.snapshot()
        release.set()
        session._input_worker.join(1)
        self.assertEqual(session.snapshot(), state)
        self.assertEqual(state["status"], "cancelled")
        self.assertEqual(state["segments"], [])
        self.assertEqual(self.translated, [])

    def test_clear_discards_results_reference_and_terminology_after_cancel(self):
        session = self.make(kind="evaluation", audio=np.ones(1600, dtype=np.float32), evaluation={
            "reference_kind": "source", "reference": {"text": "source 1", "has_cjk": False},
        })
        self.wait_for(lambda: session.snapshot()["finished"])
        self.assertTrue(session.snapshot()["segments"])
        session.close()
        state = session.snapshot()
        self.assertEqual(state["segments"], [])
        self.assertIsNone(state["evaluation"])
        self.assertIsNone(session._glossary)
        self.assertIsNone(session._audio)
        for name in ("_pipeline", "_translation", "_tencent", "_lane", "_diarization", "_speech"):
            self.assertIsNone(getattr(session, name))

    def test_cancellation_during_model_preparation_never_attaches_late_pipeline(self):
        entered, release = Event(), self.release()

        def load(pair):
            entered.set()
            release.wait(3)
            return self.transcribe

        session = self.make(dependencies={"select_transcriber": load})
        self.assertTrue(entered.wait(1))
        self.assertEqual(session.snapshot()["status"], "preparing")
        session.close()
        release.set()
        session._input_worker.join(1)
        self.assertIsNone(session._pipeline)
        self.assertEqual(session.snapshot()["status"], "cancelled")
        self.assertEqual(session.snapshot()["segments"], [])

    def test_evaluation_is_breeze_only_and_reference_never_reaches_providers(self):
        selected = []
        session = self.make(kind="evaluation", audio=np.ones(3200, dtype=np.float32), evaluation={
            "reference_kind": "source", "reference": {"text": "PRIVATE REFERENCE", "has_cjk": False},
            "reference_view": {"text": "PRIVATE ENGLISH", "name": "private.txt"},
        }, dependencies={"select_transcriber": lambda pair: selected.append(pair) or self.transcribe})
        self.wait_for(lambda: session.snapshot()["finished"])
        state = session.snapshot()
        self.assertEqual(selected, [LEGACY_PAIR])
        self.assertEqual(state["evaluation"]["status"], "Final score")
        self.assertIsNotNone(state["evaluation"]["metrics"])
        self.assertNotIn("PRIVATE", json.dumps(self.reviewed))
        self.assertNotIn("PRIVATE", repr(self.translated))
        self.assertIsNone(state["speech"])
        with self.assertRaisesRegex(ValueError, "unavailable for evaluation"):
            session.command({"type": "speech", "enabled": True})
        self.assertFalse(session.snapshot()["settings"]["speech_enabled"])
        self.assertIsNone(session.speech_snapshot())
        session.command({"type": "speech", "enabled": False})

    def test_speech_opt_in_excludes_old_segments_and_keeps_one_minute_delay(self):
        events = iter(({"start": 0}, {"end": 1024}, {"start": 1024}, {"end": 2048}))
        session = self.ready(dependencies={"load_vad": lambda: lambda chunk: next(events, None)})
        session.push_pcm(b"\x01\x00" * 1024, 16000)
        self.wait_for(lambda: session.snapshot()["segments"] and session.snapshot()["segments"][0]["english"])
        session.command({"type": "speech", "enabled": True})
        self.assertEqual(self.spoken, [])
        self.assertEqual(session._speech_start_index, 1)
        session.push_pcm(b"\x01\x00" * 1024, 16000)
        session.finish()
        self.wait_for(lambda: len(self.spoken) == 1)
        self.assertEqual(self.spoken[0][0], "English source 2")
        self.assertGreater(session.speech_snapshot()["playback_delay_ms"], 58000)

    def test_speech_follows_new_segments_in_order_and_snapshot_does_not_touch_heartbeat(self):
        settings = SessionSettings(translation_type="Fast English", diarization=False, speech_enabled=True)
        session = self.ready(settings=settings)
        session.push_pcm(b"\x01\x00" * 1024, 16000)
        session.finish()
        self.wait_for(lambda: len(self.spoken) == 1)
        self.wait_for(lambda: session.snapshot()["speech"]["buffered_seconds"] > 0)
        first_poll = session._speech._last_poll
        sleep(0.12)
        for _ in range(5):
            session.snapshot()
        self.assertEqual(session._speech._last_poll, first_poll)
        state = session.snapshot()
        self.assertNotIn("chunks", state["speech"])
        self.assertGreater(state["speech"]["playback_delay_ms"], 58000)
        playback = session.speech_snapshot()
        self.assertGreater(session._speech._last_poll, first_poll)
        self.assertTrue(playback["chunks"])
        session.acknowledge(playback["session_id"], playback["chunks"][-1]["id"])
        self.assertEqual(session.speech_snapshot()["chunks"], [])
        speech = session._speech
        session.command({"type": "speech", "enabled": False})
        self.assertTrue(speech.snapshot()["closed"])
        self.assertIsNone(session.speech_snapshot())
        self.assertTrue(session.snapshot()["segments"])

    def test_corrected_speech_waits_for_astra_and_never_replays_second_review(self):
        first_release = self.release()
        second_release = self.release()

        def review(request):
            target = request["segments"][0]
            second = request.get("review_stage") == "conversation"
            (second_release if second else first_release).wait(3)
            return {"no_change": [], "corrections": [{
                "segment_id": target["segment_id"], "base_version": target["base_version"],
                "target_text": "Final reviewed English." if second else "Accepted corrected English.",
                "change_type": ["asr_fix"], "confidence": 0.99,
                "rationale": "Confirmed from context.", "term_pairs": [],
            }]}

        session = self.ready(settings=SessionSettings(
            translation_type="Fully reviewed English", diarization=False, speech_enabled=True,
        ), dependencies={"correct": review})
        session.push_pcm(b"\x01\x00" * 512, 16000)
        session.finish()
        self.wait_for(lambda: session.snapshot()["segments"] and session.snapshot()["segments"][0]["english"])
        self.assertEqual(self.spoken, [])
        first_release.set()
        self.wait_for(lambda: len(self.spoken) == 1)
        self.assertEqual(self.spoken[0][0], "Accepted corrected English.")
        second_release.set()
        self.wait_for(lambda: session.snapshot()["finished"])
        self.assertEqual(session.snapshot()["segments"][0]["english"], "Final reviewed English.")
        self.assertEqual(len(self.spoken), 1)

    def test_exports_preserve_source_model_and_measured_caption_times(self):
        session = self.make(kind="upload", audio=np.ones(3200, dtype=np.float32))
        self.wait_for(lambda: session.snapshot()["finished"])
        for fmt in ("txt", "json", "csv", "srt", "vtt"):
            text, mime, name = session.export(fmt)
            self.assertTrue(text)
            self.assertTrue(mime)
            self.assertTrue(name.endswith("." + fmt))
        self.assertIn("OpenAI ASR · gpt-live-transcribe", session.export("txt")[0])
        self.assertIn("00:00:00,000 --> 00:00:00,100", session.export("srt")[0])
        self.assertTrue(session.export("vtt")[0].startswith("WEBVTT"))
        with self.assertRaises(ValueError):
            session.export("exe")

    def test_provider_failure_is_retryable_without_repeating_asr(self):
        attempt = []

        def translate(text):
            attempt.append(text)
            if len(attempt) == 1:
                raise RuntimeError("SECRET PROVIDER TOKEN")
            return "Recovered English"

        session = self.ready(dependencies={"translation_factory": lambda *args, **options: TranslationSession(translate)})
        session.push_pcm(b"\x01\x00" * 512, 16000)
        session.finish()
        self.wait_for(lambda: session.snapshot()["finished"])
        self.assertEqual(session.snapshot()["segments"][0]["status"], "unavailable")
        self.assertNotIn("SECRET", json.dumps(session.snapshot()))
        session.command({"type": "retry", "provider": "openai"})
        self.wait_for(lambda: session.snapshot()["segments"][0]["english"] == "Recovered English")
        self.assertEqual(len(self.transcribed), 1)

    def test_empty_upload_does_not_load_asr_and_errors_hide_credentials(self):
        def forbidden(pair):
            raise RuntimeError("SECRET ASR TOKEN")

        silent = self.make(kind="upload", audio=np.zeros(3200, dtype=np.float32), dependencies={
            "speech_segments": lambda *args, **kwargs: [], "select_transcriber": forbidden,
        })
        self.wait_for(lambda: silent.snapshot()["finished"])
        self.assertEqual(silent.snapshot()["status"], "complete")
        failed = self.make(dependencies={"select_transcriber": forbidden})
        self.wait_for(lambda: failed.snapshot()["finished"])
        self.assertEqual(failed.snapshot()["status"], "failed")
        self.assertNotIn("SECRET", json.dumps(failed.snapshot()))

    def test_realtime_opt_in_uses_current_english_as_baseline(self):
        realtime = FakeRealtime()
        session = self.ready(settings=SessionSettings(model_pair=REALTIME_PAIR),
                             dependencies={"realtime_factory": lambda: realtime})
        realtime.source, realtime.english = "Original words", "Existing English. "
        self.wait_for(lambda: session.snapshot()["realtime"]["english"])
        self.assertTrue(session.snapshot()["realtime"]["incomplete"])
        self.assertIn("Incomplete realtime translation", session.export("txt")[0])
        session.command({"type": "speech", "enabled": True})
        realtime.english += "New spoken English. "
        session.finish()
        self.wait_for(lambda: len(self.spoken) == 1)
        self.wait_for(lambda: session.snapshot()["finished"])
        self.assertFalse(session.snapshot()["realtime"]["incomplete"])
        self.assertEqual(self.spoken[0][0], "New spoken English.")
        self.assertEqual(session.snapshot()["segments"], [])
        self.assertEqual(session.snapshot()["diarization"]["status"], "disabled")
        self.assertIn("Existing English.", session.export("txt")[0])
        with self.assertRaises(ValueError):
            session.export("srt")


class ControlledSpeakers:
    def __init__(self):
        self.state = {"status": "active", "segments": [], "pending": 1, "error": None,
                      "received_seconds": 0.1, "processed_seconds": 0}

    def push(self, samples):
        pass

    def finish(self):
        pass

    def wait_for_audio(self, *args, **kwargs):
        return False

    def snapshot(self):
        return deepcopy(self.state)

    def close(self):
        self.state.update(status="disabled", pending=0)


class FakeRealtime:
    def __init__(self):
        self.source, self.english = "", ""
        self.finished = False

    def snapshot(self):
        return {"texts": [self.source] if self.source else [], "timings": [], "pending": 0,
                "accepting": not self.finished, "finished": self.finished, "error": None,
                "direct_translation": {"translations": [self.english] if self.english else [],
                                       "errors": [], "pending": int(not self.finished)}}

    def push(self, frame):
        pass

    def finish(self):
        self.finished = True

    def close(self):
        self.finished = True


if __name__ == "__main__":
    unittest.main()
