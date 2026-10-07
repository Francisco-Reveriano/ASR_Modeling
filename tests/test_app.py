"""Verify Streamlit recording state without models or microphone access."""

from io import BytesIO
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import re
from threading import Event, Thread
from time import monotonic, sleep
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import wave
import weakref

import av
import numpy as np
import streamlit as st
from streamlit.testing.v1 import AppTest


APP_FILE = Path(__file__).resolve().parents[1] / "app.py"
SOURCE_REFERENCE = (
    "# Synthetic meeting transcript\n"
    "[00:00.000 --> 00:01.250] SPK1: (overlap) 你好 Teams\n"
    "[BG 00:01.250 --> 00:02.000] (typing) keyboard noise\n"
    "[FX 00:02.000] join_chime\n"
).encode("utf-8")


def wav_bytes():
    """A short, valid WAV for exercising the actual upload decoder."""
    output = BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(np.zeros(1600, dtype="<i2").tobytes())
    return output.getvalue()


class FakeTranslationSession:
    """Controllable translation results without credentials, threads, or HTTP."""

    def __init__(self, translate=None, *, failure_message=None, glossary=None,
                 source_lang="zh-TW+en", target_lang="en", initial_results=None,
                 initial_errors=None, initial_filtered=None, on_result=None, model=None,
                 cancellable=False):
        self.translate = translate
        self.model = model
        self.failure_message = failure_message
        self.provider = "Tencent" if translate is not None else "English"
        self.initial_results = initial_results
        self.initial_errors = initial_errors or []
        self.initial_filtered = initial_filtered or []
        self.sources = []
        self.translations = []
        self.errors = []
        self.filtered = []
        self.on_result = on_result
        self.submit = Mock(side_effect=self._submit)
        self.retry_failed = Mock(side_effect=self._retry_failed)
        self.close = Mock()

    def _submit(self, texts):
        for text in texts[len(self.sources):]:
            index = len(self.sources)
            self.sources.append(text)
            self.translations.append(
                self.initial_results[index] if self.initial_results is not None and index < len(self.initial_results)
                else f"{self.provider} segment {len(self.sources)}."
            )
            self.errors.append(self.initial_errors[index] if index < len(self.initial_errors) else None)
            self.filtered.append(self.initial_filtered[index] if index < len(self.initial_filtered) else False)
            if self.on_result is not None and (self.translations[index] is not None or self.errors[index]):
                try:
                    self.on_result(self.sources.copy(), self.snapshot())
                except Exception:
                    # The real provider isolates observers and polling recovers.
                    pass

    def _retry_failed(self):
        self.errors = [None for _ in self.errors]

    def snapshot(self):
        return {
            "translations": self.translations.copy(),
            "errors": self.errors.copy(),
            "filtered": self.filtered.copy(),
            "pending": sum(
                text is None and error is None
                for text, error in zip(self.translations, self.errors)
            ),
        }

    def progress(self):
        return {"active_rows": [], "queued": self.snapshot()["pending"], "elapsed_s": 0}


class FakeRealtimeSession:
    def __init__(self):
        self._worker = Thread()
        self.state = {"texts": [], "timings": [], "pending": 0, "accepting": True,
                      "finished": False, "error": None, "realtime": True,
                      "sent_seconds": 0, "duration": 1,
                      "direct_translation": {"translations": [], "errors": [], "pending": 1}}
        self.finish = Mock(side_effect=self._finish)
        self.close = Mock(side_effect=self._finish)
        self.push = Mock(side_effect=lambda frame: frame)
        self.start_file = Mock(side_effect=self._start_file)

    def _finish(self):
        self.state.update(accepting=False, finished=True)
        self.state["direct_translation"]["pending"] = 0

    def _start_file(self, audio):
        self.state.update(texts=["你好"], sent_seconds=len(audio) / 16000, duration=len(audio) / 16000)
        self.state["direct_translation"] = {"translations": ["Hello."], "errors": [None], "pending": 0}
        self._finish()

    def snapshot(self):
        return deepcopy(self.state)


class RecordingAppTests(unittest.TestCase):
    def setUp(self):
        st.cache_resource.clear()
        self.addCleanup(st.cache_resource.clear)
        self.start_patch("dotenv.load_dotenv", return_value=False)
        self.start_patch("src.reasoning.load_dotenv", return_value=False)
        self.enterContext(patch.dict(os.environ, {
            "OPENAI_CORRECTION_MODEL": "", "OPENAI_CORRECTION_REASONING_EFFORT": "",
            "OPENAI_CORRECTION_MAX_OUTPUT_TOKENS": "",
            "OPENAI_FILTER_BACKGROUND_SPEECH": "",
        }))
        self.transcribe = Mock(return_value="A short transcript")
        self.vad = Mock(return_value=None)
        self.context = SimpleNamespace(
            state=SimpleNamespace(playing=False, signalling=False),
        )
        self.load_transcriber = self.start_patch(
            "src.pipeline.load_transcriber", return_value=self.transcribe,
        )
        self.cloud_transcribe = Mock(return_value="Cloud transcript")
        self.load_openai_transcriber = self.start_patch(
            "src.openai_transcription.load_openai_transcriber", return_value=self.cloud_transcribe,
        )
        self.make_realtime = self.start_patch(
            "src.realtime_translation.RealtimeTranslationSession", side_effect=FakeRealtimeSession,
        )
        from src.speech import SpeechSession
        self.spoken = []
        self.spoken_voices = []
        def synthesize(text, *, model, voice="coral"):
            self.spoken.append((text, model))
            self.spoken_voices.append((text, voice))
            yield b"\x00\x00" * 4800
        self.make_speech = self.start_patch(
            "src.speech.SpeechSession", side_effect=lambda model, **kwargs: SpeechSession(model, synthesize=synthesize, **kwargs),
        )
        self.speech_player = self.start_patch("src.speech_player.render_speech_player")
        self.load_vad = self.start_patch("src.pipeline.load_vad", return_value=self.vad)
        self.webrtc = self.start_patch(
            "streamlit_webrtc.webrtc_streamer", return_value=self.context,
        )
        self.make_translation = self.start_patch(
            "src.translation.TranslationSession",
            side_effect=FakeTranslationSession,
        )
        self.correct = self.start_patch("src.reasoning.correct_translations", side_effect=lambda request: {
            "corrections": [], "no_change": [
                {"segment_id": row["segment_id"], "base_version": row["base_version"]}
                for row in request["segments"]
            ],
        })
        self.diarization = Mock()
        self.diarization.snapshot.return_value = {
            "status": "complete", "segments": [], "error": None, "pending": 0,
        }
        self.make_diarization = self.start_patch(
            "src.diarization.DiarizationSession", return_value=self.diarization,
        )
        self.app = AppTest.from_file(str(APP_FILE)).run()
        self.default_model_choices = [self.app.selectbox(key=key).value
                                      for key in ("microphone_model_pair", "upload_model_pair")]
        # Existing lifecycle/evaluation fixtures exercise the preserved Breeze path.
        self.app.selectbox(key="microphone_model_pair").set_value("Breeze + OpenAI")
        self.app.selectbox(key="upload_model_pair").set_value("Breeze + OpenAI").run()
        self.addCleanup(self.finish_current_recording)
        self.assertEqual(len(self.app.exception), 0)

    def start_patch(self, target, **kwargs):
        patcher = patch(target, **kwargs)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def configure_translations(self, *, openai=None, tencent=None, openai_errors=None, tencent_errors=None,
                               openai_filtered=None):
        """Publish chosen fake outcomes on first submission, just like a real worker."""
        self.translation_sessions = {}

        def create(translate=None, **kwargs):
            session = FakeTranslationSession(
                translate, **kwargs,
                initial_results=tencent if translate is not None else openai,
                initial_errors=tencent_errors if translate is not None else openai_errors,
                initial_filtered=openai_filtered if translate is None else None,
            )
            self.translation_sessions["tencent" if translate is not None else "openai"] = session
            return session
        self.make_translation.side_effect = create

    def wait_for_corrections(self):
        """Wait for the real scheduler with a mocked model, without rerunning the UI."""
        deadline = monotonic() + 3
        while monotonic() < deadline:
            snapshot = self.app.session_state["slow_lane"].snapshot()
            if snapshot["pending"] == 0:
                return snapshot
            sleep(0.005)
        self.fail("The mocked correction worker did not drain.")

    def review_requests(self, *, conversation=False):
        return [call.args[0] for call in self.correct.call_args_list
                if (call.args[0].get("review_stage") == "conversation") == conversation]

    def finish_current_recording(self):
        if "recording" in self.app.session_state:
            session = self.app.session_state["recording"]
            session.finish()
            if session._worker.ident is not None:
                session._worker.join(timeout=5)
            self.assertFalse(session._worker.is_alive())
        if "slow_lane" in self.app.session_state:
            self.app.session_state["slow_lane"].close()
        if "realtime_translation" in self.app.session_state:
            self.app.session_state["realtime_translation"].close()
        if "speech" in self.app.session_state:
            speech = self.app.session_state["speech"]
            speech.close()
            if speech._worker:
                speech._worker.join(3)

    def start_recording(self):
        self.app.button(key="start_recording").click().run()
        self.assertEqual(len(self.app.exception), 0)
        self.app.run()
        self.assertEqual(len(self.app.exception), 0)
        return self.app.session_state["recording"]

    def rendered_text(self):
        return "\n".join(
            element.value
            for kind in ("markdown", "caption", "info", "text", "title")
            for element in self.app.get(kind)
        )

    def transcript_html(self):
        return "\n".join(element.value for element in self.app.markdown)

    def select_wav(self, name="sample.wav", content=None):
        content = wav_bytes() if content is None else content
        self.app.file_uploader(key="wav_file").set_value((name, content, "audio/wav")).run()
        self.assertEqual(len(self.app.exception), 0)

    def transcribe_file(self):
        self.app.button(key="transcribe_file").click().run()
        self.assertEqual(len(self.app.exception), 0)

    def select_evaluation(self, reference=b"English reference.", *, audio_name="evaluation.wav",
                          reference_name="reference.txt"):
        self.app.file_uploader(key="eval_wav_file").set_value(
            (audio_name, wav_bytes(), "audio/wav"),
        ).run()
        self.app.file_uploader(key="eval_reference_file").set_value(
            (reference_name, reference, "text/plain"),
        ).run()
        self.assertEqual(len(self.app.exception), 0)

    def evaluate_file(self):
        self.app.button(key="evaluate_file").click().run()
        self.assertEqual(len(self.app.exception), 0)

    def evaluation_scores(self):
        return {metric.label: metric.value for metric in self.app.metric}

    def capture_downloads(self):
        from streamlit.delta_generator import DeltaGenerator

        downloads = []
        real_download = DeltaGenerator.download_button

        def capture_download(*args, **kwargs):
            downloads.append(kwargs["data"])
            return real_download(*args, **kwargs)

        self.start_patch(
            "streamlit.delta_generator.DeltaGenerator.download_button",
            autospec=True, side_effect=capture_download,
        )
        return downloads

    def track_upload_audio(self, *, silent=False):
        """Track real decoded arrays without mock call histories retaining them."""
        from src.uploads import decode_wav

        decoded, views = [], []

        def decode(data):
            audio = decode_wav(data)
            decoded.append(weakref.ref(audio))
            return audio

        def split(audio, vad, *, with_timestamps):
            segments = []
            for start in (() if silent else (0, 800)):
                samples = audio[start:start + 800]
                views.append(weakref.ref(samples))
                segments.append({"audio": samples, "start_s": start / 16000,
                                 "end_s": (start + 800) / 16000})
            return segments

        self.start_patch("src.uploads.decode_wav", new=decode)
        self.start_patch("src.uploads.speech_segments", new=split)
        self.load_transcriber.return_value = lambda audio: "Hello"
        self.diarization.push = lambda audio: None
        return decoded, views

    def test_initial_page_waits_for_start_without_loading_models(self):
        self.assertEqual(self.app.title[0].value, "Voice transcription & translation")
        self.assertFalse(self.app.button(key="start_recording").disabled)
        self.assertTrue(self.app.button(key="stop_recording").disabled)
        self.assertTrue(self.app.download_button(key="download_transcript").disabled)
        self.assertIn("allow microphone access", self.rendered_text())
        self.load_transcriber.assert_not_called()
        self.load_vad.assert_not_called()
        self.webrtc.assert_not_called()
        self.make_translation.assert_not_called()

    def test_microphone_translation_type_is_available_before_start(self):
        control = self.app.selectbox(key="microphone_translation_type")
        self.assertEqual(control.label, "Translation type")
        self.assertEqual(control.value, "Compare all translations")
        self.assertEqual(control.options, [
            "Compare all translations", "Fast English", "Corrected English", "Fully reviewed English",
        ])
        self.assertFalse(control.disabled)
        self.make_translation.assert_not_called()

    def test_speech_controls_offer_requested_models_and_are_opt_in(self):
        for prefix in ("microphone", "upload"):
            self.assertEqual(self.app.selectbox(key=f"{prefix}_speech_model").options,
                             ["gpt-4o-mini-tts", "tts-1-hd"])
            self.assertEqual(self.app.selectbox(key=f"{prefix}_speech_model").value, "gpt-4o-mini-tts")
            self.assertFalse(self.app.toggle(key=f"{prefix}_speech_enabled").value)
        self.make_speech.assert_not_called()

    def test_player_is_armed_before_start_and_detected_speaker_selects_voice(self):
        self.app.toggle(key="microphone_speech_enabled").set_value(True).run()
        self.speech_player.assert_called_with(armed=True)
        self.assertTrue(self.app.toggle(key="microphone_speaker_voices").value)
        self.make_speech.assert_not_called()
        self.diarization.snapshot.return_value = {
            "status": "complete", "error": None, "pending": 0,
            "segments": [{"start_s": 0, "end_s": 1, "speaker_id": "Speaker 2"}],
        }
        session = self.start_recording()
        self.enterContext(patch.object(session, "snapshot", return_value={
            "texts": ["Source speech"], "timings": [{"start_s": 0, "end_s": 1}], "pending": 0,
            "accepting": False, "finished": True, "error": None,
        }))
        self.app.run()
        self.wait_for_corrections()
        self.app.run()
        self.app.session_state["speech"]._worker.join(2)
        self.assertEqual(self.spoken_voices, [("English segment 1.", "onyx")])
        self.assertIn("1-minute lead-in", self.rendered_text())
        self.assertEqual(self.app.session_state["speech"].snapshot()["voices"], {"Speaker 2": "onyx"})

    def test_realtime_microphone_speaks_only_english_and_new_session_cancels_voice(self):
        self.app.selectbox(key="microphone_model_pair").set_value("gpt-realtime-translate")
        self.app.toggle(key="microphone_speech_enabled").set_value(True).run()
        recording = self.start_recording()
        speech = self.app.session_state["speech"]
        self.assertTrue(self.app.toggle(key="microphone_speech_enabled").disabled)
        recording.state["texts"] = ["你好"]
        recording.state["direct_translation"] = {"translations": ["Hello. Next"], "errors": [None], "pending": 1}
        self.app.run()
        deadline = monotonic() + 2
        while not self.spoken and monotonic() < deadline:
            sleep(0.005)
        self.assertEqual(self.spoken, [("Hello.", "gpt-4o-mini-tts")])
        self.assertFalse(speech.snapshot()["complete"])
        self.app.button(key="stop_recording").click().run()
        self.app.toggle(key="microphone_speech_enabled").set_value(False).run()
        self.start_recording()
        self.assertTrue(speech.snapshot()["closed"])
        self.assertEqual(speech.snapshot()["chunks"], [])
        self.assertNotIn("speech", self.app.session_state)

    def test_microphone_freezes_selected_hd_speech_model_for_the_conversation(self):
        self.app.toggle(key="microphone_speech_enabled").set_value(True)
        self.app.selectbox(key="microphone_speech_model").set_value("tts-1-hd").run()
        self.start_recording()
        self.assertEqual(self.app.session_state["speech"].model, "tts-1-hd")
        self.assertEqual(self.app.session_state["translation_config"]["speech_model"], "tts-1-hd")
        self.assertTrue(self.app.selectbox(key="microphone_speech_model").disabled)

    def test_realtime_upload_uses_own_speech_setting_and_flushes_final_english(self):
        self.app.selectbox(key="upload_model_pair").set_value("gpt-realtime-translate")
        self.app.selectbox(key="upload_speech_model").set_value("tts-1-hd")
        self.app.toggle(key="upload_speech_enabled").set_value(True).run()
        self.select_wav()
        self.transcribe_file()
        speech = self.app.session_state["speech"]
        self.app.run()
        speech._worker.join(2)
        self.assertEqual(self.spoken, [("Hello.", "tts-1-hd")])
        self.app.run()
        self.assertEqual(len(self.spoken), 1)
        self.assertFalse(self.app.toggle(key="microphone_speech_enabled").value)
        self.assertEqual(self.app.session_state["translation_config"]["speech_model"], "tts-1-hd")

    def test_segmented_upload_yields_for_live_voice_and_never_speaks_tencent_or_reference(self):
        self.app.toggle(key="upload_speech_enabled").set_value(True).run()
        self.start_patch("src.uploads.speech_segments", return_value=[np.zeros(800, dtype=np.float32)] * 2)
        self.select_wav()
        self.transcribe_file()
        for _ in range(10):
            self.app.run()
            if "upload_job" not in self.app.session_state:
                break
        self.app.run()
        self.assertNotIn("upload_job", self.app.session_state)
        speech = self.app.session_state["speech"]
        speech._worker.join(2)
        self.assertEqual([text for text, _ in self.spoken], ["English segment 1.", "English segment 2."])
        self.assertEqual(self.transcribe.call_count, 2)
        self.speech_player.assert_called()
        self.app.run()
        self.assertEqual(len(self.spoken), 2)

    def test_speech_waits_for_astra_conversation_correction_and_uses_its_text(self):
        first_entered, second_entered, release_first, release_second = Event(), Event(), Event(), Event()
        self.addCleanup(release_first.set)
        self.addCleanup(release_second.set)

        def correct(request):
            second = request.get("review_stage") == "conversation"
            (second_entered if second else first_entered).set()
            if not (release_second if second else release_first).wait(10):
                raise RuntimeError("Test correction was not released")
            row = request["segments"][0]
            return {"corrections": [{
                "segment_id": row["segment_id"], "base_version": row["base_version"],
                "target_text": "Astra final English." if second else "Astra initial correction.",
                "change_type": ["style"], "confidence": 0.95,
                "rationale": "Clarify the wording.", "term_pairs": [],
            }], "no_change": []}

        self.correct.side_effect = correct
        self.app.toggle(key="upload_speech_enabled").set_value(True).run()
        self.app.selectbox(key="upload_astra_speech_mode").set_value("Full review").run()
        self.start_patch("src.uploads.speech_segments", return_value=[np.zeros(800, dtype=np.float32)])
        self.select_wav()
        downloads = self.capture_downloads()
        self.transcribe_file()
        for _ in range(5):
            self.app.run()
            if "upload_job" not in self.app.session_state:
                break
        self.assertTrue(first_entered.wait(1))
        self.app.run()
        self.assertEqual(self.spoken, [])
        self.assertIn("Astra Correction", self.rendered_text())
        self.assertIn("gpt-4o-mini-tts (Astra Correction · Full review)", downloads[-1])
        release_first.set()
        self.assertTrue(second_entered.wait(1))
        self.app.run()
        self.assertEqual(self.spoken, [])
        self.assertIn("Astra initial correction.", self.transcript_html())
        release_second.set()
        self.wait_for_corrections()
        self.app.run()
        self.app.session_state["speech"]._worker.join(2)
        self.assertEqual(self.spoken, [("Astra final English.", "gpt-4o-mini-tts")])
        self.app.run()
        self.assertEqual(len(self.spoken), 1)

    def test_live_astra_speech_arrives_while_high_reasoning_conversation_review_is_pending(self):
        entered, release = Event(), Event()
        self.addCleanup(release.set)
        requests = []

        def correct(request):
            requests.append(request)
            second = request.get("review_stage") == "conversation"
            if second:
                entered.set()
                if not release.wait(10):
                    raise RuntimeError("Test review was not released")
            row = request["segments"][0]
            return {"corrections": [{
                "segment_id": row["segment_id"], "base_version": row["base_version"],
                "target_text": "Later reviewed wording." if second else "Accepted Astra correction.",
                "change_type": ["style"], "confidence": 0.95,
                "rationale": "Clarify the wording.", "term_pairs": [],
            }], "no_change": []}

        self.enterContext(patch.dict(os.environ, {"OPENAI_CORRECTION_REASONING_EFFORT": "high"}))
        self.correct.side_effect = correct
        self.app.toggle(key="microphone_speech_enabled").set_value(True).run()
        self.assertEqual(self.app.selectbox(key="microphone_astra_speech_mode").value, "Live corrections")
        session = self.start_recording()
        self.enterContext(patch.object(session, "snapshot", return_value={
            "texts": ["Source speech"], "timings": [], "pending": 0,
            "accepting": False, "finished": True, "error": None,
        }))
        self.app.run()
        self.assertTrue(entered.wait(1))
        self.app.run()
        self.app.session_state["speech"]._worker.join(2)
        self.assertEqual(self.spoken, [("Accepted Astra correction.", "gpt-4o-mini-tts")])
        self.assertEqual([request["config"]["reasoning_effort"] for request in requests], ["low", "high"])
        # Editing the next conversation's option cannot change the current voice.
        self.app.selectbox(key="microphone_astra_speech_mode").set_value("Full review").run()
        self.assertEqual(self.app.session_state["translation_config"]["astra_speech_mode"], "Live corrections")
        release.set()
        self.wait_for_corrections()
        self.app.run()
        self.assertIn("Later reviewed wording.", self.transcript_html())
        self.assertEqual(len(self.spoken), 1)

    def test_corrected_microphone_speech_uses_first_pass_when_context_review_is_disabled(self):
        self.app.toggle(key="microphone_speech_enabled").set_value(True)
        self.app.selectbox(key="microphone_translation_type").set_value("Corrected English").run()
        session = self.start_recording()
        lane = self.app.session_state["slow_lane"]
        # Feed the same completed transcript state the recording worker exposes.
        self.enterContext(patch.object(session, "snapshot", return_value={
            "texts": ["Source speech"], "timings": [], "pending": 0,
            "accepting": False, "finished": True, "error": None,
        }))
        self.app.run()
        self.wait_for_corrections()
        self.app.run()
        self.app.session_state["speech"]._worker.join(2)
        self.assertEqual(self.spoken, [("English segment 1.", "gpt-4o-mini-tts")])
        self.assertEqual(lane.snapshot()["statuses"], ["confirmed"])
        self.assertEqual(self.review_requests(conversation=True), [])

    def test_evaluation_does_not_create_speech_from_upload_or_microphone_settings(self):
        self.app.toggle(key="upload_speech_enabled").set_value(True)
        self.app.toggle(key="microphone_speech_enabled").set_value(True).run()
        self.start_patch("src.uploads.speech_segments", return_value=[])
        self.select_evaluation()
        self.evaluate_file()
        self.make_speech.assert_not_called()

    def test_openai_pair_is_default_for_both_inputs(self):
        self.assertEqual(self.default_model_choices, ["gpt-live-transcribe + gpt-6-luna"] * 2)
        for key in ("microphone_model_pair", "upload_model_pair"):
            self.assertEqual(self.app.selectbox(key=key).options,
                             ["gpt-live-transcribe + gpt-6-luna", "Breeze + OpenAI", "gpt-realtime-translate"])
        self.load_openai_transcriber.assert_not_called()

    def test_realtime_microphone_renders_direct_english_without_text_model_workers(self):
        self.app.selectbox(key="microphone_model_pair").set_value("gpt-realtime-translate").run()
        self.assertTrue(self.app.selectbox(key="microphone_translation_type").disabled)
        session = self.start_recording()
        self.assertIs(self.app.session_state["realtime_translation"], session)
        self.assertEqual(self.app.session_state["translation_config"]["translation_model"], "gpt-realtime-translate")
        self.assertEqual(self.app.session_state["translation_config"]["providers"], ("openai",))
        session.state["texts"] = ["你好"]
        session.state["direct_translation"] = {"translations": ["Hello"], "errors": [None], "pending": 1}
        downloads = self.capture_downloads()
        self.app.run()
        self.assertIn("Hello", self.transcript_html())
        self.assertIn("Streaming", self.rendered_text())
        self.assertIn("incomplete", downloads[-1])
        self.assertNotIn("Tencent", downloads[-1])
        self.assertNotIn("Astra", downloads[-1])
        self.app.button(key="stop_recording").click().run()
        self.assertIn("English captions: Complete", downloads[-1])
        self.load_transcriber.assert_not_called()
        self.load_openai_transcriber.assert_not_called()
        self.load_vad.assert_not_called()
        self.make_translation.assert_not_called()
        self.make_diarization.assert_not_called()
        self.correct.assert_not_called()

    def test_realtime_upload_bypasses_vad_and_never_retranslates_captions(self):
        self.app.selectbox(key="upload_model_pair").set_value("gpt-realtime-translate").run()
        segments = self.start_patch("src.uploads.speech_segments")
        self.select_wav()
        downloads = self.capture_downloads()
        self.transcribe_file()
        self.app.run()
        self.assertEqual(self.app.session_state["upload_state"]["texts"], ["你好"])
        self.assertEqual(self.app.session_state["upload_state"]["direct_translation"]["translations"], ["Hello."])
        self.assertIn("gpt-realtime-translate", downloads[-1])
        self.assertIn("Hello.", downloads[-1])
        self.assertIn("English captions: Complete", downloads[-1])
        self.assertNotIn("upload_job", self.app.session_state)
        self.assertEqual(self.make_realtime.call_count, 1)
        segments.assert_not_called()
        self.load_vad.assert_not_called()
        self.make_translation.assert_not_called()
        self.make_diarization.assert_not_called()

    def test_realtime_failure_retains_partial_english_and_exports_incomplete_status(self):
        self.app.selectbox(key="microphone_model_pair").set_value("gpt-realtime-translate").run()
        session = self.start_recording()
        session.state.update(texts=["來源"], accepting=False, finished=True, error="Connection failed")
        session.state["direct_translation"] = {"translations": ["Partial English."],
                                                "errors": ["Connection failed"], "pending": 0}
        downloads = self.capture_downloads()
        self.app.run()
        self.assertIn("Partial English.", downloads[-1])
        self.assertIn("English captions: Incomplete", downloads[-1])
        self.assertNotIn("retry_translation", [button.key for button in self.app.button])
        self.assertFalse(self.app.button(key="start_recording").disabled)
        self.make_translation.assert_not_called()

    def test_switching_from_realtime_to_default_closes_old_stream(self):
        self.app.selectbox(key="microphone_model_pair").set_value("gpt-realtime-translate").run()
        old = self.start_recording()
        self.app.button(key="stop_recording").click().run()
        self.app.selectbox(key="microphone_model_pair").set_value("gpt-live-transcribe + gpt-6-luna").run()
        self.start_recording()
        old.close.assert_called_once()
        self.assertNotIn("realtime_translation", self.app.session_state)
        self.assertEqual(self.app.session_state["translation"].model, "gpt-6-luna")

    def test_realtime_upload_can_be_stopped_and_retains_incomplete_download(self):
        streaming = FakeRealtimeSession()
        streaming.state.update(texts=["來源"], accepting=False)
        streaming.state["direct_translation"] = {"translations": ["Partial English."], "errors": [None], "pending": 1}
        streaming.start_file.side_effect = None
        self.make_realtime.side_effect = None
        self.make_realtime.return_value = streaming
        self.app.selectbox(key="upload_model_pair").set_value("gpt-realtime-translate").run()
        self.select_wav()
        self.transcribe_file()
        self.assertIn("upload_job", self.app.session_state)
        self.assertTrue(self.app.selectbox(key="upload_model_pair").disabled)
        downloads = self.capture_downloads()
        self.app.button(key="stop_realtime_upload").click().run()
        self.assertNotIn("upload_job", self.app.session_state)
        streaming.close.assert_called_once()
        self.assertIn("English captions: Incomplete", downloads[-1])

    def test_openai_microphone_routes_audio_and_freezes_model_until_next_start(self):
        self.app.selectbox(key="microphone_model_pair").set_value("gpt-live-transcribe + gpt-6-luna").run()
        self.app.selectbox(key="microphone_translation_type").set_value("Fast English").run()
        self.vad.side_effect = [{"start": 0}, None]
        session = self.start_recording()
        self.assertTrue(self.app.selectbox(key="microphone_model_pair").disabled)
        self.assertTrue(self.app.selectbox(key="upload_model_pair").disabled)
        frame = av.AudioFrame.from_ndarray(np.ones((1, 800), dtype=np.float32), format="fltp", layout="mono")
        frame.sample_rate = 16_000
        session.push(frame)
        self.app.button(key="stop_recording").click().run()
        session._worker.join(timeout=5)
        downloads = self.capture_downloads()
        self.app.run()
        self.load_transcriber.assert_not_called()
        self.cloud_transcribe.assert_called_once()
        translation = self.app.session_state["translation"]
        self.assertEqual(translation.model, "gpt-6-luna")
        self.assertEqual(translation.sources, ["Cloud transcript"])
        self.assertIn("OpenAI ASR · gpt-live-transcribe", self.transcript_html())
        self.assertIn("Original transcript model: OpenAI ASR · gpt-live-transcribe", downloads[-1])
        self.assertIn("gpt-live-transcribe + gpt-6-luna", downloads[-1])
        self.app.selectbox(key="microphone_model_pair").set_value("Breeze + OpenAI").run()
        self.assertEqual(self.app.session_state["translation_config"]["model_pair"],
                         "gpt-live-transcribe + gpt-6-luna")
        self.start_recording()
        self.load_transcriber.assert_called_once_with()
        self.assertIn("Breeze ASR · Local", self.transcript_html())
        self.assertIsNone(self.app.session_state["translation"].model)
        translation.close.assert_called_once()

    def test_openai_upload_uses_own_model_choice_and_preserves_results_on_rerun(self):
        self.app.selectbox(key="upload_model_pair").set_value("gpt-live-transcribe + gpt-6-luna").run()
        self.start_patch("src.uploads.speech_segments", return_value=[np.zeros(800, dtype=np.float32)] * 2)
        self.cloud_transcribe.side_effect = ["First cloud line", "Second cloud line"]
        self.select_wav()
        self.transcribe_file()
        self.app.run()
        self.assertEqual(self.app.session_state["upload_state"]["texts"],
                         ["First cloud line", "Second cloud line"])
        self.load_transcriber.assert_not_called()
        self.assertEqual(self.cloud_transcribe.call_count, 2)
        self.assertEqual(self.app.session_state["translation"].model, "gpt-6-luna")
        self.assertEqual(self.app.session_state["translation"].sources,
                         self.app.session_state["tencent_translation"].sources)
        self.assertIn("OpenAI ASR · gpt-live-transcribe", self.transcript_html())
        self.app.selectbox(key="upload_model_pair").set_value("Breeze + OpenAI").run()
        self.assertEqual(self.cloud_transcribe.call_count, 2)
        self.assertEqual(self.app.session_state["translation_config"]["model_pair"],
                         "gpt-live-transcribe + gpt-6-luna")
        self.assertIn("OpenAI ASR · gpt-live-transcribe", self.transcript_html())

    def test_failed_openai_upload_keeps_original_empty_without_using_breeze(self):
        from src.openai_transcription import FAILED_MESSAGE, TranscriptionError

        self.app.selectbox(key="upload_model_pair").set_value("gpt-live-transcribe + gpt-6-luna").run()
        self.start_patch("src.uploads.speech_segments", return_value=[np.zeros(800, dtype=np.float32)])
        self.cloud_transcribe.side_effect = TranscriptionError(FAILED_MESSAGE)
        self.select_wav()
        self.transcribe_file()
        deadline = monotonic() + 3
        while "upload_job" in self.app.session_state and monotonic() < deadline:
            self.app.run()
        state = self.app.session_state["upload_state"]
        self.assertTrue(state["finished"])
        self.assertEqual(state["texts"], [])
        self.assertIn(FAILED_MESSAGE, state["error"])
        self.load_transcriber.assert_not_called()
        self.assertIn("OpenAI ASR · gpt-live-transcribe", self.transcript_html())

    def test_missing_openai_key_does_not_start_recording_or_load_breeze(self):
        from src.openai_transcription import MISSING_KEY_MESSAGE, TranscriptionError

        self.app.selectbox(key="microphone_model_pair").set_value("gpt-live-transcribe + gpt-6-luna").run()
        self.load_openai_transcriber.side_effect = TranscriptionError(MISSING_KEY_MESSAGE)
        self.app.button(key="start_recording").click().run()
        self.assertIn(MISSING_KEY_MESSAGE, self.app.error[0].value)
        self.assertNotIn("recording", self.app.session_state)
        self.load_transcriber.assert_not_called()

    def test_fast_microphone_mode_creates_only_openai_and_stays_frozen(self):
        self.app.selectbox(key="microphone_translation_type").set_value("Fast English").run()
        self.vad.side_effect = [{"start": 0}, None]
        session = self.start_recording()
        self.assertTrue(self.app.selectbox(key="microphone_translation_type").disabled)
        self.assertEqual(self.make_translation.call_count, 1)
        self.assertNotIn("tencent_translation", self.app.session_state)
        self.assertNotIn("slow_lane", self.app.session_state)
        self.assertIsNone(self.app.session_state["translation"].on_result)
        self.assertEqual(self.app.session_state["translation_config"]["providers"], ("openai",))
        self.app.toggle(key="enable_corrections").set_value(False).run()
        self.app.toggle(key="enable_corrections").set_value(True).run()
        self.assertNotIn("slow_lane", self.app.session_state)
        self.assertNotIn("Astra · Ready", self.rendered_text())
        frame = av.AudioFrame.from_ndarray(
            np.ones((1, 800), dtype=np.float32), format="fltp", layout="mono",
        )
        frame.sample_rate = 16_000
        session.push(frame)
        self.app.button(key="stop_recording").click().run()
        session._worker.join(timeout=5)
        downloads = self.capture_downloads()
        self.app.run()
        self.assertIn("English segment 1.", self.transcript_html())
        self.assertNotIn('class="translation-cell tencent-cell"', self.transcript_html())
        self.assertNotIn('class="translation-cell astra-cell', self.transcript_html())
        self.assertNotIn("Tencent", downloads[-1])
        self.assertNotIn("Astra", downloads[-1])
        self.correct.assert_not_called()
        self.app.selectbox(key="microphone_translation_type").set_value("Fully reviewed English").run()
        self.assertEqual(self.app.session_state["translation_config"]["type"], "Fast English")
        self.assertEqual(self.make_translation.call_count, 1)
        self.assertNotIn('class="translation-cell astra-cell', self.transcript_html())
        self.assertNotIn("Tencent", downloads[-1])
        self.assertNotIn("Astra", downloads[-1])

    def test_corrected_microphone_mode_uses_only_first_astra_review(self):
        self.app.selectbox(key="microphone_translation_type").set_value("Corrected English").run()
        self.start_recording()
        lane = self.app.session_state["slow_lane"]
        selected = self.app.session_state["translation_config"]
        self.assertEqual(selected["providers"], ("openai", "astra"))
        self.assertFalse(selected["second_review"])
        self.assertEqual(self.make_translation.call_count, 1)
        self.assertNotIn("tencent_translation", self.app.session_state)
        lane.submit(["Hello"], ["Hello"])
        snapshot = self.wait_for_corrections()
        self.assertEqual(snapshot["statuses"], ["confirmed"])
        self.assertNotIn("conversation_review", snapshot)
        self.assertEqual(self.correct.call_count, 1)

    def test_fully_reviewed_microphone_mode_uses_both_reviews_without_tencent(self):
        self.app.selectbox(key="microphone_translation_type").set_value("Fully reviewed English").run()
        self.start_recording()
        lane = self.app.session_state["slow_lane"]
        self.assertTrue(self.app.session_state["translation_config"]["second_review"])
        self.assertEqual(self.make_translation.call_count, 1)
        self.assertNotIn("tencent_translation", self.app.session_state)
        lane.submit(["Hello"], ["Hello"])
        snapshot = self.wait_for_corrections()
        self.assertEqual(snapshot["conversation_review"]["reviewed"], 1)
        self.assertEqual(snapshot["first_pass"]["statuses"], ["confirmed"])
        self.assertEqual(self.correct.call_count, 2)
        self.assertEqual(self.correct.call_args_list[1].args[0]["review_stage"], "conversation")

    def test_upload_ignores_microphone_mode_and_keeps_both_review_stages(self):
        self.app.selectbox(key="microphone_translation_type").set_value("Fast English").run()
        self.start_patch("src.uploads.speech_segments", return_value=[np.zeros(800, dtype=np.float32)])
        self.select_wav()
        self.transcribe_file()
        snapshot = self.wait_for_corrections()
        self.assertEqual(self.app.session_state["translation_config"]["type"], "Compare all translations")
        self.assertEqual(self.make_translation.call_count, 2)
        self.assertTrue(snapshot["conversation_review"]["enabled"])
        self.assertEqual(snapshot["conversation_review"]["reviewed"], 1)
        self.app.run()
        self.assertIn("Conversation review 1/1", self.rendered_text())
        self.app.selectbox(key="microphone_translation_type").set_value("Corrected English").run()
        self.assertEqual(self.app.session_state["translation_config"]["type"], "Compare all translations")

    def test_start_and_reruns_reuse_recording_without_starting_a_worker(self):
        session = self.start_recording()
        self.app.run()

        self.assertIs(self.app.session_state["recording"], session)
        self.assertEqual(self.app.session_state["recording_number"], 1)
        self.assertTrue(self.app.button(key="start_recording").disabled)
        self.assertFalse(self.app.button(key="stop_recording").disabled)
        self.assertIsNone(session._worker.ident)
        self.load_transcriber.assert_called_once_with()
        self.load_vad.assert_called_once_with()
        self.assertEqual(self.make_translation.call_count, 2)
        self.assertIsNone(self.app.session_state["translation"].translate)
        self.assertIsNotNone(self.app.session_state["tencent_translation"].translate)
        callback = self.webrtc.call_args.kwargs["audio_frame_callback"]
        self.assertIs(callback.__self__, session)
        self.assertTrue(self.webrtc.call_args.kwargs["desired_playing_state"])

    def test_stop_keeps_transcript_until_start_creates_a_fresh_recording(self):
        self.vad.side_effect = [{"start": 0}, None]
        first = self.start_recording()
        first_translation = self.app.session_state["translation"]
        first_tencent = self.app.session_state["tencent_translation"]
        frame = av.AudioFrame.from_ndarray(
            np.ones((1, 800), dtype=np.float32), format="fltp", layout="mono",
        )
        frame.sample_rate = 16_000
        first.push(frame)

        self.app.button(key="stop_recording").click().run()
        first._worker.join(timeout=5)
        self.app.run()
        self.assertEqual(len(self.app.exception), 0)
        self.assertIn("A short transcript", self.transcript_html())
        self.assertIn("English segment 1.", self.transcript_html())
        self.assertIn("Tencent segment 1.", self.transcript_html())
        self.assertEqual(first_translation.sources, ["A short transcript"])
        self.assertEqual(first_tencent.sources, ["A short transcript"])
        self.assertFalse(self.app.download_button(key="download_transcript").disabled)
        self.assertFalse(self.app.button(key="start_recording").disabled)
        self.assertTrue(self.app.button(key="stop_recording").disabled)
        self.assertFalse(self.webrtc.call_args.kwargs["desired_playing_state"])
        self.app.run()
        self.assertIn("A short transcript", self.transcript_html())
        first_translation.close.assert_not_called()
        first_tencent.close.assert_not_called()

        second = self.start_recording()
        self.assertIsNot(first, second)
        self.assertEqual(self.app.session_state["recording_number"], 2)
        self.assertEqual(second.snapshot()["texts"], [])
        self.assertNotIn("A short transcript", self.transcript_html())
        self.assertNotIn("English segment 1.", self.transcript_html())
        self.assertNotIn("Tencent segment 1.", self.transcript_html())
        self.assertTrue(self.app.download_button(key="download_transcript").disabled)
        self.assertEqual(first.snapshot()["texts"], ["A short transcript"])
        self.assertTrue(first.snapshot()["finished"])
        self.load_transcriber.assert_called_once_with()
        self.assertEqual(self.load_vad.call_count, 2)
        self.assertEqual(self.webrtc.call_args.kwargs["key"], "microphone-2")
        first_translation.close.assert_called_once_with()
        first_tencent.close.assert_called_once_with()
        self.assertIsNot(self.app.session_state["translation"], first_translation)
        self.assertIsNot(self.app.session_state["tencent_translation"], first_tencent)
        self.assertEqual(self.make_translation.call_count, 4)

    def test_model_load_failure_is_visible_and_can_be_retried(self):
        self.load_transcriber.side_effect = RuntimeError("weights missing")
        self.app.button(key="start_recording").click().run()

        self.assertEqual(len(self.app.exception), 0)
        self.assertIn("weights missing", self.app.error[0].value)
        self.assertNotIn("recording", self.app.session_state)
        self.webrtc.assert_not_called()
        self.load_vad.assert_not_called()
        self.app.run()
        self.assertIn("weights missing", self.app.error[0].value)

        self.load_transcriber.side_effect = None
        session = self.start_recording()
        self.assertTrue(session.snapshot()["accepting"])
        self.assertEqual(len(self.app.error), 0)
        self.assertEqual(self.load_transcriber.call_count, 2)

    def test_selecting_a_wav_does_not_start_transcription_or_load_models(self):
        segments = self.start_patch("src.uploads.speech_segments")
        self.select_wav()

        self.assertFalse(self.app.button(key="transcribe_file").disabled)
        self.load_transcriber.assert_not_called()
        self.load_vad.assert_not_called()
        segments.assert_not_called()
        self.transcribe.assert_not_called()
        self.assertNotIn("upload_job", self.app.session_state)
        self.make_translation.assert_not_called()

    def test_uploaded_segments_persist_without_reprocessing_until_next_action(self):
        audio_segments = [
            np.full(800, 0.25, dtype=np.float32), np.full(800, 0.5, dtype=np.float32),
        ]
        segments = self.start_patch("src.uploads.speech_segments", return_value=audio_segments)
        self.transcribe.side_effect = ["First line.", "Second line."]
        self.select_wav()
        self.transcribe_file()
        file_translation = self.app.session_state["translation"]
        file_tencent = self.app.session_state["tencent_translation"]

        state = self.app.session_state["upload_state"]
        self.assertEqual(state["texts"], ["First line.", "Second line."])
        self.assertTrue(state["finished"])
        self.assertEqual(state["pending"], 0)
        self.assertIsNone(state["error"])
        self.assertEqual(state["name"], "sample.wav")
        self.assertEqual(self.app.session_state["transcript_source"], "upload")
        self.assertNotIn("upload_job", self.app.session_state)
        self.assertFalse(self.app.download_button(key="download_transcript").disabled)
        for call, expected in zip(self.transcribe.call_args_list, audio_segments):
            np.testing.assert_array_equal(call.args[0], expected)

        self.app.run()
        self.select_wav(name="another.wav")
        self.assertIn("First line.", self.transcript_html())
        self.assertIn("Second line.", self.transcript_html())
        self.assertEqual(self.app.session_state["upload_state"]["name"], "sample.wav")
        self.assertEqual(self.transcribe.call_count, 2)
        segments.assert_called_once()
        self.load_transcriber.assert_called_once_with()
        self.load_vad.assert_called_once_with()
        file_translation.close.assert_not_called()
        file_tencent.close.assert_not_called()
        self.assertEqual(file_translation.sources, ["First line.", "Second line."])
        self.assertEqual(file_tencent.sources, ["First line.", "Second line."])
        self.assertEqual(self.make_translation.call_count, 2)

        self.start_recording()
        self.assertEqual(self.app.session_state["transcript_source"], "microphone")
        self.assertNotIn("First line.", self.transcript_html())
        file_translation.close.assert_called_once_with()
        file_tencent.close.assert_called_once_with()
        self.assertIsNot(self.app.session_state["translation"], file_translation)
        self.assertIsNot(self.app.session_state["tencent_translation"], file_tencent)

    def test_failed_upload_transcription_clears_job_and_can_be_retried(self):
        self.start_patch(
            "src.uploads.speech_segments", return_value=[np.ones(800, dtype=np.float32)],
        )
        self.transcribe.side_effect = RuntimeError("inference unavailable")
        self.select_wav()
        self.transcribe_file()
        failed_translation = self.app.session_state["translation"]
        failed_tencent = self.app.session_state["tencent_translation"]

        state = self.app.session_state["upload_state"]
        self.assertTrue(state["finished"])
        self.assertIn("inference unavailable", state["error"])
        self.assertNotIn("upload_job", self.app.session_state)
        self.assertFalse(self.app.button(key="transcribe_file").disabled)
        self.app.run()
        self.assertTrue(any("inference unavailable" in error.value for error in self.app.error))

        self.transcribe.side_effect = None
        self.transcribe_file()
        state = self.app.session_state["upload_state"]
        self.assertIsNone(state["error"])
        self.assertEqual(state["texts"], ["A short transcript"])
        self.assertNotIn("upload_job", self.app.session_state)
        self.assertEqual(self.load_vad.call_count, 2)
        failed_translation.close.assert_called_once_with()
        failed_tencent.close.assert_called_once_with()
        self.assertIsNot(self.app.session_state["translation"], failed_translation)
        self.assertIsNot(self.app.session_state["tencent_translation"], failed_tencent)

    def test_completed_upload_releases_audio_and_keeps_evaluation_after_replacement(self):
        decoded, views = self.track_upload_audio()
        downloads = self.capture_downloads()
        self.select_evaluation(b"Hello Hello")
        self.evaluate_file()
        original = self.app.session_state["upload_state"]

        gc.collect()
        self.assertTrue(all(reference() is None for reference in decoded + views))
        self.assertEqual(original["texts"], ["Hello", "Hello"])
        self.assertEqual([timing["start_s"] for timing in original["timings"]], [0.0, 0.05])
        self.assertEqual(self.evaluation_scores(), {"Breeze · 1-wMER": "100.0%"})
        self.assertFalse(self.app.download_button(key="download_transcript").disabled)
        self.assertTrue(any("Hello" in content and "English segment 1." in content
                            for content in downloads if isinstance(content, str)))

        self.select_evaluation(b"Hello Hello", audio_name="replacement.wav")
        self.evaluate_file()
        gc.collect()
        self.assertEqual(len(decoded), 2)
        self.assertTrue(all(reference() is None for reference in decoded + views))
        self.assertIsNot(self.app.session_state["upload_state"], original)
        self.assertEqual(original["name"], "evaluation.wav")
        self.assertEqual(original["texts"], ["Hello", "Hello"])
        self.assertEqual(self.app.session_state["upload_state"]["name"], "replacement.wav")

    def test_failed_upload_releases_audio_but_preserves_partial_text(self):
        decoded, views = self.track_upload_audio()
        calls = 0

        def transcribe(audio):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("inference unavailable")
            return "Hello"

        self.load_transcriber.return_value = transcribe
        self.select_evaluation(b"Hello Hello")
        self.evaluate_file()

        gc.collect()
        self.assertTrue(all(reference() is None for reference in decoded + views))
        state = self.app.session_state["upload_state"]
        self.assertEqual(state["texts"], ["Hello"])
        self.assertEqual(len(state["timings"]), 1)
        self.assertTrue(state["finished"])
        self.assertEqual(state["pending"], 0)
        self.assertIn("inference unavailable", state["error"])
        self.assertEqual(self.evaluation_scores(), {"Breeze · 1-wMER": "—"})
        self.assertFalse(self.app.download_button(key="download_transcript").disabled)
        self.assertNotIn("upload_job", self.app.session_state)

    def test_upload_resumes_after_rerun_without_repeating_completed_segments(self):
        from src.ui import render_transcript

        segments = self.start_patch(
            "src.uploads.speech_segments",
            return_value=[np.ones(800, dtype=np.float32), np.zeros(800, dtype=np.float32)],
        )
        self.transcribe.side_effect = ["First line.", "Second line."]
        interrupted = False

        def render_with_one_rerun(texts, *args, **kwargs):
            nonlocal interrupted
            if texts == ["First line."] and not interrupted:
                interrupted = True
                st.rerun()
            return render_transcript(texts, *args, **kwargs)

        self.start_patch("src.ui.render_transcript", side_effect=render_with_one_rerun)
        self.select_wav()
        self.transcribe_file()

        self.assertTrue(interrupted)
        state = self.app.session_state["upload_state"]
        self.assertEqual(state["texts"], ["First line.", "Second line."])
        self.assertTrue(state["finished"])
        self.assertEqual(state["pending"], 0)
        self.assertIsNone(state["error"])
        self.assertNotIn("upload_job", self.app.session_state)
        self.assertEqual(self.transcribe.call_count, 2)
        self.load_vad.assert_called_once_with()
        segments.assert_called_once()
        translation = self.app.session_state["translation"]
        tencent = self.app.session_state["tencent_translation"]
        self.assertEqual(translation.sources, ["First line.", "Second line."])
        self.assertEqual(tencent.sources, ["First line.", "Second line."])
        self.assertEqual(len(translation.translations), 2)
        self.assertEqual(len(tencent.translations), 2)
        self.assertEqual(self.make_translation.call_count, 2)
        translation.close.assert_not_called()
        tencent.close.assert_not_called()

    def test_each_original_segment_is_paired_with_both_provider_translations(self):
        self.configure_translations(
            openai=["Hello.", "Have you eaten?"], tencent=["Hi there.", "Did you eat?"],
        )
        self.start_patch(
            "src.uploads.speech_segments",
            return_value=[np.ones(800, dtype=np.float32), np.zeros(800, dtype=np.float32)],
        )
        self.transcribe.side_effect = ["你好。", "食飽未？"]
        self.select_wav()
        self.transcribe_file()
        translation = self.app.session_state["translation"]
        tencent = self.app.session_state["tencent_translation"]
        self.app.run()

        rows = re.findall(r"<li\b[^>]*>(.*?)</li>", self.transcript_html(), re.DOTALL)
        self.assertEqual(len(rows), 2)
        self.assertIn("你好。", rows[0])
        self.assertIn("Hello.", rows[0])
        self.assertIn("Hi there.", rows[0])
        self.assertNotIn("Have you eaten?", rows[0])
        self.assertNotIn("Did you eat?", rows[0])
        self.assertIn("食飽未？", rows[1])
        self.assertIn("Have you eaten?", rows[1])
        self.assertIn("Did you eat?", rows[1])
        self.assertEqual(translation.sources, ["你好。", "食飽未？"])
        self.assertEqual(tencent.sources, ["你好。", "食飽未？"])

    def test_upload_preserves_filtered_source_without_scheduling_astra_review(self):
        self.configure_translations(
            openai=["[Background speech filtered]"], openai_filtered=[True],
            tencent=["Unrelated background conversation."],
        )
        self.start_patch("src.uploads.speech_segments", return_value=[np.ones(800, dtype=np.float32)])
        self.transcribe.return_value = "Background source speech"
        self.select_wav()
        self.transcribe_file()
        final = self.wait_for_corrections()
        self.app.run()
        self.assertFalse(self.app.exception)
        self.assertEqual(final["statuses"], ["filtered"])
        self.assertEqual(final["pending"], 0)
        self.correct.assert_not_called()
        self.assertIn("Background source speech", self.transcript_html())
        self.assertIn("Background filtered", self.transcript_html())
        self.assertIn("Unrelated background conversation.", self.transcript_html())

    def test_pending_and_failed_translations_keep_originals_and_allow_retry(self):
        self.configure_translations(
            openai=[None, None], openai_errors=[None, "Translation service unavailable"],
            tencent=["First Tencent result.", "Second Tencent result."],
        )
        self.start_patch(
            "src.uploads.speech_segments",
            return_value=[np.ones(800, dtype=np.float32), np.zeros(800, dtype=np.float32)],
        )
        self.transcribe.side_effect = ["First original.", "Second original."]
        self.select_wav()
        self.transcribe_file()
        translation = self.app.session_state["translation"]
        tencent = self.app.session_state["tencent_translation"]
        self.app.run()

        self.assertIn("First original.", self.transcript_html())
        self.assertIn("Second original.", self.transcript_html())
        self.assertIn("First Tencent result.", self.transcript_html())
        self.assertIn("Second Tencent result.", self.transcript_html())
        self.assertIn("OpenAI · Translating 1 segment(s)", self.rendered_text())
        self.assertIn("Translation service unavailable", self.app.warning[0].value)
        self.assertFalse(self.app.download_button(key="download_transcript").disabled)
        self.app.button(key="retry_translation").click().run()
        self.assertEqual(len(self.app.exception), 0)
        translation.retry_failed.assert_called_once_with()
        tencent.retry_failed.assert_not_called()
        self.assertEqual(translation.snapshot()["pending"], 2)
        self.assertIn("First original.", self.transcript_html())
        self.assertIn("Second original.", self.transcript_html())

        translation.translations = ["First English.", "Second English."]
        self.app.run()
        self.assertIn("First English.", self.transcript_html())
        self.assertIn("Second English.", self.transcript_html())
        self.assertEqual(len(self.app.warning), 0)

    def test_uploaded_source_never_appears_as_english_when_astra_recovers_fast_failure(self):
        source = "今天的天氣很好。"
        self.configure_translations(
            openai=[None], openai_errors=["Translation service unavailable"],
            tencent=["The weather is nice today."],
        )
        self.start_patch(
            "src.uploads.speech_segments", return_value=[np.ones(800, dtype=np.float32)],
        )
        self.transcribe.return_value = source
        self.correct.side_effect = lambda request: {
            "corrections": [{
                "segment_id": row["segment_id"], "base_version": row["base_version"],
                "target_text": "The weather is lovely today.", "change_type": ["omission"],
                "confidence": 0.95, "rationale": "Translate the missing English draft.",
                "term_pairs": [],
            } for row in request["segments"]],
            "no_change": [],
        }
        downloads = self.capture_downloads()
        self.select_wav()
        self.transcribe_file()
        result = self.wait_for_corrections()
        self.assertEqual(result["statuses"], ["corrected"])
        self.assertEqual(len(self.review_requests()), 1)
        self.assertEqual(len(self.review_requests(conversation=True)), 1)
        self.assertTrue(self.review_requests()[0]["segments"][0]["source_fallback"])
        second_input = self.review_requests(conversation=True)[0]["segments"][0]
        self.assertFalse(second_input["source_fallback"])
        self.assertEqual(second_input["target_text"], "The weather is lovely today.")

        for view in ("Live subtitles", "Final record"):
            with self.subTest(view=view):
                self.app.selectbox(key="subtitle_view").set_value(view).run()
                self.assertEqual(len(self.app.exception), 0)
                html = self.transcript_html()
                self.assertEqual(html.count(source), 1)
                self.assertIn("The weather is lovely today.", html)
                self.assertIn("Corrected", html)
                self.assertFalse(self.app.download_button(key="download_transcript").disabled)
                report = next(value for value in reversed(downloads)
                              if isinstance(value, str) and "English (Astra correction):" in value)
                self.assertEqual(report.count(source), 1)
                self.assertIn("English (OpenAI): [Translation unavailable]", report)
                self.assertIn("English (Astra correction): The weather is lovely today.", report)

    def test_tencent_failure_and_retry_do_not_block_or_repeat_openai_translation(self):
        self.configure_translations(tencent=[None], tencent_errors=["Tencent translation unavailable"])
        self.start_patch(
            "src.uploads.speech_segments", return_value=[np.ones(800, dtype=np.float32)],
        )
        self.select_wav()
        self.transcribe_file()
        translation = self.app.session_state["translation"]
        tencent = self.app.session_state["tencent_translation"]
        self.app.run()

        self.assertIn("A short transcript", self.transcript_html())
        self.assertIn("English segment 1.", self.transcript_html())
        self.assertIn("Tencent translation unavailable", self.app.warning[0].value)
        self.app.button(key="retry_tencent_translation").click().run()
        self.assertEqual(len(self.app.exception), 0)
        tencent.retry_failed.assert_called_once_with()
        translation.retry_failed.assert_not_called()
        self.assertEqual(tencent.snapshot()["pending"], 1)
        self.assertIn("English segment 1.", self.transcript_html())
        self.assertEqual(translation.sources, ["A short transcript"])
        self.assertEqual(self.transcribe.call_count, 1)

        tencent.translations = ["Tencent recovered."]
        self.app.run()
        self.assertIn("Tencent recovered.", self.transcript_html())
        self.assertIn("English segment 1.", self.transcript_html())
        self.assertEqual(len(self.app.warning), 0)

    def test_corrupt_wav_is_rejected_before_loading_models_and_allows_retry(self):
        segments = self.start_patch(
            "src.uploads.speech_segments", return_value=[np.ones(800, dtype=np.float32)],
        )
        self.select_wav(content=b"this is not a WAV file")
        self.transcribe_file()

        self.assertGreater(len(self.app.error), 0)
        self.assertNotIn("upload_job", self.app.session_state)
        self.load_transcriber.assert_not_called()
        self.load_vad.assert_not_called()
        segments.assert_not_called()
        self.select_wav(content=wav_bytes())
        self.transcribe_file()
        self.assertEqual(len(self.app.error), 0)
        self.assertEqual(self.app.session_state["upload_state"]["texts"], ["A short transcript"])

    def test_silent_wav_finishes_without_loading_asr(self):
        decoded, views = self.track_upload_audio(silent=True)
        self.select_wav()
        self.transcribe_file()

        gc.collect()
        self.assertEqual(len(decoded), 1)
        self.assertIsNone(decoded[0]())
        self.assertEqual(views, [])
        state = self.app.session_state["upload_state"]
        self.assertTrue(state["finished"])
        self.assertEqual(state["texts"], [])
        self.assertIsNone(state["error"])
        self.assertNotIn("upload_job", self.app.session_state)
        self.load_vad.assert_called_once_with()
        self.load_transcriber.assert_not_called()
        self.transcribe.assert_not_called()
        self.assertTrue(self.app.download_button(key="download_transcript").disabled)
        self.assertIn("No speech", self.rendered_text())

    def test_evaluation_requires_valid_reference_before_loading_audio_or_models(self):
        decode = self.start_patch("src.uploads.decode_wav")
        segments = self.start_patch("src.uploads.speech_segments")
        self.assertTrue(self.app.button(key="evaluate_file").disabled)
        self.app.file_uploader(key="eval_wav_file").set_value(
            ("evaluation.wav", wav_bytes(), "audio/wav"),
        ).run()
        self.assertTrue(self.app.button(key="evaluate_file").disabled)

        for content in (b" \n\t", b"\xff invalid encoding", b"\xff\xfe\x00"):
            with self.subTest(content=content):
                self.app.file_uploader(key="eval_reference_file").set_value(
                    ("reference.txt", content, "text/plain"),
                ).run()
                self.evaluate_file()
                self.assertGreater(len(self.app.error), 0)
                self.assertNotIn("upload_job", self.app.session_state)
                self.assertNotIn("upload_state", self.app.session_state)

        self.select_evaluation()
        self.assertFalse(self.app.button(key="evaluate_file").disabled)
        decode.assert_not_called()
        segments.assert_not_called()
        self.load_vad.assert_not_called()
        self.load_transcriber.assert_not_called()
        self.transcribe.assert_not_called()
        self.make_translation.assert_not_called()

    def test_evaluation_scores_joined_outputs_and_keeps_submitted_files_on_reruns(self):
        from src.evaluation import word_match_score
        from src.ui import render_transcript

        segments = self.start_patch(
            "src.uploads.speech_segments",
            return_value=[np.ones(800, dtype=np.float32), np.zeros(800, dtype=np.float32)],
        )
        score = self.start_patch("src.evaluation.word_match_score", wraps=word_match_score)
        self.transcribe.side_effect = ["Hello", "wonderful world"]
        interrupted = False

        def render_with_one_rerun(texts, *args, **kwargs):
            nonlocal interrupted
            if texts == ["Hello"] and not interrupted:
                interrupted = True
                st.rerun()
            return render_transcript(texts, *args, **kwargs)

        self.start_patch("src.ui.render_transcript", side_effect=render_with_one_rerun)
        self.select_evaluation(b"Hello wonderful world.", audio_name="original.wav")
        self.evaluate_file()

        self.assertTrue(interrupted)
        self.assertEqual(self.evaluation_scores(), {"Breeze · 1-wMER": "100.0%"})
        score.assert_called_once_with("Hello wonderful world.", "Hello wonderful world")
        translation = self.app.session_state["translation"]
        tencent = self.app.session_state["tencent_translation"]
        self.assertIn("Hello", self.transcript_html())
        self.assertIn("wonderful world", self.transcript_html())
        self.assertEqual(self.app.download_button(key="download_transcript").label,
                         "Download evaluation")

        self.select_evaluation(
            b"A different reference.", audio_name="different.wav", reference_name="different.txt",
        )
        self.app.download_button(key="download_transcript").click().run()
        self.app.run()
        self.assertEqual(len(self.app.exception), 0)
        state = self.app.session_state["upload_state"]
        self.assertEqual(state["name"], "original.wav")
        self.assertEqual(state["evaluation"]["reference_name"], "reference.txt")
        self.assertEqual(state["evaluation"]["reference_text"], "Hello wonderful world.")
        self.assertTrue(state["finished"])
        self.assertNotIn("upload_job", self.app.session_state)
        self.assertEqual(score.call_count, 1)
        self.assertEqual(self.transcribe.call_count, 2)
        segments.assert_called_once()
        self.load_vad.assert_called_once_with()
        self.load_transcriber.assert_called_once_with()
        for session in (translation, tencent):
            self.assertEqual(session.sources, ["Hello", "wonderful world"])
            session.close.assert_not_called()
        self.assertEqual(self.make_translation.call_count, 2)

    def test_translation_pending_failure_and_retry_do_not_change_breeze_score(self):
        from src.evaluation import word_match_score

        self.configure_translations(openai=[None, None], tencent=["English segment 1.", "English segment 2."])
        self.start_patch(
            "src.uploads.speech_segments",
            return_value=[np.ones(800, dtype=np.float32), np.zeros(800, dtype=np.float32)],
        )
        self.transcribe.side_effect = ["First original.", "Second original."]
        score = self.start_patch("src.evaluation.word_match_score", wraps=word_match_score)
        self.select_evaluation(b"First original. Second original.")
        self.evaluate_file()
        translation = self.app.session_state["translation"]
        tencent = self.app.session_state["tencent_translation"]
        self.app.run()
        expected_scores = {"Breeze · 1-wMER": "100.0%"}
        self.assertEqual(self.evaluation_scores(), expected_scores)
        self.assertIn("Translating", self.rendered_text())

        translation.errors = [None, "Translation unavailable"]
        self.app.run()
        self.assertEqual(self.evaluation_scores(), expected_scores)
        self.assertIn("Translation unavailable", self.app.warning[0].value)
        self.app.button(key="retry_translation").click().run()
        self.assertEqual(self.evaluation_scores(), expected_scores)
        translation.retry_failed.assert_called_once_with()
        tencent.retry_failed.assert_not_called()
        self.assertEqual(score.call_count, 1)

        translation.translations = ["English segment 1.", "English segment 2. again"]
        self.app.run()
        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual(self.evaluation_scores(), expected_scores)
        score.assert_called_once_with(
            "First original. Second original.", "First original. Second original.",
        )
        self.assertEqual(self.transcribe.call_count, 2)
        self.assertEqual(translation.sources, ["First original.", "Second original."])
        self.assertEqual(tencent.sources, translation.sources)

    def test_invalid_evaluation_preserves_results_until_valid_replacement_closes_sessions(self):
        self.start_patch(
            "src.uploads.speech_segments", return_value=[np.ones(800, dtype=np.float32)],
        )
        self.select_evaluation(b"First reference.")
        self.evaluate_file()
        previous = self.app.session_state["upload_state"]
        translation = self.app.session_state["translation"]
        tencent = self.app.session_state["tencent_translation"]
        self.select_evaluation(b"", audio_name="replacement.wav", reference_name="replacement.txt")
        self.evaluate_file()

        self.assertIs(self.app.session_state["upload_state"], previous)
        self.assertEqual(self.transcribe.call_count, 1)
        self.assertEqual(self.make_translation.call_count, 2)
        translation.close.assert_not_called()
        tencent.close.assert_not_called()

        self.select_evaluation(
            b"Second reference.", audio_name="replacement.wav", reference_name="replacement.txt",
        )
        self.evaluate_file()
        state = self.app.session_state["upload_state"]
        self.assertEqual(state["name"], "replacement.wav")
        self.assertEqual(state["evaluation"]["reference_name"], "replacement.txt")
        self.assertEqual(state["evaluation"]["reference_text"], "Second reference.")
        self.assertIsNot(state, previous)
        translation.close.assert_called_once_with()
        tencent.close.assert_called_once_with()
        self.assertEqual(self.transcribe.call_count, 2)
        self.assertEqual(self.make_translation.call_count, 4)

    def test_silent_evaluation_scores_zero_without_loading_asr(self):
        self.start_patch("src.uploads.speech_segments", return_value=[])
        self.select_evaluation()
        self.evaluate_file()

        self.assertEqual(self.evaluation_scores(), {"Breeze · 1-wMER": "0.0%"})
        self.assertIn("No speech", self.rendered_text())
        self.assertFalse(self.app.download_button(key="download_transcript").disabled)
        self.load_transcriber.assert_not_called()
        self.transcribe.assert_not_called()
        for key in ("translation", "tencent_translation"):
            self.assertEqual(self.app.session_state[key].sources, [])

    def test_evaluation_withholds_breeze_score_when_asr_fails_after_partial_text(self):
        score = self.start_patch("src.evaluation.word_match_score")
        self.start_patch(
            "src.uploads.speech_segments",
            return_value=[np.ones(800, dtype=np.float32), np.zeros(800, dtype=np.float32)],
        )
        self.transcribe.side_effect = ["Partial original.", RuntimeError("inference unavailable")]
        self.select_evaluation()
        self.evaluate_file()

        self.assertEqual(self.evaluation_scores(), {"Breeze · 1-wMER": "—"})
        self.assertIn("Transcription failed · no final score", self.rendered_text())
        self.assertIn("Partial original.", self.transcript_html())
        self.assertTrue(self.app.session_state["upload_state"]["finished"])
        self.assertNotIn("upload_job", self.app.session_state)
        score.assert_not_called()

    def test_source_reference_preview_and_score_use_spoken_text_without_leaking_reference(self):
        self.configure_translations(openai=[None], tencent=[None])
        self.start_patch(
            "src.uploads.speech_segments", return_value=[np.ones(800, dtype=np.float32)],
        )
        self.transcribe.return_value = "你好 Zoom"
        self.select_evaluation(SOURCE_REFERENCE)

        self.assertTrue(any(element.value == "你好 Teams" for element in self.app.text))
        self.assertNotIn("eval_english_reference_file", [item.key for item in self.app.file_uploader])
        self.load_transcriber.assert_not_called()
        self.load_vad.assert_not_called()
        self.make_translation.assert_not_called()
        self.evaluate_file()

        evaluation = self.app.session_state["upload_state"]["evaluation"]
        self.assertEqual(evaluation["reference_kind"], "source")
        self.assertEqual(evaluation["reference_text"], "你好 Teams")
        self.assertNotIn("english_reference", evaluation)
        expected = {"Breeze · Mixed match": "66.7%"}
        self.assertEqual(self.evaluation_scores(), expected)
        self.assertNotIn("English reference needed", self.rendered_text())
        translation = self.app.session_state["translation"]
        tencent = self.app.session_state["tencent_translation"]
        for session in (translation, tencent):
            self.assertEqual(session.sources, ["你好 Zoom"])
        self.transcribe.assert_called_once()
        self.assertIsInstance(self.transcribe.call_args.args[0], np.ndarray)

        translation.errors = ["Translation unavailable"]
        self.app.run()
        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual(self.evaluation_scores(), expected)
        self.assertIn("你好 Zoom", self.transcript_html())

    def test_caption_reference_scores_breeze_and_export_keeps_unscored_translations(self):
        downloads = self.capture_downloads()
        self.start_patch(
            "src.uploads.speech_segments", return_value=[np.ones(800, dtype=np.float32)],
        )
        self.transcribe.return_value = "Welcome to Teams."
        captions = b"1\n00:00:00,000 --> 00:00:02,000\nWelcome to Teams.\n"
        self.select_evaluation(captions, reference_name="source.srt")
        self.make_translation.assert_not_called()
        self.evaluate_file()
        for key in ("translation", "tencent_translation"):
            session = self.app.session_state[key]
            self.assertEqual(session.sources, ["Welcome to Teams."])
        self.app.run()

        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual(self.evaluation_scores(), {"Breeze · 1-wMER": "100.0%"})
        evaluation = self.app.session_state["upload_state"]["evaluation"]
        self.assertEqual(evaluation["reference_text"], "Welcome to Teams.")
        self.assertEqual(evaluation["reference_name"], "source.srt")
        self.assertIn("English (OpenAI): English segment 1.", downloads[-1])
        self.assertIn("English (Tencent local): Tencent segment 1.", downloads[-1])
        report = downloads[-1].split("Evaluation: Breeze transcription score", 1)[1]
        self.assertIn("Breeze (1-wMER): 100.0%", report)
        self.assertNotIn("OpenAI", report)
        self.assertNotIn("Tencent", report)
        self.transcribe.assert_called_once()

    def test_english_source_scores_without_override_and_freezes_submitted_format(self):
        self.start_patch(
            "src.uploads.speech_segments", return_value=[np.ones(800, dtype=np.float32)],
        )
        self.transcribe.return_value = "One two."
        self.select_evaluation(b"One two.")
        self.app.selectbox(key="eval_reference_format").select("Plain text").run()
        self.evaluate_file()
        submitted = self.app.session_state["upload_state"]["evaluation"]
        translation = self.app.session_state["translation"]
        tencent = self.app.session_state["tencent_translation"]
        expected = {"Breeze · 1-wMER": "100.0%"}
        self.assertEqual(self.evaluation_scores(), expected)
        self.assertEqual(submitted["reference_kind"], "source")

        self.app.selectbox(key="eval_reference_format").select("Transcript / captions").run()
        self.app.download_button(key="download_transcript").click().run()
        self.assertEqual(len(self.app.exception), 0)
        self.assertIs(self.app.session_state["upload_state"]["evaluation"], submitted)
        self.assertEqual(submitted["reference_kind"], "source")
        self.assertEqual(submitted["reference_text"], "One two.")
        self.assertEqual(submitted["reference"]["format"], "Plain text")
        self.assertEqual(self.evaluation_scores(), expected)
        self.transcribe.assert_called_once()
        translation.close.assert_not_called()
        tencent.close.assert_not_called()

        self.evaluate_file()
        self.assertEqual(self.app.session_state["upload_state"]["evaluation"]["reference"]["format"],
                         "Annotated transcript")
        self.assertEqual(self.evaluation_scores(), expected)
        translation.close.assert_called_once_with()
        tencent.close.assert_called_once_with()

    def test_source_evaluation_withholds_score_on_asr_failure(self):
        self.start_patch(
            "src.uploads.speech_segments", return_value=[np.ones(800, dtype=np.float32)],
        )
        self.transcribe.side_effect = RuntimeError("inference unavailable")
        self.select_evaluation(SOURCE_REFERENCE)
        self.evaluate_file()

        self.assertEqual(self.evaluation_scores(), {"Breeze · Mixed match": "—"})
        self.assertIn("Transcription failed · no final score", self.rendered_text())
        self.assertNotIn("upload_job", self.app.session_state)
        for key in ("translation", "tencent_translation"):
            self.assertEqual(self.app.session_state[key].sources, [])

    def test_legacy_translation_reference_is_not_used_to_score_breeze(self):
        from src.evaluation import word_match_score

        self.start_patch(
            "src.uploads.speech_segments", return_value=[np.ones(800, dtype=np.float32)],
        )
        self.select_evaluation()
        self.evaluate_file()
        state = self.app.session_state["upload_state"]
        state["evaluation"]["reference_kind"] = "english"
        score = self.start_patch("src.evaluation.word_match_score", wraps=word_match_score)
        self.app.run()

        self.assertEqual(self.evaluation_scores(), {"Breeze · 1-wMER": "—"})
        self.assertIn("Source transcript needed", self.rendered_text())
        score.assert_not_called()

    def test_table_english_reference_is_displayed_independently_and_submitted_mapping_is_frozen(self):
        from src.evaluation import mixed_match_score

        downloads = self.capture_downloads()
        disabled_during_job = {}
        mapping_keys = {"eval_reference_sheet", "eval_header_row", "eval_text_column", "eval_display_column",
                        "microphone_translation_type"}
        for widget_name in ("selectbox", "number_input"):
            real_widget = getattr(st, widget_name)

            def capture_control(*args, _real_widget=real_widget, **kwargs):
                key = kwargs.get("key")
                if key in mapping_keys and st.session_state.get("upload_job") is not None:
                    disabled_during_job[key] = kwargs.get("disabled", False)
                return _real_widget(*args, **kwargs)

            self.start_patch(f"streamlit.{widget_name}", side_effect=capture_control)
        table = self.start_patch("src.reference_tables.read_reference_tables", return_value={
            "Transcript": [["text_zh_TW", "translation_en"], ["你好 Teams", "Hello Teams."]],
        })
        self.start_patch(
            "src.uploads.speech_segments", return_value=[np.ones(800, dtype=np.float32)],
        )
        score = self.start_patch("src.evaluation.mixed_match_score", wraps=mixed_match_score)
        self.transcribe.return_value = "你好 Zoom"
        self.select_evaluation(b"synthetic workbook", reference_name="original.xlsx")

        self.assertEqual(self.app.selectbox(key="eval_text_column").value, 0)
        self.assertEqual(self.app.selectbox(key="eval_display_column").value, 1)
        previews = [element.value for element in self.app.text]
        self.assertIn("你好 Teams", previews)
        self.assertIn("Hello Teams.", previews)
        self.load_transcriber.assert_not_called()
        self.load_vad.assert_not_called()
        self.make_translation.assert_not_called()
        self.evaluate_file()

        self.assertEqual(disabled_during_job, dict.fromkeys(mapping_keys, True))
        evaluation = self.app.session_state["upload_state"]["evaluation"]
        self.assertEqual(evaluation["reference"]["text"], "你好 Teams")
        self.assertEqual(evaluation["reference_view"]["text"], "Hello Teams.")
        self.assertEqual(evaluation["reference_view"]["title"], "English reference")
        self.assertEqual(evaluation["reference_view"]["name"], "original.xlsx")
        self.assertIn("translation_en", evaluation["reference_view"]["column"])
        self.assertEqual(self.evaluation_scores(), {"Breeze · Mixed match": "66.7%"})
        score.assert_called_once_with("你好 Teams", "你好 Zoom")
        reference_pane = re.search(
            r'<section class="reference-pane".*?</section>', self.transcript_html(), re.DOTALL,
        ).group(0)
        self.assertIn("Hello Teams.", reference_pane)
        self.assertNotIn("你好 Zoom", reference_pane)
        for key in ("translation", "tencent_translation"):
            self.assertEqual(self.app.session_state[key].sources, ["你好 Zoom"])
        self.assertIsInstance(self.transcribe.call_args.args[0], np.ndarray)

        self.app.selectbox(key="eval_display_column").select(None).run()
        table.return_value = {
            "Replacement": [["text_zh_TW", "translation_en"], ["其他文字", "Other reference."]],
        }
        self.select_evaluation(b"replacement workbook", reference_name="replacement.xlsx")
        self.app.download_button(key="download_transcript").click().run()
        self.assertEqual(len(self.app.exception), 0)
        self.assertIs(self.app.session_state["upload_state"]["evaluation"], evaluation)
        self.assertEqual(evaluation["reference_view"]["text"], "Hello Teams.")
        self.assertIn("Hello Teams.", self.transcript_html())
        self.assertIn("Hello Teams.", downloads[-1])
        self.assertIn("你好 Teams", downloads[-1])
        self.assertNotIn("Other reference.", downloads[-1])
        self.assertEqual(score.call_count, 1)
        self.transcribe.assert_called_once()
        self.load_vad.assert_called_once_with()
        self.assertEqual(self.make_translation.call_count, 2)

    def test_table_mapping_resets_when_header_sheet_or_file_changes(self):
        table = self.start_patch("src.reference_tables.read_reference_tables", return_value={
            "First": [
                ["text_zh_TW", "translation_en", "notes"],
                ["你好", "Hello", "memo"], ["謝謝", "Thanks", "more"],
            ],
            "Second": [["notes", "text_zh_TW", "translation_en"], ["memo", "再見", "Goodbye"]],
        })
        self.select_evaluation(b"synthetic workbook", reference_name="mapping.xlsx")
        self.app.selectbox(key="eval_text_column").select(2).run()
        self.app.selectbox(key="eval_display_column").select(0).run()
        self.app.number_input(key="eval_header_row").set_value(2).run()
        self.assertEqual(len(self.app.exception), 0)
        self.assertIsNone(self.app.selectbox(key="eval_text_column").value)
        self.assertIsNone(self.app.selectbox(key="eval_display_column").value)

        self.app.selectbox(key="eval_reference_sheet").select("Second").run()
        self.assertEqual(self.app.number_input(key="eval_header_row").value, 1)
        self.assertEqual(self.app.selectbox(key="eval_text_column").value, 1)
        self.assertEqual(self.app.selectbox(key="eval_display_column").value, 2)
        self.app.number_input(key="eval_header_row").set_value(0).run()
        table.return_value = {
            "Replacement": [["text_zh_TW", "translation_en"], ["替代", "Replacement"]],
        }
        self.select_evaluation(b"different workbook", reference_name="replacement.xlsx")
        self.assertEqual(self.app.selectbox(key="eval_reference_sheet").value, "Replacement")
        self.assertEqual(self.app.number_input(key="eval_header_row").value, 1)
        self.assertEqual(self.app.selectbox(key="eval_text_column").value, 0)
        self.assertEqual(self.app.selectbox(key="eval_display_column").value, 1)
        self.load_transcriber.assert_not_called()
        self.load_vad.assert_not_called()
        self.make_translation.assert_not_called()

    def test_ambiguous_or_invalid_selected_table_cells_preserve_previous_evaluation(self):
        self.start_patch(
            "src.uploads.speech_segments", return_value=[np.ones(800, dtype=np.float32)],
        )
        self.select_evaluation(b"A short transcript")
        self.evaluate_file()
        previous = self.app.session_state["upload_state"]
        table = self.start_patch("src.reference_tables.read_reference_tables", return_value={
            "Transcript": [["First", "Second"], ["你好", "Hello"]],
        })
        self.select_evaluation(b"ambiguous workbook", reference_name="ambiguous.xlsx")
        self.assertIsNone(self.app.selectbox(key="eval_text_column").value)
        self.evaluate_file()
        self.assertGreater(len(self.app.error), 0)
        self.assertIs(self.app.session_state["upload_state"], previous)

        for name, row in (("source-formula.xlsx", [None, "Hello"]),
                          ("display-error.xlsx", ["你好", None])):
            with self.subTest(name=name):
                table.return_value = {"Transcript": [["text_zh_TW", "translation_en"], row]}
                self.select_evaluation(name.encode(), reference_name=name)
                self.evaluate_file()
                self.assertGreater(len(self.app.error), 0)
                self.assertIs(self.app.session_state["upload_state"], previous)
                self.assertNotIn("upload_job", self.app.session_state)
        self.transcribe.assert_called_once()
        self.load_vad.assert_called_once_with()
        self.assertEqual(self.make_translation.call_count, 2)

    def test_csv_and_tsv_references_preview_selected_columns_without_loading_models(self):
        for suffix, delimiter in (("csv", ","), ("tsv", "\t")):
            with self.subTest(suffix=suffix):
                content = f"text_zh_TW{delimiter}translation_en\n你好 Teams{delimiter}Hello Teams.\n"
                self.select_evaluation(content.encode(), reference_name=f"reference.{suffix}")
                self.assertEqual(self.app.selectbox(key="eval_text_column").value, 0)
                self.assertEqual(self.app.selectbox(key="eval_display_column").value, 1)
                previews = [element.value for element in self.app.text]
                self.assertIn("你好 Teams", previews)
                self.assertIn("Hello Teams.", previews)
                self.assertFalse(self.app.button(key="evaluate_file").disabled)
        self.assertNotIn("upload_state", self.app.session_state)
        self.load_transcriber.assert_not_called()
        self.load_vad.assert_not_called()
        self.make_translation.assert_not_called()

    def test_live_recording_disables_upload_controls(self):
        self.select_wav()
        self.select_evaluation(SOURCE_REFERENCE)
        self.start_recording()

        self.assertTrue(self.app.file_uploader(key="wav_file").disabled)
        self.assertTrue(self.app.button(key="transcribe_file").disabled)
        self.assertTrue(self.app.file_uploader(key="eval_wav_file").disabled)
        self.assertTrue(self.app.file_uploader(key="eval_reference_file").disabled)
        self.assertTrue(self.app.selectbox(key="eval_reference_format").disabled)
        self.assertTrue(self.app.button(key="evaluate_file").disabled)
        self.assertNotIn("upload_job", self.app.session_state)

    def test_correction_settings_are_frozen_per_session_and_pause_is_immediate(self):
        settings = {
            "slow_max_tokens": 8192,
        }
        for key, value in settings.items():
            self.app.number_input(key=key).set_value(value)
        self.app.slider(key="slow_confidence").set_value(0.7)
        self.app.toggle(key="enable_diarization").set_value(False)
        self.app.run()
        self.start_recording()
        lane = self.app.session_state["slow_lane"]

        self.assertEqual(lane.config.model, "gpt-6-astra")
        self.assertEqual(lane.config.reasoning_effort, "medium")
        self.assertEqual((lane.config.max_output_tokens, lane.config.confidence_threshold), (8192, 0.7))
        for key in settings:
            self.assertTrue(self.app.number_input(key=key).disabled)
        self.assertTrue(self.app.toggle(key="enable_diarization").disabled)
        self.assertFalse(self.make_diarization.call_args.kwargs["enabled"])

        self.app.toggle(key="enable_corrections").set_value(False).run()
        self.assertEqual(lane.snapshot()["status"], "paused")
        self.app.button(key="stop_recording").click().run()
        self.app.number_input(key="slow_max_tokens").set_value(6144).run()
        self.assertEqual(lane.config.max_output_tokens, 8192)
        self.start_recording()
        replacement = self.app.session_state["slow_lane"]
        self.assertIsNot(replacement, lane)
        self.assertEqual(lane.snapshot()["status"], "closed")
        self.assertEqual(replacement.config.max_output_tokens, 6144)
        self.assertFalse(replacement.config.enabled)
        self.assertNotEqual(replacement.session_id, lane.session_id)

    def test_correction_token_cap_defaults_to_combined_reasoning_and_output_budget(self):
        control = self.app.number_input(key="slow_max_tokens")
        self.assertEqual(control.value, 16384)
        self.assertEqual(control.label, "Reasoning + output token cap")
        self.assertIn("OPENAI_CORRECTION_MAX_OUTPUT_TOKENS", control.proto.help)
        self.assertIn("reasoning", control.proto.help)
        self.start_recording()
        self.assertEqual(self.app.session_state["slow_lane"].config.max_output_tokens, 16384)

    def test_correction_token_cap_uses_environment_for_a_fresh_session(self):
        os.environ["OPENAI_CORRECTION_MAX_OUTPUT_TOKENS"] = "8192"
        self.app = AppTest.from_file(str(APP_FILE)).run()
        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual(self.app.number_input(key="slow_max_tokens").value, 8192)
        self.start_recording()
        lane = self.app.session_state["slow_lane"]
        self.assertEqual(lane.config.max_output_tokens, 8192)
        self.assertTrue(self.app.number_input(key="slow_max_tokens").disabled)
        self.app.button(key="stop_recording").click().run()
        self.app.number_input(key="slow_max_tokens").set_value(6144).run()
        self.assertEqual(lane.config.max_output_tokens, 8192)
        self.start_recording()
        self.assertEqual(self.app.session_state["slow_lane"].config.max_output_tokens, 6144)

    def test_correction_controls_have_no_deadlines_even_with_legacy_session_settings(self):
        for key in ("slow_horizon", "slow_seal_timeout", "slow_request_timeout"):
            self.app.session_state[key] = 1
        self.app.run()
        keys = {element.key for element in self.app.number_input}
        self.assertFalse(keys & {"slow_horizon", "slow_seal_timeout", "slow_request_timeout"})
        self.assertFalse(any("timeout" in element.label.casefold() for element in self.app.number_input))
        self.start_recording()
        config = self.app.session_state["slow_lane"].snapshot()["config"]
        self.assertFalse(set(config) & {"revision_horizon_s", "seal_timeout_s", "request_timeout_s"})

    def test_correction_has_no_batch_control_even_with_legacy_session_setting(self):
        self.app.session_state["slow_window_n"] = 8
        self.app.run()
        self.assertNotIn("slow_window_n", {element.key for element in self.app.number_input})
        self.assertFalse(any("Segments per review" in element.label for element in self.app.number_input))

    def test_correction_model_and_reasoning_are_configurable_and_frozen_per_session(self):
        os.environ["OPENAI_CORRECTION_MODEL"] = "configured-correction-model"
        os.environ["OPENAI_CORRECTION_REASONING_EFFORT"] = " HIGH "
        os.environ["OPENAI_FILTER_BACKGROUND_SPEECH"] = "false"
        self.start_recording()
        lane = self.app.session_state["slow_lane"]

        self.assertEqual(lane.config.model, "configured-correction-model")
        self.assertEqual(lane.config.reasoning_effort, "high")
        self.assertFalse(lane.config.filter_background_speech)
        self.assertTrue(any("configured-correction-model · High reasoning" in caption.value
                            for caption in self.app.caption))

        os.environ["OPENAI_CORRECTION_MODEL"] = "next-correction-model"
        os.environ["OPENAI_CORRECTION_REASONING_EFFORT"] = "max"
        os.environ["OPENAI_FILTER_BACKGROUND_SPEECH"] = "true"
        self.app.run()
        self.assertEqual(lane.config.model, "configured-correction-model")
        self.assertEqual(lane.config.reasoning_effort, "high")
        self.assertFalse(lane.config.filter_background_speech)
        self.assertTrue(any("Astra ·" in caption.value
                            and "configured-correction-model · High reasoning" in caption.value
                            for caption in self.app.caption))
        self.app.button(key="stop_recording").click().run()
        self.start_recording()
        replacement = self.app.session_state["slow_lane"]
        self.assertEqual(replacement.config.model, "next-correction-model")
        self.assertEqual(replacement.config.reasoning_effort, "max")
        self.assertTrue(replacement.config.filter_background_speech)

    def test_fast_completion_dispatches_astra_without_a_ui_poll_and_keeps_its_original_lane(self):
        self.configure_translations(openai=[None])
        self.start_patch("src.uploads.speech_segments", return_value=[
            {"audio": np.zeros(1600, dtype=np.float32), "start_s": 0.0, "end_s": 0.1},
        ])
        self.select_wav()
        self.transcribe_file()
        provider = self.translation_sessions["openai"]
        original_lane = self.app.session_state["slow_lane"]
        self.correct.assert_not_called()
        self.assertIsNotNone(provider.on_result)
        self.assertIsNone(self.translation_sessions["tencent"].on_result)
        provider.translations[0] = "The fast English result."
        failures = []

        def notify():
            try:
                provider.on_result(provider.sources.copy(), provider.snapshot())
            except Exception as exc:
                failures.append(type(exc).__name__)

        thread = Thread(target=notify, daemon=True)
        thread.start()
        thread.join(timeout=1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(failures, [])
        completed = self.wait_for_corrections()
        self.assertEqual(completed["statuses"], ["confirmed"])
        self.assertEqual(len(self.review_requests()), 1)
        self.assertEqual(len(self.review_requests(conversation=True)), 1)
        self.assertEqual(completed["conversation_review"]["reviewed"], 1)
        self.assertEqual(completed["segments"][0]["timing"]["start_s"], 0.0)

        self.select_wav(name="replacement.wav")
        self.transcribe_file()
        replacement = self.app.session_state["slow_lane"]
        self.assertIsNot(replacement, original_lane)
        provider.translations[0] = "A stale result from the previous conversation."
        provider.on_result(provider.sources.copy(), provider.snapshot())
        self.assertEqual(replacement.snapshot()["statuses"], ["waiting"])
        self.assertEqual(len(self.review_requests()), 1)
        self.assertEqual(len(self.review_requests(conversation=True)), 1)

    def test_upload_publishes_each_parallel_correction_as_its_own_review_finishes(self):
        first_entered, second_entered, release_first = Event(), Event(), Event()
        self.addCleanup(release_first.set)

        def correct(request):
            self.assertEqual(len(request["segments"]), 1)
            row = request["segments"][0]
            first = row["source_text"] == "First transcript"
            if first:
                first_entered.set()
                if not release_first.wait(5):
                    raise RuntimeError("The first mocked review was not released.")
            else:
                second_entered.set()
            return {"corrections": [{
                "segment_id": row["segment_id"], "base_version": row["base_version"],
                "target_text": "Reviewed first sentence." if first else "Reviewed second sentence.",
                "change_type": ["style"], "confidence": 0.95,
                "rationale": "Make the wording clear.", "term_pairs": [],
            }], "no_change": []}

        self.correct.side_effect = correct
        self.transcribe.side_effect = ["First transcript", "Second transcript"]
        self.start_patch("src.uploads.speech_segments", return_value=[
            {"audio": np.zeros(1600, dtype=np.float32), "start_s": 0.0, "end_s": 0.1},
            {"audio": np.zeros(1600, dtype=np.float32), "start_s": 0.2, "end_s": 0.3},
        ])
        self.app.session_state["slow_window_n"] = 8
        self.select_wav()
        self.transcribe_file()
        self.assertTrue(first_entered.wait(1))
        self.assertTrue(second_entered.wait(1), "The second review waited for the first review to finish.")
        deadline = monotonic() + 2
        while monotonic() < deadline:
            snapshot = self.app.session_state["slow_lane"].snapshot()
            if snapshot["segments"][1]["status"] == "corrected":
                break
            sleep(0.005)
        self.assertEqual(snapshot["statuses"], ["draft", "corrected"])
        self.assertEqual(snapshot["first_pass"]["pending"], 1)
        self.assertEqual(snapshot["conversation_review"]["pending"], 1)
        self.assertEqual(snapshot["pending"], 2)
        self.assertEqual(snapshot["active_reviews"], 1)
        self.assertEqual(self.review_requests(conversation=True), [])
        self.app.run()
        self.assertIn("Reviewed second sentence.", self.transcript_html())
        self.assertIn("Corrected", self.transcript_html())
        self.assertNotIn("Reviewed first sentence.", self.transcript_html())
        self.assertIn("Astra · 1 segment(s) under review", self.rendered_text())
        self.assertIn("1 awaiting review", self.rendered_text())
        self.assertIn("Conversation review 0/2", self.rendered_text())
        release_first.set()
        final = self.wait_for_corrections()
        self.assertEqual(final["statuses"], ["corrected", "corrected"])
        self.assertEqual(final["translations"], ["Reviewed first sentence.", "Reviewed second sentence."])
        self.assertEqual(len(self.review_requests()), 2)
        self.assertEqual([request["segments"][0]["source_text"]
                          for request in self.review_requests(conversation=True)],
                         ["First transcript", "Second transcript"])
        self.assertEqual(final["conversation_review"]["reviewed"], 2)

    def test_fourth_lane_replaces_live_draft_when_late_correction_finishes(self):
        entered, release = Event(), Event()
        self.addCleanup(release.set)

        def correct(request):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("Test correction was not released.")
            row = request["segments"][0]
            return {"corrections": [{
                "segment_id": row["segment_id"], "base_version": row["base_version"],
                "target_text": "The reviewed English sentence.", "change_type": ["word_order"],
                "confidence": 0.9, "rationale": "Preserve sentence order.", "term_pairs": [],
            }], "no_change": []}

        self.correct.side_effect = correct
        self.start_patch("src.uploads.speech_segments", return_value=[{
            "audio": np.zeros(1600, dtype=np.float32), "start_s": 0.0, "end_s": 0.1,
        }])
        self.select_wav()
        self.transcribe_file()
        self.assertTrue(entered.wait(1))
        self.assertIn("Astra · Correction", self.transcript_html())
        self.assertIn("English segment 1.", self.transcript_html())
        self.assertIn(">Draft</span>", self.transcript_html())
        self.app.selectbox(key="subtitle_view").select("Final record").run()
        self.assertIn("Awaiting final text…", self.transcript_html())
        self.assertNotIn("The reviewed English sentence.", self.transcript_html())
        self.app.selectbox(key="subtitle_view").select("Live subtitles").run()

        release.set()
        final = self.wait_for_corrections()
        self.app.run()
        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual(final["authoritative"], ["The reviewed English sentence."])
        self.assertEqual(final["translations"], ["The reviewed English sentence."])
        self.assertIn(">Corrected</span>", self.transcript_html())
        self.assertIn('class="translation-cell astra-cell astra-corrected"', self.transcript_html())
        self.assertIn("The reviewed English sentence.", self.transcript_html())
        self.assertFalse(self.app.download_button(key="download_final_srt").disabled)
        self.app.selectbox(key="subtitle_view").select("Final record").run()
        self.assertIn("Corrected · Final", self.transcript_html())
        self.assertIn("The reviewed English sentence.", self.transcript_html())
        self.assertIn("A short transcript", self.transcript_html())
        self.assertIn("Tencent segment 1.", self.transcript_html())
        self.assertEqual(self.transcribe.call_count, 1)
        self.assertEqual(len(self.review_requests()), 1)
        self.assertEqual(len(self.review_requests(conversation=True)), 1)
        self.assertEqual(final["conversation_review"]["reviewed"], 1)

    def test_upload_splits_speaker_turns_before_asr_and_preserves_offsets(self):
        audio = np.zeros(32000, dtype=np.float32)
        self.start_patch("src.uploads.speech_segments", return_value=[{
            "audio": audio, "start_s": 10.0, "end_s": 12.0,
        }])
        self.diarization.snapshot.return_value = {
            "status": "complete", "pending": 0, "error": None,
            "processed_seconds": 12, "received_seconds": 12,
            "segments": [
                {"start_s": 10, "end_s": 11, "speaker_id": "Speaker 2"},
                {"start_s": 11, "end_s": 12, "speaker_id": "Speaker 1"},
            ],
        }
        self.transcribe.side_effect = ["First speaker.", "Second speaker."]
        self.select_wav()
        self.transcribe_file()
        deadline = monotonic() + 5
        while "upload_job" in self.app.session_state and monotonic() < deadline:
            self.app.run()
        self.assertEqual(len(self.app.exception), 0)
        state = self.app.session_state["upload_state"]
        self.assertTrue(state["finished"])
        self.assertIsNone(state["error"])
        self.assertEqual(state["texts"], ["First speaker.", "Second speaker."])
        self.assertEqual([t["start_s"] for t in state["timings"]], [10, 11])
        self.assertEqual([t["speaker_id"] for t in state["timings"]], ["Speaker 2", "Speaker 1"])
        np.testing.assert_array_equal(np.concatenate([call.args[0] for call in self.transcribe.call_args_list]), audio)
        self.assertTrue(all("_turns_prepared" not in t for t in state["timings"]))

    def test_local_asr_loading_does_not_hold_the_upload_render_thread(self):
        entered, release = Event(), Event()
        self.addCleanup(release.set)

        def load():
            entered.set()
            if not release.wait(5):
                raise RuntimeError("Model loading was not released.")
            return self.transcribe

        self.load_transcriber.side_effect = load
        self.start_patch("src.uploads.speech_segments", return_value=[np.zeros(1600, dtype=np.float32)])
        self.select_wav()
        self.app.button(key="transcribe_file").click().run(timeout=3)
        self.assertTrue(entered.wait(1))
        self.assertFalse(self.app.session_state["upload_job"]["asr_task"].done())
        self.assertEqual(len(self.app.exception), 0)
        self.assertTrue(any("Tencent · Local" in item.value for item in self.app.caption))
        release.set()
        deadline = monotonic() + 5
        while "upload_job" in self.app.session_state and monotonic() < deadline:
            self.app.run()
        self.assertTrue(self.app.session_state["upload_state"]["finished"])
        self.assertIsNone(self.app.session_state["upload_state"]["error"])

    def test_completed_fast_draft_is_rendered_and_reviewed_while_next_asr_is_blocked(self):
        from src.ui import render_transcript

        second_started, release, rendered_while_blocked = Event(), Event(), Event()
        interrupted = False
        calls = 0
        pending_task_id = None
        self.addCleanup(release.set)
        self.configure_translations(openai=[None, None])
        decoded, views = self.track_upload_audio()

        def transcribe(audio):
            nonlocal calls
            calls += 1
            if calls == 1:
                return "First original."
            second_started.set()
            self.translation_sessions["openai"].translations[0] = "Fast English arrived."
            if not release.wait(5):
                raise RuntimeError("The UI did not publish fast output during blocked ASR.")
            np.testing.assert_array_equal(audio, np.zeros(800, dtype=np.float32))
            return "Second original."

        def observe_render(texts, *args, **kwargs):
            nonlocal interrupted, pending_task_id
            if second_started.is_set() and not interrupted and st.session_state.get("upload_job"):
                interrupted = True
                job = st.session_state.upload_job
                pending_task_id = id(job["asr_task"])
                self.assertFalse(job["asr_task"].done())
                self.assertIs(job["audio"], decoded[0]())
                st.rerun()
            slow = kwargs.get("slow_lane", {})
            if (second_started.is_set() and not release.is_set() and texts == ["First original."]
                    and st.session_state.get("upload_job")
                    and args[0][0] == "Fast English arrived."
                    and slow.get("translations") == ["Fast English arrived."] and self.correct.call_count):
                job = st.session_state.upload_job
                self.assertEqual(id(job["asr_task"]), pending_task_id)
                self.assertEqual(job["next_segment"], 1)
                self.assertIsNotNone(decoded[0]())
                self.assertTrue(np.shares_memory(views[1](), decoded[0]()))
                rendered_while_blocked.set()
                release.set()
            return render_transcript(texts, *args, **kwargs)

        self.load_transcriber.return_value = transcribe
        self.start_patch("src.ui.render_transcript", side_effect=observe_render)
        self.select_wav()
        self.app.button(key="transcribe_file").click().run(timeout=8)
        deadline = monotonic() + 6
        while "upload_job" in self.app.session_state and monotonic() < deadline:
            self.app.run(timeout=8)
            sleep(0.01)

        self.assertEqual(len(self.app.exception), 0)
        self.assertTrue(interrupted)
        state = self.app.session_state["upload_state"]
        self.assertIsNone(state["error"])
        self.assertTrue(rendered_while_blocked.is_set())
        self.assertEqual(state["texts"], ["First original.", "Second original."])
        self.assertEqual(calls, 2)
        self.assertIn("Fast English arrived.", self.transcript_html())
        gc.collect()
        self.assertEqual(len(decoded), 1)
        self.assertTrue(all(reference() is None for reference in decoded + views))

    def test_speaker_offsets_reach_corrections_and_references_stay_out_of_model_inputs(self):
        source_reference, english_reference = "來源參考秘密", "PRIVATE ENGLISH REFERENCE"
        self.start_patch("src.reference_tables.read_reference_tables", return_value={
            "Transcript": [
                ["text_zh_TW", "translation_en"],
                [source_reference, english_reference], ["第二參考行", "Second reference line"],
                ["第三參考行", "Third reference line"],
            ],
        })
        audio_segments = [
            {"audio": np.zeros(800, dtype=np.float32), "start_s": 0.0, "end_s": 0.05},
            {"audio": np.zeros(640, dtype=np.float32), "start_s": 0.06, "end_s": 0.1},
        ]
        split = self.start_patch("src.uploads.speech_segments", return_value=audio_segments)
        self.diarization.snapshot.return_value = {
            "status": "complete", "pending": 0, "error": None,
            "segments": [
                {"start_s": 0.005, "end_s": 0.045, "speaker_id": "Speaker 1"},
                {"start_s": 0.06, "end_s": 0.1, "speaker_id": "Speaker 2"},
            ],
        }
        self.transcribe.side_effect = ["模型輸出一", "模型輸出二"]
        self.select_evaluation(b"synthetic workbook", reference_name="reference.xlsx")
        self.evaluate_file()
        snapshot = self.wait_for_corrections()
        self.app.run()

        self.assertEqual(len(self.app.exception), 0)
        split.assert_called_once()
        self.assertTrue(split.call_args.kwargs["with_timestamps"])
        self.diarization.push.assert_called_once()
        np.testing.assert_array_equal(self.diarization.push.call_args.args[0], np.zeros(1600, dtype=np.float32))
        self.diarization.finish.assert_called_once_with()
        rows = re.findall(r'<li class="transcript-row"[^>]*>(.*?)</li>', self.transcript_html(), re.DOTALL)
        self.assertEqual(len(rows), 2)
        for index, row in enumerate(rows, 1):
            self.assertIn(f"Speaker {index}", row)
        requests = [call.args[0] for call in self.correct.call_args_list]
        self.assertTrue(requests)
        sent = {row["source_text"]: row for request in requests for row in request["segments"]}
        stored = {row["source_text"]: row for row in snapshot["segments"]}
        for text, label, start in (("模型輸出一", "Speaker 1", 0.0), ("模型輸出二", "Speaker 2", 0.06)):
            self.assertEqual(sent[text]["speaker_id"], label)
            self.assertEqual(stored[text]["timing"]["start_s"], start)
        serialized = json.dumps(requests, ensure_ascii=False)
        self.assertNotIn(source_reference, serialized)
        self.assertNotIn(english_reference, serialized)
        self.assertNotIn('"audio"', serialized)
        for key in ("translation", "tencent_translation"):
            self.assertEqual(self.app.session_state[key].sources, ["模型輸出一", "模型輸出二"])
        reference_pane = re.search(r'<section class="reference-pane".*?</section>', self.transcript_html(), re.DOTALL).group(0)
        self.assertIn(english_reference, reference_pane)
        self.assertIn("R03", reference_pane)
        self.assertNotIn("模型輸出一", reference_pane)
        self.assertEqual(set(self.evaluation_scores()), {"Breeze · Mixed match"})

    def test_conversation_review_failure_keeps_first_english_and_retries_only_the_second_pass(self):
        fail_second = Event()
        fail_second.set()
        exported = []
        real_download = st.download_button

        def capture_download(*args, **kwargs):
            if kwargs.get("key") == "download_session":
                data = kwargs["data"] if "data" in kwargs else args[1]
                exported.append(json.loads(data))
            return real_download(*args, **kwargs)

        def correct(request):
            row = request["segments"][0]
            first = row["source_text"] == "Original first."
            second_pass = request.get("review_stage") == "conversation"
            if second_pass and first and fail_second.is_set():
                raise RuntimeError("PRIVATE SECOND PASS ERROR")
            prefix = "Conversation" if second_pass else "First accepted"
            target = f"{prefix} {'first' if first else 'second'} sentence."
            return {"corrections": [{
                "segment_id": row["segment_id"], "base_version": row["base_version"],
                "target_text": target, "change_type": ["style"], "confidence": 0.95,
                "rationale": "Clarify the sentence.", "term_pairs": [],
            }], "no_change": []}

        self.correct.side_effect = correct
        self.start_patch("streamlit.download_button", side_effect=capture_download)
        self.start_patch("src.uploads.speech_segments", return_value=[
            {"audio": np.zeros(800, dtype=np.float32), "start_s": 0.0, "end_s": 0.05},
            {"audio": np.zeros(800, dtype=np.float32), "start_s": 0.05, "end_s": 0.1},
        ])
        self.transcribe.side_effect = ["Original first.", "Original second."]
        self.select_wav()
        self.transcribe_file()
        deadline = monotonic() + 3
        while monotonic() < deadline:
            failed = self.app.session_state["slow_lane"].snapshot()
            if (failed["status"] == "degraded" and len(failed["segments"]) == 2
                    and failed["segments"][1]["first_pass_status"] == "corrected"):
                break
            sleep(0.005)
        self.assertEqual(failed["authoritative"], [
            "First accepted first sentence.", "First accepted second sentence.",
        ])
        self.assertEqual(failed["segments"][0]["conversation_review_status"], "failed")
        self.assertEqual(failed["segments"][1]["first_pass_status"], "corrected")
        self.app.run()
        self.assertIn("First accepted second sentence.", self.transcript_html())
        self.assertIn("astra-corrected", self.transcript_html())
        self.assertIn("Conversation review 0/2", self.rendered_text())
        self.assertTrue(any("Conversation review:" in warning.value for warning in self.app.warning))
        self.assertNotIn("PRIVATE SECOND PASS ERROR", json.dumps(exported[-1]))
        self.assertEqual(exported[-1]["first_pass"]["authoritative"], failed["authoritative"])

        fail_second.clear()
        self.app.button(key="retry_corrections").click().run()
        reviewed = self.wait_for_corrections()
        self.app.run()
        self.assertEqual(reviewed["authoritative"], [
            "Conversation first sentence.", "Conversation second sentence.",
        ])
        self.assertEqual(reviewed["first_pass"]["authoritative"], failed["authoritative"])
        self.assertEqual(reviewed["conversation_review"]["reviewed"], 2)
        self.assertEqual(exported[-1]["authoritative"], reviewed["authoritative"])
        self.assertEqual(exported[-1]["translation_config"]["type"], "Compare all translations")
        first_calls = [call for call in self.correct.call_args_list
                       if call.args[0].get("review_stage") != "conversation"]
        self.assertEqual(len(first_calls), 2)

    def test_correction_and_speaker_failures_preserve_fast_results_and_allow_correction_retry(self):
        confirm = self.correct.side_effect
        self.correct.side_effect = RuntimeError("PRIVATE API ERROR")
        self.diarization.snapshot.return_value = {
            "status": "failed", "segments": [], "pending": 0,
            "error": "Speaker diarization is unavailable.",
        }
        self.start_patch("src.uploads.speech_segments", return_value=[np.zeros(800, dtype=np.float32)])
        self.select_wav()
        self.transcribe_file()
        degraded = self.wait_for_corrections()
        self.app.run()

        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual(degraded["status"], "degraded")
        self.assertIn("Some reviews need retry", self.rendered_text())
        self.assertIn("Speaker detection failed", self.rendered_text())
        self.assertIn("Speaker unknown", self.transcript_html())
        self.assertIn("A short transcript", self.transcript_html())
        self.assertIn("English segment 1.", self.transcript_html())
        self.assertIn("Tencent segment 1.", self.transcript_html())
        self.assertNotIn("PRIVATE API ERROR", json.dumps(degraded))
        self.correct.side_effect = confirm
        self.app.button(key="retry_corrections").click().run()
        recovered = self.wait_for_corrections()
        self.app.run()
        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual(recovered["authoritative"], ["English segment 1."])
        self.assertIn(">Confirmed</span>", self.transcript_html())
        self.assertEqual(self.transcribe.call_count, 1)
        self.assertEqual(len(self.review_requests()), 2)
        self.assertEqual(len(self.review_requests(conversation=True)), 1)
        self.assertEqual(recovered["conversation_review"]["reviewed"], 1)


if __name__ == "__main__":
    unittest.main()
