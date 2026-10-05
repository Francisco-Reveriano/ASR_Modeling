"""Exercise the endpoint-only Streamlit workflow without HTTP or model weights."""

from concurrent.futures import Future
from dataclasses import dataclass
from io import BytesIO
import gc
import json
import os
from pathlib import Path
from threading import Event, Lock, Thread
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

from src.translation import TranslationSession as RealTranslationSession


APP_FILE = Path(__file__).resolve().parents[1] / "app.py"
SOURCE_REFERENCE = (
    "# Synthetic meeting transcript\n"
    "[00:00.000 --> 00:01.250] SPK1: (overlap) 你好 Teams\n"
    "[BG 00:01.250 --> 00:02.000] (typing) keyboard noise\n"
).encode()


@dataclass(frozen=True)
class FakeConfig:
    label: str
    model: str

    def public(self):
        return {"label": self.label, "model": self.model}


def wav_bytes():
    output = BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(np.zeros(1600, dtype="<i2").tobytes())
    return output.getvalue()


class FakeTranslationSession:
    """Thread-safe controllable outcomes with the real cumulative-submit shape."""

    def __init__(self, translate, *, contextual, owned_client, glossary, initial_results=None,
                 initial_errors=None, initial_filtered=None):
        self.translate = translate
        self.initial_results = initial_results
        self.initial_errors = initial_errors or []
        self.initial_filtered = initial_filtered or []
        self.sources, self.translations, self.errors, self.filtered = [], [], [], []
        self._lock = Lock()
        self._closed = False
        self.submit = Mock(side_effect=self._submit)
        self.retry_failed = Mock(side_effect=self._retry_failed)
        self.close = Mock(side_effect=self._close)

    def _submit(self, texts, *, allow_prefix=False):
        with self._lock:
            if self._closed:
                return
            overlap = min(len(texts), len(self.sources))
            if texts[:overlap] != self.sources[:overlap] or (len(texts) < len(self.sources) and not allow_prefix):
                raise ValueError("Source segments are append-only.")
            for text in texts[len(self.sources):]:
                index = len(self.sources)
                self.sources.append(text)
                self.translations.append(
                    self.initial_results[index] if self.initial_results is not None and index < len(self.initial_results)
                    else f"English segment {index + 1}."
                )
                self.errors.append(self.initial_errors[index] if index < len(self.initial_errors) else None)
                self.filtered.append(self.initial_filtered[index] if index < len(self.initial_filtered) else False)

    def _retry_failed(self):
        with self._lock:
            for index, error in enumerate(self.errors):
                if error:
                    self.errors[index] = None
                    self.translations[index] = f"English segment {index + 1}."

    def _close(self):
        with self._lock:
            self._closed = True
        self.translate.close()

    def snapshot(self):
        with self._lock:
            return {
                "translations": self.translations.copy(), "errors": self.errors.copy(),
                "filtered": self.filtered.copy(),
                "pending": sum(text is None and error is None for text, error in zip(self.translations, self.errors)),
            }


class RecordingAppTests(unittest.TestCase):
    def setUp(self):
        st.cache_resource.clear()
        st.cache_data.clear()
        self.start_patch("dotenv.load_dotenv", return_value=False)
        self.enterContext(patch.dict(os.environ, {
            "WEBRTC_ICE_SERVERS_JSON": "", "GLOSSARY_FILE": "", "DNT_FILE": "",
        }))
        self.profiles = {"alpha": FakeConfig("Alpha English", "alpha-model"),
                         "beta": FakeConfig("Beta English", "beta-model")}
        self.asr_config = FakeConfig("Remote ASR", "speech-model")
        self.load_profiles = self.start_patch("src.vllm.load_fast_profiles", return_value=self.profiles)
        self.load_asr_config = self.start_patch("src.vllm.load_asr_config", return_value=self.asr_config)
        self.start_patch("src.vllm.default_fast_profile", return_value="alpha")
        self.transcribe = Mock(return_value="A short transcript")
        self.make_asr = self.start_patch("src.vllm.create_asr", return_value=self.transcribe)
        self.make_translator = self.start_patch("src.vllm.create_translator", side_effect=lambda profile: Mock())
        self.vad = Mock(return_value=None)
        self.load_vad = self.start_patch("src.pipeline.load_vad", return_value=self.vad)
        self.context = SimpleNamespace(state=SimpleNamespace(playing=False, signalling=False))
        self.webrtc = self.start_patch("streamlit_webrtc.webrtc_streamer", return_value=self.context)
        self.make_translation = self.start_patch("src.translation.TranslationSession", side_effect=FakeTranslationSession)
        self.app = AppTest.from_file(str(APP_FILE)).run()
        self.addCleanup(self.close_current)
        self.assert_no_exception()

    def start_patch(self, target, **kwargs):
        patcher = patch(target, **kwargs)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def close_current(self):
        if "recording" in self.app.session_state:
            recording = self.app.session_state["recording"]
            recording.close()
            if recording._worker.ident is not None:
                recording._worker.join(timeout=3)
            self.assertFalse(recording._worker.is_alive())
        if "translation" in self.app.session_state:
            self.app.session_state["translation"].close()

    def assert_no_exception(self):
        self.assertEqual(len(self.app.exception), 0, [item.message for item in self.app.exception])

    def start_recording(self):
        self.app.button(key="start_recording").click().run()
        self.assert_no_exception()
        self.app.run()
        self.assert_no_exception()
        return self.app.session_state["recording"]

    def finish_recording(self, recording):
        recording.finish()
        if recording._worker.ident is not None:
            recording._worker.join(timeout=3)
        self.app.run()
        self.assert_no_exception()

    def push_speech(self, recording):
        frame = av.AudioFrame.from_ndarray(np.ones((1, 800), dtype=np.float32), format="fltp", layout="mono")
        frame.sample_rate = 16000
        recording.push(frame)

    def rendered_text(self):
        return "\n".join(item.value for kind in ("markdown", "caption", "info", "text", "title", "error", "warning")
                         for item in self.app.get(kind))

    def transcript_html(self):
        return "\n".join(item.value for item in self.app.markdown)

    def select_wav(self, name="sample.wav", content=None):
        self.app.file_uploader(key="wav_file").set_value((name, wav_bytes() if content is None else content, "audio/wav")).run()
        self.assert_no_exception()

    def transcribe_file(self):
        self.app.button(key="transcribe_file").click().run()
        self.assert_no_exception()

    def select_evaluation(self, reference=b"English reference.", *, audio_name="evaluation.wav", reference_name="reference.txt"):
        self.app.file_uploader(key="eval_wav_file").set_value((audio_name, wav_bytes(), "audio/wav")).run()
        self.app.file_uploader(key="eval_reference_file").set_value((reference_name, reference, "text/plain")).run()
        self.assert_no_exception()

    def evaluate_file(self):
        self.app.button(key="evaluate_file").click().run()
        self.assert_no_exception()

    def evaluation_scores(self):
        return {metric.label: metric.value for metric in self.app.metric}

    def segments(self, count=1):
        return self.start_patch("src.uploads.speech_segments", return_value=[
            {"audio": np.ones(800, dtype=np.float32), "start_s": index * 0.05, "end_s": (index + 1) * 0.05}
            for index in range(count)
        ])

    def configure_translations(self, *, results=None, errors=None, filtered=None):
        self.make_translation.side_effect = lambda *args, **kwargs: FakeTranslationSession(
            *args, **kwargs, initial_results=results, initial_errors=errors, initial_filtered=filtered,
        )

    def capture_downloads(self):
        from streamlit.delta_generator import DeltaGenerator
        downloads = {}
        real_download = DeltaGenerator.download_button
        real_st_download = st.download_button

        def capture(*args, **kwargs):
            downloads[kwargs.get("key")] = kwargs.get("data", args[2] if len(args) > 2 else None)
            return real_download(*args, **kwargs)
        def capture_st(*args, **kwargs):
            downloads[kwargs.get("key")] = kwargs.get("data", args[1] if len(args) > 1 else None)
            return real_st_download(*args, **kwargs)
        self.start_patch("streamlit.delta_generator.DeltaGenerator.download_button", autospec=True, side_effect=capture)
        self.start_patch("streamlit.download_button", side_effect=capture_st)
        return downloads

    def test_initial_page_has_one_shared_selector_and_no_inference(self):
        self.assertEqual(self.app.selectbox(key="next_fast_profile").label, "Translation model")
        self.assertEqual(self.app.selectbox(key="next_fast_profile").options, ["Alpha English", "Beta English"])
        self.assertFalse(self.app.button(key="start_recording").disabled)
        self.assertTrue(self.app.button(key="stop_recording").disabled)
        self.assertTrue(self.app.download_button(key="download_transcript").disabled)
        self.make_asr.assert_not_called()
        self.make_translator.assert_not_called()
        self.load_vad.assert_not_called()
        self.webrtc.assert_not_called()
        for obsolete in ("Astra", "Tencent", "Nemotron", "OpenAI", "on this Mac"):
            self.assertNotIn(obsolete, self.rendered_text())
        self.assertEqual(len(self.app.toggle), 0)

    def test_terminology_paths_resolve_against_repository_and_blank_stays_none(self):
        glossary = self.start_patch("src.glossary.load_glossary", return_value=Mock())
        self.app.text_input(key="glossary_path").set_value("terms/project.csv").run()
        recording = self.start_recording()
        glossary.assert_called_once_with(APP_FILE.parent / "terms/project.csv", None)
        self.finish_recording(recording)
        self.app.text_input(key="glossary_path").set_value("").run()
        self.app.text_input(key="dnt_path").set_value("/tmp/project-identifiers.txt").run()
        self.start_recording()
        self.assertEqual(glossary.call_args.args, (None, Path("/tmp/project-identifiers.txt")))

    def test_missing_endpoint_configuration_is_actionable_without_model_loading(self):
        from src.vllm import VllmConfigError
        self.load_asr_config.side_effect = VllmConfigError("Set VLLM_ASR_BASE_URL in .env.")
        self.app.run()
        self.assert_no_exception()
        self.assertTrue(self.app.button(key="start_recording").disabled)
        self.assertIn("VLLM_ASR_BASE_URL", self.rendered_text())
        self.make_asr.assert_not_called()
        self.make_translator.assert_not_called()
        self.load_vad.assert_not_called()
        self.load_asr_config.side_effect = None
        self.app.run()
        self.assertFalse(self.app.button(key="start_recording").disabled)

    def test_microphone_profile_and_asr_configuration_are_frozen_per_conversation(self):
        self.app.selectbox(key="next_fast_profile").select("beta").run()
        recording = self.start_recording()
        config = self.app.session_state["conversation_config"]
        self.assertIs(config["fast"], self.profiles["beta"])
        self.assertIs(config["asr"], self.asr_config)
        self.make_asr.assert_called_once_with(self.asr_config)
        self.make_translator.assert_called_once_with(self.profiles["beta"])
        self.assertTrue(self.app.selectbox(key="next_fast_profile").disabled)
        self.assertTrue(self.make_translation.call_args.kwargs["contextual"])
        self.assertTrue(self.make_translation.call_args.kwargs["owned_client"])
        self.app.run()
        self.assertIs(self.app.session_state["recording"], recording)
        self.assertIsNone(recording._worker.ident)
        self.make_asr.assert_called_once()
        self.finish_recording(recording)
        self.app.selectbox(key="next_fast_profile").select("alpha").run()
        self.assertIs(self.app.session_state["conversation_config"], config)
        self.assertIn("Beta English", self.transcript_html())
        self.start_recording()
        self.assertIs(self.app.session_state["conversation_config"]["fast"], self.profiles["alpha"])

    def test_live_asr_dispatches_translation_before_ui_poll(self):
        self.vad.side_effect = [{"start": 0}, None]
        recording = self.start_recording()
        translation = self.app.session_state["translation"]
        self.push_speech(recording)
        recording.finish()
        recording._worker.join(timeout=3)
        self.assertEqual(translation.sources, ["A short transcript"])
        self.assertEqual(len(recording.snapshot()["timings"]), 1)
        self.app.run()
        self.assertIn("A short transcript", self.transcript_html())
        self.assertIn("English segment 1.", self.transcript_html())
        first = translation
        self.start_recording()
        first.close.assert_called_once()
        first.submit(["A short transcript", "Late result"])
        self.assertEqual(first.sources, ["A short transcript"])
        self.assertEqual(self.app.session_state["translation"].sources, [])

    def test_prepare_failure_preserves_previous_results_and_redacts_error(self):
        self.segments()
        self.select_wav()
        self.transcribe_file()
        previous = self.app.session_state["upload_state"]
        translation = self.app.session_state["translation"]
        self.load_vad.side_effect = RuntimeError("secret-key private request body")
        self.app.button(key="start_recording").click().run()
        self.assert_no_exception()
        self.assertIs(self.app.session_state["upload_state"], previous)
        translation.close.assert_not_called()
        self.assertNotIn("secret-key", self.rendered_text())
        self.assertIn("Could not prepare", self.rendered_text())
        self.load_vad.side_effect = None
        self.start_recording()
        self.assertNotIn("Could not prepare", self.rendered_text())

    def test_selecting_wav_or_reference_does_not_start_processing(self):
        self.select_wav()
        self.select_evaluation(SOURCE_REFERENCE)
        self.make_asr.assert_not_called()
        self.make_translator.assert_not_called()
        self.load_vad.assert_not_called()
        self.make_translation.assert_not_called()

    def test_upload_and_evaluation_use_the_shared_profile(self):
        self.segments()
        self.app.selectbox(key="next_fast_profile").select("beta").run()
        self.select_wav()
        self.transcribe_file()
        first = self.app.session_state["translation"]
        self.assertIs(self.app.session_state["conversation_config"]["fast"], self.profiles["beta"])
        self.app.selectbox(key="next_fast_profile").select("alpha").run()
        self.assertIn("Beta English", self.transcript_html())
        self.select_evaluation()
        self.evaluate_file()
        self.assertIs(self.app.session_state["conversation_config"]["fast"], self.profiles["alpha"])
        first.close.assert_called_once()
        self.assertEqual(self.make_translation.call_count, 2)
        for absent in ("slow_lane", "diarization", "tencent_translation"):
            self.assertNotIn(absent, self.app.session_state)

    def test_upload_retains_order_offsets_and_does_not_repeat_on_rerun_or_selection(self):
        segments = self.segments(2)
        self.transcribe.side_effect = ["First line.", "Second line."]
        self.select_wav()
        self.transcribe_file()
        state = self.app.session_state["upload_state"]
        self.assertEqual(state["texts"], ["First line.", "Second line."])
        self.assertEqual([row["start_s"] for row in state["timings"]], [0.0, 0.05])
        self.assertTrue(all(row["asr_final_ms"] >= row["asr_start_ms"] for row in state["timings"]))
        self.assertTrue(state["finished"])
        self.assertNotIn("upload_job", self.app.session_state)
        self.assertEqual(self.app.session_state["translation"].sources, state["texts"])
        self.app.run()
        self.select_wav("replacement.wav")
        self.assertIs(self.app.session_state["upload_state"], state)
        self.assertEqual(self.transcribe.call_count, 2)
        segments.assert_called_once()
        self.make_asr.assert_called_once_with(self.asr_config)
        self.transcribe.close.assert_called_once()

    def test_upload_checkpoint_survives_rerun_without_repeating_remote_request(self):
        from src.ui import render_transcript
        self.segments(2)
        self.transcribe.side_effect = ["First line.", "Second line."]
        interrupted = False

        def render(texts, *args, **kwargs):
            nonlocal interrupted
            if texts == ["First line."] and not interrupted:
                interrupted = True
                st.rerun()
            return render_transcript(texts, *args, **kwargs)
        self.start_patch("src.ui.render_transcript", side_effect=render)
        self.select_wav()
        self.transcribe_file()
        self.assertTrue(interrupted)
        self.assertEqual(self.app.session_state["upload_state"]["texts"], ["First line.", "Second line."])
        self.assertEqual(self.app.session_state["translation"].sources, ["First line.", "Second line."])
        self.assertEqual(self.transcribe.call_count, 2)
        self.make_asr.assert_called_once()
        self.transcribe.close.assert_called_once()

    def test_upload_callback_can_publish_before_future_while_ui_polls_older_source(self):
        self.segments()
        self.make_translator.side_effect = lambda profile: Mock(return_value="Completed English.")
        self.make_translation.side_effect = RealTranslationSession
        submitted, release = Event(), Event()
        prefix_polls = []
        workers = []
        original_submit = RealTranslationSession.submit

        def submit(session, texts, **kwargs):
            result = original_submit(session, texts, **kwargs)
            if submitted.is_set() and texts == []:
                prefix_polls.append(kwargs.get("allow_prefix"))
                release.set()
            return result

        def task(transcribe, audio, *, on_result):
            future = Future()
            def run():
                text = transcribe(audio)
                on_result(text, 100.0, 101.0)
                submitted.set()
                release.wait(3)
                future.set_result((text, 100.0, 101.0))
            worker = Thread(target=run, daemon=True)
            workers.append(worker)
            worker.start()
            return future

        self.enterContext(patch.object(RealTranslationSession, "submit", new=submit))
        self.start_patch("src.uploads.transcribe_in_background", side_effect=task)
        try:
            self.select_wav()
            self.transcribe_file()
            self.assertTrue(prefix_polls)
            self.assertTrue(all(prefix_polls))
            state = self.app.session_state["upload_state"]
            self.assertIsNone(state["error"])
            self.assertEqual(state["texts"], ["A short transcript"])
            deadline = monotonic() + 2
            while self.app.session_state["translation"].snapshot()["pending"] and monotonic() < deadline:
                sleep(0.005)
            self.assertEqual(self.app.session_state["translation"].snapshot()["translations"], ["Completed English."])
            self.transcribe.assert_called_once()
        finally:
            release.set()
            for worker in workers:
                worker.join(timeout=3)

    def test_clear_during_active_asr_discards_late_results_without_waiting(self):
        entered, release = Event(), Event()
        def transcribe(audio):
            entered.set()
            release.wait(3)
            return "Late source"
        self.transcribe.side_effect = transcribe
        self.vad.side_effect = [{"start": 0}, None]
        recording = self.start_recording()
        translation = self.app.session_state["translation"]
        try:
            self.push_speech(recording)
            recording.finish()
            self.assertTrue(entered.wait(2))
            self.app.run()
            self.assertFalse(self.app.button(key="clear_conversation").disabled)
            self.app.button(key="clear_conversation").click().run()
            self.assert_no_exception()
            self.assertNotIn("recording", self.app.session_state)
            self.assertNotIn("translation", self.app.session_state)
            translation.close.assert_called_once()
        finally:
            release.set()
            recording._worker.join(timeout=3)
        self.assertEqual(translation.sources, [])
        self.assertEqual(recording.snapshot()["texts"], [])
        self.app.run()
        self.assertNotIn("Late source", self.transcript_html())

    def test_upload_error_is_redacted_preserves_partial_source_and_allows_explicit_retry(self):
        self.segments(2)
        self.transcribe.side_effect = ["First line.", RuntimeError("Bearer private-token request-body")]
        self.select_wav()
        self.transcribe_file()
        state = self.app.session_state["upload_state"]
        self.assertEqual(state["texts"], ["First line."])
        self.assertTrue(state["error"])
        self.assertNotIn("private-token", self.rendered_text())
        self.assertNotIn("upload_job", self.app.session_state)
        self.assertEqual(self.transcribe.call_count, 2)
        self.app.run()
        self.assertEqual(self.transcribe.call_count, 2)
        self.transcribe.side_effect = None
        self.transcribe_file()
        self.assertIsNone(self.app.session_state["upload_state"]["error"])
        self.assertEqual(self.transcribe.call_count, 4)

    def test_completed_and_failed_uploads_release_decoded_audio(self):
        from src.uploads import decode_wav
        references = []

        def decode(data):
            audio = decode_wav(data)
            references.append(weakref.ref(audio))
            return audio

        def split(audio, vad, *, with_timestamps):
            part = audio[:800]
            references.append(weakref.ref(part))
            return [{"audio": part, "start_s": 0, "end_s": 0.05}]
        self.start_patch("src.uploads.decode_wav", new=decode)
        self.start_patch("src.uploads.speech_segments", new=split)
        # No Mock call history retains the arrays under this memory assertion.
        def transcribe(audio):
            return "Hello"
        transcribe.close = Mock()
        self.make_asr.return_value = transcribe
        self.select_wav()
        self.transcribe_file()
        gc.collect()
        self.assertTrue(all(reference() is None for reference in references))
        def failing(audio):
            raise RuntimeError("unavailable")
        failing.close = Mock()
        self.make_asr.return_value = failing
        self.transcribe_file()
        gc.collect()
        self.assertTrue(all(reference() is None for reference in references))
        self.assertTrue(self.app.session_state["upload_state"]["error"])

    def test_silence_makes_no_remote_asr_or_translation_request(self):
        self.segments(0)
        self.select_evaluation()
        self.evaluate_file()
        self.assertEqual(self.evaluation_scores(), {"ASR · 1-wMER": "0.0%"})
        self.make_asr.assert_not_called()
        self.transcribe.assert_not_called()
        self.app.session_state["translation"].translate.assert_not_called()
        self.assertEqual(self.app.session_state["translation"].sources, [])

    def test_invalid_wav_or_reference_preserves_existing_conversation(self):
        self.segments()
        self.select_wav()
        self.transcribe_file()
        previous = self.app.session_state["upload_state"]
        self.select_wav(content=b"corrupt audio")
        self.transcribe_file()
        self.assertIs(self.app.session_state["upload_state"], previous)
        self.select_evaluation(b"\xff", reference_name="bad.txt")
        self.evaluate_file()
        self.assertIs(self.app.session_state["upload_state"], previous)
        self.assertEqual(self.transcribe.call_count, 1)

    def test_failed_translation_keeps_source_and_retries_only_translation(self):
        self.configure_translations(results=[None], errors=["Translation unavailable. Check the endpoint."])
        self.segments()
        self.select_wav()
        self.transcribe_file()
        translation = self.app.session_state["translation"]
        self.assertIn("A short transcript", self.transcript_html())
        self.assertIn("Translation unavailable", self.rendered_text())
        self.app.button(key="retry_translation").click().run()
        self.assert_no_exception()
        translation.retry_failed.assert_called_once()
        self.transcribe.assert_called_once()
        self.make_translator.assert_called_once()
        self.assertIn("English segment 1.", self.transcript_html())

    def test_clear_closes_translation_and_removes_old_output_and_errors(self):
        self.configure_translations(results=[None], errors=["Translation unavailable"])
        self.segments()
        self.select_wav()
        self.transcribe_file()
        translation = self.app.session_state["translation"]
        self.app.button(key="clear_conversation").click().run()
        self.assert_no_exception()
        translation.close.assert_called_once()
        self.assertNotIn("translation", self.app.session_state)
        self.assertNotIn("upload_state", self.app.session_state)
        self.assertNotIn("Translation unavailable", self.rendered_text())
        self.assertTrue(self.app.download_button(key="download_transcript").disabled)

    def test_exports_use_frozen_profile_and_safe_selected_results_only(self):
        downloads = self.capture_downloads()
        self.segments()
        self.app.selectbox(key="next_fast_profile").select("beta").run()
        self.select_wav()
        self.transcribe_file()
        self.app.selectbox(key="next_fast_profile").select("alpha").run()
        record = json.loads(downloads["download_session"])
        self.assertEqual(record["config"]["profile"], "beta")
        self.assertEqual(record["config"]["translation"]["model"], "beta-model")
        self.assertEqual(record["segments"][0]["translation"], "English segment 1.")
        self.assertIn("Beta English", downloads["download_transcript"])
        for obsolete in ("Astra", "Tencent", "OpenAI", "api_key", "base_url"):
            self.assertNotIn(obsolete, json.dumps(downloads))
        self.assertIn("English segment 1.", downloads["download_srt"])

    def test_reference_stays_local_and_only_source_transcript_is_scored(self):
        self.segments()
        self.transcribe.return_value = "你好 Zoom"
        self.select_evaluation(SOURCE_REFERENCE)
        self.evaluate_file()
        self.assertEqual(self.evaluation_scores(), {"ASR · Mixed match": "66.7%"})
        self.assertEqual(self.app.session_state["translation"].sources, ["你好 Zoom"])
        self.assertIsInstance(self.transcribe.call_args.args[0], np.ndarray)
        self.assertNotIn("Teams", str(self.make_translator.call_args))
        self.assertNotIn("Teams", str(self.make_asr.call_args))
        self.app.session_state["translation"].errors = ["Unavailable"]
        self.app.run()
        self.assertEqual(self.evaluation_scores(), {"ASR · Mixed match": "66.7%"})

    def test_partial_asr_failure_has_no_final_evaluation_score(self):
        self.segments(2)
        self.transcribe.side_effect = ["Partial source.", RuntimeError("private-body")]
        self.select_evaluation(b"Partial source.")
        self.evaluate_file()
        self.assertEqual(self.evaluation_scores(), {"ASR · 1-wMER": "—"})
        self.assertIn("no final score", self.rendered_text())
        self.assertIn("Partial source.", self.transcript_html())
        self.assertNotIn("private-body", self.rendered_text())

    def test_submitted_reference_and_score_are_frozen_until_next_evaluation(self):
        self.segments()
        self.transcribe.return_value = "One two."
        self.select_evaluation(b"One two.")
        self.evaluate_file()
        original = self.app.session_state["upload_state"]
        self.select_evaluation(b"Different reference.", reference_name="replacement.txt")
        self.assertIs(self.app.session_state["upload_state"], original)
        self.assertEqual(self.evaluation_scores(), {"ASR · 1-wMER": "100.0%"})
        self.transcribe.assert_called_once()
        self.evaluate_file()
        self.assertIsNot(self.app.session_state["upload_state"], original)
        self.assertEqual(self.app.session_state["upload_state"]["evaluation"]["reference_text"], "Different reference.")

    def test_table_reference_display_is_separate_and_mapping_is_frozen(self):
        self.segments()
        self.transcribe.return_value = "你好 Zoom"
        table = self.start_patch("src.reference_tables.read_reference_tables", return_value={
            "Transcript": [["text_zh_TW", "translation_en"], ["你好 Teams", "Hello Teams."]],
        })
        self.select_evaluation(b"workbook", reference_name="original.xlsx")
        self.assertEqual(self.app.selectbox(key="eval_text_column").value, 0)
        self.assertEqual(self.app.selectbox(key="eval_display_column").value, 1)
        self.evaluate_file()
        original = self.app.session_state["upload_state"]["evaluation"]
        self.assertEqual(original["reference"]["text"], "你好 Teams")
        self.assertEqual(original["reference_view"]["text"], "Hello Teams.")
        self.assertEqual(self.evaluation_scores(), {"ASR · Mixed match": "66.7%"})
        self.assertIn("Hello Teams.", self.transcript_html())
        self.assertEqual(self.app.session_state["translation"].sources, ["你好 Zoom"])
        table.return_value = {"Replacement": [["text_zh_TW", "translation_en"], ["替代", "Replacement"]]}
        self.select_evaluation(b"replacement", reference_name="new.xlsx")
        self.assertIs(self.app.session_state["upload_state"]["evaluation"], original)
        self.assertIn("Hello Teams.", self.transcript_html())
        self.transcribe.assert_called_once()

    def test_live_recording_disables_all_other_input_starts_and_profile_changes(self):
        recording = self.start_recording()
        self.assertTrue(self.app.file_uploader(key="wav_file").disabled)
        self.assertTrue(self.app.file_uploader(key="eval_wav_file").disabled)
        self.assertTrue(self.app.selectbox(key="next_fast_profile").disabled)
        self.finish_recording(recording)
        self.assertFalse(self.app.file_uploader(key="wav_file").disabled)
        self.assertFalse(self.app.selectbox(key="next_fast_profile").disabled)

    def test_explicit_ice_settings_reach_browser_and_remain_frozen(self):
        servers = [{"urls": ["turn:relay.example:3478"], "username": "browser", "credential": "private-turn-secret"}]
        with patch.dict(os.environ, {"WEBRTC_ICE_SERVERS_JSON": json.dumps(servers)}):
            recording = self.start_recording()
        self.assertEqual(self.webrtc.call_args.kwargs["rtc_configuration"], {"iceServers": servers})
        self.app.run()
        self.assertEqual(self.webrtc.call_args.kwargs["rtc_configuration"], {"iceServers": servers})
        self.assertNotIn("private-turn-secret", self.rendered_text())
        self.finish_recording(recording)

    def test_invalid_ice_configuration_blocks_microphone_safely_but_not_upload(self):
        for value in ('{"credential":"private-turn-secret"}', '[{"urls": "https://invalid", "credential":"private-turn-secret"}]',
                      '[{"urls": [], "credential": 7}]', '[{"urls":"stun"}]', '[{"urls":"turn:"}]',
                      'not-json-private-turn-secret'):
            with self.subTest(value=value), patch.dict(os.environ, {"WEBRTC_ICE_SERVERS_JSON": value}):
                self.app.run()
                self.assert_no_exception()
                self.assertTrue(self.app.button(key="start_recording").disabled)
                self.assertIn("WEBRTC_ICE_SERVERS_JSON", self.rendered_text())
                self.assertNotIn("private-turn-secret", self.rendered_text())
        self.make_asr.assert_not_called()
        self.load_vad.assert_not_called()


if __name__ == "__main__":
    unittest.main()
