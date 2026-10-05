"""Verify Streamlit recording state without models or microphone access."""

from io import BytesIO
from pathlib import Path
import re
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import wave

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

    def __init__(self, translate=None, *, failure_message=None):
        self.translate = translate
        self.failure_message = failure_message
        self.provider = "Tencent" if translate is not None else "English"
        self.sources = []
        self.translations = []
        self.errors = []
        self.submit = Mock(side_effect=self._submit)
        self.retry_failed = Mock(side_effect=self._retry_failed)
        self.close = Mock()

    def _submit(self, texts):
        for text in texts[len(self.sources):]:
            self.sources.append(text)
            self.translations.append(f"{self.provider} segment {len(self.sources)}.")
            self.errors.append(None)

    def _retry_failed(self):
        self.errors = [None for _ in self.errors]

    def snapshot(self):
        return {
            "translations": self.translations.copy(),
            "errors": self.errors.copy(),
            "pending": sum(
                text is None and error is None
                for text, error in zip(self.translations, self.errors)
            ),
        }


class RecordingAppTests(unittest.TestCase):
    def setUp(self):
        st.cache_resource.clear()
        self.addCleanup(st.cache_resource.clear)
        self.transcribe = Mock(return_value="A short transcript")
        self.vad = Mock(return_value=None)
        self.context = SimpleNamespace(
            state=SimpleNamespace(playing=False, signalling=False),
        )
        self.load_transcriber = self.start_patch(
            "src.pipeline.load_transcriber", return_value=self.transcribe,
        )
        self.load_vad = self.start_patch("src.pipeline.load_vad", return_value=self.vad)
        self.webrtc = self.start_patch(
            "streamlit_webrtc.webrtc_streamer", return_value=self.context,
        )
        self.make_translation = self.start_patch(
            "src.translation.TranslationSession",
            side_effect=FakeTranslationSession,
        )
        self.app = AppTest.from_file(str(APP_FILE)).run()
        self.addCleanup(self.finish_current_recording)
        self.assertEqual(len(self.app.exception), 0)

    def start_patch(self, target, **kwargs):
        patcher = patch(target, **kwargs)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def finish_current_recording(self):
        if "recording" in self.app.session_state:
            session = self.app.session_state["recording"]
            session.finish()
            if session._worker.ident is not None:
                session._worker.join(timeout=5)
            self.assertFalse(session._worker.is_alive())

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
        self.start_patch(
            "src.uploads.speech_segments",
            return_value=[np.ones(800, dtype=np.float32), np.zeros(800, dtype=np.float32)],
        )
        self.transcribe.side_effect = ["你好。", "食飽未？"]
        self.select_wav()
        self.transcribe_file()
        translation = self.app.session_state["translation"]
        tencent = self.app.session_state["tencent_translation"]
        translation.translations = ["Hello.", "Have you eaten?"]
        tencent.translations = ["Hi there.", "Did you eat?"]
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

    def test_pending_and_failed_translations_keep_originals_and_allow_retry(self):
        self.start_patch(
            "src.uploads.speech_segments",
            return_value=[np.ones(800, dtype=np.float32), np.zeros(800, dtype=np.float32)],
        )
        self.transcribe.side_effect = ["First original.", "Second original."]
        self.select_wav()
        self.transcribe_file()
        translation = self.app.session_state["translation"]
        tencent = self.app.session_state["tencent_translation"]
        translation.translations = [None, None]
        translation.errors = [None, "Translation service unavailable"]
        tencent.translations = ["First Tencent result.", "Second Tencent result."]
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

    def test_tencent_failure_and_retry_do_not_block_or_repeat_openai_translation(self):
        self.start_patch(
            "src.uploads.speech_segments", return_value=[np.ones(800, dtype=np.float32)],
        )
        self.select_wav()
        self.transcribe_file()
        translation = self.app.session_state["translation"]
        tencent = self.app.session_state["tencent_translation"]
        tencent.translations = [None]
        tencent.errors = ["Tencent translation unavailable"]
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
        self.start_patch("src.uploads.speech_segments", return_value=[])
        self.select_wav()
        self.transcribe_file()

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
        translation.translations = [None, None]
        tencent.translations = ["English segment 1.", "English segment 2."]
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

        translation.translations = [None]
        translation.errors = ["Translation unavailable"]
        tencent.translations = [None]
        self.app.run()
        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual(self.evaluation_scores(), expected)
        self.assertIn("你好 Zoom", self.transcript_html())

    def test_caption_reference_scores_breeze_and_export_keeps_unscored_translations(self):
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


if __name__ == "__main__":
    unittest.main()
