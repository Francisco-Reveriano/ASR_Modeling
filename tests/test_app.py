"""Verify Streamlit recording state without models or microphone access."""

from io import BytesIO
import json
from pathlib import Path
import re
from threading import Event
from time import monotonic, sleep
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

    def __init__(self, translate=None, *, failure_message=None, glossary=None,
                 source_lang="zh-TW+en", target_lang="en", initial_results=None,
                 initial_errors=None):
        self.translate = translate
        self.failure_message = failure_message
        self.provider = "Tencent" if translate is not None else "English"
        self.initial_results = initial_results
        self.initial_errors = initial_errors or []
        self.sources = []
        self.translations = []
        self.errors = []
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
        self.start_patch("dotenv.load_dotenv", return_value=False)
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
        self.addCleanup(self.finish_current_recording)
        self.assertEqual(len(self.app.exception), 0)

    def start_patch(self, target, **kwargs):
        patcher = patch(target, **kwargs)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def configure_translations(self, *, openai=None, tencent=None, openai_errors=None, tencent_errors=None):
        """Publish chosen fake outcomes on first submission, just like a real worker."""
        self.translation_sessions = {}

        def create(translate=None, **kwargs):
            session = FakeTranslationSession(
                translate, **kwargs,
                initial_results=tencent if translate is not None else openai,
                initial_errors=tencent_errors if translate is not None else openai_errors,
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

    def finish_current_recording(self):
        if "recording" in self.app.session_state:
            session = self.app.session_state["recording"]
            session.finish()
            if session._worker.ident is not None:
                session._worker.join(timeout=5)
            self.assertFalse(session._worker.is_alive())
        if "slow_lane" in self.app.session_state:
            self.app.session_state["slow_lane"].close()

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
        mapping_keys = {"eval_reference_sheet", "eval_header_row", "eval_text_column", "eval_display_column"}
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
            "slow_window_n": 6, "slow_horizon": 10, "slow_seal_timeout": 25,
            "slow_request_timeout": 5, "slow_max_tokens": 8192,
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
        self.assertEqual((lane.config.window_n, lane.config.revision_horizon_s), (6, 10))
        self.assertEqual((lane.config.seal_timeout_s, lane.config.request_timeout_s), (25, 5))
        self.assertEqual((lane.config.max_output_tokens, lane.config.confidence_threshold), (8192, 0.7))
        for key in settings:
            self.assertTrue(self.app.number_input(key=key).disabled)
        self.assertTrue(self.app.toggle(key="enable_diarization").disabled)
        self.assertFalse(self.make_diarization.call_args.kwargs["enabled"])

        self.app.toggle(key="enable_corrections").set_value(False).run()
        self.assertEqual(lane.snapshot()["status"], "paused")
        self.app.button(key="stop_recording").click().run()
        self.app.number_input(key="slow_window_n").set_value(2).run()
        self.assertEqual(lane.config.window_n, 6)
        self.start_recording()
        replacement = self.app.session_state["slow_lane"]
        self.assertIsNot(replacement, lane)
        self.assertEqual(lane.snapshot()["status"], "closed")
        self.assertEqual(replacement.config.window_n, 2)
        self.assertFalse(replacement.config.enabled)
        self.assertNotEqual(replacement.session_id, lane.session_id)

    def test_fourth_lane_keeps_live_draft_when_late_correction_is_shown_in_final_view(self):
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
        self.app.number_input(key="slow_horizon").set_value(0).run()
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

        release.set()
        final = self.wait_for_corrections()
        self.app.run()
        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual(final["authoritative"], ["The reviewed English sentence."])
        self.assertEqual(final["translations"], ["English segment 1."])
        self.assertIn("Corrected · Final", self.transcript_html())
        self.assertIn("The reviewed English sentence.", self.transcript_html())
        self.assertFalse(self.app.download_button(key="download_final_srt").disabled)
        self.app.selectbox(key="subtitle_view").select("Live subtitles").run()
        self.assertIn("Draft · Final correction available", self.transcript_html())
        self.assertNotIn("The reviewed English sentence.", self.transcript_html())
        self.assertIn("A short transcript", self.transcript_html())
        self.assertIn("Tencent segment 1.", self.transcript_html())
        self.assertEqual(self.transcribe.call_count, 1)
        self.assertEqual(self.correct.call_count, 1)

    def test_completed_fast_draft_is_rendered_and_reviewed_while_next_asr_is_blocked(self):
        from src.ui import render_transcript

        second_started, release, rendered_while_blocked = Event(), Event(), Event()
        interrupted = False
        self.addCleanup(release.set)
        self.configure_translations(openai=[None, None])
        self.start_patch("src.uploads.speech_segments", return_value=[
            np.ones(800, dtype=np.float32), np.zeros(800, dtype=np.float32),
        ])

        def transcribe(audio):
            if self.transcribe.call_count == 1:
                return "First original."
            second_started.set()
            self.translation_sessions["openai"].translations[0] = "Fast English arrived."
            if not release.wait(5):
                raise RuntimeError("The UI did not publish fast output during blocked ASR.")
            return "Second original."

        def observe_render(texts, *args, **kwargs):
            nonlocal interrupted
            if second_started.is_set() and not interrupted:
                interrupted = True
                st.rerun()
            slow = kwargs.get("slow_lane", {})
            if (second_started.is_set() and not release.is_set() and texts == ["First original."]
                    and args[0][0] == "Fast English arrived."
                    and slow.get("translations") == ["Fast English arrived."] and self.correct.call_count):
                rendered_while_blocked.set()
                release.set()
            return render_transcript(texts, *args, **kwargs)

        self.transcribe.side_effect = transcribe
        self.start_patch("src.ui.render_transcript", side_effect=observe_render)
        self.select_wav()
        self.app.button(key="transcribe_file").click().run(timeout=8)

        self.assertEqual(len(self.app.exception), 0)
        self.assertTrue(interrupted)
        self.assertTrue(rendered_while_blocked.is_set())
        state = self.app.session_state["upload_state"]
        self.assertIsNone(state["error"])
        self.assertEqual(state["texts"], ["First original.", "Second original."])
        self.assertEqual(self.transcribe.call_count, 2)
        self.assertIn("Fast English arrived.", self.transcript_html())

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
        self.assertIn("Corrections unavailable", self.rendered_text())
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
        self.assertEqual(self.correct.call_count, 2)


if __name__ == "__main__":
    unittest.main()
