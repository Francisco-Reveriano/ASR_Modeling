"""Verify continuous English translation and shutdown without real API calls."""

import asyncio
import base64
import json
import os
from threading import Event
from time import monotonic, sleep
import unittest
from unittest.mock import patch

import av
import numpy as np

from src.realtime_translation import (
    EMPTY_TRANSLATION_MESSAGE, FAILED_MESSAGE, MISSING_KEY_MESSAGE,
    TRANSLATION_URL, RealtimeTranslationSession,
)


class FakeSocket:
    def __init__(self):
        self.sent = []
        self.audio_received = Event()
        self.closed = Event()
        self.translate = True
        self.fail = False
        self.send_final = True

    async def __aenter__(self):
        self.events = asyncio.Queue()
        return self

    async def __aexit__(self, *args):
        self.closed.set()

    async def send(self, payload):
        event = json.loads(payload)
        self.sent.append(event)
        if self.fail:
            raise RuntimeError("private-api-key")
        if event["type"] == "session.update":
            self.events.put_nowait({"type": "session.created"})
            self.events.put_nowait({"type": "session.updated"})
        elif event["type"] == "session.input_audio_buffer.append":
            if not self.audio_received.is_set():
                self.events.put_nowait({"type": "session.input_transcript.delta", "delta": "你好"})
                if self.translate:
                    # Equal timing values are not duplicate text identifiers.
                    for delta in ("Hel", "lo"):
                        self.events.put_nowait({"type": "session.output_transcript.delta", "delta": delta, "elapsed_ms": 200})
                self.events.put_nowait({"type": "session.output_audio.delta", "delta": "ignored"})
                self.audio_received.set()
        elif event["type"] == "session.close" and self.send_final:
            if self.translate:
                self.events.put_nowait({"type": "session.output_transcript.delta", "delta": " world."})
            self.events.put_nowait({"type": "session.closed"})

    async def recv(self):
        return json.dumps(await self.events.get())


class RealtimeTranslationTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {"OPENAI_API_KEY": "private-api-key"}))
        self.enterContext(patch("src.realtime_translation.load_dotenv"))
        self.socket = FakeSocket()
        self.connect = self.enterContext(patch("websockets.asyncio.client.connect", return_value=self.socket))

    def session(self, **kwargs):
        session = RealtimeTranslationSession(**kwargs)
        self.addCleanup(self.cleanup, session)
        return session

    def cleanup(self, session):
        session.close()
        if session._worker.ident is not None:
            session._worker.join(3)
            self.assertFalse(session._worker.is_alive())

    def frame(self, samples=1600):
        frame = av.AudioFrame.from_ndarray(np.zeros((1, samples), dtype=np.float32), format="fltp", layout="mono")
        frame.sample_rate = 16000
        return frame

    def wait_until(self, predicate):
        deadline = monotonic() + 3
        while monotonic() < deadline:
            if predicate():
                return
            sleep(0.005)
        self.fail("Streaming worker did not reach expected state")

    def test_microphone_publishes_english_before_stop_and_drains_final_words(self):
        session = self.session()
        self.connect.assert_not_called()
        session.push(self.frame())
        self.wait_until(lambda: session.snapshot()["direct_translation"]["translations"] == ["Hello"])
        self.assertTrue(session.snapshot()["accepting"])
        self.assertFalse(session.snapshot()["finished"])
        self.assertEqual(session.snapshot()["texts"], ["你好"])
        session.finish()
        session.finish()
        self.wait_until(lambda: session.snapshot()["finished"])
        state = session.snapshot()
        self.assertIsNone(state["error"])
        self.assertEqual(state["direct_translation"]["translations"], ["Hello world."])
        self.assertEqual(state["direct_translation"]["pending"], 0)
        self.assertEqual(self.connect.call_args.args, (TRANSLATION_URL,))
        self.assertEqual(self.socket.sent[0], {
            "type": "session.update", "session": {"audio": {
                "input": {"transcription": {"model": "gpt-realtime-whisper"}},
                "output": {"language": "en"},
            }},
        })
        self.assertEqual(sum(event["type"] == "session.close" for event in self.socket.sent), 1)
        self.assertTrue(self.socket.closed.is_set())
        self.assertEqual(session._queued_bytes, 0)
        self.assertIsNone(session._resampler)
        self.assertEqual(session._api_key, "")

    def test_upload_preserves_silence_and_sends_every_sample_at_24khz(self):
        session = self.session()
        session.start_file(np.zeros(3200, dtype=np.float32))
        self.wait_until(lambda: session.snapshot()["finished"])
        pcm = b"".join(base64.b64decode(event["audio"]) for event in self.socket.sent
                       if event["type"] == "session.input_audio_buffer.append")
        self.assertEqual(len(pcm), 4800 * 2)
        self.assertTrue((np.frombuffer(pcm, dtype="<i2") == 0).all())
        self.assertAlmostEqual(session.snapshot()["sent_seconds"], 0.2)
        self.assertIsNone(session._file_audio)

    def test_empty_recording_opens_no_connection(self):
        session = self.session()
        session.finish()
        self.assertTrue(session.snapshot()["finished"])
        self.connect.assert_not_called()

    def test_cancel_suppresses_late_output_and_releases_file_audio(self):
        session = self.session()
        session.start_file(np.zeros(160000, dtype=np.float32))
        self.wait_until(lambda: session.snapshot()["direct_translation"]["translations"] == ["Hello"])
        session.close()
        before = session.snapshot()["direct_translation"]["translations"]
        self.wait_until(lambda: session.snapshot()["finished"])
        self.assertEqual(session.snapshot()["direct_translation"]["translations"], before)
        self.assertIsNone(session._file_audio)
        self.assertTrue(self.socket.closed.is_set())

    def test_connection_failure_is_safe_and_preserves_no_fake_english(self):
        self.socket.fail = True
        session = self.session()
        session.push(self.frame())
        self.wait_until(lambda: session.snapshot()["finished"])
        self.assertEqual(session.snapshot()["error"], FAILED_MESSAGE)
        self.assertNotIn("private-api-key", str(session.snapshot()))

    def test_missing_key_fails_before_start(self):
        os.environ["OPENAI_API_KEY"] = " "
        with self.assertRaisesRegex(ValueError, MISSING_KEY_MESSAGE):
            self.session()
        self.connect.assert_not_called()

    def test_missing_translation_keeps_source_separate_and_marks_failure(self):
        self.socket.translate = False
        session = self.session()
        session.start_file(np.zeros(1600, dtype=np.float32))
        self.wait_until(lambda: session.snapshot()["finished"])
        self.assertEqual(session.snapshot()["error"], EMPTY_TRANSLATION_MESSAGE)
        self.assertEqual(session.snapshot()["direct_translation"]["translations"], [None])
        self.assertEqual(session.snapshot()["texts"], ["你好"])

    def test_close_deadline_does_not_hang_on_missing_session_closed(self):
        self.socket.send_final = False
        self.enterContext(patch("src.realtime_translation.DRAIN_TIMEOUT", 0.01))
        session = self.session()
        session.start_file(np.zeros(1600, dtype=np.float32))
        self.wait_until(lambda: session.snapshot()["finished"])
        self.assertEqual(session.snapshot()["error"], FAILED_MESSAGE)
        self.assertEqual(session.snapshot()["direct_translation"]["translations"], ["Hello"])

    def test_overflow_stops_capture_with_visible_error(self):
        session = self.session(max_pending_seconds=0.001)
        session.push(self.frame())
        self.wait_until(lambda: session.snapshot()["finished"])
        self.assertIn("fell behind", session.snapshot()["error"])
        self.assertFalse(session.snapshot()["accepting"])

    def test_source_script_never_appears_in_english_output(self):
        session = self.session()
        session._source.append("中文")
        session._english.append("English 中文")
        self.assertEqual(session.snapshot()["direct_translation"]["translations"], [None])
        self.assertEqual(session.snapshot()["error"], "English translation unavailable.")
