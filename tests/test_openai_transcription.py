"""Exercise audio conversion and Realtime protocol without network requests."""

import base64
import json
import os
import traceback
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src.openai_transcription import (
    FAILED_MESSAGE, MISSING_KEY_MESSAGE, REALTIME_URL, TRANSCRIPTION_MODEL,
    TranscriptionError, load_openai_transcriber,
)


class OpenAITranscriptionTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {"OPENAI_API_KEY": "test-secret"}))
        self.env = self.enterContext(patch("src.openai_transcription.load_dotenv"))
        self.connect = self.enterContext(patch("websockets.sync.client.connect"))
        self.connection = MagicMock()
        self.connect.return_value.__enter__.return_value = self.connection
        self.events([
            {"type": "session.created"}, {"type": "session.updated"},
            {"type": "input_audio_buffer.committed", "item_id": "utterance-1"},
            {"type": "conversation.item.input_audio_transcription.delta", "delta": "Partial"},
            {"type": "conversation.item.input_audio_transcription.completed",
             "item_id": "other-item", "transcript": "Wrong text"},
            {"type": "conversation.item.input_audio_transcription.completed",
             "item_id": "utterance-1", "transcript": "  你好 world  "},
        ])

    def events(self, events):
        self.connection.recv.side_effect = [json.dumps(event) for event in events]

    def sent(self):
        return [json.loads(call.args[0]) for call in self.connection.send.call_args_list]

    def test_commits_pcm_audio_and_returns_only_matching_final_transcript(self):
        transcribe = load_openai_transcriber()
        self.connect.assert_not_called()
        result = transcribe(np.ones(32000, dtype=np.float32) * 0.25)
        self.assertEqual(result, "你好 world")
        self.assertEqual(self.connect.call_args.args, (REALTIME_URL,))
        self.assertEqual(self.connect.call_args.kwargs["additional_headers"],
                         {"Authorization": "Bearer test-secret"})
        sent = self.sent()
        session = sent[0]["session"]
        self.assertEqual(session["type"], "transcription")
        config = session["audio"]["input"]
        self.assertEqual(config["transcription"]["model"], TRANSCRIPTION_MODEL)
        self.assertEqual(config["format"], {"type": "audio/pcm", "rate": 24000})
        self.assertIsNone(config["turn_detection"])
        self.assertEqual(sent[-1], {"type": "input_audio_buffer.commit"})
        chunks = [base64.b64decode(event["audio"]) for event in sent[1:-1]]
        self.assertTrue(all(len(chunk) <= 48000 for chunk in chunks))
        pcm = np.frombuffer(b"".join(chunks), dtype="<i2")
        self.assertEqual(len(pcm), 48000)
        np.testing.assert_allclose(pcm[100:-100], 8192, atol=10)
        self.assertTrue(all(call.kwargs["timeout"] > 0 for call in self.connection.recv.call_args_list))
        self.connect.return_value.__exit__.assert_called_once()

    def test_short_final_tail_is_padded_and_extreme_samples_are_clipped(self):
        load_openai_transcriber()(np.ones(800, dtype=np.float32) * 2)
        pcm = np.frombuffer(base64.b64decode(self.sent()[1]["audio"]), dtype="<i2")
        self.assertEqual(len(pcm), 2400)
        self.assertTrue((pcm[:1100] == 32767).all())
        self.assertTrue((pcm[1200:] == 0).all())

    def test_empty_audio_does_not_open_connection(self):
        self.assertEqual(load_openai_transcriber()(np.empty(0)), "")
        self.connect.assert_not_called()

    def test_missing_key_fails_before_connection(self):
        os.environ["OPENAI_API_KEY"] = " "
        with self.assertRaisesRegex(TranscriptionError, MISSING_KEY_MESSAGE):
            load_openai_transcriber()
        self.connect.assert_not_called()

    def test_api_failure_and_timeout_close_connection_without_exposing_payloads(self):
        for failure in (TimeoutError("test-secret"), RuntimeError("test-secret")):
            with self.subTest(failure=type(failure)):
                self.connection.recv.side_effect = failure
                try:
                    load_openai_transcriber()(np.zeros(1600))
                except TranscriptionError as exc:
                    self.assertEqual(str(exc), FAILED_MESSAGE)
                    self.assertNotIn("test-secret", "".join(traceback.format_exception(exc)))
                else:
                    self.fail("Expected a safe transcription failure")
        self.assertEqual(self.connect.return_value.__exit__.call_count, 2)

    def test_server_error_or_failed_transcription_is_not_returned_as_text(self):
        for kind in ("error", "conversation.item.input_audio_transcription.failed"):
            with self.subTest(kind=kind):
                self.events([{"type": kind, "error": {"message": "test-secret"}}])
                with self.assertRaisesRegex(TranscriptionError, FAILED_MESSAGE):
                    load_openai_transcriber()(np.zeros(1600))

    def test_session_events_cannot_extend_overall_deadline(self):
        self.events([{"type": "session.created"}] * 5)
        with patch("src.openai_transcription.monotonic", side_effect=[0, 10, 46]):
            with self.assertRaisesRegex(TranscriptionError, FAILED_MESSAGE):
                load_openai_transcriber()(np.zeros(1600))
        self.assertEqual(self.connection.recv.call_count, 1)

    def test_invalid_or_oversized_audio_never_reaches_api(self):
        for audio in (np.array([np.nan]), np.zeros((2, 100)), np.zeros(240001)):
            with self.subTest(shape=audio.shape):
                with self.assertRaises(TranscriptionError):
                    load_openai_transcriber()(audio)
        self.connect.assert_not_called()
