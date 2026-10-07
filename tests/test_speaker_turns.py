"""Speaker boundaries, progress and worker cancellation without model weights."""

from threading import Event, Thread
import unittest
from unittest.mock import Mock

import numpy as np

from src.diarization import DiarizationSession, prepare_speaker_turns, split_speaker_turns
from src.uploads import prepare_speaker_turns_in_background


class SpeakerTurnTests(unittest.TestCase):
    def setUp(self):
        self.audio = np.arange(6 * 16000, dtype=np.float32)
        self.timing = {"start_s": 10, "end_s": 16, "server_endpoint_ms": 123}

    def span(self, start, end, speaker):
        return {"start_s": start, "end_s": end, "speaker_id": f"Speaker {speaker}"}

    def split(self, spans):
        parts = split_speaker_turns(self.audio, self.timing, spans)
        np.testing.assert_array_equal(np.concatenate([p["audio"] for p in parts]), self.audio)
        self.assertTrue(all(np.shares_memory(p["audio"], self.audio) for p in parts))
        self.assertEqual(parts[0]["start_s"], 10)
        self.assertEqual(parts[-1]["end_s"], 16)
        for left, right in zip(parts, parts[1:]):
            self.assertEqual(left["end_s"], right["start_s"])
        self.assertTrue(all(p["server_endpoint_ms"] == 123 for p in parts))
        return parts

    def test_three_sequential_speakers_split_in_time_order_with_intact_audio(self):
        parts = self.split([self.span(10, 11.9, 2), self.span(12.1, 14, 3), self.span(14, 16, 1)])
        self.assertEqual([p["speaker_id"] for p in parts], ["Speaker 2", "Speaker 3", "Speaker 1"])
        self.assertEqual([p["start_s"] for p in parts], [10, 12, 14])

    def test_brief_flicker_does_not_create_tiny_asr_requests(self):
        parts = self.split([self.span(10, 12, 1), self.span(12, 12.1, 2), self.span(12.1, 16, 1)])
        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0]["speaker_id"], "Speaker 1")

    def test_substantial_overlap_retains_combined_label(self):
        parts = self.split([self.span(10, 14, 1), self.span(12, 16, 2)])
        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0]["speaker_id"], "Speaker 1 / Speaker 2")

    def test_same_speaker_after_silence_is_not_split(self):
        self.assertEqual(len(self.split([self.span(10, 11, 1), self.span(14, 16, 1)])), 1)

    def test_no_speaker_or_timing_keeps_audio_unchanged(self):
        self.assertEqual(len(self.split([])), 1)
        self.assertIs(split_speaker_turns(self.audio, None, [])[0]["audio"], self.audio)

    def test_timeout_keeps_unsplit_audio_instead_of_using_partial_labels(self):
        diarization = Mock()
        diarization.snapshot.return_value = {"status": "active", "processed_seconds": 12,
                                             "segments": [self.span(10, 12, 1)]}
        diarization.wait_for_audio.return_value = False
        parts = prepare_speaker_turns(self.audio, self.timing, diarization, timeout=0.02)
        self.assertEqual(len(parts), 1)
        self.assertNotIn("speaker_id", parts[0])
        diarization.wait_for_audio.assert_called_once_with(16, timeout=0.02)

    def test_upload_wait_runs_off_caller_thread_and_uses_completed_timing(self):
        entered, release = Event(), Event()
        self.addCleanup(release.set)
        diarization = Mock()
        pending = {"status": "active", "processed_seconds": 10, "segments": []}
        complete = {"status": "complete", "processed_seconds": 16,
                    "segments": [self.span(10, 13, 1), self.span(13, 16, 2)]}
        diarization.snapshot.side_effect = [pending, complete]

        def wait(*args, **kwargs):
            entered.set()
            return release.wait(2)

        diarization.wait_for_audio.side_effect = wait
        future = prepare_speaker_turns_in_background(dict(self.timing, audio=self.audio), diarization)
        self.assertTrue(entered.wait(1))
        self.assertFalse(future.done())
        release.set()
        self.assertEqual([p["speaker_id"] for p in future.result(2)], ["Speaker 1", "Speaker 2"])

    def test_silent_audio_advances_progress_and_close_releases_waiter(self):
        processed, release = Event(), Event()
        self.addCleanup(release.set)

        def backend(audio, **kwargs):
            processed.set()
            release.wait(2)
            return []

        session = DiarizationSession(backend)
        self.addCleanup(session.close)
        session.push(np.zeros(16000, dtype=np.float32))
        self.assertTrue(processed.wait(1))
        self.assertEqual(session.snapshot()["received_seconds"], 1)
        self.assertFalse(session.wait_for_audio(1, timeout=0.01))
        release.set()
        self.assertTrue(session.wait_for_audio(1, timeout=2))
        self.assertEqual(session.snapshot()["segments"], [])
        result = []
        waiter = Thread(target=lambda: result.append(session.wait_for_audio(2, timeout=20)))
        waiter.start()
        worker = session._worker
        session.close()
        waiter.join(1)
        self.assertFalse(waiter.is_alive())
        self.assertEqual(result, [False])
        worker.join(2)
