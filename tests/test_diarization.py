"""Exercise real streaming feature windows with fake models and bounded workers."""

from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from transformers import Nemotron3DiarizationProcessor, NemotronAsrStreamingFeatureExtractor

from src import diarization
from src.diarization import DiarizationSession, NemotronDiarizer, SAMPLE_RATE, speaker_labels


class NativeStreamingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.processor = Nemotron3DiarizationProcessor(NemotronAsrStreamingFeatureExtractor())

    def fake_model(self):
        calls = []

        def run(**inputs):
            calls.append(inputs)
            frames = inputs["input_features"].shape[1] - inputs.get("num_lookahead_frames", 0) * 8
            logits = torch.full((1, frames, 8), -10.0)
            logits[:, :, 0 if len(calls) == 1 else 1] = 10.0
            return SimpleNamespace(logits=logits, speaker_cache=f"cache-{len(calls)}")

        return Mock(side_effect=run), calls

    def test_irregular_audio_preserves_native_overlap_tail_timing_and_speaker_cache(self):
        model, calls = self.fake_model()
        backend = NemotronDiarizer(self.processor, model)
        segments = []
        for size in (211, 8000, 502, 16500, 12800):
            segments.extend(backend(np.zeros(size, dtype=np.float32)))
        segments.extend(backend(np.empty(0, dtype=np.float32), final=True))

        self.assertEqual([item["input_features"].shape[1] for item in calls], [104, 104, 93])
        self.assertEqual([item["num_lookahead_frames"] for item in calls], [4, 4, 0])
        self.assertEqual([item["speaker_cache"] for item in calls], [None, "cache-1", "cache-2"])
        self.assertTrue(all(item["input_features"].device.type == "cpu" for item in calls))
        self.assertEqual([item["speaker_id"] for item in segments], ["Speaker 1", "Speaker 2", "Speaker 2"])
        self.assertAlmostEqual(segments[0]["start_s"], 0.0)
        self.assertAlmostEqual(segments[0]["end_s"], 0.72)
        self.assertAlmostEqual(segments[1]["start_s"], 0.72)
        self.assertAlmostEqual(segments[-1]["end_s"], 2.37)
        self.assertLessEqual(segments[-1]["end_s"], 38013 / SAMPLE_RATE)
        self.assertEqual(backend(np.zeros(800, dtype=np.float32), final=True), [])
        self.assertEqual(model.call_count, 3)

    def test_short_first_and_last_chunk_uses_streaming_mode_without_lookahead(self):
        model, calls = self.fake_model()
        backend = NemotronDiarizer(self.processor, model)
        self.assertEqual(backend(np.zeros(800, dtype=np.float32)), [])

        result = backend(np.empty(0, dtype=np.float32), final=True)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["num_lookahead_frames"], 0)
        self.assertEqual(calls[0]["input_features"].shape[1], 5)
        self.assertEqual(result, [{"start_s": 0.0, "end_s": 0.05, "speaker_id": "Speaker 1"}])

    def test_audio_shorter_than_a_model_frame_has_no_invented_speaker(self):
        model, _ = self.fake_model()
        backend = NemotronDiarizer(self.processor, model)

        self.assertEqual(backend(np.zeros(100, dtype=np.float32), final=True), [])
        model.assert_not_called()

    def test_local_model_cache_uses_cpu_once_and_streams_keep_separate_state(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            for filename in diarization.REQUIRED_FILES:
                (path / filename).touch()
            processor, model = Mock(), Mock()
            model.to.return_value = model
            model.eval.return_value = model
            with patch.object(diarization, "MODEL_DIR", path), patch.object(diarization, "_MODEL_CACHE", None), \
                    patch("transformers.AutoProcessor.from_pretrained", return_value=processor) as make_processor, \
                    patch("transformers.AutoModelForAudioFrameClassification.from_pretrained", return_value=model) as make_model:
                first, second = NemotronDiarizer(), NemotronDiarizer()

            make_processor.assert_called_once_with(path, local_files_only=True)
            make_model.assert_called_once_with(path, local_files_only=True, dtype=torch.float32)
            processor.set_streaming_mode.assert_called_once_with("low_latency")
            model.to.assert_called_once_with("cpu")
            self.assertIs(first._model, second._model)
            first._speaker_cache = "first-stream-state"
            self.assertIsNone(second._speaker_cache)


class DiarizationSessionTests(unittest.TestCase):
    def new_session(self, *args, **kwargs):
        session = DiarizationSession(*args, **kwargs)

        def cleanup():
            worker = session._worker
            session.close()
            if worker is not None:
                worker.join(timeout=3)
                self.assertFalse(worker.is_alive())

        self.addCleanup(cleanup)
        return session

    def wait_for(self, session, status):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            snapshot = session.snapshot()
            if snapshot["status"] == status:
                return snapshot
            time.sleep(0.005)
        self.fail(f"Diarization did not become {status}: {session.snapshot()}")

    def test_disabled_and_empty_finished_sessions_do_not_load_models(self):
        with patch("src.diarization.NemotronDiarizer") as make_backend:
            disabled = self.new_session(enabled=False)
            disabled.push(np.ones(800, dtype=np.float32))
            disabled.finish()
            empty = self.new_session()
            empty.finish()
            empty.finish()

        self.assertEqual(disabled.snapshot(), {"status": "disabled", "segments": [], "pending": 0, "error": None,
                                                     "processed_seconds": 0.0, "received_seconds": 0.0})
        self.assertEqual(empty.snapshot()["status"], "complete")
        make_backend.assert_not_called()

    def test_full_upload_is_fed_incrementally_and_adjacent_speaker_intervals_merge(self):
        calls, offset = [], 0

        def backend(audio, *, final=False):
            nonlocal offset
            calls.append((len(audio), final))
            if final:
                return []
            start = offset / SAMPLE_RATE
            offset += len(audio)
            return [{"start_s": start, "end_s": offset / SAMPLE_RATE, "speaker_id": "Speaker 1"}]

        session = self.new_session(backend)
        session.push(np.ones(20000, dtype=np.float32))
        session.finish()
        session.finish()
        state = self.wait_for(session, "complete")

        self.assertEqual(calls, [(16000, False), (4000, False), (0, True)])
        self.assertEqual(state["pending"], 0)
        self.assertEqual(state["segments"], [{"start_s": 0.0, "end_s": 1.25, "speaker_id": "Speaker 1"}])
        state["segments"][0]["speaker_id"] = "changed externally"
        self.assertEqual(session.labels([{"start_s": 0, "end_s": 1}, {"start_s": 2, "end_s": 3}]),
                         ["Speaker 1", None])

    def test_push_copies_capture_audio_without_waiting_for_inference(self):
        started, release = Event(), Event()
        seen = []

        def backend(audio, *, final=False):
            if not final:
                started.set()
                if not release.wait(timeout=2):
                    raise RuntimeError("test release was missing")
                seen.append(audio.copy())
            return []

        session = self.new_session(backend)
        self.addCleanup(release.set)
        audio = np.ones(800, dtype=np.float32)
        session.push(audio)
        self.assertTrue(started.wait(timeout=2))
        audio[:] = 9
        self.assertGreater(session.snapshot()["pending"], 0)
        session.finish()
        release.set()
        self.wait_for(session, "complete")
        np.testing.assert_array_equal(seen[0], np.ones(800, dtype=np.float32))

    def test_live_frames_reuse_one_waiting_worker_between_arrivals(self):
        processed = Event()

        def backend(audio, *, final=False):
            processed.set()
            return []

        session = self.new_session(backend)
        session.push(np.ones(512, dtype=np.float32))
        self.assertTrue(processed.wait(timeout=2))
        worker = session._worker
        processed.clear()
        session.push(np.ones(512, dtype=np.float32))
        self.assertTrue(processed.wait(timeout=2))
        self.assertIs(session._worker, worker)
        self.assertTrue(worker.is_alive())
        session.finish()
        self.wait_for(session, "complete")

    def test_audio_overflow_fails_explicitly_and_ignores_inflight_result(self):
        started, release = Event(), Event()

        def backend(audio, *, final=False):
            started.set()
            release.wait(timeout=2)
            return [{"start_s": 0, "end_s": 0.05, "speaker_id": "Speaker 1"}]

        session = self.new_session(backend, max_pending_seconds=0.1)
        self.addCleanup(release.set)
        session.push(np.ones(1000, dtype=np.float32))
        self.assertTrue(started.wait(timeout=2))
        session.push(np.ones(1000, dtype=np.float32))
        state = session.snapshot()
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["error"], diarization.OVERFLOW_MESSAGE)
        self.assertEqual(state["pending"], 0)
        worker = session._worker
        release.set()
        worker.join(timeout=2)
        self.assertEqual(session.snapshot()["segments"], [])

    def test_close_cancels_after_model_loading_without_calling_backend(self):
        loading, release = Event(), Event()
        backend = Mock(return_value=[])

        def load():
            loading.set()
            release.wait(timeout=2)
            return backend

        session = self.new_session()
        self.addCleanup(release.set)
        with patch("src.diarization.NemotronDiarizer", side_effect=load):
            session.push(np.ones(800, dtype=np.float32))
            self.assertTrue(loading.wait(timeout=2))
            worker = session._worker
            session.close()
            self.assertFalse(release.is_set())
            release.set()
            worker.join(timeout=2)
        backend.assert_not_called()
        self.assertEqual(session.snapshot(), {"status": "disabled", "segments": [], "pending": 0, "error": None,
                                                     "processed_seconds": 0.0, "received_seconds": 0.0})

    def test_invalid_audio_after_close_or_finish_does_not_replace_terminal_state(self):
        for action, expected in (("close", "disabled"), ("finish", "complete")):
            with self.subTest(action=action):
                validating, release = Event(), Event()
                backend = Mock(return_value=[])
                session = self.new_session(backend)
                invalid_audio = np.array([np.nan], dtype=np.float32)

                def delayed_conversion(*args, **kwargs):
                    validating.set()
                    release.wait(timeout=2)
                    return invalid_audio

                with patch("src.diarization.np.asarray", side_effect=delayed_conversion):
                    producer = Thread(target=session.push, args=([np.nan],))
                    producer.start()
                    try:
                        self.assertTrue(validating.wait(timeout=2))
                        getattr(session, action)()
                    finally:
                        release.set()
                        producer.join(timeout=2)

                self.assertFalse(producer.is_alive())
                self.assertEqual(session.snapshot(), {
                    "status": expected, "segments": [], "pending": 0, "error": None,
                    "processed_seconds": 0.0, "received_seconds": 0.0,
                })
                backend.assert_not_called()

    def test_model_exception_is_a_safe_failure_without_exception_text(self):
        session = self.new_session(Mock(side_effect=RuntimeError("private audio details / local path")))
        session.push(np.ones(800, dtype=np.float32))

        state = self.wait_for(session, "failed")

        self.assertEqual(state["error"], diarization.FAILED_MESSAGE)
        self.assertNotIn("private", state["error"])
        self.assertEqual(state["pending"], 0)

    def test_labels_keep_meaningful_mixed_speech_and_ignore_tiny_boundary_overlap(self):
        intervals = [
            {"start_s": 0, "end_s": 1, "speaker_id": "Speaker 1"},
            {"start_s": 0.6, "end_s": 2, "speaker_id": "Speaker 2"},
        ]
        timings = [{"start_s": 0, "end_s": 0.62}, {"start_s": 0, "end_s": 2},
                   {"start_s": 3, "end_s": 4}, None]

        self.assertEqual(speaker_labels(timings, intervals),
                         ["Speaker 1", "Speaker 2 / Speaker 1", None, None])


if __name__ == "__main__":
    unittest.main()
