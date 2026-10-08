"""Exercise recording behavior without loading the speech models."""

from fractions import Fraction
import gc
from queue import Empty, Queue
from threading import Event
import unittest
from unittest.mock import Mock, patch
import weakref

import av
import numpy as np

from src.pipeline import FRAME_SIZE, SAMPLE_RATE, LiveTranscriber


def audio_frame(samples, *, sample_rate=SAMPLE_RATE, offset=0):
    samples = np.ascontiguousarray(samples, dtype=np.float32)
    if samples.ndim == 1:
        samples = samples[np.newaxis, :]
    layout = "mono" if samples.shape[0] == 1 else "stereo"
    frame = av.AudioFrame.from_ndarray(samples, format="fltp", layout=layout)
    frame.sample_rate = sample_rate
    frame.pts = offset
    frame.time_base = Fraction(1, sample_rate)
    return frame


class ScriptedVAD:
    def __init__(self, events=None):
        self.events = events or {}
        self.chunks = []

    def __call__(self, chunk):
        self.chunks.append(chunk.detach().cpu().numpy().copy())
        return self.events.get(len(self.chunks))


class LiveTranscriberTests(unittest.TestCase):
    def test_speaker_turns_are_transcribed_separately_without_losing_audio(self):
        speakers = Mock()
        speakers.snapshot.return_value = {"status": "complete", "segments": [
            {"start_s": 0, "end_s": 1, "speaker_id": "Speaker 1"},
            {"start_s": 1, "end_s": 2, "speaker_id": "Speaker 2"},
        ]}
        vad = ScriptedVAD({1: {"start": 0}})
        pipeline = self.make_pipeline(vad, diarization=speakers)
        audio = np.arange(32000, dtype=np.float32) / 32000
        pipeline.push(audio_frame(audio))
        self.finish(pipeline)
        state = pipeline.snapshot()
        self.assertEqual(state["texts"], ["segment 1", "segment 2"])
        self.assertEqual([t["speaker_id"] for t in state["timings"]], ["Speaker 1", "Speaker 2"])
        np.testing.assert_array_equal(np.concatenate(self.segments), audio)

    def setUp(self):
        self.segments = []

    def record(self, samples):
        self.segments.append(samples.copy())
        return f"segment {len(self.segments)}"

    def make_pipeline(self, vad, **kwargs):
        transcribe = kwargs.pop("transcribe", self.record)
        pipeline = LiveTranscriber(transcribe, vad, **kwargs)
        self.addCleanup(self.finish, pipeline)
        return pipeline

    def finish(self, pipeline):
        pipeline.finish()
        if pipeline._worker.ident is not None:
            pipeline._worker.join(timeout=5)
        self.assertFalse(pipeline._worker.is_alive(), "worker did not finish")
        self.assertTrue(pipeline.snapshot()["finished"])
        self.assertEqual(pipeline._buffer.size, 0)
        self.assertEqual(pipeline._tail.size, 0)

    def test_stop_without_audio_finishes_without_a_worker(self):
        pipeline = self.make_pipeline(ScriptedVAD())
        self.finish(pipeline)
        self.finish(pipeline)

        snapshot = pipeline.snapshot()
        self.assertFalse(snapshot["accepting"])
        self.assertEqual(snapshot["pending"], 0)
        self.assertEqual(snapshot["texts"], [])
        self.assertIsNone(snapshot["error"])
        self.assertIsNone(pipeline._worker.ident)

    def test_irregular_frames_preserve_samples_and_feed_complete_vad_chunks(self):
        vad = ScriptedVAD({1: {"start": 0}, 3: {"end": 1300}})
        pipeline = self.make_pipeline(vad)
        samples = np.arange(1800, dtype=np.float32) / 1800
        offset = 0
        for size in (123, 389, 31, 870, 387):
            frame = audio_frame(samples[offset:offset + size], offset=offset)
            self.assertIs(pipeline.push(frame), frame)
            offset += size

        self.assertEqual(len(vad.chunks), 3)
        self.assertTrue(all(chunk.shape == (FRAME_SIZE,) for chunk in vad.chunks))
        np.testing.assert_array_equal(np.concatenate(vad.chunks), samples[:1536])
        self.finish(pipeline)
        self.assertEqual(len(self.segments), 1)
        np.testing.assert_array_equal(self.segments[0], samples[:1300])

    def test_silence_does_not_submit_transcription(self):
        pipeline = self.make_pipeline(ScriptedVAD())
        pipeline.push(audio_frame(np.zeros(2 * FRAME_SIZE + 47)))
        self.finish(pipeline)

        self.assertEqual(self.segments, [])
        self.assertEqual(pipeline.snapshot()["texts"], [])
        self.assertEqual(pipeline.snapshot()["pending"], 0)
        self.assertIsNone(pipeline.snapshot()["error"])

    def test_vad_boundaries_already_include_padding(self):
        vad = ScriptedVAD({4: {"start": 256}, 7: {"end": 3000}})
        pipeline = self.make_pipeline(vad)
        samples = np.arange(4096, dtype=np.float32) / 4096
        pipeline.push(audio_frame(samples))
        self.finish(pipeline)

        self.assertEqual(len(self.segments), 1)
        np.testing.assert_array_equal(self.segments[0], samples[256:3000])

    def test_long_speech_splits_contiguously_without_losing_the_tail(self):
        pipeline = self.make_pipeline(
            ScriptedVAD({1: {"start": 0}}),
            max_segment_seconds=1024 / SAMPLE_RATE,
        )
        samples = np.arange(2248, dtype=np.float32) / 2248
        pipeline.push(audio_frame(samples))
        self.finish(pipeline)

        self.assertEqual([len(segment) for segment in self.segments], [1024, 1024, 200])
        np.testing.assert_array_equal(np.concatenate(self.segments), samples)

    def test_speech_end_in_split_frame_still_respects_maximum_duration(self):
        pipeline = self.make_pipeline(
            ScriptedVAD({1: {"start": 0}, 2: {"end": 1024}}),
            max_segment_seconds=1000 / SAMPLE_RATE,
        )
        samples = np.arange(1024, dtype=np.float32) / 1024
        pipeline.push(audio_frame(samples))
        self.finish(pipeline)

        self.assertEqual([len(segment) for segment in self.segments], [1000, 24])
        np.testing.assert_array_equal(np.concatenate(self.segments), samples)

    def test_stop_flushes_partial_speech_once_and_snapshots_are_independent(self):
        pipeline = self.make_pipeline(ScriptedVAD({1: {"start": 64}}))
        samples = np.arange(800, dtype=np.float32) / 800
        pipeline.push(audio_frame(samples))
        self.finish(pipeline)
        self.finish(pipeline)
        pipeline.push(audio_frame(np.ones(FRAME_SIZE), offset=len(samples)))

        self.assertEqual(len(self.segments), 1)
        np.testing.assert_array_equal(self.segments[0], samples[64:])
        snapshot = pipeline.snapshot()
        self.assertFalse(snapshot["accepting"])
        snapshot["texts"].append("external edit")
        self.assertEqual(pipeline.snapshot()["texts"], ["segment 1"])

    def test_stereo_48khz_input_is_resampled_and_resampler_tail_is_flushed(self):
        pipeline = self.make_pipeline(ScriptedVAD({1: {"start": 0}}))
        samples = np.full((2, 4800), 0.25, dtype=np.float32)
        offset = 0
        for size in (960, 1920, 1920):
            pipeline.push(audio_frame(
                samples[:, offset:offset + size], sample_rate=48000, offset=offset,
            ))
            offset += size
        self.finish(pipeline)

        self.assertEqual(len(self.segments), 1)
        self.assertEqual(self.segments[0].shape, (1600,))
        self.assertTrue(np.isfinite(self.segments[0]).all())
        self.assertGreater(float(self.segments[0].mean()), 0.1)

    def test_diarization_receives_continuous_audio_including_silence_with_asr_offsets(self):
        diarization = Mock()
        pipeline = self.make_pipeline(ScriptedVAD({2: {"start": 512}, 4: {"end": 1800}}), diarization=diarization)
        samples = np.arange(2400, dtype=np.float32) / 2400
        for start, end in ((0, 300), (300, 1400), (1400, 2400)):
            pipeline.push(audio_frame(samples[start:end], offset=start))
        self.finish(pipeline)
        captured = np.concatenate([call.args[0] for call in diarization.push.call_args_list])
        np.testing.assert_array_equal(captured, samples)
        diarization.finish.assert_called()
        timing = pipeline.snapshot()["timings"][0]
        self.assertEqual((timing["start_s"], timing["end_s"]), (512 / SAMPLE_RATE, 1800 / SAMPLE_RATE))
        self.assertGreaterEqual(timing["asr_final_ms"], timing["asr_start_ms"])
        self.assertIsNone(timing["t_capture_ms"])
        np.testing.assert_array_equal(self.segments[0], samples[512:1800])

    def test_completed_speech_is_released_while_capture_continues(self):
        started, release, collected = Event(), Event(), Event()
        references = []

        def transcribe(samples):
            references.append(weakref.ref(samples, lambda _: collected.set()))
            started.set()
            self.assertTrue(release.wait(timeout=5))
            return self.record(samples)

        pipeline = self.make_pipeline(
            ScriptedVAD({1: {"start": 0}, 2: {"end": 700}}),
            transcribe=transcribe,
        )
        self.addCleanup(release.set)
        samples = np.arange(1024, dtype=np.float32) / 1024
        pipeline.push(audio_frame(samples))
        self.assertTrue(started.wait(timeout=5))
        self.assertIsNotNone(references[0]())
        self.assertEqual(pipeline.snapshot()["pending"], 1)

        release.set()
        self.assertTrue(collected.wait(timeout=5), "completed speech is still retained")
        snapshot = pipeline.snapshot()
        self.assertTrue(snapshot["accepting"])
        self.assertEqual(snapshot["texts"], ["segment 1"])
        self.assertEqual(snapshot["pending"], 0)
        self.assertEqual(snapshot["timings"][0]["end_s"], 700 / SAMPLE_RATE)
        np.testing.assert_array_equal(self.segments[0], samples[:700])

    def test_stop_drains_audio_enqueued_immediately_after_worker_timeout(self):
        timed_out, release = Event(), Event()

        class PausedTimeoutQueue(Queue):
            def __init__(self):
                super().__init__()
                self.first_get = True

            def get(self, *args, **kwargs):
                if self.first_get:
                    self.first_get = False
                    timed_out.set()
                    release.wait(timeout=5)
                    raise Empty
                return super().get(*args, **kwargs)

        # Reproduce a queue timeout followed by a final enqueue before the
        # worker checks whether capture ended. Stop must still drain the tail.
        with patch("src.pipeline.Queue", return_value=PausedTimeoutQueue()):
            pipeline = self.make_pipeline(ScriptedVAD({1: {"start": 0}}))
        self.addCleanup(release.set)
        samples = np.ones(800, dtype=np.float32)
        pipeline.push(audio_frame(samples))
        self.assertTrue(timed_out.wait(timeout=5))
        pipeline.finish()
        self.assertEqual(pipeline.snapshot()["pending"], 1)
        release.set()
        self.finish(pipeline)

        self.assertEqual(pipeline.snapshot()["pending"], 0)
        self.assertEqual(len(self.segments), 1)
        np.testing.assert_array_equal(self.segments[0], samples)

    def test_segments_are_transcribed_in_order_while_recording_continues(self):
        started, release = Event(), Event()

        def transcribe(samples):
            if not self.segments:
                started.set()
                self.assertTrue(release.wait(timeout=5))
            return self.record(samples)

        vad = ScriptedVAD({
            1: {"start": 0}, 2: {"end": 700},
            3: {"start": 1100}, 4: {"end": 1700},
        })
        pipeline = self.make_pipeline(vad, transcribe=transcribe)
        self.addCleanup(release.set)
        samples = np.arange(2048, dtype=np.float32) / 2048
        pipeline.push(audio_frame(samples[:1024]))
        self.assertTrue(started.wait(timeout=5))
        pipeline.push(audio_frame(samples[1024:], offset=1024))
        self.assertEqual(pipeline.snapshot()["pending"], 2)
        self.assertEqual(pipeline.snapshot()["texts"], [])
        release.set()
        self.finish(pipeline)

        self.assertEqual(pipeline.snapshot()["texts"], ["segment 1", "segment 2"])
        np.testing.assert_array_equal(self.segments[0], samples[:700])
        np.testing.assert_array_equal(self.segments[1], samples[1100:1700])

    def test_queue_overflow_stops_capture_and_drains_accepted_segments(self):
        started, release = Event(), Event()

        def transcribe(samples):
            started.set()
            self.assertTrue(release.wait(timeout=5))
            return self.record(samples)

        vad = ScriptedVAD({
            1: {"start": 0}, 2: {"end": 600},
            3: {"start": 1024}, 4: {"end": 1600},
            5: {"start": 2048}, 6: {"end": 2600},
        })
        pipeline = self.make_pipeline(vad, transcribe=transcribe, max_pending=1)
        self.addCleanup(release.set)
        samples = np.arange(4096, dtype=np.float32) / 4096
        pipeline.push(audio_frame(samples[:1024]))
        self.assertTrue(started.wait(timeout=5))
        pipeline.push(audio_frame(samples[1024:2048], offset=1024))
        self.assertEqual(pipeline.snapshot()["pending"], 2)
        pipeline.push(audio_frame(samples[2048:3072], offset=2048))

        snapshot = pipeline.snapshot()
        self.assertFalse(snapshot["accepting"])
        self.assertTrue(snapshot["error"])
        pipeline.push(audio_frame(samples[3072:], offset=3072))
        release.set()
        self.finish(pipeline)

        self.assertEqual(pipeline.snapshot()["pending"], 0)
        self.assertEqual(len(self.segments), 2)
        np.testing.assert_array_equal(self.segments[0], samples[:600])
        np.testing.assert_array_equal(self.segments[1], samples[1024:1600])

    def test_inference_failure_is_visible_and_worker_stops(self):
        references = []

        def transcribe(samples):
            references.append(weakref.ref(samples))
            raise RuntimeError("decode broke")

        pipeline = self.make_pipeline(
            ScriptedVAD({1: {"start": 0}, 2: {"end": 700}}),
            transcribe=transcribe,
        )
        with self.assertLogs("src.pipeline", level="ERROR"):
            pipeline.push(audio_frame(np.ones(1024)))
            self.finish(pipeline)

        snapshot = pipeline.snapshot()
        self.assertIn("Transcription failed", snapshot["error"])
        self.assertNotIn("decode broke", snapshot["error"])
        self.assertFalse(snapshot["accepting"])
        self.assertEqual(snapshot["texts"], [])
        self.assertEqual(snapshot["pending"], 0)
        gc.collect()
        self.assertIsNone(references[0](), "failed speech is still retained")

    def test_vad_failure_is_visible_and_worker_stops(self):
        def vad(chunk):
            raise RuntimeError("VAD broke")

        pipeline = self.make_pipeline(vad)
        with self.assertLogs("src.pipeline", level="ERROR"):
            pipeline.push(audio_frame(np.ones(FRAME_SIZE)))
            self.finish(pipeline)

        snapshot = pipeline.snapshot()
        self.assertIn("Audio processing failed", snapshot["error"])
        self.assertNotIn("VAD broke", snapshot["error"])
        self.assertFalse(snapshot["accepting"])
        self.assertEqual(snapshot["pending"], 0)

    def test_close_discards_queued_and_in_flight_transcription(self):
        entered, release = Event(), Event()
        calls = []

        def transcribe(samples):
            calls.append(len(samples))
            entered.set()
            release.wait(3)
            return "must not appear"

        pipeline = self.make_pipeline(ScriptedVAD({1: {"start": 0}}),
                                      transcribe=transcribe, max_segment_seconds=0.032)
        try:
            pipeline.push(audio_frame(np.ones(FRAME_SIZE * 3)))
            self.assertTrue(entered.wait(1))
            pipeline.close()
            pipeline.close()
            self.assertFalse(pipeline.snapshot()["accepting"])
            self.assertEqual(pipeline.snapshot()["pending"], 0)
        finally:
            release.set()
        self.finish(pipeline)
        self.assertEqual(len(calls), 1)
        self.assertEqual(pipeline.snapshot()["texts"], [])


if __name__ == "__main__":
    unittest.main()
