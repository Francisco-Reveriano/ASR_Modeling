"""Check uploaded audio preparation without loading either speech model."""

from io import BytesIO
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf

from src.uploads import MAX_SEGMENT_SECONDS, SAMPLE_RATE, decode_wav, speech_segments


def audio_bytes(samples, sample_rate=SAMPLE_RATE, *, format="WAV", subtype="FLOAT"):
    buffer = BytesIO()
    sf.write(buffer, samples, sample_rate, format=format, subtype=subtype)
    return buffer.getvalue()


class DecodeWavTests(unittest.TestCase):
    def test_mono_float_wav_preserves_samples_and_output_contract(self):
        samples = np.linspace(-0.75, 0.75, 1001, dtype=np.float32)

        audio = decode_wav(audio_bytes(samples))

        np.testing.assert_array_equal(audio, samples)
        self.assertEqual(audio.dtype, np.float32)
        self.assertEqual(audio.ndim, 1)
        self.assertTrue(audio.flags.c_contiguous)

    def test_stereo_pcm_wav_is_scaled_and_averaged(self):
        samples = np.column_stack((np.full(1600, 0.25), np.full(1600, 0.75)))

        audio = decode_wav(audio_bytes(samples, subtype="PCM_16"))

        np.testing.assert_array_equal(audio, np.full(1600, 0.5, dtype=np.float32))

    def test_resampling_preserves_duration_and_tone(self):
        expected = 0.25 * np.sin(2 * np.pi * 1000 * np.arange(SAMPLE_RATE) / SAMPLE_RATE)
        for sample_rate in (8000, 22050, 44100, 48000):
            with self.subTest(sample_rate=sample_rate):
                samples = 0.25 * np.sin(2 * np.pi * 1000 * np.arange(sample_rate) / sample_rate)

                audio = decode_wav(audio_bytes(samples, sample_rate))

                self.assertEqual(audio.shape, (SAMPLE_RATE,))
                self.assertEqual(audio.dtype, np.float32)
                self.assertTrue(audio.flags.c_contiguous)
                np.testing.assert_allclose(audio[100:-100], expected[100:-100], atol=0.001)

    def test_wav_variants_are_supported(self):
        for format in ("WAVEX", "RF64"):
            with self.subTest(format=format):
                audio = decode_wav(audio_bytes(np.full(500, 0.25), format=format))
                np.testing.assert_array_equal(audio, np.full(500, 0.25, dtype=np.float32))

    def test_silence_is_valid_audio(self):
        audio = decode_wav(audio_bytes(np.zeros(3200), subtype="PCM_16"))

        np.testing.assert_array_equal(audio, np.zeros(3200, dtype=np.float32))

    def test_corrupt_or_missing_file_data_has_a_clear_error(self):
        for data in (b"", b"this is not audio", b"RIFF\x00\x00\x00\x00WAVE"):
            with self.subTest(data=data):
                with self.assertRaisesRegex(ValueError, "Could not read this WAV"):
                    decode_wav(data)

    def test_empty_wav_has_a_clear_error(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            decode_wav(audio_bytes(np.empty(0, dtype=np.float32)))

    def test_non_wav_audio_is_rejected_even_if_decodable(self):
        data = audio_bytes(np.zeros(100), format="FLAC", subtype="PCM_16")

        with self.assertRaisesRegex(ValueError, "upload a WAV"):
            decode_wav(data)

    def test_nonfinite_float_audio_is_rejected(self):
        for invalid in (np.nan, np.inf, -np.inf):
            with self.subTest(invalid=invalid):
                data = audio_bytes(np.array([0.0, invalid, 0.0], dtype=np.float32))
                with self.assertRaisesRegex(ValueError, "nonfinite"):
                    decode_wav(data)


class SpeechSegmentsTests(unittest.TestCase):
    def split(self, audio, timestamps):
        detector = Mock(return_value=timestamps)
        vad = SimpleNamespace(model=object())
        with patch.dict("sys.modules", {"silero_vad": SimpleNamespace(get_speech_timestamps=detector)}):
            segments = speech_segments(audio, vad)
        return segments, detector, vad

    def test_detected_boundaries_preserve_order_padding_and_shared_memory(self):
        audio = np.arange(3200, dtype=np.float32)

        segments, detector, vad = self.split(audio, [
            {"start": 100, "end": 700}, {"start": 1200, "end": 3000},
        ])

        self.assertEqual(len(segments), 2)
        np.testing.assert_array_equal(segments[0], audio[100:700])
        np.testing.assert_array_equal(segments[1], audio[1200:3000])
        self.assertTrue(all(np.shares_memory(segment, audio) for segment in segments))
        arguments, settings = detector.call_args
        np.testing.assert_array_equal(arguments[0].numpy(), audio)
        self.assertIs(arguments[1], vad.model)
        self.assertEqual(settings, {
            "sampling_rate": SAMPLE_RATE, "threshold": 0.5,
            "min_speech_duration_ms": 250, "max_speech_duration_s": 15,
            "min_silence_duration_ms": 500, "speech_pad_ms": 150,
        })

    def test_silence_produces_no_segments(self):
        audio = decode_wav(audio_bytes(np.zeros(1600)))

        segments, _, _ = self.split(audio, [])

        self.assertEqual(segments, [])

    def test_timestamped_segments_keep_original_file_offsets_for_speaker_alignment(self):
        audio = np.arange(2 * SAMPLE_RATE, dtype=np.float32)
        detector = Mock(return_value=[{"start": 4000, "end": 12000}, {"start": 16000, "end": 30000}])
        with patch.dict("sys.modules", {"silero_vad": SimpleNamespace(get_speech_timestamps=detector)}):
            segments = speech_segments(audio, SimpleNamespace(model=object()), with_timestamps=True)
        self.assertEqual([(row["start_s"], row["end_s"]) for row in segments], [(0.25, 0.75), (1.0, 1.875)])
        np.testing.assert_array_equal(segments[1]["audio"], audio[16000:30000])

    def test_oversized_span_splits_without_losing_or_repeating_samples(self):
        limit = MAX_SEGMENT_SECONDS * SAMPLE_RATE
        audio = np.arange(2 * limit + 273, dtype=np.float32)

        segments, _, _ = self.split(audio, [{"start": 100, "end": len(audio)}])

        self.assertEqual([len(segment) for segment in segments], [limit, limit, 173])
        np.testing.assert_array_equal(np.concatenate(segments), audio[100:])
        self.assertTrue(all(np.shares_memory(segment, audio) for segment in segments))


if __name__ == "__main__":
    unittest.main()
