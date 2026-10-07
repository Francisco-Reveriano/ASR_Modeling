"""Verify padding removal never cuts audible PCM or collapses internal pauses."""

import unittest

from src.speech_audio import FRAME_BYTES, trim_speech_padding


class SpeechAudioTests(unittest.TestCase):
    def test_quiet_speech_and_partial_frames_are_preserved_exactly(self):
        # Just above the conservative silence threshold, including negative PCM.
        pcm = (b"\x41\x00\xbf\xff" * 500) + b"\xff\x7f"
        chunks = (pcm[i:i + 97] for i in range(0, len(pcm), 97))
        self.assertEqual(b"".join(trim_speech_padding(chunks)), pcm)

    def test_long_internal_silence_is_preserved_with_bounded_lookbehind(self):
        voiced = b"\x00\x10" * (FRAME_BYTES // 2)
        quiet = bytes(FRAME_BYTES)
        consumed = []
        def source():
            for frame in [voiced] + [quiet] * 300 + [voiced]:
                consumed.append(frame)
                yield frame
        output = trim_speech_padding(source())
        self.assertEqual(next(output), voiced)
        first_quiet = next(output)
        self.assertLessEqual(len(consumed), 202)
        self.assertEqual(voiced + first_quiet + b"".join(output), voiced + quiet * 300 + voiced)

    def test_streaming_does_not_wait_for_the_end_of_a_request(self):
        voiced = b"\x00\x10" * (FRAME_BYTES // 2)
        def source():
            yield voiced
            self.fail("First audio should be delivered before requesting the next frame")
        output = trim_speech_padding(source())
        self.assertEqual(next(output), voiced)
        output.close()

    def test_incomplete_samples_fail_instead_of_creating_a_click(self):
        with self.assertRaisesRegex(ValueError, "Incomplete PCM sample"):
            b"".join(trim_speech_padding([b"\x00"]))


if __name__ == "__main__":
    unittest.main()
