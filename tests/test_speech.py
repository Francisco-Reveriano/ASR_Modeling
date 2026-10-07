"""Exercise streaming, ordered English input, cancellation and bounded playback."""

import base64
from threading import Event
from time import monotonic, sleep
import unittest
from unittest.mock import MagicMock, patch

from src.speech import (
    FAILED_MESSAGE, PCM_CHUNK_BYTES, PREFETCH_BYTES, SPEAKER_VOICES,
    SpeechSession, astra_speech_result, stream_speech, voice_for_speaker,
)


class SpeechTests(unittest.TestCase):
    def setUp(self):
        self.spoken = []

    def synthesize(self, text, *, model, voice="coral"):
        self.spoken.append((text, model))
        yield b"\x01\x00" * 4800

    def session(self, **kwargs):
        session = SpeechSession(synthesize=kwargs.pop("synthesize", self.synthesize), **kwargs)
        self.addCleanup(self.cleanup, session)
        return session

    def cleanup(self, session):
        session.close()
        if session._worker:
            session._worker.join(3)
            self.assertFalse(session._worker.is_alive())
        for worker in tuple(session._producers):
            worker.join(3)
            self.assertFalse(worker.is_alive())

    def wait_for(self, predicate):
        deadline = monotonic() + 3
        while monotonic() < deadline:
            if predicate():
                return
            sleep(0.005)
        self.fail("Speech worker did not reach expected state")

    def result(self, *translations, errors=None, filtered=None):
        return {"translations": list(translations), "errors": errors or [], "filtered": filtered or []}

    def test_audio_arrives_before_request_finishes_and_before_translation_finishes(self):
        release = Event()
        self.addCleanup(release.set)
        def streaming(text, *, model, voice="coral"):
            yield b"\x01\x00" * 4800
            release.wait(2)
            yield b"\x02\x00" * 4800
        session = self.session(synthesize=streaming)
        session.submit(self.result("Hello. More words"), realtime=True)
        self.wait_for(lambda: len(session.snapshot()["chunks"]) == 1)
        state = session.snapshot()
        self.assertFalse(state["complete"])
        self.assertEqual(state["pending"], 1)
        self.assertEqual(base64.b64decode(state["chunks"][0]["pcm"]), b"\x01\x00" * 4800)
        release.set()

    def test_prefetches_next_voice_before_first_finishes_but_publishes_in_order(self):
        first_entered, second_entered, release = Event(), Event(), Event()
        self.addCleanup(release.set)
        voices = {}
        def synthesize(text, *, model, voice):
            voices[text] = voice
            if text == "First.":
                first_entered.set()
                release.wait(2)
                yield b"\x01\x00" * 4800
            else:
                second_entered.set()
                yield b"\x02\x00" * 4800
        session = self.session(synthesize=synthesize)
        session.submit(self.result("First.", "Second."), speakers=["Speaker 1", "Speaker 2"], final=True)
        self.assertTrue(first_entered.wait(1))
        self.assertTrue(second_entered.wait(1))
        self.assertEqual(session.snapshot()["chunks"], [])
        release.set()
        self.wait_for(lambda: session.snapshot()["pending"] == 0)
        chunks = session.snapshot()["chunks"]
        self.assertEqual([base64.b64decode(chunk["pcm"])[:2] for chunk in chunks], [b"\x01\x00", b"\x02\x00"])
        self.assertEqual(voices, {"First.": "coral", "Second.": "onyx"})

    def test_prefetch_memory_and_concurrency_stay_bounded_when_playback_waits(self):
        started = []
        def synthesize(text, *, model, voice):
            started.append(text)
            for _ in range(100):
                yield b"\x01\x00" * 4800
        session = self.session(synthesize=synthesize, max_audio_seconds=0.2)
        session.submit(self.result("First.", "Second.", "Third."), final=True)
        self.wait_for(lambda: len(started) == 2 and all(job["bytes"] == PREFETCH_BYTES for job in session._jobs))
        self.assertLessEqual(session._audio_bytes, PCM_CHUNK_BYTES)
        self.assertEqual(len(session._producers), 2)
        self.assertEqual(len(started), 2)
        session.close()
        self.wait_for(lambda: not session._producers)

    def test_speaker_voices_are_stable_and_mixed_rows_use_dominant_channel(self):
        self.assertEqual(len({voice_for_speaker(f"Speaker {i}") for i in range(1, 9)}), 8)
        self.assertEqual(voice_for_speaker("Speaker 3 / Speaker 2 / Speaker 1"), "nova")
        for unknown in (None, "", "Unknown", "Speaker 99", "<script>"):
            self.assertEqual(voice_for_speaker(unknown), "coral")
        calls = []
        def synthesize(text, *, model, voice):
            calls.append((text, voice))
            yield b"\x00\x00" * 4800
        session = self.session(synthesize=synthesize)
        session.submit(self.result("Second.", "Third.", "Second again."),
                       speakers=["Speaker 2", "Speaker 3", "Speaker 2"], final=True)
        self.wait_for(lambda: session.snapshot()["pending"] == 0)
        self.assertEqual(dict(calls), {"Second.": "onyx", "Third.": "nova", "Second again.": "onyx"})
        self.assertEqual(session.snapshot()["voices"], {"Speaker 2": "onyx", "Speaker 3": "nova"})
        session.submit(self.result("Revised."), speakers=["Speaker 4"], final=True)
        self.assertEqual(len(calls), 3)

    def test_single_voice_option_keeps_default_and_lead_in_does_not_delay_generation(self):
        calls = []
        def synthesize(text, *, model, voice):
            calls.append(voice)
            yield b"\x00\x00" * 4800
        with patch("src.speech.monotonic", return_value=100):
            session = self.session(synthesize=synthesize, speaker_voices=False)
            self.assertIsNone(session.snapshot()["playback_delay_ms"])
        with patch("src.speech.monotonic", return_value=200):
            session.submit(self.result("A question?"), speakers=["Speaker 2"], final=True)
        self.wait_for(lambda: session.snapshot()["pending"] == 0)
        self.assertEqual(calls, ["coral"])
        with patch("src.speech.monotonic", return_value=210):
            state = session.snapshot()
        self.assertEqual(state["playback_delay_ms"], 50000)
        self.assertAlmostEqual(state["buffered_seconds"], 0.2)
        self.assertEqual(state["minimum_buffer_seconds"], 2)
        self.assertTrue(state["generation_complete"])
        self.assertFalse(state["complete"])  # Playback has not acknowledged it yet.
        with patch("src.speech.monotonic", return_value=260):
            self.assertEqual(session.snapshot()["playback_delay_ms"], 0)

    def test_lead_in_waits_for_accepted_text_and_later_rows_do_not_restart_it(self):
        now = [100]
        with patch("src.speech.monotonic", side_effect=lambda: now[0]):
            session = self.session()
            session.submit(self.result(None, "[Filtered]", filtered=[False, True]))
            now[0] = 300
            self.assertIsNone(session.snapshot()["playback_delay_ms"])
            session.submit(self.result("First accepted translation.", "[Filtered]", filtered=[False, True]))
            self.assertEqual(session.snapshot()["playback_delay_ms"], 60000)
            now[0] = 320
            session.submit(self.result("First accepted translation.", "[Filtered]", "Another turn.",
                                       filtered=[False, True, False]), final=True)
            self.wait_for(lambda: session.snapshot()["pending"] == 0)
            self.assertEqual(session.snapshot()["playback_delay_ms"], 40000)

    def test_empty_or_filtered_conversation_finishes_without_starting_countdown(self):
        session = self.session()
        session.submit(self.result("[Filtered]", "   ", filtered=[True, False]), final=True)
        self.assertIsNone(session.snapshot()["playback_delay_ms"])
        self.assertTrue(session.snapshot()["complete"])

    def test_segment_order_deduplication_and_no_replaying_revisions(self):
        session = self.session()
        session.submit(self.result(None, "Second."))
        self.assertEqual(self.spoken, [])
        session.submit(self.result("First.", "Second."))
        self.wait_for(lambda: len(self.spoken) == 2)
        session.submit(self.result("First revised.", "Second."), final=True)
        self.wait_for(lambda: session.snapshot()["pending"] == 0)
        self.assertEqual([text for text, _ in self.spoken], ["First.", "Second."])

    def test_realtime_deltas_join_without_duplication_and_final_tail_flushes(self):
        session = self.session()
        for text in ("Hel", "Hello", "Hello there. Last", "Hello there. Last words"):
            session.submit(self.result(text), realtime=True)
        session.submit(self.result("Hello there. Last words"), realtime=True, final=True)
        self.wait_for(lambda: len(self.spoken) == 2)
        self.assertEqual([text for text, _ in self.spoken], ["Hello there.", "Last words"])
        self.assertTrue(all(model == "gpt-4o-mini-tts" for _, model in self.spoken))

    def test_decimal_split_across_deltas_is_not_spoken_as_separate_sentences(self):
        session = self.session()
        session.submit(self.result("The value is 98."), realtime=True)
        self.assertEqual(self.spoken, [])
        session.submit(self.result("The value is 98.5 percent."), realtime=True, final=True)
        self.wait_for(lambda: len(self.spoken) == 1)
        self.assertEqual(self.spoken[0][0], "The value is 98.5 percent.")

    def test_unpunctuated_stream_flushes_whole_words_after_bounded_delay(self):
        session = self.session()
        text = "We are testing a long English phrase with no punctuation yet"
        session.submit(self.result(text), realtime=True)
        session._tail_since = monotonic() - 2
        session.submit(self.result(text), realtime=True)
        self.wait_for(lambda: len(self.spoken) == 1)
        session.submit(self.result(text), realtime=True, final=True)
        self.wait_for(lambda: len(self.spoken) == 2)
        self.assertEqual(" ".join(text for text, _ in self.spoken), text)

    def test_failed_filtered_and_source_script_text_is_never_spoken(self):
        session = self.session()
        session.submit(self.result(None, "你好", "[Background speech filtered]", "English.",
                                   errors=["failed", None, None, None],
                                   filtered=[False, False, True, False]), final=True)
        self.wait_for(lambda: len(self.spoken) == 1)
        self.assertEqual(self.spoken[0][0], "English.")
        self.assertEqual(session.snapshot()["skipped"], 2)

    def test_long_translation_stays_within_phrase_limit_without_losing_words(self):
        text = " ".join(["English"] * 700)
        session = self.session()
        session.submit(self.result(text), final=True)
        self.wait_for(lambda: session.snapshot()["pending"] == 0)
        phrases = [text for text, _ in self.spoken]
        self.assertEqual(" ".join(phrases), text)
        self.assertTrue(all(len(text) <= 300 for text in phrases))

    def test_short_sentences_in_one_speaker_turn_use_one_speech_request(self):
        text = "Can everyone hear me? It sounds a little laggy. Ah Ming, have you joined?"
        session = self.session()
        session.submit(self.result(text), speakers=["Speaker 1"], final=True)
        self.wait_for(lambda: session.snapshot()["pending"] == 0)
        self.assertEqual(self.spoken, [(text, "gpt-4o-mini-tts")])

    def test_acknowledgements_release_audio_and_ignore_other_sessions(self):
        session = self.session()
        session.submit(self.result("Hello."), final=True)
        self.wait_for(lambda: len(session.snapshot()["chunks"]) == 1)
        session.acknowledge("old-session", 100)
        self.assertEqual(len(session.snapshot()["chunks"]), 1)
        session.acknowledge(session.session_id, 1)
        self.wait_for(lambda: session.snapshot()["complete"])
        self.assertEqual(session._audio_bytes, 0)

    def test_backpressure_bounds_audio_and_close_unblocks_worker(self):
        def many_chunks(text, *, model, voice="coral"):
            for _ in range(10):
                yield b"\x01\x00" * 4800
        session = self.session(synthesize=many_chunks, max_audio_seconds=0.2)
        session.submit(self.result("Hello."), final=True)
        self.wait_for(lambda: len(session.snapshot()["chunks"]) == 1)
        self.assertEqual(session._audio_bytes, PCM_CHUNK_BYTES)
        session.acknowledge(session.session_id, 1)
        self.wait_for(lambda: any(chunk["id"] == 2 for chunk in session.snapshot()["chunks"]))

        session.close()
        session._worker.join(2)
        self.assertFalse(session._worker.is_alive())
        self.assertEqual(session.snapshot()["chunks"], [])

    def test_close_suppresses_late_audio_and_closes_response_generator(self):
        entered, release, closed = Event(), Event(), Event()
        self.addCleanup(release.set)
        def delayed(text, *, model, voice="coral"):
            try:
                entered.set()
                release.wait(2)
                yield b"\x00\x00" * 4800
            finally:
                closed.set()
        session = self.session(synthesize=delayed)
        session.submit(self.result("Hello."))
        self.assertTrue(entered.wait(1))
        session.close()
        release.set()
        self.assertTrue(closed.wait(1))
        self.assertEqual(session.snapshot()["chunks"], [])

    def test_failure_is_safe_and_does_not_retry_or_leak_credential(self):
        def failure(text, *, model, voice="coral"):
            raise RuntimeError("secret-key-and-private-text")
            yield b""
        session = self.session(synthesize=failure)
        session.submit(self.result("Hello."))
        self.wait_for(lambda: session.snapshot()["closed"])
        self.assertEqual(session.snapshot()["error"], FAILED_MESSAGE)
        self.assertNotIn("secret", str(session.snapshot()))

    def test_prefetched_stream_cleanup_failure_is_reported_without_private_details(self):
        class FailingClose:
            def __iter__(self):
                yield b"\x00\x00" * 4800
            def close(self):
                raise RuntimeError("private-api-key")
        session = self.session(synthesize=lambda *args, **kwargs: FailingClose())
        session.submit(self.result("Hello."), final=True)
        self.wait_for(lambda: session.snapshot()["closed"])
        self.assertEqual(session.snapshot()["error"], FAILED_MESSAGE)
        self.assertNotIn("private", str(session.snapshot()))

    def test_changed_realtime_prefix_stops_voice_instead_of_repeating(self):
        session = self.session()
        session.submit(self.result("First sentence. Next"), realtime=True)
        session.submit(self.result("Replacement sentence."), realtime=True)
        self.assertTrue(session.snapshot()["closed"])

    def test_text_overflow_and_empty_pcm_are_visible_failures(self):
        session = self.session(max_text_chars=2)
        session.submit(self.result("Hello."))
        self.assertIn("fell behind", session.snapshot()["error"])
        empty = self.session(synthesize=lambda *args, **kwargs: iter(()))
        empty.submit(self.result("Hello."))
        self.wait_for(lambda: empty.snapshot()["closed"])
        self.assertEqual(empty.snapshot()["error"], FAILED_MESSAGE)

    def test_short_sentence_plays_during_pause_without_waiting_for_session_end(self):
        session = self.session()
        session.submit(self.result("Hello."), realtime=True)
        session._last_delta = monotonic() - 1
        session.submit(self.result("Hello."), realtime=True)
        self.wait_for(lambda: len(self.spoken) == 1)
        self.assertFalse(session.snapshot()["complete"])

    def test_sdk_streaming_configuration_uses_requested_model_and_pcm(self):
        client = MagicMock()
        client.__enter__.return_value = client
        response = client.audio.speech.with_streaming_response.create.return_value.__enter__.return_value
        response.iter_bytes.return_value = [b"\x00\x00"]
        with patch("src.speech.load_dotenv"), patch.dict("os.environ", {"OPENAI_API_KEY": "private-key"}), \
             patch("openai.OpenAI", return_value=client) as factory:
            self.assertEqual(list(stream_speech("Hello.", model="gpt-4o-mini-tts", voice="onyx")), [b"\x00\x00"])
        self.assertEqual(factory.call_args.kwargs["max_retries"], 0)
        request = client.audio.speech.with_streaming_response.create.call_args.kwargs
        self.assertEqual(request["model"], "gpt-4o-mini-tts")
        self.assertEqual(request["response_format"], "pcm")
        self.assertEqual(request["input"], "Hello.")
        self.assertEqual(request["voice"], "onyx")
        response.iter_bytes.assert_called_once_with(chunk_size=PCM_CHUNK_BYTES)

    def test_hd_streaming_omits_unsupported_instructions(self):
        client = MagicMock()
        client.__enter__.return_value = client
        response = client.audio.speech.with_streaming_response.create.return_value.__enter__.return_value
        response.iter_bytes.return_value = [b"\x00\x00"]
        with patch("src.speech.load_dotenv"), patch.dict("os.environ", {"OPENAI_API_KEY": "private-key"}), \
             patch("openai.OpenAI", return_value=client):
            self.assertEqual(list(stream_speech("A question?", model="tts-1-hd", voice="onyx")), [b"\x00\x00"])
        request = client.audio.speech.with_streaming_response.create.call_args.kwargs
        self.assertEqual(request["model"], "tts-1-hd")
        self.assertEqual(request["voice"], "onyx")
        self.assertEqual(request["response_format"], "pcm")
        self.assertNotIn("instructions", request)

    def test_sdk_stream_shortens_edge_padding_but_preserves_speech_and_internal_pauses(self):
        silence = lambda ms: b"\x00\x00" * (24 * ms)
        speech = b"\x00\x10" * 4800
        original = silence(400) + speech + silence(300) + speech + silence(1300)
        client = MagicMock()
        client.__enter__.return_value = client
        response = client.audio.speech.with_streaming_response.create.return_value.__enter__.return_value
        # Network chunks need not align with samples, frames or silence edges.
        response.iter_bytes.return_value = (original[i:i + 997] for i in range(0, len(original), 997))
        with patch("src.speech.load_dotenv"), patch.dict("os.environ", {"OPENAI_API_KEY": "private-key"}), \
             patch("openai.OpenAI", return_value=client):
            pcm = b"".join(stream_speech("Two short sentences.", model="gpt-4o-mini-tts"))
        self.assertEqual(pcm, silence(80) + speech + silence(300) + speech + silence(120))

    def test_astra_drafts_and_sealed_paused_fallbacks_never_reach_speech(self):
        state = {"translations": ["Fast draft.", "Paused draft."],
                 "authoritative": [None, "Paused draft."], "statuses": ["draft", "paused"],
                 "segments": [{}, {"sealed": True}], "pending": 0}
        result = astra_speech_result(state, segment_count=3)
        self.assertEqual(result["translations"], [None, None, None])
        self.assertEqual(result["pending"], 3)

    def test_astra_speech_uses_accepted_text_even_if_screen_pointer_is_older(self):
        state = {"translations": ["Old screen draft.", "Another draft."],
                 "authoritative": ["Astra corrected.", "Astra confirmed."],
                 "statuses": ["corrected", "confirmed"], "segments": [{}, {}]}
        result = astra_speech_result(state, segment_count=2)
        session = self.session()
        session.submit(result, final=result["pending"] == 0)
        self.wait_for(lambda: len(self.spoken) == 2)
        self.assertEqual([text for text, _ in self.spoken], ["Astra corrected.", "Astra confirmed."])

    def test_conversation_review_waits_for_final_accepted_wording(self):
        state = {"authoritative": ["First review."], "statuses": ["corrected"],
                 "segments": [{"conversation_review_status": "reviewing"}]}
        session = self.session()
        result = astra_speech_result(state, segment_count=1, conversation_review=True)
        session.submit(result, final=result["pending"] == 0)
        self.assertEqual(result["pending"], 1)
        self.assertIsNone(session._worker)
        state["authoritative"][0] = "Reviewed in context."
        state["segments"][0]["conversation_review_status"] = "confirmed"
        result = astra_speech_result(state, segment_count=1, conversation_review=True)
        session.submit(result, final=result["pending"] == 0)
        self.wait_for(lambda: len(self.spoken) == 1)
        self.assertEqual(self.spoken[0][0], "Reviewed in context.")

    def test_live_speech_uses_stable_first_correction_even_after_later_review(self):
        first = {"authoritative": ["First correction."], "statuses": ["corrected"], "segments": [{}]}
        state = {"authoritative": ["Later wording."], "statuses": ["corrected"],
                 "segments": [{"conversation_review_status": "corrected"}], "first_pass": first}
        result = astra_speech_result(state, segment_count=1)
        self.assertEqual(result["translations"], ["First correction."])
        self.assertEqual(result["pending"], 0)
        final = astra_speech_result(state, segment_count=1, conversation_review=True)
        self.assertEqual(final["translations"], ["Later wording."])

    def test_astra_failure_waits_for_retry_and_preserves_order_without_repeats(self):
        state = {"authoritative": [None, "Second correction."], "statuses": ["failed", "corrected"],
                 "segments": [{"error": "Failed"}, {}], "pending": 0}
        session = self.session()
        result = astra_speech_result(state, segment_count=2)
        session.submit(result, final=result["pending"] == 0)
        self.assertEqual(self.spoken, [])
        self.assertEqual(result["errors"], [None, None])
        state["authoritative"][0] = "First correction."
        state["statuses"][0] = "corrected"
        result = astra_speech_result(state, segment_count=2)
        session.submit(result, final=result["pending"] == 0)
        self.wait_for(lambda: len(self.spoken) == 2)
        session.submit(result, final=True)
        self.assertEqual([text for text, _ in self.spoken], ["First correction.", "Second correction."])

    def test_astra_filters_and_invalid_english_are_not_read_aloud(self):
        state = {"authoritative": ["[Background speech filtered]", "你好", "Original source"],
                 "statuses": ["filtered", "confirmed", "confirmed"],
                 "segments": [{"filtered": True}, {}, {"source_fallback": True}]}
        result = astra_speech_result(state, segment_count=3)
        self.assertEqual(result["translations"], [None, None, None])
        self.assertEqual(result["filtered"], [True, False, False])
        self.assertEqual(result["pending"], 2)


if __name__ == "__main__":
    unittest.main()
