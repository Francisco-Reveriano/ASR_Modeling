"""Two review layers with controlled providers, no network or local models."""

from copy import deepcopy
import json
from threading import Event
import time
import unittest
from unittest.mock import patch

from src.conversation_review import ConversationReviewSession, _SequentialReviewSession
from src.glossary import Glossary
from src.reasoning import CorrectionProviderError
from src.slow_lane import SlowLaneConfig, SlowLaneSession
import src.slow_lane as slow_lane


def corrected(request, text, **changes):
    row = request["segments"][0]
    return {"no_change": [], "corrections": [{
        "segment_id": row["segment_id"], "base_version": row["base_version"],
        "target_text": text, "change_type": ["asr_fix"], "confidence": 0.9,
        "rationale": "Supported by the conversation.", "term_pairs": [], **changes,
    }]}


def confirmed(request):
    row = request["segments"][0]
    return {"corrections": [], "no_change": [{
        "segment_id": row["segment_id"], "base_version": row["base_version"],
    }]}


class ConversationReviewTests(unittest.TestCase):
    def setUp(self):
        self.sessions = []
        self.releases = []

    def tearDown(self):
        for session in self.sessions:
            session.close()
        for event in self.releases:
            event.set()
        for session in self.sessions:
            self.wait_for(lambda: not session._first._calls and
                          (session._second is None or not session._second._calls))

    def wait_for(self, predicate):
        deadline = time.monotonic() + 3
        while not predicate():
            if time.monotonic() > deadline:
                self.fail("asynchronous review did not complete")
            time.sleep(0.002)

    def release(self):
        event = Event()
        self.releases.append(event)
        return event

    def session(self, correct=confirmed, **kwargs):
        session = ConversationReviewSession(correct, **kwargs)
        self.sessions.append(session)
        return session

    def reviewed(self, session, count):
        self.wait_for(lambda: session.snapshot()["conversation_review"]["reviewed"] == count)
        return session.snapshot()

    def test_independent_first_reviews_and_ordered_second_reviews_publish_without_polling(self):
        first_started, first_later_finished = Event(), Event()
        second_started = [Event(), Event()]
        first_release, second_releases = self.release(), [self.release(), self.release()]
        calls = []
        def review(request):
            calls.append(deepcopy(request))
            source = request["segments"][0]["source_text"]
            index = ["one", "two"].index(source)
            if request.get("review_stage") == "conversation":
                second_started[index].set()
                second_releases[index].wait(timeout=5)
                return corrected(request, f"Conversation {source}")
            if index == 0:
                first_started.set()
                first_release.wait(timeout=5)
            else:
                first_later_finished.set()
            return corrected(request, f"First {source}")
        session = self.session(review)
        session.submit(["one", "two"], ["Draft one", "Draft two"],
                       timings=[{"speaker_id": "A"}, {"speaker_id": "B"}])
        self.assertTrue(first_started.wait(timeout=1))
        self.assertTrue(first_later_finished.wait(timeout=1))
        self.wait_for(lambda: session._first.snapshot()["statuses"][1] == "corrected")
        visible = session.snapshot()
        self.assertEqual(visible["translations"], ["Draft one", "First two"])
        self.assertEqual(visible["pending"], 2)
        self.assertEqual(visible["active_reviews"], 1)
        self.assertFalse(any(event.is_set() for event in second_started))

        first_release.set()
        # Provider events, not UI snapshots, drive both stage transitions.
        self.assertTrue(second_started[0].wait(timeout=1))
        self.assertFalse(second_started[1].is_set())
        frozen_first = deepcopy(session.snapshot()["first_pass"])
        second_releases[0].set()
        self.assertTrue(second_started[1].wait(timeout=1))
        visible = session.snapshot()
        self.assertEqual(visible["translations"], ["Conversation one", "First two"])
        self.assertEqual([row["conversation_review_status"] for row in visible["segments"]],
                         ["corrected", "reviewing"])
        self.assertEqual(visible["pending"], 1)
        self.assertEqual(visible["active_reviews"], 1)
        second_calls = [request for request in calls if request.get("review_stage") == "conversation"]
        self.assertEqual([request["segments"][0]["source_text"] for request in second_calls], ["one", "two"])
        self.assertEqual(second_calls[0]["context"][0]["target_text"], "First two")
        self.assertEqual(second_calls[0]["context"][0]["target_status"], "corrected")
        self.assertEqual(second_calls[1]["context"][0]["target_text"], "Conversation one")
        self.assertEqual(second_calls[1]["context"][0]["target_status"], "corrected")
        second_releases[1].set()
        final = self.reviewed(session, 2)
        self.assertEqual(final["translations"], ["Conversation one", "Conversation two"])
        self.assertEqual(final["authoritative"], final["translations"])
        self.assertEqual(final["first_pass"], frozen_first)
        self.assertEqual([row["version"] for row in final["segments"]], [3, 3])
        self.assertEqual([row["replacements"] for row in final["segments"]], [2, 2])
        self.assertEqual(final["review_stage"], "combined")
        self.assertEqual([row["version"] for row in final["first_pass"]["segments"]], [2, 2])
        self.assertEqual([row["timing"]["speaker_id"] for row in final["segments"]], ["A", "B"])
        self.assertEqual([row["segment_id"] for row in final["segments"]],
                         [row["segment_id"] for row in final["second_pass"]["segments"]])
        self.assertTrue(all(len(request["segments"]) == 1 for request in calls))
        self.assertEqual([event["sequence"] for event in final["events"]], list(range(1, final["last_sequence"] + 1)))
        self.assertEqual(session.events_after(final["last_sequence"]), [])

    def test_adopting_filtered_following_rows_is_atomic_before_second_context_dispatch(self):
        first = SlowLaneSession(confirmed)
        self.addCleanup(first.close)
        first.submit(["one", "Background announcement", "two"],
                     ["First English", "[Background speech filtered]", "Second English"],
                     draft_filtered=[False, True, False])
        self.wait_for(lambda: first.snapshot()["pending"] == 0 and not first._calls)
        calls = []
        second = _SequentialReviewSession(lambda request: calls.append(request) or confirmed(request),
                                          session_id=first.session_id)
        self.addCleanup(second.close)
        second.sync_first_pass(first.snapshot())
        self.wait_for(lambda: second.snapshot()["pending"] == 0 and not second._calls)
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(row["source_text"] != "Background announcement"
                            for request in calls for row in request["segments"] + request["context"]))
        self.assertEqual(second.snapshot()["statuses"], ["confirmed", "filtered", "confirmed"])

    def test_second_pass_keeps_context_budget_and_late_prefix_metadata_updates(self):
        calls = []
        session = self.session(lambda request: calls.append(request) or confirmed(request),
                               config=SlowLaneConfig(max_source_tokens=6))
        sources = ["aaa", "bbb", "ccc"]
        session.submit(sources, [], timings=[{"speaker_id": "A"}, {"speaker_id": "B"}, {"speaker_id": "C"}])
        session.submit(sources[:1], ["First English"], allow_prefix=True)
        self.reviewed(session, 1)
        session.submit(sources, ["First English", "Second English", "Third English"])
        before = self.reviewed(session, 3)
        session.submit(sources[:1], ["First English"], allow_prefix=True, timings=[{"speaker_id": "Merged"}])
        after = session.snapshot()
        self.assertEqual(after["translations"], before["translations"])
        self.assertEqual([row["version"] for row in after["segments"]], [row["version"] for row in before["segments"]])
        self.assertEqual([row["timing"]["speaker_id"] for row in after["segments"]], ["Merged", "B", "C"])
        self.assertEqual(after["first_pass"]["segments"][0]["timing"]["speaker_id"], "Merged")
        self.assertEqual(after["second_pass"]["segments"][0]["timing"]["speaker_id"], "Merged")
        for request in calls:
            self.assertLessEqual(sum(len(row["source_text"].encode())
                                     for row in request["segments"] + request["context"]), 6)

    def test_second_confirmation_keeps_first_correction_and_version(self):
        def review(request):
            return confirmed(request) if request.get("review_stage") else corrected(request, "Corrected English")
        session = self.session(review)
        session.submit(["source"], ["English draft"])
        final = self.reviewed(session, 1)
        row = final["segments"][0]
        self.assertEqual(row["status"], "corrected")
        self.assertEqual(row["first_pass_status"], "corrected")
        self.assertEqual(row["conversation_review_status"], "confirmed")
        self.assertEqual(row["version"], 2)
        self.assertEqual(final["translations"], ["Corrected English"])

    def test_second_failure_keeps_first_english_blocks_successor_and_manual_retry_recovers(self):
        good = Event()
        failed = Event()
        calls = []
        def review(request):
            calls.append(request)
            if not request.get("review_stage"):
                return corrected(request, "First " + request["segments"][0]["source_text"])
            if request["segments"][0]["source_text"] == "one" and not good.is_set():
                failed.set()
                raise RuntimeError("SECRET provider exception")
            return corrected(request, "Reviewed " + request["segments"][0]["source_text"])
        session = self.session(review)
        session.submit(["one", "two"], ["Draft one", "Draft two"])
        self.assertTrue(failed.wait(timeout=1))
        self.wait_for(lambda: session.snapshot()["conversation_review"]["status"] == "degraded" and
                      session._first.snapshot()["pending"] == 0)
        result = session.snapshot()
        self.assertEqual(result["translations"], ["First one", "First two"])
        self.assertEqual(result["authoritative"], result["translations"])
        self.assertEqual(result["statuses"], ["corrected", "corrected"])
        self.assertEqual([row["conversation_review_status"] for row in result["segments"]], ["failed", "blocked"])
        self.assertIsNone(result["segments"][0]["error"])
        self.assertEqual(result["segments"][0]["conversation_review_error_details"], {"category": "error"})
        self.assertNotIn("SECRET", json.dumps(result))
        self.assertEqual(len([request for request in calls if request.get("review_stage")]), 1)
        for _ in range(3):
            session.submit(["one", "two"], ["Draft one", "Draft two"])
        self.assertEqual(len(calls), 3)
        good.set()
        session.retry_failed()
        final = self.reviewed(session, 2)
        self.assertEqual(final["translations"], ["Reviewed one", "Reviewed two"])
        self.assertEqual(len(calls), 5)

    def test_failed_first_review_blocks_conversation_order_until_first_retry_succeeds(self):
        good = Event()
        calls = []
        def review(request):
            calls.append(request)
            if not request.get("review_stage") and request["segments"][0]["source_text"] == "one" and not good.is_set():
                raise RuntimeError("failure")
            return confirmed(request)
        session = self.session(review)
        session.submit(["one", "two"], ["Draft one", "Draft two"])
        self.wait_for(lambda: session._first.snapshot()["status"] == "degraded" and session._first.snapshot()["pending"] == 0)
        result = session.snapshot()
        self.assertEqual([row["conversation_review_status"] for row in result["segments"]], ["blocked", "blocked"])
        self.assertFalse(any(request.get("review_stage") for request in calls))
        good.set()
        session.retry_failed()
        self.reviewed(session, 2)
        self.assertEqual([request["segments"][0]["source_text"] for request in calls if request.get("review_stage")], ["one", "two"])

    def test_second_review_preserves_validation_guards_and_one_automatic_retry(self):
        failures = [
            (lambda request: corrected(request, "中文 ETCH-07"), "language"),
            (lambda request: corrected(request, "Changed ETCH-08"), "dnt"),
            (lambda request: corrected(request, "Correct ETCH-07", confidence=0.53), "low_confidence"),
            (lambda request: corrected(request, "Correct ETCH-07", base_version=0), "stale"),
            (lambda request: {"corrections": [], "no_change": []}, "schema"),
        ]
        for response, category in failures:
            with self.subTest(category=category):
                calls = []
                def review(request):
                    calls.append(request)
                    return response(request) if request.get("review_stage") else confirmed(request)
                session = self.session(review)
                session.submit(["Keep ETCH-07"], ["Keep ETCH-07"])
                self.wait_for(lambda: session.snapshot()["segments"][0]["conversation_review_status"] == "failed")
                result = session.snapshot()
                self.assertEqual(result["translations"], ["Keep ETCH-07"])
                self.assertEqual(result["statuses"], ["confirmed"])
                self.assertEqual(result["segments"][0]["conversation_review_error_details"], {"category": category})
                self.assertEqual(len(calls), 3)
                self.assertEqual(result["second_pass"]["metrics"]["automatic_retries"], 1)

    def test_filtered_and_explicitly_paused_rows_skip_without_claiming_conversation_review(self):
        second_started, release = Event(), self.release()
        calls = []
        def review(request):
            calls.append(request)
            if request.get("review_stage") and request["segments"][0]["source_text"] == "one":
                second_started.set()
                release.wait(timeout=5)
                return corrected(request, "Canceled change")
            return confirmed(request)
        session = self.session(review)
        sources = ["Background announcement", "one"]
        drafts = ["[Background speech filtered]", "First English"]
        session.submit(sources, drafts, draft_filtered=[True, False])
        self.assertTrue(second_started.wait(timeout=1))
        session.set_enabled(False)
        paused = session.snapshot()
        self.assertEqual([row["conversation_review_status"] for row in paused["segments"]], ["disabled", "disabled"])
        self.assertEqual(paused["conversation_review"]["reviewed"], 0)
        self.assertEqual(paused["pending"], 0)
        session.set_enabled(True)
        session.submit(sources + ["two"], drafts + ["Second English"], draft_filtered=[True, False, False])
        final = self.reviewed(session, 1)
        self.assertEqual(final["translations"][1:], ["First English", "Second English"])
        self.assertEqual(final["segments"][2]["conversation_review_status"], "confirmed")
        release.set()
        self.wait_for(lambda: not session._second._calls)
        self.assertEqual(session.snapshot()["translations"][1], "First English")
        self.assertTrue(all(request["segments"][0]["source_text"] != "Background announcement" for request in calls))

    def test_close_ignores_both_active_layers_and_does_not_retry_truncation(self):
        second_started, later_started = Event(), Event()
        release = self.release()
        calls = []
        def review(request):
            calls.append(request)
            if request.get("review_stage"):
                second_started.set()
                release.wait(timeout=5)
                raise CorrectionProviderError("incomplete", incomplete_reason="max_output_tokens")
            if request["segments"][0]["source_text"] == "two":
                later_started.set()
                release.wait(timeout=5)
            return corrected(request, "First reviewed English")
        session = self.session(review)
        session.submit(["one", "two"], ["Draft one", "Draft two"])
        self.assertTrue(second_started.wait(timeout=1))
        self.assertTrue(later_started.wait(timeout=1))
        session.close()
        frozen = deepcopy(session.snapshot()["segments"])
        release.set()
        self.wait_for(lambda: not session._first._calls and not session._second._calls)
        final = session.snapshot()
        self.assertEqual(final["segments"], frozen)
        self.assertEqual(final["status"], "closed")
        self.assertEqual(final["active_reviews"], 0)
        self.assertEqual(final["pending"], 0)
        self.assertEqual(final["metrics"]["automatic_retries"], 0)
        self.assertEqual(len(calls), 3)

    def test_same_utterance_cannot_double_promote_glossary_across_layers(self):
        glossary = Glossary()
        def review(request):
            return corrected(request, "The wafer is ready", change_type=["terminology"],
                             term_pairs=[{"source": "晶圓", "target": "wafer"}])
        session = self.session(review, glossary=glossary)
        session.submit(["晶圓好了"], ["The draft"])
        first = self.reviewed(session, 1)
        self.assertEqual(first["learned_terms"], [])
        self.assertEqual(glossary.learned_terms(), [])
        session.submit(["晶圓好了", "晶圓準備好了"], ["The draft", "Another draft"])
        final = self.reviewed(session, 2)
        self.assertEqual(len(final["learned_terms"]), 1)
        self.assertEqual(final["metrics"]["learned_terms"], 1)

    def test_stage_uses_same_global_capacity_and_drains_without_polling(self):
        first_started, release, all_done = Event(), self.release(), Event()
        calls = []
        def review(request):
            calls.append(request)
            if len(calls) == 1:
                first_started.set()
                release.wait(timeout=5)
            if request.get("review_stage") and request["segments"][0]["source_text"] == "two":
                all_done.set()
            return confirmed(request)
        with patch.object(slow_lane, "_MAX_ACTIVE_REVIEWS", 1):
            session = self.session(review)
            session.submit(["one", "two"], ["First English", "Second English"])
            self.assertTrue(first_started.wait(timeout=1))
            self.assertEqual(session.snapshot()["active_reviews"], 1)
            release.set()
            self.assertTrue(all_done.wait(timeout=2))
            final = self.reviewed(session, 2)
            self.wait_for(lambda: not session._first._calls and not session._second._calls)
            self.assertEqual(final["metrics"]["requests"], 4)
            self.assertEqual(slow_lane._ACTIVE_REVIEW_CALLS, 0)

    def test_disabled_second_layer_and_prefix_callbacks_remain_compatible(self):
        calls = []
        session = self.session(lambda request: calls.append(request) or confirmed(request), second_pass_enabled=False)
        session.submit(["one", "two"], [], timings=[{"speaker_id": "A"}, {"speaker_id": "B"}])
        session.submit(["one"], ["First English"], allow_prefix=True)
        self.wait_for(lambda: session._first.snapshot()["statuses"][0] == "confirmed")
        result = session.snapshot()
        self.assertEqual(result["translations"], ["First English", None])
        self.assertFalse(result["config"]["second_pass_enabled"])
        self.assertEqual(result["conversation_review"]["status"], "disabled")
        self.assertEqual([row["conversation_review_status"] for row in result["segments"]], ["disabled", "disabled"])
        self.assertEqual(result["segments"][1]["timing"], {"speaker_id": "B"})
        self.assertIsNone(result["second_pass"])
        self.assertEqual(len(calls), 1)
        self.assertNotIn("review_stage", calls[0])

    def test_skipped_second_placeholders_keep_the_combined_current_version_in_audit(self):
        calls = []
        session = self.session(lambda request: calls.append(request) or confirmed(request),
                               config=SlowLaneConfig(enabled=False))
        session.submit(["one"], ["English draft"])
        result = session.snapshot()
        self.assertEqual(result["segments"][0]["version"], 1)
        seals = [event for event in result["events"]
                 if event["type"] == "Seal" and event["review_stage"] == "conversation"]
        self.assertEqual(len(seals), 1)
        self.assertEqual(seals[0]["segment"]["version"], 1)
        self.assertEqual(seals[0]["segment"]["screen_version"], 1)
        self.assertEqual(result["second_pass"]["segments"][0]["version"], 0)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
