"""Deterministic slow-lane faults without model, audio, or network dependencies."""

from copy import deepcopy
from dataclasses import replace
import json
import re
from threading import Event
import time
import unittest
from unittest.mock import patch

import src.slow_lane as slow_lane

from src.slow_lane import SlowLaneConfig, SlowLaneSession
from src.reasoning import CorrectionProviderError
from src.glossary import Glossary


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class FakeGlossary:
    def __init__(self):
        self.observations = {}
        self.promoted = []

    def retrieve(self, text, limit=40):
        return [{"source": "晶圓", "target": "wafer"}][:limit]

    def dnt_hits(self, text):
        return re.findall(r"\bETCH-\d+\b", text)

    def compare_dnt(self, source, target):
        return {"ok": self.dnt_hits(source) == self.dnt_hits(target)}

    def observe(self, source, target, segment_id):
        ids = self.observations.setdefault((source, target), set())
        ids.add(segment_id)
        entry = {"source": source, "target": target}
        if len(ids) >= 2 and entry not in self.promoted:
            self.promoted.append(entry)
            return True
        return False

    def learned_terms(self):
        return deepcopy(self.promoted)


def confirmed(request):
    return {"corrections": [], "no_change": [
        {"segment_id": item["segment_id"], "base_version": item["base_version"]}
        for item in request["segments"]
    ]}


def corrected(request, target="Corrected translation", **changes):
    return {"no_change": [], "corrections": [
        {"segment_id": item["segment_id"], "base_version": item["base_version"],
         "target_text": target, "change_type": ["asr_fix"], "confidence": 0.9,
         "rationale": "The source supports this repair.", "term_pairs": [], **changes}
        for item in request["segments"]
    ]}


class SlowLaneTestCase(unittest.TestCase):
    def setUp(self):
        self.sessions = []
        self.releases = []
        self.clock = FakeClock()
        self.glossary = FakeGlossary()

    def tearDown(self):
        for session in self.sessions:
            session.close()
        for release in self.releases:
            release.set()
        for session in self.sessions:
            self.wait_for(lambda: not session._calls)

    def session(self, correct=confirmed, **kwargs):
        session = SlowLaneSession(correct, clock=self.clock, glossary=kwargs.pop("glossary", self.glossary), **kwargs)
        self.sessions.append(session)
        return session

    def wait_for(self, predicate):
        deadline = time.monotonic() + 2
        while not predicate():
            if time.monotonic() > deadline:
                self.fail("asynchronous condition did not finish")
            time.sleep(0.002)

    def idle(self, session):
        self.wait_for(lambda: session.snapshot()["pending"] == 0)
        return session.snapshot()

    def blocked(self, result=confirmed):
        started, release = Event(), Event()
        self.releases.append(release)
        def correct(request):
            started.set()
            release.wait(timeout=5)
            return result(request)
        return correct, started, release


class ConfigTests(unittest.TestCase):
    def test_defaults_are_bounded_and_config_is_immutable(self):
        config = SlowLaneConfig()
        self.assertEqual((config.model, config.reasoning_effort), ("gpt-6-astra", "medium"))
        self.assertEqual(config.max_output_tokens, 16_384)
        for removed in ("request_timeout_s", "seal_timeout_s", "revision_horizon_s",
                        "queue_depth", "max_window_seconds", "window_n"):
            self.assertFalse(hasattr(config, removed), removed)
        with self.assertRaises(AttributeError):
            config.max_output_tokens = 512

    def test_invalid_limits_and_types_are_rejected(self):
        for options in ({"earlier_context_n": 101}, {"max_source_tokens": 100_001},
                        {"confidence_threshold": -0.1}, {"enabled": "yes"}, {"model": " "},
                        {"filter_background_speech": "true"},
                        {"reasoning_effort": "unbounded"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                SlowLaneConfig(**options)

    def test_output_cap_supports_explicit_lower_limits(self):
        config = SlowLaneConfig(max_output_tokens=4096, confidence_threshold=0)
        self.assertEqual(config.max_output_tokens, 4096)

    def test_reasoning_efforts_allow_model_specific_api_values(self):
        for effort in ("none", "minimal", "low", "medium", "high", "xhigh", "max"):
            with self.subTest(effort=effort):
                self.assertEqual(SlowLaneConfig(reasoning_effort=effort).reasoning_effort, effort)


class StoreTests(SlowLaneTestCase):
    def test_glossary_reload_does_not_revalidate_sealed_filtered_rows_on_polling(self):
        for close_before_reload in (False, True):
            with self.subTest(close_before_reload=close_before_reload):
                glossary = Glossary()
                session = self.session(glossary=glossary)
                sources, drafts = ["Background announcement"], ["[Background speech filtered]"]
                session.submit(sources, drafts, draft_filtered=[True])
                if close_before_reload:
                    session.close()
                frozen = deepcopy(session.snapshot()["segments"][0])
                glossary.replace_master(Glossary([{
                    "term_src": "announcement", "term_tgt": "announcement", "dnt": True,
                }]))

                session.submit(sources, drafts, draft_filtered=[True])

                self.assertEqual(session.snapshot()["segments"][0], frozen)
                self.assertEqual(session.snapshot()["metrics"]["requests"], 0)
                if not close_before_reload:
                    with self.assertRaisesRegex(ValueError, "append-only"):
                        session.submit(["Changed announcement"], drafts, draft_filtered=[True])

    def test_fast_filter_retry_resolves_failed_review_without_stale_degraded_status(self):
        calls = []
        def failed_review(request):
            calls.append(request)
            raise RuntimeError("private provider failure")
        session = self.session(failed_review)
        session.submit(["Background speech"], [None], draft_errors=["fast failure"])
        failed = self.idle(session)
        self.assertEqual(failed["status"], "degraded")

        session.submit(["Background speech"], ["[Background speech filtered]"], draft_filtered=[True])

        result = session.snapshot()
        self.assertEqual(result["statuses"], ["filtered"])
        self.assertEqual(result["status"], "idle")
        self.assertEqual(result["pending"], 0)
        self.assertIsNone(result["segments"][0]["error"])
        self.assertIsNone(result["segments"][0]["review_error"])
        session.retry_failed()
        self.assertEqual(len(calls), 1)

    def test_invalid_filtered_flags_do_not_mutate_transcript(self):
        cases = [
            (["one"], ["[Background speech filtered]"], [1], None),
            (["one"], ["[Background speech filtered]"], [True, True], None),
            (["one"], [None], [True], None),
            (["one"], ["Ordinary English draft"], [True], None),
            (["one"], ["[Background speech filtered]"], [True], ["private failure"]),
            (["one", "Keep ETCH-07"], ["English draft", "[Background speech filtered]"], [False, True], None),
        ]
        for sources, drafts, flags, errors in cases:
            with self.subTest(flags=flags, sources=sources):
                session = self.session()
                with self.assertRaises(ValueError):
                    session.submit(sources, drafts, draft_filtered=flags, draft_errors=errors)
                self.assertEqual(session.snapshot()["segments"], [])

    def test_late_review_cannot_replace_an_explicitly_filtered_record(self):
        correct, started, release = self.blocked(corrected)
        session = self.session(correct)
        session.submit(["Background speech"], [None], draft_errors=["failure"])
        self.assertTrue(started.wait(timeout=1))
        session.submit(["Background speech"], ["[Background speech filtered]"], draft_filtered=[True])
        frozen = deepcopy(session.snapshot()["segments"][0])
        release.set()
        self.wait_for(lambda: not session._calls)
        result = session.snapshot()
        self.assertEqual(result["segments"][0], frozen)
        self.assertEqual(result["statuses"], ["filtered"])
        self.assertEqual(result["status"], "idle")
        self.assertEqual(result["pending"], 0)

    def test_filtered_background_is_sealed_without_review_or_retry_and_keeps_source(self):
        calls = []
        session = self.session(lambda request: calls.append(request) or confirmed(request))
        session.submit(["Background announcement"], ["[Background speech filtered]"], draft_filtered=[True])
        result = self.idle(session)
        self.assertEqual(calls, [])
        self.assertEqual(result["statuses"], ["filtered"])
        self.assertEqual(result["authoritative"], ["[Background speech filtered]"])
        record = result["segments"][0]
        self.assertEqual(record["source_text"], "Background announcement")
        self.assertTrue(record["filtered"])
        self.assertTrue(record["sealed"])
        self.assertEqual(record["seal_reason"], "filtered")
        self.assertFalse(record["source_fallback"])
        frozen = deepcopy(record)
        session.retry_failed()
        session.close()
        self.assertEqual(session.snapshot()["segments"][0], frozen)
        self.assertTrue(any(event["type"] == "Filtered" for event in result["events"]))

    def test_filtered_background_is_excluded_from_later_correction_context(self):
        calls = []
        session = self.session(lambda request: calls.append(request) or confirmed(request))
        session.submit(["Background announcement", "Main speech"],
                       ["[Background speech filtered]", "Main translation"], draft_filtered=[True, False])
        result = self.idle(session)
        self.assertEqual(result["statuses"], ["filtered", "confirmed"])
        self.assertEqual(len(calls), 1)
        self.assertEqual([row["source_text"] for row in calls[0]["segments"]], ["Main speech"])
        self.assertEqual(calls[0]["context"], [])

    def test_fast_draft_is_immediate_and_append_only_submission_is_idempotent(self):
        correct, started, release = self.blocked()
        session = self.session(correct)
        session.submit(["first source", "second source"], ["First draft", None])
        self.assertTrue(started.wait(timeout=1))

        initial = session.snapshot()
        self.assertEqual(initial["translations"], ["First draft", None])
        self.assertEqual(initial["statuses"], ["draft", "waiting"])
        self.assertEqual([record["version"] for record in initial["segments"]], [1, 0])
        self.assertEqual(initial["authoritative"], [None, None])
        self.assertEqual(initial["model"], "gpt-6-astra")
        self.assertEqual(initial["segments"][1]["segment_id"], session.session_id + ":2")
        session.submit(["first source", "second source"], ["First draft", None])
        self.assertEqual(session.snapshot()["metrics"]["requests"], 1)
        session.submit(["first source", "second source"], ["First draft", "Second draft"])
        self.assertEqual(session.snapshot()["translations"], ["First draft", "Second draft"])
        release.set()
        result = self.idle(session)
        self.assertEqual(result["authoritative"], ["First draft", "Second draft"])
        self.assertEqual(result["statuses"], ["confirmed", "confirmed"])

    def test_existing_source_or_fast_draft_replacement_is_rejected_without_appending(self):
        session = self.session(config=SlowLaneConfig(enabled=False))
        session.submit(["source"], ["draft"])
        for sources, drafts in ((["edited", "extra"], ["draft", "extra"]), (["source"], ["edited"]), ([], [])):
            with self.subTest(sources=sources), self.assertRaises(ValueError):
                session.submit(sources, drafts)
        self.assertEqual(len(session.snapshot()["segments"]), 1)

    def test_correction_versions_seal_once_and_events_replay_the_visible_state(self):
        session = self.session(corrected)
        session.submit(["source"], ["draft"])
        result = self.idle(session)
        record = result["segments"][0]

        self.assertEqual((record["version"], record["replacements"], record["seal_reason"]), (2, 1, "correction"))
        self.assertEqual(result["statuses"], ["corrected"])
        self.assertEqual(result["authoritative"], ["Corrected translation"])
        events = session.events_after(0)
        self.assertEqual([event["sequence"] for event in events], list(range(1, len(events) + 1)))
        self.assertTrue(all(event["schema_version"] == 1 and event["session_id"] == session.session_id for event in events))
        record_events = [event for event in events if "segment" in event]
        self.assertEqual([event["type"] for event in record_events], ["Draft", "Correction", "Seal"])
        self.assertEqual(record_events[-1]["segment"], record)
        self.assertEqual(session.events_after(events[1]["sequence"]), events[2:])
        frozen = deepcopy(record)
        session.set_enabled(False)
        session.close()
        self.clock.advance(100)
        self.assertEqual(session.snapshot()["segments"][0], frozen)

    def test_snapshot_and_request_mutation_cannot_change_the_store(self):
        def mutate(request):
            response = confirmed(request)
            request["segments"][0]["source_text"] = "external change"
            return response
        session = self.session(mutate)
        session.submit(["source"], ["draft"])
        result = self.idle(session)
        result["segments"][0]["history"][0]["target_text"] = "external change"
        result["events"].clear()
        self.assertEqual(session.snapshot()["segments"][0]["source_text"], "source")
        self.assertEqual(session.snapshot()["segments"][0]["history"][0]["target_text"], "draft")
        self.assertTrue(session.events_after(0))

    def test_elapsed_time_never_seals_a_draft_or_discards_a_slow_review(self):
        correct, started, release = self.blocked(corrected)
        session = self.session(correct)
        session.submit(["source"], ["draft"])
        self.assertTrue(started.wait(timeout=1))
        for elapsed in (21, 31, 121, 3600):
            self.clock.advance(elapsed)
            result = session.snapshot()
            self.assertEqual(result["statuses"], ["draft"])
            self.assertEqual(result["authoritative"], [None])
            self.assertFalse(result["segments"][0]["sealed"])
            self.assertEqual(result["pending"], 1)
        release.set()
        result = self.idle(session)
        self.assertEqual(result["statuses"], ["corrected"])
        self.assertEqual(result["translations"], ["Corrected translation"])
        self.assertEqual(result["authoritative"], ["Corrected translation"])

    def test_late_correction_publishes_sealed_version_to_screen(self):
        correct, started, release = self.blocked(corrected)
        session = self.session(correct)
        session.submit(["source"], ["screen draft"])
        self.assertTrue(started.wait(timeout=1))
        self.clock.advance(21)
        release.set()
        result = self.idle(session)
        self.assertEqual(result["translations"], ["Corrected translation"])
        self.assertEqual(result["authoritative"], ["Corrected translation"])
        self.assertEqual(result["statuses"], ["corrected"])
        self.assertEqual(result["segments"][0]["screen_version"], 2)
        self.assertEqual(result["segments"][0]["version"], 2)
        self.assertEqual(result["segments"][0]["replacements"], 1)
        self.assertEqual(result["metrics"]["screen_replacements"], 1)
        self.assertTrue(result["segments"][0]["history"][-1]["screen_updated"])
        self.assertEqual(session.events_after(0)[-2]["segment"], result["segments"][0])

    def test_missing_fast_draft_stays_unsealed_until_explicit_close(self):
        session = self.session()
        session.submit(["source"], [None])
        self.clock.advance(3600)
        result = session.snapshot()
        self.assertEqual(result["translations"], [None])
        self.assertEqual(result["authoritative"], [None])
        self.assertEqual(result["statuses"], ["waiting"])
        self.assertFalse(result["segments"][0]["sealed"])
        session.close()
        self.assertTrue(session.snapshot()["segments"][0]["sealed"])
        self.assertEqual(session.snapshot()["segments"][0]["seal_reason"], "closed")

    def test_fast_draft_arriving_hours_later_still_receives_review(self):
        calls = []
        session = self.session(lambda request: calls.append(request) or corrected(request))
        session.submit(["source"], [None])
        self.clock.advance(3600)
        session.submit(["source"], ["late fast draft"])
        result = self.idle(session)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["segments"][0]["target_text"], "late fast draft")
        self.assertEqual(result["authoritative"], ["Corrected translation"])
        self.assertEqual(result["segments"][0]["seal_reason"], "correction")

    def test_old_live_endpoint_does_not_hide_accepted_correction_and_timing_stays_audit_only(self):
        calls = []
        self.clock.advance(21)
        session = self.session(lambda request: calls.append(request) or corrected(request))
        timing = {"start_s": 9000, "end_s": 9001, "server_endpoint_ms": 0,
                  "asr_start_ms": 1000, "asr_final_ms": 20000, "t_capture_ms": None,
                  "speaker_id": "speaker-1"}
        session.submit(["source"], ["screen draft"], timings=[timing])
        result = self.idle(session)
        self.assertEqual(result["translations"], ["Corrected translation"])
        self.assertEqual(result["authoritative"], ["Corrected translation"])
        self.assertEqual(result["segments"][0]["timing"], timing)
        self.assertEqual(calls[0]["segments"][0]["speaker_id"], "speaker-1")
        self.assertNotIn("server_endpoint_ms", json.dumps(calls[0]))
        self.assertNotIn("asr_final_ms", json.dumps(calls[0]))

    def test_speaker_arriving_with_delayed_draft_reaches_the_request_and_audit_record(self):
        calls = []
        session = self.session(lambda request: calls.append(request) or confirmed(request))
        session.submit(["source"], [None], timings=[{"start_s": 0, "end_s": 1}])
        session.submit(["source"], ["draft"], timings=[{"start_s": 0, "end_s": 1, "speaker_id": "speaker-1"}])
        result = self.idle(session)
        self.assertEqual(calls[0]["segments"][0]["speaker_id"], "speaker-1")
        self.assertEqual(result["segments"][0]["timing"]["speaker_id"], "speaker-1")

    def test_sealed_speaker_metadata_can_change_without_changing_subtitle_versions(self):
        session = self.session()
        session.submit(["source"], ["draft"], timings=[{"start_s": 0, "end_s": 1}])
        before = self.idle(session)
        session.submit(["source"], ["draft"], timings=[{"start_s": 0, "end_s": 1, "speaker_id": "speaker-2"}])
        after = session.snapshot()
        original = before["segments"][0]
        updated = after["segments"][0]
        self.assertEqual(updated["timing"]["speaker_id"], "speaker-2")
        self.assertEqual({key: value for key, value in updated.items() if key != "timing"},
                         {key: value for key, value in original.items() if key != "timing"})
        self.assertEqual(after["metrics"]["requests"], 1)
        events = session.events_after(before["last_sequence"])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "SessionStatus")
        self.assertEqual(events[0]["metadata_updates"][0]["timing"]["speaker_id"], "speaker-2")

    def test_late_timing_enrichment_is_audit_only_and_drift_is_rejected(self):
        session = self.session()
        session.submit(["source"], [None], timings=[{"start_s": 0, "end_s": 1}])
        for changed in ({"start_s": 1, "end_s": 2}, {"start": 1, "end": 2}):
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "cannot change"):
                session.submit(["source"], [None], timings=[changed])
        self.clock.advance(3600)
        session.submit(["source"], [None], timings=[{"server_endpoint_ms": 25_000, "speaker_id": "speaker-1"}])
        self.assertEqual(session.snapshot()["segments"][0]["created_at"], 0)
        with self.assertRaisesRegex(ValueError, "cannot change"):
            session.submit(["source"], [None], timings=[{"server_endpoint_ms": 26_000}])
        result = session.snapshot()
        self.assertEqual(result["authoritative"], [None])
        self.assertFalse(result["segments"][0]["sealed"])

    def test_explicit_fast_failure_remains_retryable_without_promoting_source(self):
        session = self.session()
        session.submit(["原文"], [None], draft_errors=["private fast error details"])
        result = self.idle(session)
        self.assertEqual(result["translations"], ["原文"])
        self.assertEqual(result["statuses"], ["failed"])
        self.assertTrue(result["segments"][0]["source_fallback"])
        self.assertEqual(result["segments"][0]["version"], 1)
        self.assertNotIn("private fast error", json.dumps(result))
        self.clock.advance(3600)
        self.assertEqual(session.snapshot()["authoritative"], [None])
        session.submit(["原文"], ["Late successful English"])
        result = self.idle(session)
        self.assertEqual(result["translations"], ["Late successful English"])
        self.assertEqual(result["statuses"], ["confirmed"])

    def test_fast_retry_before_sealing_can_replace_source_fallback_then_be_corrected(self):
        correct, started, release = self.blocked(corrected)
        session = self.session(correct)
        session.submit(["source"], [None], draft_errors=["failure"])
        self.assertTrue(started.wait(timeout=1))
        session.submit(["source"], ["Fast English"])
        release.set()
        result = self.idle(session)
        self.assertEqual(result["segments"][0]["version"], 3)
        self.assertEqual(result["segments"][0]["replacements"], 2)
        self.assertFalse(result["segments"][0]["source_fallback"])
        self.assertEqual(result["metrics"]["screen_replacements"], 2)

    def test_automatic_correction_recovers_source_fallback_without_mislabeling_either_view(self):
        for age in (0, 21):
            with self.subTest(age=age):
                correct, started, release = self.blocked(corrected)
                session = self.session(correct)
                session.submit(["原文"], [None], draft_errors=["failure"])
                self.assertTrue(started.wait(timeout=1))
                self.clock.advance(age)
                release.set()
                result = self.idle(session)
                record = result["segments"][0]
                self.assertEqual(result["authoritative"], ["Corrected translation"])
                self.assertEqual(result["statuses"], ["corrected"])
                self.assertFalse(record["source_fallback"])
                self.assertFalse(record["screen_source_fallback"])
                self.assertEqual(result["translations"], ["Corrected translation"])
                self.assertTrue(record["history"][0]["source_fallback"])

    def test_late_successful_fast_retry_publishes_after_astra_confirmation(self):
        correct, started, release = self.blocked()
        session = self.session(correct)
        session.submit(["原文"], [None], draft_errors=["failure"])
        self.assertTrue(started.wait(timeout=1))
        self.clock.advance(21)
        session.submit(["原文"], ["Successful fast English"])
        self.assertTrue(started.wait(timeout=1))
        pending = session.snapshot()
        self.assertEqual(pending["translations"], ["Successful fast English"])
        self.assertEqual(pending["authoritative"], [None])
        release.set()
        result = self.idle(session)
        record = result["segments"][0]
        self.assertEqual(result["translations"], ["Successful fast English"])
        self.assertEqual(result["authoritative"], ["Successful fast English"])
        self.assertEqual(result["statuses"], ["confirmed"])
        self.assertEqual((record["screen_version"], record["version"]), (2, 2))
        self.assertFalse(record["screen_source_fallback"])
        self.assertFalse(record["source_fallback"])
        self.assertEqual(record["replacements"], 1)
        self.assertEqual(result["metrics"]["screen_replacements"], 1)

    def test_disabled_session_seals_drafts_and_resume_only_affects_new_records(self):
        session = self.session(config=SlowLaneConfig(enabled=False))
        session.submit(["first"], ["First draft"])
        first = session.snapshot()["segments"][0]
        self.assertEqual(first["status"], "paused")
        self.assertEqual(session.snapshot()["pending"], 0)
        session.set_enabled(True)
        session.submit(["first", "second"], ["First draft", "Second draft"])
        result = self.idle(session)
        self.assertEqual(result["statuses"], ["paused", "confirmed"])
        self.assertEqual(result["segments"][0], first)

    def test_close_and_pause_discard_active_output_without_waiting(self):
        for action in ("close", "pause"):
            with self.subTest(action=action):
                correct, started, release = self.blocked(corrected)
                session = self.session(correct)
                session.submit(["source"], ["draft"])
                self.assertTrue(started.wait(timeout=1))
                if action == "close":
                    session.close()
                else:
                    session.set_enabled(False)
                result = session.snapshot()
                self.assertEqual(result["pending"], 0)
                self.assertEqual(result["authoritative"], ["draft"])
                frozen = result["segments"][0]
                release.set()
                self.wait_for(lambda: not session._calls)
                self.assertEqual(session.snapshot()["segments"][0], frozen)


class ValidationTests(SlowLaneTestCase):
    def test_pause_or_close_discards_a_blocked_automatic_re_review(self):
        for action in ("pause", "close"):
            with self.subTest(action=action):
                calls = []
                started, release = Event(), Event()
                self.releases.append(release)
                def correct(request):
                    calls.append(request)
                    if len(calls) == 1:
                        return corrected(request, confidence=0.53)
                    started.set()
                    release.wait(timeout=5)
                    return corrected(request)
                session = self.session(correct)
                session.submit(["source"], ["English draft"])
                self.assertTrue(started.wait(timeout=1))
                session.close() if action == "close" else session.set_enabled(False)
                frozen = deepcopy(session.snapshot()["segments"][0])
                release.set()
                self.wait_for(lambda: not session._calls)
                self.assertEqual(session.snapshot()["segments"][0], frozen)
                self.assertEqual(session.snapshot()["translations"], ["English draft"])
                self.assertEqual(len(calls), 2)

    def test_weak_row_retries_independently_without_delaying_valid_row(self):
        calls = []
        started, release = Event(), Event()
        self.releases.append(release)
        def review(request):
            calls.append(request)
            if request["segments"][0]["source_text"] == "one":
                return confirmed(request)
            if "review_feedback" not in request:
                return corrected(request, "Rejected speculative sentence", confidence=0.53)
            started.set()
            release.wait(timeout=5)
            return corrected(request)
        session = self.session(review)
        session.submit(["one", "two"], ["First draft", "Second draft"])
        self.assertTrue(started.wait(timeout=1))
        self.wait_for(lambda: session.snapshot()["statuses"][0] == "confirmed")
        pending = session.snapshot()
        self.assertEqual(pending["authoritative"], ["First draft", None])
        self.assertEqual(pending["translations"], ["First draft", "Second draft"])
        self.assertEqual(pending["pending"], 1)
        release.set()
        result = self.idle(session)
        self.assertEqual(result["statuses"], ["confirmed", "corrected"])
        self.assertEqual(len(calls), 3)
        retry = next(request for request in calls if "review_feedback" in request)
        self.assertEqual(retry["review_feedback"], {"category": "low_confidence"})
        self.assertNotIn("Rejected speculative sentence", json.dumps(retry))
        self.assertTrue(all(len(request["segments"]) == 1 for request in calls))
        self.assertEqual(result["metrics"]["automatic_retries"], 1)
        self.assertEqual(result["metrics"]["failure_reasons"], {"low_confidence": 1})

    def test_repeated_weak_review_stops_after_one_automatic_retry_with_safe_reason(self):
        calls = []
        session = self.session(lambda request: calls.append(request) or corrected(request, confidence=0.53))
        session.submit(["source"], ["English draft"])
        result = self.idle(session)
        self.assertEqual(len(calls), 2)
        self.assertEqual(result["statuses"], ["failed"])
        self.assertEqual(result["authoritative"], [None])
        self.assertEqual(result["segments"][0]["review_error"], {"category": "low_confidence"})
        self.assertIn("confidence", result["segments"][0]["error"])
        session.submit(["source"], ["English draft"])
        self.assertEqual(len(calls), 2)

    def test_missing_row_response_exhausts_one_repair_then_requires_explicit_retry(self):
        calls = []
        valid = Event()
        def correct(request):
            calls.append(request)
            return confirmed(request) if valid.is_set() else {"corrections": [], "no_change": []}
        session = self.session(correct)
        session.submit(["one", "two"], ["Draft one", "Draft two"])
        result = self.idle(session)
        self.assertEqual(result["statuses"], ["failed", "failed"])
        self.assertEqual(result["authoritative"], [None, None])
        self.assertEqual(result["metrics"]["schema_rejections"], 4)
        self.clock.advance(3600)
        session.submit(["one", "two"], ["Draft one", "Draft two"])
        self.assertEqual(len(calls), 4)
        valid.set()
        session.retry_failed()
        result = self.idle(session)
        self.assertEqual(result["statuses"], ["confirmed", "confirmed"])
        self.assertEqual(len(calls), 6)

    def test_non_english_correction_rejects_entire_response_without_publishing_it(self):
        def mixed_response(request):
            response = corrected(request, "A valid English correction")
            response["corrections"][-1]["target_text"] = "That batch of 布拉吉 from last Wednesday."
            return response

        session = self.session(mixed_response)
        session.submit(["第一句", "上禮拜三那批布拉吉"], ["First draft", "Second draft"])
        result = self.idle(session)

        self.assertEqual(result["translations"], ["First draft", "Second draft"])
        self.assertEqual(result["authoritative"], [None, None])
        self.assertEqual([record["version"] for record in result["segments"]], [1, 1])
        self.assertEqual(result["metrics"]["language_rejections"], 4)
        self.assertFalse(any(event["type"] == "Correction" for event in result["events"]))

    def test_no_change_cannot_confirm_mixed_language_fast_draft(self):
        session = self.session(confirmed)
        session.submit(["上禮拜三那批布拉吉"], ["That batch of 布拉吉 from last Wednesday."])
        result = self.idle(session)

        self.assertEqual(result["statuses"], ["failed"])
        self.assertEqual(result["authoritative"], [None])
        self.assertEqual(result["metrics"]["no_change"], 0)
        self.assertEqual(result["metrics"]["language_rejections"], 2)
        self.assertFalse(result["segments"][0]["sealed"])

    def test_no_change_cannot_seal_source_fallback_even_when_source_is_ascii(self):
        for source in ("原文", "ASCII source transcript"):
            with self.subTest(source=source):
                calls = []
                session = self.session(lambda request: calls.append(request) or confirmed(request))
                session.submit([source], [None], draft_errors=["failure"])
                result = self.idle(session)

                self.assertTrue(calls[0]["segments"][0]["source_fallback"])
                self.assertEqual(result["metrics"]["no_change"], 0)
                self.assertEqual(result["metrics"]["language_rejections"], 2)
                self.assertEqual(result["authoritative"], [None])
                self.assertFalse(result["segments"][0]["sealed"])

    def test_failed_fast_translation_gets_one_automatic_english_recovery(self):
        calls = []
        session = self.session(lambda request: calls.append(request) or corrected(request, "The original speech."))
        session.submit(["原文"], [None], draft_errors=["private failure"])
        result = self.idle(session)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["segments"][0]["source_text"], "原文")
        self.assertTrue(calls[0]["segments"][0]["source_fallback"])
        self.assertEqual(result["translations"], ["The original speech."])
        self.assertEqual(result["statuses"], ["corrected"])
        self.assertEqual(result["segments"][0]["version"], 2)
        frozen = deepcopy(result["segments"][0])
        session.submit(["原文"], ["Late fast English."])
        self.assertEqual(session.snapshot()["segments"][0], frozen)

    def test_rejected_automatic_recovery_waits_for_manual_retry(self):
        calls = []
        def recover(request):
            calls.append(request)
            return confirmed(request) if len(calls) <= 2 else corrected(request, "Recovered English.")

        session = self.session(recover)
        session.submit(["原文"], [None], draft_errors=["failure"])
        result = self.idle(session)
        self.assertEqual(result["metrics"]["language_rejections"], 2)
        self.assertFalse(result["segments"][0]["sealed"])
        for _ in range(3):
            session.submit(["原文"], [None], draft_errors=["failure"])
            self.idle(session)
        self.assertEqual(len(calls), 2)
        session.retry_failed()
        result = self.idle(session)
        self.assertEqual(len(calls), 3)
        self.assertEqual(result["authoritative"], ["Recovered English."])
        self.assertEqual(result["statuses"], ["corrected"])

    def test_disabled_corrections_do_not_schedule_failed_fast_translation_recovery(self):
        calls = []
        session = self.session(
            lambda request: calls.append(request) or corrected(request),
            config=SlowLaneConfig(enabled=False),
        )
        session.submit(["原文"], [None], draft_errors=["failure"])
        result = self.idle(session)

        self.assertEqual(calls, [])
        self.assertEqual(result["statuses"], ["failed"])
        self.assertTrue(result["segments"][0]["sealed"])
        self.assertEqual(result["segments"][0]["seal_reason"], "paused")
        session.set_enabled(True)
        self.assertEqual(self.idle(session)["metrics"]["requests"], 0)

    def test_english_correction_can_repair_mixed_language_fast_draft(self):
        session = self.session(lambda request: corrected(request, "That batch of Bragi from last Wednesday."))
        session.submit(["上禮拜三那批布拉吉"], ["That batch of 布拉吉 from last Wednesday."])
        result = self.idle(session)

        self.assertEqual(result["statuses"], ["corrected"])
        self.assertEqual(result["translations"], ["That batch of Bragi from last Wednesday."])
        self.assertEqual(result["authoritative"], result["translations"])
        self.assertEqual(result["segments"][0]["source_text"], "上禮拜三那批布拉吉")

    def test_any_malformed_entry_prevents_all_subtitle_writes(self):
        def invalid(request):
            response = corrected(request)
            response["corrections"][-1]["target_text"] = " "
            return response
        session = self.session(invalid)
        session.submit(["one", "two"], ["Draft one", "Draft two"])
        result = self.idle(session)
        self.assertEqual(result["translations"], ["Draft one", "Draft two"])
        self.assertEqual([record["version"] for record in result["segments"]], [1, 1])
        self.assertEqual(result["metrics"]["schema_rejections"], 4)
        self.assertFalse(any(event["type"] == "Correction" for event in result["events"]))

    def test_schema_whitelist_duplicate_confidence_and_stale_faults(self):
        def duplicate(request):
            response = corrected(request)
            response["no_change"] = confirmed(request)["no_change"]
            return response
        scenarios = [
            (lambda request: None, "schema_rejections"),
            (lambda request: {"corrections": [], "no_change": [], "extra": True}, "schema_rejections"),
            (duplicate, "schema_rejections"),
            (lambda request: corrected(request, change_type=["invented"]), "schema_rejections"),
            (lambda request: corrected(request, confidence=float("nan")), "schema_rejections"),
            (lambda request: corrected(request, confidence=True), "schema_rejections"),
            (lambda request: corrected(request, confidence=0.59), "low_confidence_rejections"),
            (lambda request: corrected(request, base_version=0), "stale_rejections"),
            (lambda request: corrected(request, segment_id="other-session:1"), "stale_rejections"),
        ]
        for correct, metric in scenarios:
            with self.subTest(metric=metric):
                session = self.session(correct)
                session.submit(["source"], ["draft"])
                result = self.idle(session)
                self.assertEqual(result["translations"], ["draft"])
                self.assertEqual(result["metrics"][metric], 2)
                self.assertEqual(result["statuses"], ["failed"])

    def test_dnt_guard_rejects_identifier_changes_and_unchanged_bad_drafts(self):
        for correct, draft in ((lambda request: corrected(request, "ETCH-08 chamber"), "ETCH-07 chamber"),
                               (confirmed, "ETCH-08 chamber")):
            with self.subTest(draft=draft):
                session = self.session(correct)
                session.submit(["ETCH-07 chamber"], [draft])
                result = self.idle(session)
                self.assertEqual(result["translations"], [draft])
                self.assertEqual(result["metrics"]["dnt_rejections"], 2)

    def test_accepted_correction_publishes_after_any_elapsed_duration(self):
        for elapsed in (0, 3600):
            with self.subTest(elapsed=elapsed):
                correct, started, release = self.blocked(corrected)
                session = self.session(correct)
                session.submit(["source"], ["draft"], timings=[{"start_s": 5000, "end_s": 5001, "speaker_id": "s1"}])
                self.assertTrue(started.wait(timeout=1))
                self.clock.advance(elapsed)
                release.set()
                result = self.idle(session)
                self.assertEqual(result["translations"], ["Corrected translation"])
                self.assertEqual(result["authoritative"], ["Corrected translation"])
                self.assertEqual(result["metrics"]["stale_rejections"], 0)
                self.assertEqual(result["segments"][0]["timing"]["speaker_id"], "s1")

    def test_glossary_learning_requires_evidence_in_accepted_corrections(self):
        def terminology(request):
            return corrected(request, "wafer ready", change_type=["terminology"],
                             term_pairs=[{"source": "晶圓", "target": "wafer"}, {"source": "absent", "target": "guess"}])
        session = self.session(terminology)
        session.submit(["晶圓一", "晶圓二"], ["draft one", "draft two"])
        self.wait_for(lambda: bool(session.snapshot()["learned_terms"]))
        result = self.idle(session)
        self.assertEqual(result["learned_terms"], [{"source": "晶圓", "target": "wafer"}])
        self.assertEqual(result["metrics"]["learned_terms"], 1)
        self.assertNotIn(("absent", "guess"), self.glossary.observations)


class WorkerTests(SlowLaneTestCase):
    def test_later_segment_publishes_while_earlier_call_is_blocked(self):
        calls = []
        first_started, second_finished, release = Event(), Event(), Event()
        self.releases.append(release)
        def review(request):
            calls.append(request)
            if request["segments"][0]["source_text"] == "First source":
                first_started.set()
                release.wait(timeout=5)
            else:
                second_finished.set()
            return corrected(request)
        session = self.session(review)
        sources, drafts = ["First source", "Second source"], ["First draft", "Second draft"]
        session.submit(sources[:1], drafts[:1])
        self.assertTrue(first_started.wait(timeout=1))
        session.submit(sources, drafts)
        self.assertTrue(second_finished.wait(timeout=1))
        self.wait_for(lambda: session.snapshot()["statuses"][1] == "corrected")
        result = session.snapshot()
        self.assertEqual(result["translations"], ["First draft", "Corrected translation"])
        self.assertEqual(result["authoritative"], [None, "Corrected translation"])
        self.assertEqual(result["pending"], 1)
        self.assertEqual(result["active_reviews"], 1)
        for _ in range(5):
            session.submit(sources, drafts)
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(len(request["segments"]) == 1 for request in calls))
        release.set()
        self.assertEqual(self.idle(session)["statuses"], ["corrected", "corrected"])

    def test_truncated_rows_each_get_one_fresh_review_with_same_cap(self):
        calls = []
        release = Event()
        self.releases.append(release)
        def review(request):
            calls.append(request)
            if "review_feedback" not in request:
                raise CorrectionProviderError("incomplete", incomplete_reason="max_output_tokens")
            release.wait(timeout=5)
            return corrected(request)
        session = self.session(review, config=SlowLaneConfig(reasoning_effort="high", max_output_tokens=4096))
        sources = [f"Source {index}" for index in range(4)]
        drafts = [f"English draft {index}" for index in range(4)]
        session.submit(sources, drafts)
        self.wait_for(lambda: len(calls) == 8)
        pending = session.snapshot()
        self.assertEqual(pending["pending"], 4)
        self.assertEqual(pending["active_reviews"], 4)
        self.assertEqual(pending["translations"], drafts)
        self.assertEqual(pending["authoritative"], [None] * 4)
        release.set()
        result = self.idle(session)

        self.assertEqual(len(calls), 8)
        self.assertEqual(result["statuses"], ["corrected"] * 4)
        self.assertEqual(result["metrics"]["automatic_retries"], 4)
        self.assertEqual(result["metrics"]["failure_reasons"], {"incomplete": 4})
        retries = [request for request in calls if "review_feedback" in request]
        self.assertCountEqual([request["segments"][0]["source_text"] for request in retries], sources)
        for request in retries:
            self.assertEqual(len(request["segments"]), 1)
            index = sources.index(request["segments"][0]["source_text"])
            self.assertEqual(request["segments"][0]["target_text"], drafts[index])
            self.assertEqual(request["review_feedback"], {"category": "output_budget"})
            self.assertEqual(request["config"]["max_output_tokens"], 4096)
            self.assertEqual(request["config"]["reasoning_effort"], "high")
        events = [event for event in result["events"] if event["type"] == "ReviewRetry"]
        self.assertEqual(len(events), 4)
        self.assertTrue(all(event["review_feedback"] == {"category": "output_budget"} for event in events))

    def test_repeated_truncation_stops_after_one_repair_and_later_rows_still_finish(self):
        calls = []
        def review(request):
            calls.append(request)
            if any(row["source_text"] == "First source" for row in request["segments"]):
                raise CorrectionProviderError("incomplete", incomplete_reason="max_output_tokens")
            return confirmed(request)
        session = self.session(review)
        sources, drafts = ["First source", "Second source"], ["First draft", "Second draft"]
        session.submit(sources, drafts)
        result = self.idle(session)
        self.assertEqual(len(calls), 3)
        self.assertEqual(result["statuses"], ["failed", "confirmed"])
        self.assertEqual(result["translations"], drafts)
        self.assertEqual(result["authoritative"], [None, "Second draft"])
        failed = result["segments"][0]
        self.assertEqual(failed["review_error"], {"category": "incomplete", "incomplete_reason": "max_output_tokens"})
        self.assertEqual(failed["repair_count"], 1)
        self.assertIn("output token limit", failed["error"])
        self.assertIn("reasoning", failed["error"])
        session.submit(sources, drafts)
        self.assertEqual(len(calls), 3)

    def test_truncation_and_validation_rejections_share_one_automatic_repair_budget(self):
        for truncate_first in (True, False):
            with self.subTest(truncate_first=truncate_first):
                calls = []
                def review(request):
                    calls.append(request)
                    if (len(calls) == 1) == truncate_first:
                        raise CorrectionProviderError("incomplete", incomplete_reason="max_output_tokens")
                    return corrected(request, confidence=0.53)
                session = self.session(review)
                session.submit(["source"], ["English draft"])
                result = self.idle(session)
                self.assertEqual(len(calls), 2)
                self.assertEqual(result["statuses"], ["failed"])
                self.assertEqual(result["authoritative"], [None])
                self.assertEqual(result["segments"][0]["repair_count"], 1)
                self.assertEqual(result["metrics"]["automatic_retries"], 1)
                self.assertEqual(result["metrics"]["failure_reasons"], {"incomplete": 1, "low_confidence": 1})

    def test_pending_draft_gap_becomes_context_without_blocking_later_ready_rows(self):
        calls = []
        session = self.session(lambda request: calls.append(request) or confirmed(request))
        sources = ["First phrase", "Intervening source clue", "Last phrase"]
        session.submit(sources, ["First draft", None, "Last draft"])
        result = self.idle(session)

        self.assertCountEqual([[row["source_text"] for row in request["segments"]] for request in calls],
                              [["First phrase"], ["Last phrase"]])
        first_request = next(request for request in calls if request["segments"][0]["source_text"] == "First phrase")
        clue = first_request["context"][0]
        self.assertEqual(clue["source_text"], "Intervening source clue")
        self.assertEqual(clue["context_position"], "after")
        self.assertEqual(clue["target_status"], "unavailable")
        self.assertTrue(clue["read_only"])
        self.assertEqual(result["statuses"], ["confirmed", "waiting", "confirmed"])

        session.submit(sources, ["First draft", "Delayed English clue", "Last draft"])
        result = self.idle(session)
        self.assertEqual(len(calls), 3)
        self.assertEqual(result["statuses"], ["confirmed"] * 3)

    def test_failed_review_gap_becomes_context_and_does_not_starve_ready_rows(self):
        calls = []
        def review(request):
            calls.append(request)
            if len(calls) == 1:
                raise RuntimeError("provider failure")
            return confirmed(request)
        session = self.session(review)
        sources = ["First phrase", "Intervening source clue", "Last phrase"]
        session.submit(sources, [None, "English clue draft", None])
        self.idle(session)
        session.submit(sources, ["First draft", "English clue draft", "Last draft"])
        result = self.idle(session)

        self.assertCountEqual([[row["source_text"] for row in request["segments"]] for request in calls[1:]],
                              [["First phrase"], ["Last phrase"]])
        first_request = next(request for request in calls if request["segments"][0]["source_text"] == "First phrase")
        clue = first_request["context"][0]
        self.assertEqual(clue["source_text"], "Intervening source clue")
        self.assertEqual(clue["context_position"], "after")
        self.assertEqual(clue["target_status"], "draft")
        self.assertEqual(result["statuses"], ["confirmed", "failed", "confirmed"])
        session.retry_failed()
        result = self.idle(session)
        self.assertEqual(len(calls), 4)
        self.assertEqual(result["statuses"], ["confirmed"] * 3)

    def test_filtered_gap_can_be_skipped_without_entering_correction_context(self):
        calls = []
        session = self.session(lambda request: calls.append(request) or confirmed(request))
        session.submit(["First phrase", "Background announcement", "Last phrase"],
                       ["First draft", "[Background speech filtered]", "Last draft"],
                       draft_filtered=[False, True, False])
        result = self.idle(session)
        self.assertEqual(len(calls), 2)
        self.assertCountEqual([request["segments"][0]["source_text"] for request in calls], ["First phrase", "Last phrase"])
        self.assertTrue(all(row["source_text"] != "Background announcement"
                            for request in calls for row in request["context"]))
        self.assertEqual(result["statuses"], ["confirmed", "filtered", "confirmed"])

    def test_single_row_review_retains_nearest_reviewed_context(self):
        calls = []
        session = self.session(lambda request: calls.append(request) or confirmed(request),
                               config=SlowLaneConfig(max_source_tokens=24))
        session.submit(["p" * 8], ["Reviewed prior English"])
        self.idle(session)
        prior = deepcopy(session.snapshot()["segments"][0])

        session.submit(["p" * 8, "a" * 12, "b" * 12],
                       ["Reviewed prior English", "First draft", "Second draft"])
        result = self.idle(session)

        request = next(request for request in calls if request["segments"][0]["source_text"] == "a" * 12)
        self.assertEqual([row["source_text"] for row in request["segments"]], ["a" * 12])
        self.assertEqual(request["context"][0]["source_text"], "p" * 8)
        self.assertEqual(request["context"][0]["target_text"], "Reviewed prior English")
        self.assertEqual(request["context"][0]["target_status"], "confirmed")
        self.assertEqual(request["context"][0]["context_position"], "before")
        self.assertTrue(request["context"][0]["read_only"])
        self.assertLessEqual(sum(len(row["source_text"].encode())
                                 for row in request["segments"] + request["context"]), 24)
        self.assertEqual(result["statuses"], ["confirmed"] * 3)
        self.assertEqual(result["segments"][0], prior)

    def test_following_source_context_is_bounded_even_while_fast_drafts_are_pending(self):
        calls = []
        session = self.session(lambda request: calls.append(request) or confirmed(request),
                               config=SlowLaneConfig())
        session.submit(["Current phrase", "Background announcement", "First later clue", "Second later clue", "Distant clue"],
                       ["Current draft", "[Background speech filtered]", None, None, None],
                       draft_filtered=[False, True, False, False, False])
        self.idle(session)
        self.assertEqual(len(calls), 1)
        request = calls[0]
        self.assertEqual([row["source_text"] for row in request["context"]], ["First later clue", "Second later clue"])
        self.assertTrue(all(row["context_position"] == "after" for row in request["context"]))
        self.assertTrue(all(row["target_status"] == "unavailable" and row["target_text"] is None
                            for row in request["context"]))
        self.assertTrue(all(row["read_only"] for row in request["context"]))
        self.assertTrue({row["segment_id"] for row in request["segments"]}.isdisjoint(
            row["segment_id"] for row in request["context"]))

    def test_context_is_chronological_and_identifies_reviewed_and_provisional_english(self):
        calls = []
        def review(request):
            calls.append(request)
            if request["segments"][0]["source_text"] in {"Older phrase", "Preceding phrase"}:
                return corrected(request, "Reviewed preceding English")
            return confirmed(request)
        session = self.session(review, config=SlowLaneConfig())
        session.submit(["Older phrase", "Preceding phrase"], ["Older draft", "Old draft"])
        self.idle(session)
        session.submit(["Older phrase", "Preceding phrase", "Current phrase", "Following phrase"],
                       ["Older draft", "Old draft", "Current draft", "Following draft"])
        self.idle(session)
        context = next(request for request in calls if request["segments"][0]["source_text"] == "Current phrase")["context"]
        self.assertEqual([row["source_text"] for row in context], ["Older phrase", "Preceding phrase", "Following phrase"])
        self.assertEqual([row["context_position"] for row in context], ["before", "before", "after"])
        self.assertEqual([row["target_status"] for row in context], ["corrected", "corrected", "draft"])
        self.assertEqual([row["target_text"] for row in context],
                         ["Reviewed preceding English", "Reviewed preceding English", "Following draft"])

    def test_context_reservation_never_starves_a_long_single_current_segment(self):
        calls = []
        session = self.session(lambda request: calls.append(request) or confirmed(request),
                               config=SlowLaneConfig(max_source_tokens=10))
        session.submit(["a" * 10, "b"], ["English draft", None])
        result = self.idle(session)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["context"], [])
        self.assertEqual(result["statuses"], ["confirmed", "waiting"])
        self.assertEqual(result["metrics"]["budget_rejections"], 0)

    def test_failed_source_fallback_remains_useful_context_without_claiming_english(self):
        calls = []
        def review(request):
            calls.append(request)
            if len(calls) == 1:
                raise RuntimeError("provider failure")
            return confirmed(request)
        session = self.session(review, config=SlowLaneConfig())
        session.submit(["先前原文"], [None], draft_errors=["fast failure"])
        self.idle(session)
        failed = deepcopy(session.snapshot()["segments"][0])
        session.submit(["先前原文", "Current phrase"], [None, "Current draft"])
        result = self.idle(session)
        context = calls[1]["context"][0]
        self.assertEqual(context["source_text"], "先前原文")
        self.assertEqual(context["target_status"], "unavailable")
        self.assertIsNone(context["target_text"])
        self.assertTrue(context["source_fallback"])
        self.assertEqual(result["segments"][0], failed)

    def test_typed_provider_errors_propagate_only_safe_diagnostics_without_auto_retry(self):
        failures = [
            CorrectionProviderError("incomplete", incomplete_reason="content_filter"),
            CorrectionProviderError("incomplete", incomplete_reason="max_messages"),
            CorrectionProviderError("incomplete", incomplete_reason="steered"),
            CorrectionProviderError("incomplete"),
            CorrectionProviderError("incomplete", incomplete_reason="unknown-private-value"),
            CorrectionProviderError("http_status", status_code=429),
            CorrectionProviderError("connection"),
            CorrectionProviderError("transport_timeout"),
            CorrectionProviderError("PRIVATE-RAW-ERROR", incomplete_reason="SECRET", status_code="SECRET"),
        ]
        for failure in failures:
            with self.subTest(category=failure.category):
                calls = []
                def correct(request):
                    calls.append(request)
                    raise failure
                session = self.session(correct)
                session.submit(["source"], ["English draft"])
                result = self.idle(session)
                self.assertEqual(len(calls), 1)
                self.assertEqual(result["segments"][0]["review_error"], failure.diagnostics())
                self.assertEqual(result["metrics"]["failure_reasons"], {failure.category: 1})
                self.assertEqual(result["translations"], ["English draft"])
                self.assertEqual(result["authoritative"], [None])
                self.assertNotIn("SECRET", json.dumps(result))
                self.assertNotIn("PRIVATE-RAW-ERROR", json.dumps(result))

    def test_failed_older_row_remains_retryable_after_later_reviews_succeed(self):
        calls = []
        retry = Event()
        def correct(request):
            calls.append(request)
            if request["segments"][0]["source_text"] == "one" and not retry.is_set():
                raise RuntimeError("request failed")
            return confirmed(request)
        session = self.session(correct, config=SlowLaneConfig())
        session.submit(["one", "two", "three"], ["First draft", "Second draft", "Third draft"])
        result = self.idle(session)
        self.assertEqual(result["statuses"], ["failed", "confirmed", "confirmed"])
        self.assertEqual(result["status"], "degraded")
        self.assertEqual(result["pending"], 0)
        self.assertEqual(len(calls), 3)
        retry.set()
        session.retry_failed()
        result = self.idle(session)
        self.assertEqual(result["statuses"], ["confirmed"] * 3)
        self.assertEqual(result["status"], "idle")
        self.assertEqual(len(calls), 4)

    def test_source_budget_splits_backlog_into_bounded_requests_without_dropping_rows(self):
        calls = []
        session = self.session(lambda request: calls.append(request) or confirmed(request),
                               config=SlowLaneConfig(max_source_tokens=6))
        sources = [f"s{index:02}" for index in range(9)]
        session.submit(sources, [f"draft {index}" for index in range(9)])
        result = self.idle(session)
        self.assertEqual(result["statuses"], ["confirmed"] * 9)
        self.assertCountEqual([item["source_text"] for request in calls for item in request["segments"]], sources)
        self.assertEqual(len(calls), 9)  # one editable source plus nearby evidence fits each request
        self.assertTrue(all(request["context"] for request in calls))
        for request in calls:
            self.assertLessEqual(sum(len(item["source_text"].encode("utf-8"))
                                     for item in request["segments"] + request["context"]), 6)

    def test_all_31_ready_rows_start_without_waiting_for_older_requests(self):
        release = Event()
        self.releases.append(release)
        calls = []
        def correct(request):
            calls.append(request)
            release.wait(timeout=5)
            return confirmed(request)
        session = self.session(correct)
        sources = [f"source {index}" for index in range(31)]
        drafts = [f"draft {index}" for index in range(31)]
        for length in range(1, 32):
            session.submit(sources[:length], drafts[:length])
        self.wait_for(lambda: len(calls) == 31)
        self.clock.advance(3600)
        pending = session.snapshot()
        self.assertEqual(pending["pending"], 31)
        self.assertEqual(pending["active_reviews"], 31)
        for _ in range(5):
            session.submit(sources, drafts)
        self.assertEqual(len(calls), 31)
        release.set()
        result = self.idle(session)
        self.wait_for(lambda: not session._calls)
        self.assertEqual(result["metrics"]["stale_rejections"], 0)
        self.assertEqual(result["statuses"], ["confirmed"] * 31)
        self.assertEqual(result["authoritative"], drafts)
        self.assertCountEqual([request["segments"][0]["source_text"] for request in calls], sources)
        self.assertTrue(all(len(request["segments"]) == 1 for request in calls))

    def test_source_budget_also_bounds_read_only_earlier_context(self):
        calls = []
        session = self.session(lambda request: calls.append(request) or confirmed(request),
                               config=SlowLaneConfig(enabled=False, max_source_tokens=16))
        sources = ["aa"] * 10
        drafts = ["draft"] * 10
        session.submit(sources, drafts)
        session.set_enabled(True)
        session.submit(sources + ["bb"] * 4, drafts + ["draft"] * 4)
        self.idle(session)
        request = calls[0]
        self.assertEqual(len(request["segments"]), 1)
        self.assertEqual(len(request["context"]), 7)
        self.assertTrue(all(item["read_only"] for item in request["context"]))
        self.assertLessEqual(sum(len(item["source_text"].encode()) for item in request["segments"] + request["context"]), 16)
        self.assertEqual(request["config"]["model"], "gpt-6-astra")
        self.assertIn("dnt_hits", request["segments"][0])

    def test_distant_audio_offsets_do_not_drop_segments_and_oversized_text_fails_once(self):
        calls = []
        session = self.session(lambda request: calls.append(request) or confirmed(request))
        session.submit(["one", "two", "three", "four"], ["a", "b", "c", "d"], timings=[
            {"start_s": start, "end_s": start + 10} for start in (0, 400, 800, 1200)
        ])
        self.idle(session)
        self.assertCountEqual([request["segments"][0]["source_text"] for request in calls], ["one", "two", "three", "four"])
        small_calls = []
        oversized = self.session(lambda request: small_calls.append(request) or confirmed(request),
                                  config=SlowLaneConfig(max_source_tokens=3))
        oversized.submit(["too long", "ok"], ["full draft", "okay"])
        result = self.idle(oversized)
        self.assertEqual(result["statuses"], ["failed", "confirmed"])
        self.assertEqual(result["translations"], ["full draft", "okay"])
        self.assertEqual(result["authoritative"], [None, "okay"])
        self.assertEqual(len(small_calls), 1)
        self.assertEqual(result["metrics"]["budget_rejections"], 1)
        self.clock.advance(3600)
        oversized.submit(["too long", "ok"], ["full draft", "okay"])
        self.assertEqual(self.idle(oversized)["metrics"]["budget_rejections"], 1)

    def test_error_details_stay_private_and_explicit_retry_recovers(self):
        attempts = []
        def correct(request):
            attempts.append(request)
            if len(attempts) == 1:
                raise RuntimeError("SECRET credential and source details")
            return corrected(request)
        session = self.session(correct)
        session.submit(["source"], ["draft"])
        result = self.idle(session)
        self.assertNotIn("SECRET", json.dumps(result))
        self.assertEqual(result["statuses"], ["failed"])
        self.clock.advance(3600)
        session.submit(["source"], ["draft"])
        self.assertEqual(len(attempts), 1)
        session.retry_failed()
        result = self.idle(session)
        self.assertEqual(result["statuses"], ["corrected"])
        self.assertEqual(len(attempts), 2)

    def test_slow_request_accepts_late_result_after_later_row_finishes(self):
        started, release = Event(), Event()
        self.releases.append(release)
        calls = []
        def correct(request):
            calls.append(request)
            if len(calls) == 1:
                started.set()
                release.wait(timeout=5)
                return corrected(request, "late response")
            return confirmed(request)
        session = self.session(correct)
        session.submit(["one"], ["draft one"])
        self.assertTrue(started.wait(timeout=1))
        self.clock.advance(3600)
        session.submit(["one", "two"], ["draft one", "draft two"])
        self.wait_for(lambda: session.snapshot()["statuses"][1] == "confirmed")
        self.assertEqual(session.snapshot()["pending"], 1)
        self.assertEqual(len(calls), 2)
        release.set()
        result = self.idle(session)
        self.assertEqual(result["statuses"], ["corrected", "confirmed"])
        self.assertEqual(result["translations"], ["late response", "draft two"])
        self.assertEqual(len(calls), 2)

    def test_stale_prefix_completion_preserves_later_source_records_and_metadata(self):
        calls = []
        session = self.session(lambda request: calls.append(request) or confirmed(request))
        sources = ["first", "second", "third"]
        session.submit(sources, [], timings=[{"speaker_id": "A"}, {"speaker_id": "B"}, {"speaker_id": "C"}])
        session.submit(sources[:1], ["First draft"], allow_prefix=True)
        self.idle(session)
        session.submit(sources[:2], [None, "Second draft"], allow_prefix=True)
        result = self.idle(session)
        self.assertEqual([row["source_text"] for row in result["segments"]], sources)
        self.assertEqual(result["statuses"], ["confirmed", "confirmed", "waiting"])
        self.assertEqual(result["segments"][2]["timing"], {"speaker_id": "C"})
        self.assertEqual(len(calls), 2)
        with self.assertRaisesRegex(ValueError, "append-only"):
            session.submit(sources[:1], ["First draft"])
        with self.assertRaisesRegex(ValueError, "append-only"):
            session.submit(["changed"], [None], allow_prefix=True)
        with self.assertRaisesRegex(ValueError, "fast draft"):
            session.submit(sources[:1], ["Changed draft"], allow_prefix=True)

    def test_late_fast_draft_starts_new_review_while_stale_recovery_is_still_running(self):
        for old_fails in (False, True):
            with self.subTest(old_fails=old_fails):
                old_started, old_release = Event(), Event()
                new_started, new_release = Event(), Event()
                self.releases.extend([old_release, new_release])
                calls = []
                def review(request):
                    calls.append(request)
                    if request["segments"][0]["source_fallback"]:
                        old_started.set()
                        old_release.wait(timeout=5)
                        if old_fails:
                            raise CorrectionProviderError("incomplete", incomplete_reason="max_output_tokens")
                        return corrected(request, "Obsolete source recovery")
                    new_started.set()
                    new_release.wait(timeout=5)
                    return corrected(request, "Reviewed new English draft")
                session = self.session(review)
                session.submit(["source"], [None], draft_errors=["fast failure"])
                self.assertTrue(old_started.wait(timeout=1))
                session.submit(["source"], ["New English draft"])
                self.assertTrue(new_started.wait(timeout=1))
                old_release.set()
                self.wait_for(lambda: len(session._calls) == 1)
                frozen = session.snapshot()
                self.assertEqual(frozen["translations"], ["New English draft"])
                self.assertEqual(frozen["active_reviews"], 1)
                self.assertEqual(frozen["pending"], 1)
                self.assertEqual(frozen["metrics"]["automatic_retries"], 0)
                for _ in range(5):
                    session.submit(["source"], ["New English draft"])
                self.assertEqual(len(calls), 2)
                new_release.set()
                self.assertEqual(self.idle(session)["translations"], ["Reviewed new English draft"])

    def test_close_and_pause_ignore_multiple_outcomes_without_automatic_retries(self):
        for action in ("close", "pause"):
            with self.subTest(action=action):
                release = Event()
                self.releases.append(release)
                calls = []
                def review(request):
                    calls.append(request)
                    release.wait(timeout=5)
                    if request["segments"][0]["source_text"] == "second":
                        raise CorrectionProviderError("incomplete", incomplete_reason="max_output_tokens")
                    return corrected(request)
                session = self.session(review)
                session.submit(["first", "second"], ["First draft", "Second draft"])
                self.wait_for(lambda: len(calls) == 2)
                session.close() if action == "close" else session.set_enabled(False)
                frozen = session.snapshot()["segments"]
                self.assertEqual(session.snapshot()["pending"], 0)
                self.assertEqual(session.snapshot()["active_reviews"], 0)
                release.set()
                self.wait_for(lambda: not session._calls)
                result = session.snapshot()
                self.assertEqual(result["segments"], frozen)
                self.assertEqual(result["metrics"]["automatic_retries"], 0)
                self.assertEqual(result["metrics"]["cancelled"], 2)
                self.assertEqual(len(calls), 2)

    def test_capacity_waiters_drain_fairly_across_sessions_without_polling(self):
        release = Event()
        self.releases.append(release)
        calls = []
        def review(request):
            source = request["segments"][0]["source_text"]
            calls.append(source)
            if source == "held":
                release.wait(timeout=5)
            return confirmed(request)
        with patch.object(slow_lane, "_MAX_ACTIVE_REVIEWS", 1):
            held = self.session(review)
            held.submit(["held"], ["Held draft"])
            self.wait_for(lambda: calls == ["held"])
            first = self.session(review)
            first.submit(["first one", "first two"], ["Draft one", "Draft two"])
            second = self.session(review)
            second.submit(["second one"], ["Other draft"])
            self.assertEqual(first.snapshot()["pending"], 2)
            self.assertEqual(first.snapshot()["active_reviews"], 0)
            release.set()
            self.idle(held)
            self.idle(first)
            self.idle(second)
            self.wait_for(lambda: not held._calls and not first._calls and not second._calls)
            self.assertEqual(calls, ["held", "first one", "second one", "first two"])
            self.assertEqual(slow_lane._ACTIVE_REVIEW_CALLS, 0)
            self.assertFalse(slow_lane._REVIEW_WAITERS)

    def test_thread_creation_or_start_failure_releases_slot_and_is_safely_retryable(self):
        for failure_site in ("constructor", "start"):
            with self.subTest(failure_site=failure_site), patch.object(slow_lane, "_MAX_ACTIVE_REVIEWS", 1):
                session = self.session()
                target = "src.slow_lane.Thread" if failure_site == "constructor" else "src.slow_lane.Thread.start"
                with patch(target, side_effect=RuntimeError("SECRET native thread error")):
                    session.submit(["first"], ["First draft"])
                failed = session.snapshot()
                self.assertEqual(failed["statuses"], ["failed"])
                self.assertNotIn("SECRET", json.dumps(failed))
                self.assertFalse(session._calls)
                self.assertEqual(slow_lane._ACTIVE_REVIEW_CALLS, 0)
                following = self.session()
                following.submit(["second"], ["Second draft"])
                self.assertEqual(self.idle(following)["statuses"], ["confirmed"])
                session.retry_failed()
                self.assertEqual(self.idle(session)["statuses"], ["confirmed"])
                self.wait_for(lambda: not session._calls and not following._calls)
                self.assertEqual(slow_lane._ACTIVE_REVIEW_CALLS, 0)

    def test_completion_observer_threads_remain_inside_the_process_capacity_limit(self):
        observer_started, release = Event(), Event()
        self.releases.append(release)
        def observer():
            observer_started.set()
            release.wait(timeout=5)
        with patch.object(slow_lane, "_MAX_ACTIVE_REVIEWS", 1):
            first = self.session(on_update=observer)
            first.submit(["first"], ["First draft"])
            self.assertTrue(observer_started.wait(timeout=1))
            later = self.session()
            later.submit(["later"], ["Later draft"])
            self.assertEqual(later.snapshot()["active_reviews"], 0)
            self.assertEqual(later.snapshot()["pending"], 1)
            self.assertEqual(slow_lane._ACTIVE_REVIEW_CALLS, 1)
            release.set()
            self.assertEqual(self.idle(later)["statuses"], ["confirmed"])
            self.wait_for(lambda: not first._calls and not later._calls)
            self.assertEqual(slow_lane._ACTIVE_REVIEW_CALLS, 0)

    def test_canceled_calls_retain_process_slots_until_return_and_queue_drains_without_polling(self):
        release = Event()
        self.releases.append(release)
        calls = []
        def correct(request):
            calls.append(request)
            if request["segments"][0]["source_text"].startswith("old"):
                release.wait(timeout=5)
            return confirmed(request)
        with patch.object(slow_lane, "_MAX_ACTIVE_REVIEWS", 2):
            session = self.session(correct)
            session.submit(["old one", "old two"], ["Old draft one", "Old draft two"])
            self.wait_for(lambda: len(calls) == 2)
            session.set_enabled(False)
            session.set_enabled(True)
            sources = ["old one", "old two", "new one", "new two"]
            drafts = ["Old draft one", "Old draft two", "New draft one", "New draft two"]
            session.submit(sources, drafts)
            queued = session.snapshot()
            self.assertEqual(queued["pending"], 2)
            self.assertEqual(queued["active_reviews"], 0)
            self.assertEqual(len(calls), 2)
            self.assertEqual(len(session._calls), 2)
            other = self.session(correct)
            other.submit(["other session"], ["Other draft"])
            self.assertEqual(other.snapshot()["active_reviews"], 0)
            release.set()
            self.assertEqual(self.idle(session)["statuses"], ["paused", "paused", "confirmed", "confirmed"])
            self.assertEqual(self.idle(other)["statuses"], ["confirmed"])
            self.wait_for(lambda: not session._calls and not other._calls)
            self.assertEqual(len(calls), 5)
            self.assertEqual(session.snapshot()["metrics"]["cancelled"], 2)
            self.assertEqual(slow_lane._ACTIVE_REVIEW_CALLS, 0)
            self.assertFalse(slow_lane._REVIEW_WAITERS)


if __name__ == "__main__":
    unittest.main()
