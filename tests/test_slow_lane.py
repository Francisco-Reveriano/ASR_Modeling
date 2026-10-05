"""Deterministic slow-lane faults without model, audio, or network dependencies."""

from copy import deepcopy
from dataclasses import replace
import json
import re
from threading import Event
import time
import unittest

from src.slow_lane import SlowLaneConfig, SlowLaneSession


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
            worker = session._worker
            if worker is not None:
                worker.join(timeout=2)
                self.assertFalse(worker.is_alive(), "drain worker did not stop")
            for _, thread in session._calls:
                thread.join(timeout=2)
                self.assertFalse(thread.is_alive(), "injected request was not released")

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
        self.assertEqual((config.model, config.reasoning_effort, config.window_n, config.queue_depth),
                         ("gpt-6-astra", "medium", 4, 2))
        with self.assertRaises(AttributeError):
            config.window_n = 5

    def test_invalid_limits_types_and_horizons_are_rejected(self):
        for options in ({"window_n": 9}, {"queue_depth": 0}, {"queue_depth": True},
                        {"earlier_context_n": 101}, {"max_source_tokens": 100_001},
                        {"request_timeout_s": float("nan")}, {"seal_timeout_s": 4},
                        {"confidence_threshold": -0.1}, {"enabled": "yes"}, {"model": " "},
                        {"reasoning_effort": "unbounded"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                SlowLaneConfig(**options)

    def test_supported_ui_choices_can_exceed_defaults_and_display_horizon_can_exceed_seal(self):
        config = SlowLaneConfig(window_n=8, max_output_tokens=16_384, revision_horizon_s=120,
                                seal_timeout_s=5, request_timeout_s=60, confidence_threshold=0)
        self.assertEqual(config.revision_horizon_s, 120)


class StoreTests(SlowLaneTestCase):
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

    def test_commit_timeout_seals_draft_and_late_output_cannot_change_it(self):
        correct, started, release = self.blocked(corrected)
        session = self.session(correct)
        session.submit(["source"], ["draft"])
        self.assertTrue(started.wait(timeout=1))
        self.clock.advance(30)
        result = session.snapshot()
        self.assertEqual(result["statuses"], ["timeout"])
        self.assertEqual(result["authoritative"], ["draft"])
        frozen = deepcopy(result["segments"][0])
        release.set()
        self.idle(session)
        self.assertEqual(session.snapshot()["segments"][0], frozen)

    def test_late_correction_seals_authoritative_version_without_replacing_screen(self):
        correct, started, release = self.blocked(corrected)
        session = self.session(correct, config=SlowLaneConfig(request_timeout_s=30))
        session.submit(["source"], ["screen draft"])
        self.assertTrue(started.wait(timeout=1))
        self.clock.advance(21)
        release.set()
        result = self.idle(session)
        self.assertEqual(result["translations"], ["screen draft"])
        self.assertEqual(result["authoritative"], ["Corrected translation"])
        self.assertEqual(result["statuses"], ["corrected"])
        self.assertEqual(result["segments"][0]["screen_version"], 1)
        self.assertEqual(result["segments"][0]["version"], 2)
        self.assertEqual(result["segments"][0]["replacements"], 0)
        self.assertFalse(result["segments"][0]["history"][-1]["screen_updated"])

    def test_missing_fast_draft_seals_source_at_original_commit_deadline(self):
        session = self.session()
        session.submit(["source"], [None])
        self.clock.advance(30)
        result = session.snapshot()
        self.assertEqual(result["translations"], ["source"])
        self.assertEqual(result["authoritative"], ["source"])
        self.assertEqual(result["statuses"], ["failed"])
        self.assertEqual(result["segments"][0]["seal_reason"], "timeout")
        self.assertTrue(result["segments"][0]["source_fallback"])

    def test_late_fast_draft_does_not_restart_the_commit_timeout(self):
        correct, started, release = self.blocked(corrected)
        session = self.session(correct)
        session.submit(["source"], [None])
        self.clock.advance(25)
        session.submit(["source"], ["late fast draft"])
        self.assertTrue(started.wait(timeout=1))
        self.clock.advance(5)
        result = session.snapshot()
        self.assertEqual(result["authoritative"], ["late fast draft"])
        self.assertEqual(result["segments"][0]["seal_reason"], "timeout")
        release.set()

    def test_live_endpoint_sets_display_age_and_timing_stays_audit_only(self):
        calls = []
        self.clock.advance(21)
        session = self.session(lambda request: calls.append(request) or corrected(request))
        timing = {"start_s": 9000, "end_s": 9001, "server_endpoint_ms": 0,
                  "asr_start_ms": 1000, "asr_final_ms": 20000, "t_capture_ms": None,
                  "speaker_id": "speaker-1"}
        session.submit(["source"], ["screen draft"], timings=[timing])
        result = self.idle(session)
        self.assertEqual(result["translations"], ["screen draft"])
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

    def test_late_timing_enrichment_never_restarts_deadline_and_drift_is_rejected(self):
        session = self.session()
        session.submit(["source"], [None], timings=[{"start_s": 0, "end_s": 1}])
        for changed in ({"start_s": 1, "end_s": 2}, {"start": 1, "end": 2}):
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "cannot change"):
                session.submit(["source"], [None], timings=[changed])
        self.clock.advance(25)
        session.submit(["source"], [None], timings=[{"server_endpoint_ms": 25_000, "speaker_id": "speaker-1"}])
        self.assertEqual(session.snapshot()["segments"][0]["deadline_origin_at"], 0)
        with self.assertRaisesRegex(ValueError, "cannot change"):
            session.submit(["source"], [None], timings=[{"server_endpoint_ms": 26_000}])
        self.clock.advance(5)
        result = session.snapshot()
        self.assertEqual(result["authoritative"], ["source"])
        self.assertEqual(result["segments"][0]["seal_reason"], "timeout")

    def test_explicit_fast_failure_has_a_sealable_honest_source_fallback(self):
        session = self.session()
        session.submit(["原文"], [None], draft_errors=["private fast error details"])
        result = session.snapshot()
        self.assertEqual(result["translations"], ["原文"])
        self.assertEqual(result["statuses"], ["failed"])
        self.assertTrue(result["segments"][0]["source_fallback"])
        self.assertEqual(result["segments"][0]["version"], 1)
        self.assertNotIn("private fast error", json.dumps(result))
        self.clock.advance(30)
        self.assertEqual(session.snapshot()["authoritative"], ["原文"])
        session.submit(["原文"], ["Late successful English"])
        self.assertEqual(session.snapshot()["translations"], ["原文"])

    def test_fast_retry_before_sealing_can_replace_source_fallback_then_be_corrected(self):
        session = self.session(corrected)
        session.submit(["source"], [None], draft_errors=["failure"])
        session.submit(["source"], ["Fast English"])
        result = self.idle(session)
        self.assertEqual(result["segments"][0]["version"], 3)
        self.assertEqual(result["segments"][0]["replacements"], 2)
        self.assertFalse(result["segments"][0]["source_fallback"])
        self.assertEqual(result["metrics"]["screen_replacements"], 2)

    def test_correction_retry_can_recover_source_fallback_without_mislabeling_either_view(self):
        for age in (0, 21):
            with self.subTest(age=age):
                session = self.session(corrected)
                session.submit(["原文"], [None], draft_errors=["failure"])
                self.clock.advance(age)
                session.retry_failed()
                result = self.idle(session)
                record = result["segments"][0]
                self.assertEqual(result["authoritative"], ["Corrected translation"])
                self.assertEqual(result["statuses"], ["corrected"])
                self.assertFalse(record["source_fallback"])
                self.assertEqual(record["screen_source_fallback"], age == 21)
                self.assertEqual(result["translations"], ["原文" if age == 21 else "Corrected translation"])
                self.assertTrue(record["history"][0]["source_fallback"])

    def test_late_successful_fast_retry_keeps_existing_screen_pointer_frozen(self):
        session = self.session()
        session.submit(["原文"], [None], draft_errors=["failure"])
        self.clock.advance(21)
        session.submit(["原文"], ["Successful fast English"])
        result = self.idle(session)
        record = result["segments"][0]
        self.assertEqual(result["translations"], ["原文"])
        self.assertEqual(result["authoritative"], ["Successful fast English"])
        self.assertEqual((record["screen_version"], record["version"]), (1, 2))
        self.assertTrue(record["screen_source_fallback"])
        self.assertFalse(record["source_fallback"])
        self.assertEqual(record["replacements"], 0)

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
                self.wait_for(lambda: session._worker is None)
                self.assertEqual(session.snapshot()["segments"][0], frozen)


class ValidationTests(SlowLaneTestCase):
    def test_any_malformed_entry_prevents_all_subtitle_writes(self):
        def invalid(request):
            response = corrected(request)
            response["corrections"][1]["target_text"] = " "
            return response
        session = self.session(invalid)
        session.submit(["one", "two"], ["Draft one", "Draft two"])
        result = self.idle(session)
        self.assertEqual(result["translations"], ["Draft one", "Draft two"])
        self.assertEqual([record["version"] for record in result["segments"]], [1, 1])
        self.assertEqual(result["metrics"]["schema_rejections"], 1)
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
                self.assertEqual(result["metrics"][metric], 1)
                self.assertEqual(result["statuses"], ["failed"])

    def test_dnt_guard_rejects_identifier_changes_and_unchanged_bad_drafts(self):
        for correct, draft in ((lambda request: corrected(request, "ETCH-08 chamber"), "ETCH-07 chamber"),
                               (confirmed, "ETCH-08 chamber")):
            with self.subTest(draft=draft):
                session = self.session(correct)
                session.submit(["ETCH-07 chamber"], [draft])
                result = self.idle(session)
                self.assertEqual(result["translations"], [draft])
                self.assertEqual(result["metrics"]["dnt_rejections"], 1)

    def test_revision_horizon_freezes_screen_not_authoritative_and_uses_monotonic_time(self):
        correct, started, release = self.blocked(corrected)
        session = self.session(correct, config=SlowLaneConfig(revision_horizon_s=1))
        session.submit(["source"], ["draft"], timings=[{"start_s": 5000, "end_s": 5001, "speaker_id": "s1"}])
        self.assertTrue(started.wait(timeout=1))
        self.clock.advance(2)
        release.set()
        result = self.idle(session)
        self.assertEqual(result["translations"], ["draft"])
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
    def test_queue_replaces_oldest_waiting_windows_and_bounds_pending_work(self):
        started, release = Event(), Event()
        self.releases.append(release)
        calls = []
        def correct(request):
            calls.append(request)
            if len(calls) == 1:
                started.set()
                release.wait(timeout=5)
            return confirmed(request)
        session = self.session(correct)
        session.submit(["source 1"], ["draft 1"])
        self.assertTrue(started.wait(timeout=1))
        for length in range(2, 7):
            session.submit([f"source {index}" for index in range(1, length + 1)],
                           [f"draft {index}" for index in range(1, length + 1)])
        result = session.snapshot()
        self.assertEqual(result["pending"], 3)  # one active plus two queued windows
        self.assertEqual(result["metrics"]["queue_dropped"], 3)
        release.set()
        result = self.idle(session)
        self.assertEqual(result["metrics"]["stale_rejections"], 0)
        self.assertEqual(result["statuses"][0], "confirmed")
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(len(request["segments"]) <= 4 for request in calls))

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
        self.assertEqual(len(request["segments"]), 4)
        self.assertEqual(len(request["context"]), 4)
        self.assertTrue(all(item["read_only"] for item in request["context"]))
        self.assertLessEqual(sum(len(item["source_text"].encode()) for item in request["segments"] + request["context"]), 16)
        self.assertEqual(request["config"]["model"], "gpt-6-astra")
        self.assertIn("dnt_hits", request["segments"][0])

    def test_audio_window_and_oversized_individual_segments_are_not_truncated(self):
        calls = []
        session = self.session(lambda request: calls.append(request) or confirmed(request))
        session.submit(["one", "two", "three", "four"], ["a", "b", "c", "d"], timings=[
            {"start_s": start, "end_s": start + 10} for start in (0, 40, 80, 120)
        ])
        self.idle(session)
        self.assertEqual([item["source_text"] for item in calls[0]["segments"]], ["two", "three", "four"])
        oversized = self.session(lambda request: self.fail("oversized text must not be dispatched"),
                                  config=SlowLaneConfig(max_source_tokens=3))
        oversized.submit(["too long"], ["full draft"])
        self.assertEqual(oversized.snapshot()["pending"], 0)
        self.assertEqual(oversized.snapshot()["translations"], ["full draft"])
        self.clock.advance(30)
        self.assertEqual(oversized.snapshot()["authoritative"], ["full draft"])

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
        session.submit(["source"], ["draft"])
        self.assertEqual(len(attempts), 1)
        session.retry_failed()
        result = self.idle(session)
        self.assertEqual(result["statuses"], ["corrected"])
        self.assertEqual(len(attempts), 2)

    def test_timeout_unblocks_drain_worker_and_late_response_is_discarded(self):
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
        session = self.session(correct, config=SlowLaneConfig(request_timeout_s=1))
        session.submit(["one"], ["draft one"])
        self.assertTrue(started.wait(timeout=1))
        self.clock.advance(2)
        result = self.idle(session)
        self.assertEqual(result["metrics"]["timeouts"], 1)
        session.submit(["one", "two"], ["draft one", "draft two"])
        result = self.idle(session)
        self.assertEqual(result["statuses"], ["timeout", "confirmed"])
        release.set()
        self.wait_for(lambda: session._calls[0][0].is_set())
        self.assertEqual(session.snapshot()["translations"], ["draft one", "draft two"])

    def test_ignored_timeouts_cannot_spawn_unbounded_request_threads(self):
        starts = [Event(), Event()]
        release = Event()
        self.releases.append(release)
        calls = []
        def correct(request):
            calls.append(request)
            starts[len(calls) - 1].set()
            release.wait(timeout=5)
            return confirmed(request)
        session = self.session(correct, config=SlowLaneConfig(request_timeout_s=1))
        for index in range(2):
            session.submit([f"source {number}" for number in range(index + 1)],
                           [f"draft {number}" for number in range(index + 1)])
            self.assertTrue(starts[index].wait(timeout=1))
            self.clock.advance(2)
            self.idle(session)
        session.submit(["source 0", "source 1", "source 2"], ["draft 0", "draft 1", "draft 2"])
        result = self.idle(session)
        self.assertEqual(len(calls), 2)
        self.assertEqual(result["metrics"]["timeouts"], 2)
        self.assertEqual(result["metrics"]["errors"], 1)
        self.assertEqual(result["status"], "degraded")


if __name__ == "__main__":
    unittest.main()
