"""Test ordered translation and client configuration without making API requests."""

from collections import Counter
from contextlib import contextmanager
import json
from threading import Event
import unittest
from unittest.mock import Mock, patch

from src.translation import (
    BACKGROUND_FILTERED_TEXT, FAILED_MESSAGE, TranslationSession, _FilteredBackgroundTranslation,
)


@contextmanager
def mocked_default(**kwargs):
    translator = Mock(**kwargs)
    with patch("src.translation.create_default_translator", return_value=translator):
        yield translator


class SessionTestCase(unittest.TestCase):
    def make_session(self, translate=None, **kwargs):
        session = TranslationSession(translate, **kwargs)
        self.addCleanup(self.cleanup_session, session)
        return session

    def cleanup_session(self, session):
        session.close()
        self.join_worker(session)

    def join_worker(self, session):
        with session._lock:
            worker = session._worker
        if worker is not None:
            worker.join(timeout=5)
            self.assertFalse(worker.is_alive(), "translation worker did not finish")
        self.assertEqual(session.snapshot()["pending"], 0)


class TranslationSessionTests(SessionTestCase):
    def test_injected_reasoning_markup_is_never_published_or_used_as_context(self):
        calls = []
        def translate(text, *, context=None):
            calls.append(context)
            return "<think>Private reasoning</think>English answer" if text == "one" else "Valid English"
        session = self.make_session(translate)
        session.submit(["one", "two"])
        self.join_worker(session)
        result = session.snapshot()
        self.assertEqual(result["translations"], [None, "Valid English"])
        self.assertEqual(result["errors"], [FAILED_MESSAGE, None])
        self.assertEqual(calls[1]["previous"], [])
        self.assertNotIn("Private reasoning", str(result))

    def test_injected_contextual_adapter_gets_previous_text_and_owned_close_outside_lock(self):
        calls, closed = [], []
        glossary = Mock()
        glossary.retrieve.return_value = [{"term_src": "晶圓", "term_tgt": "wafer"}]
        glossary.dnt_hits.return_value = []
        glossary.compare_dnt.return_value = {"ok": True}
        class Adapter:
            def __call__(self, text, *, context=None):
                calls.append((text, context))
                return "English " + text

            def close(self):
                closed.append(session.snapshot())  # Must not hold session._lock.
        session = self.make_session(Adapter(), glossary=glossary, owned_client=True)
        session.submit(["one", "two", "three", "four"])
        self.join_worker(session)
        self.assertEqual([row["target_text"] for row in calls[-1][1]["previous"]], ["English two", "English three"])
        self.assertEqual(calls[-1][1]["glossary"], glossary.retrieve.return_value)
        self.assertEqual(calls[-1][1]["dnt_hits"], [])
        session.close()
        session.close()
        self.assertEqual(len(closed), 1)

    def test_default_client_is_created_once_and_owned_but_injected_unowned_client_is_not_closed(self):
        with mocked_default(return_value="English") as adapter:
            session = self.make_session()
            session.submit(["one"])
            self.join_worker(session)
            session.submit(["one", "two"])
            self.join_worker(session)
            session.close()
            adapter.close.assert_called_once_with()
            self.assertEqual(adapter.call_count, 2)
        external = Mock(return_value="English")
        unowned = self.make_session(external, owned_client=False)
        unowned.submit(["source"])
        self.join_worker(unowned)
        unowned.close()
        external.close.assert_not_called()

    def test_owned_adapter_close_cancels_inflight_and_discards_its_late_result(self):
        started, released = Event(), Event()
        closed = []
        class Adapter:
            def __call__(self, text, *, context=None):
                started.set()
                released.wait(timeout=5)
                return "Late English"

            def close(self):
                closed.append(True)
                released.set()
        observer = Mock()
        session = self.make_session(Adapter(), owned_client=True, on_result=observer)
        self.addCleanup(released.set)
        session.submit(["one", "two"])
        self.assertTrue(started.wait(1))
        session.close()
        self.join_worker(session)
        self.assertEqual(closed, [True])
        self.assertEqual(session.snapshot()["translations"], [None, None])
        observer.assert_not_called()

    def test_thread_constructor_or_start_failure_is_safe_observed_and_retryable(self):
        for failure_site in ("constructor", "start"):
            with self.subTest(failure_site=failure_site):
                observed = []
                session = self.make_session(lambda text: "English " + text,
                                            on_result=lambda texts, result: observed.append(result))
                target = "src.translation.Thread" if failure_site == "constructor" else "src.translation.Thread.start"
                with patch(target, side_effect=RuntimeError("PRIVATE thread resource failure")):
                    session.submit(["one", "two"])
                result = session.snapshot()
                self.assertIsNone(session._worker)
                self.assertEqual(result["pending"], 0)
                self.assertEqual(result["errors"], [FAILED_MESSAGE, FAILED_MESSAGE])
                self.assertNotIn("PRIVATE", str(observed))
                self.assertEqual(len(observed), 1)
                session.retry_failed()
                self.join_worker(session)
                self.assertEqual(session.snapshot()["translations"], ["English one", "English two"])
                self.assertEqual(session.snapshot()["errors"], [None, None])

    def test_matching_older_ui_prefix_after_callback_submission_does_not_replace_or_duplicate_work(self):
        calls = []
        session = self.make_session(lambda text: calls.append(text) or "English " + text)
        # The ASR callback commits the newer cumulative transcript before its
        # Future becomes visible to the UI's older completed-transcript snapshot.
        session.submit(["first", "second"])
        self.join_worker(session)
        session.submit(["first"], allow_prefix=True)
        session.submit([], allow_prefix=True)
        self.join_worker(session)
        self.assertEqual(calls, ["first", "second"])
        self.assertEqual(session.snapshot()["translations"], ["English first", "English second"])
        with self.assertRaisesRegex(ValueError, "append-only"):
            session.submit(["changed"], allow_prefix=True)
        with self.assertRaisesRegex(ValueError, "append-only"):
            session.submit(["first"])

    def test_completion_observer_runs_without_polling_or_holding_translation_lock(self):
        observed = []
        completed = Event()

        def on_result(texts, result):
            # Reentrant snapshot access would deadlock if the observer held the lock.
            self.assertEqual(session.snapshot()["translations"], result["translations"])
            observed.append((texts, result))
            completed.set()

        session = self.make_session(lambda text: "English " + text, on_result=on_result)
        session.submit(["first"])
        self.assertTrue(completed.wait(2))
        self.join_worker(session)
        session.submit(["first"])
        self.join_worker(session)
        self.assertEqual(len(observed), 1)
        texts, result = observed[0]
        self.assertEqual(texts, ["first"])
        self.assertEqual(result, {"translations": ["English first"], "errors": [None],
                                  "filtered": [False], "pending": 0})
        texts.append("Changed observer source")
        result["translations"][0] = "Changed observer result"
        self.assertEqual(session.snapshot()["translations"], ["English first"])
        session.submit(["first", "second"])
        self.join_worker(session)
        self.assertEqual(observed[1][0], ["first", "second"])

    def test_completion_observer_receives_failures_filters_and_retries(self):
        observed = []
        with mocked_default( side_effect=[
            RuntimeError("private provider detail"), _FilteredBackgroundTranslation(), "Recovered English.",
        ]):
            session = self.make_session(on_result=lambda texts, result: observed.append((texts, result)))
            session.submit(["失敗來源", "背景語音"])
            self.join_worker(session)
            self.assertEqual(observed[0][1]["errors"], [FAILED_MESSAGE, None])
            self.assertEqual(observed[1][1]["filtered"], [False, True])
            self.assertNotIn("private provider detail", str(observed))
            session.retry_failed()
            self.join_worker(session)
            self.assertEqual(len(observed), 3)
            self.assertEqual(observed[-1][1]["translations"], ["Recovered English.", BACKGROUND_FILTERED_TEXT])
            self.assertEqual(observed[-1][1]["errors"], [None, None])

    def test_observer_failure_preserves_fast_results_and_queue_progress(self):
        observer = Mock(side_effect=RuntimeError("private observer detail"))
        session = self.make_session(lambda text: "English " + text, on_result=observer)
        session.submit(["first", "second"])
        self.join_worker(session)
        self.assertEqual(observer.call_count, 2)
        self.assertEqual(session.snapshot()["translations"], ["English first", "English second"])
        self.assertEqual(session.snapshot()["errors"], [None, None])

    def test_close_suppresses_completion_observer_for_late_result(self):
        started, release = Event(), Event()
        observer = Mock()

        def translate(text):
            started.set()
            release.wait()
            return "Late English."

        session = self.make_session(translate, on_result=observer)
        self.addCleanup(release.set)
        session.submit(["source"])
        self.assertTrue(started.wait(2))
        session.close()
        release.set()
        self.join_worker(session)
        observer.assert_not_called()

    def test_filtered_background_is_successful_ordered_and_excluded_from_main_context(self):
        glossary = Mock()
        glossary.retrieve.return_value = []
        glossary.dnt_hits.return_value = []
        glossary.compare_dnt.return_value = {"ok": True}
        with mocked_default( side_effect=[
            "The process is stable.", _FilteredBackgroundTranslation(), RuntimeError("private error"),
            "The next batch is ready.", "Recovered speech.",
        ]) as translate:
            session = self.make_session(glossary=glossary)
            texts = ["製程穩定", "背景閒聊", "故障句", "下一批好了"]
            session.submit(texts)
            self.join_worker(session)

            snapshot = session.snapshot()
            self.assertEqual(snapshot["filtered"], [False, True, False, False])
            self.assertEqual(snapshot["translations"], [
                "The process is stable.", "[Background speech filtered]", None, "The next batch is ready.",
            ])
            self.assertEqual(snapshot["errors"], [None, None, FAILED_MESSAGE, None])
            self.assertEqual(translate.call_args_list[3].kwargs["context"]["previous"], [
                {"source_text": texts[0], "target_text": "The process is stable."},
            ])
            self.assertEqual([call.args[0] for call in translate.call_args_list], texts)
            snapshot["filtered"][1] = False
            self.assertTrue(session.snapshot()["filtered"][1])
            session.retry_failed()
            self.join_worker(session)
            self.assertEqual(translate.call_count, 5)
            self.assertEqual(translate.call_args.args, (texts[2],))
            self.assertEqual(session.snapshot()["filtered"], [False, True, False, False])

    def test_custom_provider_marker_is_not_interpreted_as_filtered_background(self):
        session = self.make_session(lambda text: "[Background speech filtered]")
        session.submit(["source"])
        self.join_worker(session)

        self.assertEqual(session.snapshot()["translations"], ["[Background speech filtered]"])
        self.assertEqual(session.snapshot()["filtered"], [False])

    def test_contextual_fast_lane_uses_only_previous_two_segments_and_local_glossary(self):
        glossary = Mock()
        glossary.retrieve.return_value = [{"term_src": "腔體", "term_tgt": "chamber"}]
        glossary.dnt_hits.return_value = ["ETCH-07"]
        glossary.compare_dnt.return_value = {"ok": True}
        with mocked_default(
                   side_effect=lambda text, **kwargs: f"English ETCH-07 sentence {text[-1]}") as translate:
            session = self.make_session(glossary=glossary, source_lang="zh-TW+en", target_lang="en")
            texts = [f"ETCH-07 句{i}" for i in range(4)]
            session.submit(texts)
            self.join_worker(session)

        self.assertEqual(translate.call_count, 4)
        for index, call in enumerate(translate.call_args_list):
            self.assertEqual(call.args, (texts[index],))
            context = call.kwargs["context"]
            self.assertEqual(set(context), {"previous", "source_lang", "target_lang", "glossary", "dnt_hits"})
            self.assertEqual(context["previous"], [
                {"source_text": texts[i], "target_text": f"English ETCH-07 sentence {i}"}
                for i in range(max(0, index - 2), index)
            ])
            self.assertEqual(context["source_lang"], "zh-TW+en")
            self.assertEqual(context["target_lang"], "en")
            self.assertEqual(context["glossary"], glossary.retrieve.return_value)
            self.assertEqual(context["dnt_hits"], ["ETCH-07"])
            self.assertEqual(glossary.retrieve.call_args_list[index].args, (texts[index],))
            self.assertEqual(glossary.retrieve.call_args_list[index].kwargs, {"limit": 40})
        self.assertEqual(session.snapshot()["errors"], [None] * 4)

    def test_contextual_worker_rejects_changed_or_added_identifiers_without_leaking_details(self):
        glossary = Mock()
        glossary.retrieve.return_value = []
        glossary.dnt_hits.return_value = ["ETCH-07"]
        glossary.compare_dnt.return_value = {"ok": False, "details": "private glossary details"}
        with mocked_default( return_value="Check ETCH-070."):
            session = self.make_session(glossary=glossary)
            session.submit(["檢查 ETCH-07"])
            self.join_worker(session)

        glossary.compare_dnt.assert_called_once_with("檢查 ETCH-07", "Check ETCH-070.")
        self.assertEqual(session.snapshot(), {
            "translations": [None], "errors": [FAILED_MESSAGE], "filtered": [False], "pending": 0,
        })

    def test_custom_provider_keeps_its_simple_text_only_interface(self):
        glossary = Mock()
        translate = Mock(return_value="Local English")
        session = self.make_session(translate, glossary=glossary)

        session.submit(["source"])
        self.join_worker(session)

        translate.assert_called_once_with("source")
        glossary.retrieve.assert_not_called()
        glossary.compare_dnt.assert_not_called()

    def test_any_provider_mixed_language_output_is_rejected_without_exposing_it(self):
        for output in ("The 工序 is complete.", "Return to the 会議 room.", "PRIVATE 한글"):
            with self.subTest(output=output):
                session = self.make_session(lambda text: output)
                session.submit(["PRIVATE SOURCE"])
                self.join_worker(session)

                self.assertEqual(session.snapshot(), {
                    "translations": [None], "errors": [FAILED_MESSAGE], "filtered": [False], "pending": 0,
                })
                self.assertNotIn("PRIVATE", json.dumps(session.snapshot()))

    def test_providers_progress_independently_and_keep_their_own_errors(self):
        started, release = Event(), Event()

        def wait_for_cloud(text):
            started.set()
            release.wait(timeout=5)
            raise RuntimeError("private provider details")

        cloud = TranslationSession(wait_for_cloud, failure_message="Cloud unavailable.")
        self.addCleanup(self.cleanup_session, cloud)
        self.addCleanup(release.set)
        local = self.make_session(lambda text: "Local English")
        cloud.submit(["source text"])
        self.assertTrue(started.wait(timeout=2))
        local.submit(["source text"])
        self.join_worker(local)

        self.assertEqual(local.snapshot()["translations"], ["Local English"])
        self.assertEqual(cloud.snapshot()["pending"], 1)
        release.set()
        self.join_worker(cloud)
        self.assertEqual(cloud.snapshot()["errors"], ["Cloud unavailable."])
        self.assertEqual(local.snapshot()["errors"], [None])

    def test_cumulative_submissions_are_idempotent_ordered_and_nonblocking(self):
        started, release = Event(), Event()
        calls = []

        def translate(text):
            calls.append(text)
            if len(calls) == 1:
                started.set()
                release.wait(timeout=5)
            return f"English: {text}"

        session = self.make_session(translate)
        self.addCleanup(release.set)
        texts = ["first", "first"]
        session.submit(texts)
        self.assertTrue(started.wait(timeout=2))
        session.submit(texts)
        session.submit([*texts, "third"])

        self.assertEqual(session.snapshot(), {
            "translations": [None, None, None], "errors": [None, None, None], "filtered": [False] * 3, "pending": 3,
        })
        self.assertEqual(calls, ["first"])
        self.assertEqual(texts, ["first", "first"])
        release.set()
        self.join_worker(session)

        self.assertEqual(calls, ["first", "first", "third"])
        self.assertEqual(session.snapshot()["translations"], [
            "English: first", "English: first", "English: third",
        ])

    def test_worker_exits_when_idle_and_restarts_for_new_segments(self):
        translate = Mock(side_effect=lambda text: text.upper())
        session = self.make_session(translate)
        session.submit([])
        self.assertIsNone(session._worker)

        session.submit(["one"])
        self.join_worker(session)
        self.assertIsNone(session._worker)
        session.submit(["one", "two"])
        self.join_worker(session)

        self.assertIsNone(session._worker)
        self.assertEqual(session.snapshot()["translations"], ["ONE", "TWO"])
        self.assertEqual(translate.call_count, 2)

    def test_failure_keeps_source_positions_and_requires_explicit_retry(self):
        attempts = Counter()
        retry_started, release_retry = Event(), Event()

        def translate(text):
            attempts[text] += 1
            if text == "one":
                if attempts[text] == 1:
                    raise RuntimeError("private request details must not be shown")
                retry_started.set()
                release_retry.wait(timeout=5)
            return text.upper()

        session = self.make_session(translate)
        self.addCleanup(release_retry.set)
        session.submit(["one", "two"])
        self.join_worker(session)
        self.assertEqual(session.snapshot(), {
            "translations": [None, "TWO"], "errors": [FAILED_MESSAGE, None], "filtered": [False] * 2, "pending": 0,
        })
        session.submit(["one", "two"])
        self.assertEqual(attempts, {"one": 1, "two": 1})

        session.retry_failed()
        self.assertTrue(retry_started.wait(timeout=2))
        session.retry_failed()
        self.assertEqual(session.snapshot()["pending"], 1)
        self.assertEqual(session.snapshot()["errors"], [None, None])
        release_retry.set()
        self.join_worker(session)

        self.assertEqual(attempts, {"one": 2, "two": 1})
        self.assertEqual(session.snapshot()["translations"], ["ONE", "TWO"])

    def test_blank_or_invalid_translator_output_is_a_safe_segment_error(self):
        translate = Mock(side_effect=["  ", None, 42, " good result "])
        session = self.make_session(translate)

        session.submit(["one", "two", "three", "four"])
        self.join_worker(session)

        self.assertEqual(session.snapshot(), {
            "translations": [None, None, None, "good result"],
            "errors": [FAILED_MESSAGE, FAILED_MESSAGE, FAILED_MESSAGE, None], "filtered": [False] * 4, "pending": 0,
        })

    def test_snapshots_do_not_expose_mutable_session_lists(self):
        session = self.make_session(str.upper)
        session.submit(["one"])
        self.join_worker(session)

        snapshot = session.snapshot()
        snapshot["translations"][0] = "edited externally"
        snapshot["errors"].append("external error")

        self.assertEqual(session.snapshot(), {
            "translations": ["ONE"], "errors": [None], "filtered": [False], "pending": 0,
        })

    def test_close_cancels_waiting_work_and_discards_the_active_result(self):
        started, release = Event(), Event()
        calls = []

        def translate(text):
            calls.append(text)
            started.set()
            release.wait(timeout=5)
            return "old result"

        session = self.make_session(translate)
        self.addCleanup(release.set)
        session.submit(["active", "waiting"])
        self.assertTrue(started.wait(timeout=2))
        session.close()
        session.close()
        session.submit(["active", "waiting", "new"])
        session.retry_failed()
        self.assertEqual(session.snapshot()["pending"], 0)

        replacement = self.make_session(str.upper)
        replacement.submit(["new recording"])
        self.join_worker(replacement)
        release.set()
        self.join_worker(session)

        self.assertEqual(calls, ["active"])
        self.assertEqual(session.snapshot()["translations"], [None, None])
        self.assertEqual(replacement.snapshot()["translations"], ["NEW RECORDING"])




if __name__ == "__main__":
    unittest.main()
