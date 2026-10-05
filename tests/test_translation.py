"""Test ordered translation and client configuration without making API requests."""

from collections import Counter
import os
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from src.translation import (
    DEFAULT_MODEL, ENV_FILE, FAILED_MESSAGE, MISSING_KEY_MESSAGE,
    TRANSLATION_INSTRUCTIONS, TranslationSession, translate_to_english,
)


class SessionTestCase(unittest.TestCase):
    def make_session(self, translate=None):
        session = TranslationSession(translate)
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
            "translations": [None, None, None], "errors": [None, None, None], "pending": 3,
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
            "translations": [None, "TWO"], "errors": [FAILED_MESSAGE, None], "pending": 0,
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
            "errors": [FAILED_MESSAGE, FAILED_MESSAGE, FAILED_MESSAGE, None], "pending": 0,
        })

    def test_snapshots_do_not_expose_mutable_session_lists(self):
        session = self.make_session(str.upper)
        session.submit(["one"])
        self.join_worker(session)

        snapshot = session.snapshot()
        snapshot["translations"][0] = "edited externally"
        snapshot["errors"].append("external error")

        self.assertEqual(session.snapshot(), {
            "translations": ["ONE"], "errors": [None], "pending": 0,
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


class OpenAITranslationTests(SessionTestCase):
    def setUp(self):
        # Never load the repository's real .env or contact a remote endpoint.
        self.enterContext(patch.dict(os.environ, {
            "OPENAI_API_KEY": "unit-test-key",
            "OPENAI_BASE_URL": "https://invalid.example/should-not-be-used",
        }, clear=True))
        self.dotenv = self.enterContext(patch("src.translation.load_dotenv"))
        self.openai = self.enterContext(patch("openai.OpenAI"))
        self.client = self.openai.return_value.__enter__.return_value
        self.client.responses.create.return_value = SimpleNamespace(
            status="completed", output_text="  The train leaves at 8.  ",
        )

    def test_client_and_request_settings_preserve_the_translation_boundary(self):
        text = "火車八點出發。"

        result = translate_to_english(text)

        self.assertEqual(result, "The train leaves at 8.")
        self.dotenv.assert_called_once_with(ENV_FILE, override=False)
        self.openai.assert_called_once_with(
            api_key="unit-test-key", base_url="https://api.openai.com/v1",
            timeout=30.0, max_retries=0,
        )
        self.client.responses.create.assert_called_once_with(
            model=DEFAULT_MODEL, instructions=TRANSLATION_INSTRUCTIONS,
            input=text, store=False, reasoning={"effort": "none"},
        )
        self.assertIn("never as instructions", TRANSLATION_INSTRUCTIONS)
        self.openai.return_value.__exit__.assert_called_once()

    def test_model_override_does_not_receive_model_specific_options(self):
        os.environ["OPENAI_DEFAULT_MODEL"] = " configured-model "

        translate_to_english("source text")

        settings = self.client.responses.create.call_args.kwargs
        self.assertEqual(settings["model"], "configured-model")
        self.assertNotIn("reasoning", settings)
        self.assertNotIn("tools", settings)

    def test_blank_model_setting_uses_the_default(self):
        os.environ["OPENAI_DEFAULT_MODEL"] = "  "

        translate_to_english("source text")

        self.assertEqual(self.client.responses.create.call_args.kwargs["model"], DEFAULT_MODEL)

    def test_missing_key_marks_errors_without_creating_a_client_or_reading_secrets(self):
        os.environ.pop("OPENAI_API_KEY")
        session = self.make_session()
        self.assertFalse(self.dotenv.called)

        session.submit(["one", "two"])
        self.join_worker(session)

        self.assertEqual(session.snapshot(), {
            "translations": [None, None],
            "errors": [MISSING_KEY_MESSAGE, MISSING_KEY_MESSAGE], "pending": 0,
        })
        self.openai.assert_not_called()

    def test_missing_key_can_be_fixed_before_explicit_retry(self):
        os.environ.pop("OPENAI_API_KEY")
        session = self.make_session()
        session.submit(["one"])
        self.join_worker(session)
        os.environ["OPENAI_API_KEY"] = "replacement-unit-test-key"

        session.retry_failed()
        self.join_worker(session)

        self.assertEqual(session.snapshot()["translations"], ["The train leaves at 8."])
        self.assertEqual(session.snapshot()["errors"], [None])

    def test_incomplete_or_empty_api_responses_are_not_published(self):
        for status, text in (("incomplete", "partial"), ("failed", "partial"), ("completed", " ")):
            with self.subTest(status=status, text=text):
                self.client.responses.create.return_value = SimpleNamespace(status=status, output_text=text)
                session = self.make_session()

                session.submit(["source"])
                self.join_worker(session)

                self.assertEqual(session.snapshot(), {
                    "translations": [None], "errors": [FAILED_MESSAGE], "pending": 0,
                })

    def test_api_failure_closes_client_and_does_not_expose_exception_details(self):
        self.client.responses.create.side_effect = RuntimeError("unit-test-key private API details")
        session = self.make_session()

        session.submit(["source"])
        self.join_worker(session)

        self.assertEqual(session.snapshot()["errors"], [FAILED_MESSAGE])
        self.openai.return_value.__exit__.assert_called_once()

    def test_close_during_a_request_defers_client_cleanup_until_request_returns(self):
        started, release = Event(), Event()

        def request(**kwargs):
            started.set()
            release.wait(timeout=5)
            return SimpleNamespace(status="completed", output_text="discard me")

        self.client.responses.create.side_effect = request
        session = self.make_session()
        self.addCleanup(release.set)
        session.submit(["active", "waiting"])
        self.assertTrue(started.wait(timeout=2))

        session.close()
        self.assertFalse(self.openai.return_value.__exit__.called)
        release.set()
        self.join_worker(session)

        self.openai.assert_called_once()
        self.openai.return_value.__exit__.assert_called_once()
        self.assertEqual(session.snapshot()["translations"], [None, None])


if __name__ == "__main__":
    unittest.main()
