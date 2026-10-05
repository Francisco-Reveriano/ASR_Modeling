"""Test ordered translation and client configuration without making API requests."""

from collections import Counter
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from dotenv import load_dotenv

from src.translation import (
    BACKGROUND_FILTER_INSTRUCTIONS, BACKGROUND_FILTERED_TEXT,
    DEFAULT_MODEL, ENV_FILE, FAILED_MESSAGE, MISSING_KEY_MESSAGE,
    TRANSLATION_INSTRUCTIONS, TranslationSession, _FilteredBackgroundTranslation, translate_to_english,
)


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
        with patch("src.translation.translate_to_english", side_effect=[
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
        with patch("src.translation.translate_to_english", side_effect=[
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
        with patch("src.translation.translate_to_english",
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
        with patch("src.translation.translate_to_english", return_value="Check ETCH-070."):
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

    def test_background_filter_prompt_is_conservative_and_uses_only_main_text_context(self):
        translate_to_english("目前這批資料", context={
            "previous": [
                {"source_text": "製程資料", "target_text": "Process data.", "speaker_id": "PRIVATE SPEAKER"},
                {"source_text": "背景", "target_text": "[Background speech filtered]", "filtered": True},
                {"source_text": "失敗", "target_text": None},
            ],
            "speaker_id": "PRIVATE SPEAKER", "reference": "PRIVATE REFERENCE", "audio": "PRIVATE AUDIO",
        })

        request = self.client.responses.create.call_args.kwargs
        instructions = request["instructions"]
        self.assertIn("clearly unrelated background", instructions)
        for safeguard in ("topic changes", "technical fragments", "brief replies", "uncertain", "speaker"):
            self.assertIn(safeguard, instructions)
        payload = json.loads(request["input"])
        self.assertEqual(payload["current_source"], "目前這批資料")
        self.assertEqual(payload["previous"], [{"source_text": "製程資料", "target_text": "Process data."}])
        self.assertNotIn("PRIVATE", request["input"])

    def test_entire_background_is_an_explicit_success_not_translation_failure(self):
        self.client.responses.create.return_value.output_text = "[[NO_MAIN_CONVERSATION_SPEECH]]"
        session = self.make_session()
        session.submit(["背景交談"])
        self.join_worker(session)

        self.assertEqual(session.snapshot(), {
            "translations": ["[Background speech filtered]"], "filtered": [True], "errors": [None], "pending": 0,
        })
        self.client.responses.create.assert_called_once()

    def test_literal_display_marker_stays_unfiltered_and_remains_main_context(self):
        os.environ["OPENAI_FILTER_BACKGROUND_SPEECH"] = "false"
        self.client.responses.create.side_effect = [
            SimpleNamespace(status="completed", output_text=BACKGROUND_FILTERED_TEXT),
            SimpleNamespace(status="completed", output_text="Continue."),
        ]
        session = self.make_session()
        session.submit([BACKGROUND_FILTERED_TEXT, "Continue."])
        self.join_worker(session)

        self.assertEqual(session.snapshot()["filtered"], [False, False])
        self.assertEqual(session.snapshot()["translations"], [BACKGROUND_FILTERED_TEXT, "Continue."])
        payload = json.loads(self.client.responses.create.call_args.kwargs["input"])
        self.assertEqual(payload["previous"], [{"source_text": BACKGROUND_FILTERED_TEXT, "target_text": BACKGROUND_FILTERED_TEXT}])

    def test_closed_session_discards_late_filtered_outcome(self):
        started, release = Event(), Event()
        self.addCleanup(release.set)

        def delayed_response(**kwargs):
            started.set()
            release.wait(timeout=5)
            return SimpleNamespace(status="completed", output_text="[[NO_MAIN_CONVERSATION_SPEECH]]")

        self.client.responses.create.side_effect = delayed_response
        session = self.make_session()
        session.submit(["background source"])
        self.assertTrue(started.wait(timeout=2))
        session.close()
        release.set()
        self.join_worker(session)

        self.assertEqual(session.snapshot(), {
            "translations": [None], "filtered": [False], "errors": [None], "pending": 0,
        })

    def test_filter_setting_can_disable_filtering_and_rejects_invalid_values_safely(self):
        for value in (" false ", "0", "NO", "off"):
            with self.subTest(value=value):
                os.environ["OPENAI_FILTER_BACKGROUND_SPEECH"] = value
                translate_to_english("source")
                self.assertNotIn("clearly unrelated background", self.client.responses.create.call_args.kwargs["instructions"])
        os.environ["OPENAI_FILTER_BACKGROUND_SPEECH"] = "PRIVATE-invalid-setting"
        self.openai.reset_mock()
        with self.assertRaisesRegex(ValueError, "OPENAI_FILTER_BACKGROUND_SPEECH") as raised:
            translate_to_english("source")
        self.assertNotIn("PRIVATE", str(raised.exception))
        self.openai.assert_not_called()

    def test_disabled_filter_never_accepts_a_background_sentinel(self):
        os.environ["OPENAI_FILTER_BACKGROUND_SPEECH"] = "false"
        self.client.responses.create.return_value.output_text = "[[NO_MAIN_CONVERSATION_SPEECH]]"

        with self.assertRaises(ValueError):
            translate_to_english("source")

        self.client.responses.create.assert_called_once()

    def test_background_filter_dotenv_setting_respects_existing_environment(self):
        self.dotenv.side_effect = load_dotenv
        with TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text("OPENAI_FILTER_BACKGROUND_SPEECH=false\n", encoding="utf-8")
            with patch("src.translation.ENV_FILE", env_file):
                translate_to_english("source")
                self.assertNotIn("clearly unrelated background", self.client.responses.create.call_args.kwargs["instructions"])
                os.environ["OPENAI_FILTER_BACKGROUND_SPEECH"] = "true"
                translate_to_english("source")
                self.assertIn("clearly unrelated background", self.client.responses.create.call_args.kwargs["instructions"])

    def test_protected_identifiers_prevent_whole_segment_filtering_with_one_retention_retry(self):
        self.client.responses.create.side_effect = [
            SimpleNamespace(status="completed", output_text="[[NO_MAIN_CONVERSATION_SPEECH]]"),
            SimpleNamespace(status="completed", output_text="Keep ETCH-07 running."),
        ]

        self.assertEqual(translate_to_english("維持 ETCH-07", context={"dnt_hits": ["ETCH-07"]}),
                         "Keep ETCH-07 running.")

        calls = self.client.responses.create.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].kwargs["input"], calls[1].kwargs["input"])
        self.assertIn("protected identifiers", calls[1].kwargs["instructions"])
        self.assertEqual(calls[1].kwargs["max_output_tokens"], 512)
        self.assertIs(calls[1].kwargs["store"], False)

    def test_repeated_filtering_of_protected_identifiers_is_a_safe_error(self):
        self.client.responses.create.return_value.output_text = "[[NO_MAIN_CONVERSATION_SPEECH]]"

        with self.assertRaisesRegex(ValueError, "protected identifier"):
            translate_to_english("維持 ETCH-07", context={"dnt_hits": ["ETCH-07"]})

        self.assertEqual(self.client.responses.create.call_count, 2)

    def test_superscript_identifier_change_gets_one_repair_using_exact_source(self):
        text = "R2 接近 0.7"
        self.client.responses.create.side_effect = [
            SimpleNamespace(status="completed", output_text="The R² value is close to 0.7."),
            SimpleNamespace(status="completed", output_text="The R2 value is close to 0.7."),
        ]

        self.assertEqual(translate_to_english(text, context={"dnt_hits": ["R2"]}), "The R2 value is close to 0.7.")

        calls = self.client.responses.create.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].kwargs["input"], calls[1].kwargs["input"])
        self.assertEqual(json.loads(calls[1].kwargs["input"])["current_source"], text)
        self.assertIn("exact spelling, case, punctuation, and occurrence count", calls[1].kwargs["instructions"])

    def test_identifier_repair_preserves_counts_and_case_and_rejects_invented_ids(self):
        valid = "Check ETCH-07, ETCH-07, and R2."
        for invalid in ("Check ETCH-07 and R2.", "Check etch-07, ETCH-07, and R2.",
                        "Check ETCH-07, ETCH-07, R2, and ETCH-08."):
            with self.subTest(invalid=invalid):
                self.client.responses.create.reset_mock()
                self.client.responses.create.side_effect = [
                    SimpleNamespace(status="completed", output_text=invalid),
                    SimpleNamespace(status="completed", output_text=valid),
                ]

                self.assertEqual(translate_to_english("檢查 ETCH-07 ETCH-07 R2", context={"dnt_hits": ["R2"]}), valid)
                self.assertEqual(self.client.responses.create.call_count, 2)

    def test_hyphenated_array_dimensions_get_one_repair_without_explicit_dnt_hits(self):
        text = "這是八乘四的陣列。"
        self.client.responses.create.side_effect = [
            SimpleNamespace(status="completed", output_text="This is an 8-by-4 array."),
            SimpleNamespace(status="completed", output_text="This is an 8 by 4 array."),
        ]

        result = translate_to_english(text, context={"dnt_hits": []})

        self.assertEqual(result, "This is an 8 by 4 array.")
        calls = self.client.responses.create.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].kwargs["input"], calls[1].kwargs["input"])
        payload = json.loads(calls[1].kwargs["input"])
        self.assertEqual(payload["current_source"], text)
        self.assertEqual(payload["dnt_hits"], [])
        self.assertIn("Do not introduce identifiers absent from the source", calls[1].kwargs["instructions"])
        self.assertEqual(calls[1].kwargs["max_output_tokens"], 512)
        self.assertIs(calls[1].kwargs["store"], False)

    def test_repeated_identifier_mismatch_stays_an_error_after_single_repair(self):
        self.client.responses.create.return_value.output_text = "Check ETCH-08."

        with self.assertRaisesRegex(ValueError, "protected identifier"):
            translate_to_english("檢查 ETCH-07")

        self.assertEqual(self.client.responses.create.call_count, 2)

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
            model=DEFAULT_MODEL, instructions=TRANSLATION_INSTRUCTIONS + BACKGROUND_FILTER_INSTRUCTIONS,
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

    def test_configured_reasoning_is_sent_for_default_and_custom_models(self):
        for model in (DEFAULT_MODEL, "configured-model"):
            for effort in ("none", "minimal", "low", "medium", "high", "xhigh", "max"):
                with self.subTest(model=model, effort=effort):
                    os.environ["OPENAI_DEFAULT_MODEL"] = model
                    os.environ["OPENAI_DEFAULT_REASONING_EFFORT"] = effort

                    translate_to_english("source text", context={})

                    settings = self.client.responses.create.call_args.kwargs
                    self.assertEqual(settings["model"], model)
                    self.assertEqual(settings["reasoning"], {"effort": effort})
                    self.assertEqual(settings["max_output_tokens"], 512)
                    self.assertIs(settings["store"], False)

    def test_reasoning_setting_is_trimmed_and_case_normalized(self):
        os.environ["OPENAI_DEFAULT_REASONING_EFFORT"] = "  LoW \t"

        translate_to_english("source text")

        self.assertEqual(self.client.responses.create.call_args.kwargs["reasoning"], {"effort": "low"})

    def test_missing_or_blank_reasoning_preserves_model_defaults(self):
        for model in (DEFAULT_MODEL, "configured-model"):
            for effort in (None, "", " \t "):
                with self.subTest(model=model, effort=effort):
                    os.environ["OPENAI_DEFAULT_MODEL"] = model
                    if effort is None:
                        os.environ.pop("OPENAI_DEFAULT_REASONING_EFFORT", None)
                    else:
                        os.environ["OPENAI_DEFAULT_REASONING_EFFORT"] = effort

                    translate_to_english("source text")

                    settings = self.client.responses.create.call_args.kwargs
                    if model == DEFAULT_MODEL:
                        self.assertEqual(settings["reasoning"], {"effort": "none"})
                    else:
                        self.assertNotIn("reasoning", settings)

    def test_reasoning_loads_from_dotenv_and_existing_environment_takes_precedence(self):
        self.dotenv.side_effect = load_dotenv
        with TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text("OPENAI_DEFAULT_REASONING_EFFORT=medium\n", encoding="utf-8")
            with patch("src.translation.ENV_FILE", env_file):
                translate_to_english("source text")
                self.assertEqual(self.client.responses.create.call_args.kwargs["reasoning"], {"effort": "medium"})

                os.environ["OPENAI_DEFAULT_REASONING_EFFORT"] = "low"
                translate_to_english("source text")
                self.assertEqual(self.client.responses.create.call_args.kwargs["reasoning"], {"effort": "low"})

    def test_invalid_reasoning_fails_before_client_creation_without_echoing_value(self):
        os.environ["OPENAI_DEFAULT_REASONING_EFFORT"] = "private-invalid-value"

        with self.assertRaisesRegex(ValueError, "OPENAI_DEFAULT_REASONING_EFFORT") as caught:
            translate_to_english("source text")

        self.assertNotIn("private-invalid-value", str(caught.exception))
        self.openai.assert_not_called()

        session = self.make_session()
        session.submit(["source text"])
        self.join_worker(session)

        self.assertEqual(session.snapshot(), {
            "translations": [None], "errors": [FAILED_MESSAGE], "filtered": [False], "pending": 0,
        })
        self.openai.assert_not_called()

    def test_fast_reasoning_is_independent_of_correction_reasoning(self):
        os.environ["OPENAI_CORRECTION_REASONING_EFFORT"] = "max"
        os.environ["OPENAI_DEFAULT_REASONING_EFFORT"] = "low"

        translate_to_english("source text")

        self.assertEqual(self.client.responses.create.call_args.kwargs["reasoning"], {"effort": "low"})
        os.environ.pop("OPENAI_DEFAULT_REASONING_EFFORT")

        translate_to_english("source text")

        self.assertEqual(self.client.responses.create.call_args.kwargs["reasoning"], {"effort": "none"})

    def test_context_request_bounds_and_allowlists_its_data_without_replacing_source(self):
        text = "檢查 ETCH-07。"
        context = {
            "current_source": "WRONG-SOURCE",
            "evaluation": "PRIVATE-REFERENCE", "audio": "PRIVATE-AUDIO",
            "source_lang": "zh-TW+en", "target_lang": "en", "dnt_hits": ["ETCH-07"],
            "previous": [
                {"source_text": f"source-{i}", "target_text": f"target-{i}", "reference": "PRIVATE-REFERENCE"}
                for i in range(4)
            ],
            "glossary": [
                {"entry_id": str(i), "term_src": f"term-{i}", "term_tgt": f"word-{i}",
                 "reference": "PRIVATE-REFERENCE", "audio": "PRIVATE-AUDIO"}
                for i in range(45)
            ],
        }
        self.client.responses.create.return_value = SimpleNamespace(status="completed", output_text="Check ETCH-07.")

        self.assertEqual(translate_to_english(text, context=context), "Check ETCH-07.")

        settings = self.client.responses.create.call_args.kwargs
        payload = json.loads(settings["input"])
        self.assertEqual(payload["current_source"], text)
        self.assertEqual(payload["previous"], [
            {"source_text": "source-2", "target_text": "target-2"},
            {"source_text": "source-3", "target_text": "target-3"},
        ])
        self.assertEqual(len(payload["glossary"]), 40)
        self.assertEqual(payload["dnt_hits"], ["ETCH-07"])
        self.assertNotIn("PRIVATE", settings["input"])
        self.assertNotIn("WRONG-SOURCE", settings["input"])
        self.assertEqual(settings["max_output_tokens"], 512)
        self.assertIs(settings["store"], False)
        self.assertIn("Translate only current_source", settings["instructions"])

    def test_context_adapter_rejects_missing_case_changed_or_punctuation_changed_dnt(self):
        for output in ("Check the chamber.", "Check etch-07.", "Check ETCH 07."):
            with self.subTest(output=output):
                self.client.responses.create.return_value = SimpleNamespace(status="completed", output_text=output)

                with self.assertRaisesRegex(ValueError, "protected identifier"):
                    translate_to_english("檢查 ETCH-07", context={"dnt_hits": ["ETCH-07"]})

    def test_mixed_language_response_gets_one_bounded_repair_from_original_context(self):
        os.environ["OPENAI_DEFAULT_REASONING_EFFORT"] = "low"
        text = "檢查 ETCH-07 工序。"
        context = {
            "dnt_hits": ["ETCH-07"], "target_lang": "en",
            "previous": [{"source_text": "上一句", "target_text": "Previous sentence."}],
            "reference": "PRIVATE REFERENCE", "audio": "PRIVATE AUDIO",
        }
        self.client.responses.create.side_effect = [
            SimpleNamespace(status="completed", output_text="Check ETCH-07 工序."),
            SimpleNamespace(status="completed", output_text="Check the ETCH-07 process."),
        ]

        self.assertEqual(translate_to_english(text, context=context), "Check the ETCH-07 process.")

        self.assertEqual(self.client.responses.create.call_count, 2)
        first, repaired = [call.kwargs for call in self.client.responses.create.call_args_list]
        self.assertEqual({key: value for key, value in first.items() if key != "instructions"},
                         {key: value for key, value in repaired.items() if key != "instructions"})
        self.assertEqual(json.loads(repaired["input"])["current_source"], text)
        self.assertNotIn("PRIVATE", repaired["input"])
        self.assertNotIn("Check ETCH-07 工序.", repaired["input"])
        self.assertIn("English", first["instructions"])
        self.assertIn("Latin", first["instructions"])
        self.assertNotEqual(first["instructions"], repaired["instructions"])
        self.assertEqual(repaired["reasoning"], {"effort": "low"})
        self.assertEqual(repaired["max_output_tokens"], 512)
        self.assertIs(repaired["store"], False)
        self.openai.assert_called_once()
        self.assertEqual(self.openai.call_args.kwargs["max_retries"], 0)
        self.openai.return_value.__exit__.assert_called_once()

    def test_plain_mixed_language_response_also_has_a_bounded_repair(self):
        self.client.responses.create.side_effect = [
            SimpleNamespace(status="completed", output_text="Welcome 王先生."),
            SimpleNamespace(status="completed", output_text="Welcome Mr. Wang."),
        ]

        self.assertEqual(translate_to_english("歡迎王先生。"), "Welcome Mr. Wang.")

        repair = self.client.responses.create.call_args.kwargs
        self.assertEqual(repair["input"], "歡迎王先生。")
        self.assertEqual(repair["max_output_tokens"], 512)

    def test_failed_language_repair_is_not_published_or_retried_again(self):
        self.client.responses.create.side_effect = [
            SimpleNamespace(status="completed", output_text="PRIVATE 工序"),
            SimpleNamespace(status="completed", output_text="Still PRIVATE 工序"),
        ]
        session = self.make_session()

        session.submit(["PRIVATE SOURCE"])
        self.join_worker(session)

        self.assertEqual(self.client.responses.create.call_count, 2)
        self.assertEqual(session.snapshot(), {
            "translations": [None], "errors": [FAILED_MESSAGE], "filtered": [False], "pending": 0,
        })
        self.assertNotIn("PRIVATE", json.dumps(session.snapshot()))

    def test_language_repair_preserves_identifier_validation(self):
        self.client.responses.create.side_effect = [
            SimpleNamespace(status="completed", output_text="Check ETCH-07 工序."),
            SimpleNamespace(status="completed", output_text="Check the ETCH-08 process."),
        ]

        with self.assertRaisesRegex(ValueError, "protected identifier"):
            translate_to_english("檢查 ETCH-07 工序", context={"dnt_hits": ["ETCH-07"]})

        self.assertEqual(self.client.responses.create.call_count, 2)

    def test_english_output_and_api_errors_never_trigger_language_repair(self):
        self.client.responses.create.return_value.output_text = "José checks ETCH-07 at 8."
        self.assertEqual(translate_to_english("José 檢查 ETCH-07 在八點"), "José checks ETCH-07 at 8.")
        self.client.responses.create.assert_called_once()
        self.client.responses.create.reset_mock()
        self.client.responses.create.side_effect = RuntimeError("PRIVATE API DETAILS")
        session = self.make_session()

        session.submit(["source"])
        self.join_worker(session)

        self.client.responses.create.assert_called_once()
        self.assertEqual(session.snapshot()["errors"], [FAILED_MESSAGE])

    def test_incomplete_mixed_language_output_is_not_repaired(self):
        self.client.responses.create.return_value = SimpleNamespace(status="incomplete", output_text="工序")

        with self.assertRaisesRegex(ValueError, "incomplete"):
            translate_to_english("source")

        self.client.responses.create.assert_called_once()

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
            "errors": [MISSING_KEY_MESSAGE, MISSING_KEY_MESSAGE], "filtered": [False] * 2, "pending": 0,
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
                    "translations": [None], "errors": [FAILED_MESSAGE], "filtered": [False], "pending": 0,
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
