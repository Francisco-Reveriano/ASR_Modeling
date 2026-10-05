"""Verify Astra request boundaries without loading credentials or making API calls."""

from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from openai import APIConnectionError, APIStatusError, APITimeoutError

from src import reasoning
from src.reasoning import (
    CHANGE_TYPES, DEFAULT_REASONING_EFFORT, DEFAULT_REASONING_MODEL,
    DEFAULT_CORRECTION_MAX_OUTPUT_TOKENS, INSTRUCTIONS, RESPONSE_SCHEMA,
    correct_translations, correction_output_token_limit, correction_settings,
)
from src.translation import ENV_FILE, _MissingAPIKeyError
from src.slow_lane import FAILED_MESSAGE, SlowLaneConfig, SlowLaneSession


class CorrectionSettingsTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {}, clear=True))
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.env_file = Path(directory) / ".env"
        self.enterContext(patch("src.reasoning.ENV_FILE", self.env_file))

    def test_missing_correction_settings_use_defaults_without_fast_model_fallback(self):
        self.env_file.write_text("OPENAI_DEFAULT_MODEL=fast-only-model\n", encoding="utf-8")

        self.assertEqual(correction_settings(), ("gpt-6-astra", "medium"))
        self.assertEqual(DEFAULT_REASONING_MODEL, "gpt-6-astra")
        self.assertEqual(DEFAULT_REASONING_EFFORT, "medium")

    def test_real_dotenv_loads_and_normalizes_correction_settings(self):
        self.env_file.write_text(
            'OPENAI_CORRECTION_MODEL=" custom-correction-model "\n'
            'OPENAI_CORRECTION_REASONING_EFFORT=" HIGH "\n',
            encoding="utf-8",
        )

        self.assertEqual(correction_settings(), ("custom-correction-model", "high"))

    def test_environment_settings_take_precedence_over_real_dotenv(self):
        self.env_file.write_text(
            "OPENAI_CORRECTION_MODEL=dotenv-model\n"
            "OPENAI_CORRECTION_REASONING_EFFORT=low\n",
            encoding="utf-8",
        )
        os.environ["OPENAI_CORRECTION_MODEL"] = " environment-model "
        os.environ["OPENAI_CORRECTION_REASONING_EFFORT"] = " MAX "

        self.assertEqual(correction_settings(), ("environment-model", "max"))

    def test_blank_values_fall_back_to_defaults_including_environment_precedence(self):
        for environment_blanks in (False, True):
            with self.subTest(environment_blanks=environment_blanks):
                os.environ.pop("OPENAI_CORRECTION_MODEL", None)
                os.environ.pop("OPENAI_CORRECTION_REASONING_EFFORT", None)
                if environment_blanks:
                    self.env_file.write_text(
                        "OPENAI_CORRECTION_MODEL=dotenv-model\n"
                        "OPENAI_CORRECTION_REASONING_EFFORT=high\n",
                        encoding="utf-8",
                    )
                    os.environ["OPENAI_CORRECTION_MODEL"] = "  "
                    os.environ["OPENAI_CORRECTION_REASONING_EFFORT"] = "  "
                else:
                    self.env_file.write_text(
                        'OPENAI_CORRECTION_MODEL="  "\n'
                        'OPENAI_CORRECTION_REASONING_EFFORT="  "\n',
                        encoding="utf-8",
                    )

                self.assertEqual(correction_settings(), ("gpt-6-astra", "medium"))

    def test_generic_reasoning_efforts_are_accepted(self):
        for effort in ("none", "minimal", "low", "medium", "high", "xhigh", "max"):
            with self.subTest(effort=effort):
                os.environ["OPENAI_CORRECTION_REASONING_EFFORT"] = effort

                self.assertEqual(correction_settings(), ("gpt-6-astra", effort))

    def test_invalid_reasoning_setting_has_safe_key_specific_error(self):
        private_value = "PRIVATE-CONFIG-VALUE unit-test-key"
        os.environ["OPENAI_CORRECTION_REASONING_EFFORT"] = private_value

        with self.assertRaises(ValueError) as raised:
            correction_settings()

        message = str(raised.exception)
        self.assertIn("OPENAI_CORRECTION_REASONING_EFFORT", message)
        self.assertIn("none, minimal, low, medium, high, xhigh, or max", message)
        self.assertNotIn(private_value, message)
        self.assertNotIn("unit-test-key", message)

    def test_missing_or_blank_output_limit_reserves_room_for_reasoning(self):
        self.assertEqual(DEFAULT_CORRECTION_MAX_OUTPUT_TOKENS, 16_384)
        self.assertEqual(correction_output_token_limit(), 16_384)
        self.env_file.write_text('OPENAI_CORRECTION_MAX_OUTPUT_TOKENS="  "\n', encoding="utf-8")
        self.assertEqual(correction_output_token_limit(), 16_384)

    def test_output_limit_reads_dotenv_and_keeps_explicit_smaller_caps(self):
        self.env_file.write_text('OPENAI_CORRECTION_MAX_OUTPUT_TOKENS=" 4096 "\n', encoding="utf-8")
        self.assertEqual(correction_output_token_limit(), 4096)
        os.environ["OPENAI_CORRECTION_MAX_OUTPUT_TOKENS"] = "512"
        self.assertEqual(correction_output_token_limit(), 512)
        os.environ["OPENAI_CORRECTION_MAX_OUTPUT_TOKENS"] = "16384"
        self.assertEqual(correction_output_token_limit(), 16_384)

    def test_environment_output_limit_overrides_dotenv_including_blank(self):
        self.env_file.write_text("OPENAI_CORRECTION_MAX_OUTPUT_TOKENS=4096\n", encoding="utf-8")
        os.environ["OPENAI_CORRECTION_MAX_OUTPUT_TOKENS"] = " 8192 "
        self.assertEqual(correction_output_token_limit(), 8192)
        os.environ["OPENAI_CORRECTION_MAX_OUTPUT_TOKENS"] = " "
        self.assertEqual(correction_output_token_limit(), 16_384)

    def test_invalid_output_limit_has_safe_key_specific_error(self):
        for value in ("511", "16385", "-1", "4096.0", "true", "PRIVATE-CONFIG unit-test-key", "9" * 5000):
            with self.subTest(value=value[:30]):
                os.environ["OPENAI_CORRECTION_MAX_OUTPUT_TOKENS"] = value
                with self.assertRaises(ValueError) as raised:
                    correction_output_token_limit()
                self.assertEqual(str(raised.exception),
                                 "OPENAI_CORRECTION_MAX_OUTPUT_TOKENS must be an integer between 512 and 16384.")


class ReasoningRequestTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {
            "OPENAI_API_KEY": "unit-test-key",
            "OPENAI_BASE_URL": "https://invalid.example/never-use",
            "OPENAI_DEFAULT_MODEL": "fast-model-must-not-override-astra",
        }, clear=True))
        self.dotenv = self.enterContext(patch("src.reasoning.load_dotenv"))
        self.openai = self.enterContext(patch("openai.OpenAI"))
        self.client = self.openai.return_value.__enter__.return_value
        self.result = {
            "corrections": [{
                "segment_id": "segment-1", "base_version": 2,
                "target_text": "Check ETCH-07.", "change_type": ["terminology"],
                "confidence": 0.95, "rationale": "The glossary names this chamber.",
                "term_pairs": [{"source": "蝕刻", "target": "etch"}],
            }],
            "no_change": [{"segment_id": "segment-2", "base_version": 1}],
        }
        self.client.responses.create.return_value = SimpleNamespace(
            status="completed", output_text=json.dumps(self.result),
        )
        self.request = {
            "session_id": "unit-test-session",
            "config": asdict(SlowLaneConfig()),
            "segments": [{
                "segment_id": "segment-1", "base_version": 2,
                "source_text": "檢查 ETCH-07。", "target_text": "Check the chamber.",
                "dnt_hits": ["ETCH-07"], "speaker_id": "Speaker 1",
            }],
            "context": [{
                "segment_id": "previous-0", "base_version": 3,
                "source_text": "前一句", "target_text": "Earlier context.",
            }],
            "glossary": [{
                "entry_id": "etch", "term_src": "蝕刻", "term_tgt": "etch",
                "example_src": "蝕刻腔體", "example_tgt": "etch chamber",
            }],
            "learned_terms": [{
                "entry_id": "chamber", "term_src": "腔體", "term_tgt": "chamber",
                "evidence_segment_ids": ["previous-0"],
            }],
        }

    def assert_safe_provider_failure(self, category, **details):
        with self.assertRaises(ValueError) as raised:
            correct_translations(self.request)

        error = raised.exception
        self.assertIsInstance(error, reasoning.CorrectionProviderError)
        self.assertEqual(error.diagnostics(), {"category": category, **details})
        serialized = json.dumps(error.diagnostics()) + str(error) + repr(error)
        self.assertNotIn("PRIVATE", serialized)
        self.assertNotIn("unit-test-key", serialized)
        self.assertIsNone(error.__cause__)
        self.assertIsNone(error.__context__)
        for field in ("request", "response", "body", "doc"):
            self.assertFalse(hasattr(error, field))
        return error

    def test_astra_medium_uses_official_request_without_deadline_and_strict_schema(self):
        for key in ("request_timeout_s", "seal_timeout_s", "revision_horizon_s"):
            self.request["config"].pop(key, None)
        before = deepcopy(self.request)

        result = correct_translations(self.request)

        self.assertEqual(result, self.result)
        self.assertEqual(self.request, before)
        self.assertEqual(DEFAULT_REASONING_MODEL, "gpt-6-astra")
        self.dotenv.assert_called_once_with(ENV_FILE, override=False)
        self.openai.assert_called_once_with(
            api_key="unit-test-key", base_url="https://api.openai.com/v1",
            timeout=None, max_retries=0,
        )
        settings = self.client.responses.create.call_args.kwargs
        self.assertEqual(settings["model"], "gpt-6-astra")
        self.assertEqual(settings["reasoning"], {"effort": "medium"})
        self.assertEqual(settings["max_output_tokens"], 16_384)
        self.assertIs(settings["store"], False)
        self.assertTrue(settings["instructions"].startswith(INSTRUCTIONS))
        self.assertEqual(settings["prompt_cache_key"], "subtitle-unit-test-session")
        self.assertEqual(settings["text"], {"format": {
            "type": "json_schema", "name": "subtitle_corrections",
            "strict": True, "schema": RESPONSE_SCHEMA,
        }})
        self.assertNotIn("tools", settings)
        self.assertNotIn("timeout", settings)
        self.openai.return_value.__exit__.assert_called_once()

    def test_legacy_timeout_config_cannot_restore_a_deadline_or_enter_the_payload(self):
        self.request["config"].update(
            request_timeout_s=0.001, seal_timeout_s=1, revision_horizon_s=1,
        )
        before = deepcopy(self.request)

        self.assertEqual(correct_translations(self.request), self.result)

        self.assertEqual(self.request, before)
        self.assertIsNone(self.openai.call_args.kwargs["timeout"])
        settings = self.client.responses.create.call_args.kwargs
        self.assertNotIn("timeout", settings)
        payload = json.loads(settings["input"])
        for key in ("request_timeout_s", "seal_timeout_s", "revision_horizon_s"):
            self.assertNotIn(key, payload)
            self.assertNotIn(key, settings["input"])

    def test_schema_requires_each_field_and_rejects_unknown_keys_recursively(self):
        def inspect_schema(schema):
            if schema.get("type") == "object":
                self.assertIs(schema["additionalProperties"], False)
                self.assertEqual(set(schema["required"]), set(schema["properties"]))
                for field in schema["properties"].values():
                    inspect_schema(field)
            elif schema.get("type") == "array":
                inspect_schema(schema["items"])

        inspect_schema(RESPONSE_SCHEMA)
        self.assertEqual(set(RESPONSE_SCHEMA["properties"]), {"corrections", "no_change"})
        correction = RESPONSE_SCHEMA["properties"]["corrections"]["items"]["properties"]
        self.assertEqual(correction["confidence"], {"type": "number", "minimum": 0, "maximum": 1})
        self.assertEqual(correction["change_type"]["items"]["enum"], CHANGE_TYPES)
        self.assertEqual(correction["base_version"]["type"], "integer")

    def test_prompt_requires_every_editable_segment_and_translates_source_fallbacks(self):
        self.assertIn("Include every editable segment exactly once", INSTRUCTIONS)
        self.assertIn("never put a\nsource fallback", INSTRUCTIONS)
        self.assertIn("Every target_text must be entirely in English", INSTRUCTIONS)

    def test_confidence_floor_is_explicit_without_requesting_inflated_confidence(self):
        self.request["config"]["confidence_threshold"] = 0.85

        correct_translations(self.request)

        payload = json.loads(self.client.responses.create.call_args.kwargs["input"])
        self.assertEqual(payload["minimum_correction_confidence"], 0.85)
        self.assertIn("never inflate confidence", INSTRUCTIONS)
        self.assertIn("[unclear]", INSTRUCTIONS)
        normalized = " ".join(INSTRUCTIONS.lower().split())
        self.assertIn("never invent an identifier", normalized)
        self.assertIn("never use it to bypass an uncertain correction", normalized)

    def test_frozen_background_filter_setting_preserves_main_speech_in_mixed_segments(self):
        self.request["segments"][0]["source_text"] = "原本的主談話及背景片段"
        self.request["segments"][0]["target_text"] = "The main conversation."
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                self.request["config"]["filter_background_speech"] = enabled

                correct_translations(self.request)

                settings = self.client.responses.create.call_args.kwargs
                payload = json.loads(settings["input"])
                self.assertIs(payload["filter_background_speech"], enabled)
                self.assertEqual(payload["segments"], self.request["segments"])
                if enabled:
                    self.assertIn("clearly unrelated background", settings["instructions"])
                    self.assertIn("do not restore background fragments", settings["instructions"])
                    self.assertIn("all uncertain content", settings["instructions"])
                    self.assertIn("protected identifiers", settings["instructions"])
                else:
                    self.assertNotIn("clearly unrelated background", settings["instructions"])

    def test_review_feedback_sends_only_an_allowlisted_retry_category(self):
        for category in ("schema", "low_confidence", "language", "dnt", "stale", "output_budget"):
            with self.subTest(category=category):
                self.request["review_feedback"] = {
                    "category": category, "target_text": "PRIVATE rejected target",
                    "rationale": "PRIVATE rejected rationale", "response": "PRIVATE response",
                }

                correct_translations(self.request)

                raw = self.client.responses.create.call_args.kwargs["input"]
                payload = json.loads(raw)
                self.assertEqual(payload["review_feedback"], {"category": category})
                self.assertEqual(payload["segments"], self.request["segments"])
                self.assertNotIn("PRIVATE", raw)
        self.assertIn("independent evidence-based review", INSTRUCTIONS)

    def test_unrecognized_review_feedback_is_omitted(self):
        for feedback in (None, "PRIVATE", {"category": "PRIVATE"}, {"category": ["PRIVATE"]}, {}):
            with self.subTest(feedback=feedback):
                self.request["review_feedback"] = feedback

                correct_translations(self.request)

                raw = self.client.responses.create.call_args.kwargs["input"]
                self.assertNotIn("review_feedback", json.loads(raw))
                self.assertNotIn("PRIVATE", raw)

    def test_conversation_stage_reviews_first_accepted_text_with_final_earlier_context(self):
        self.request["review_stage"] = "conversation"
        self.request["segments"][0].update(target_text="First reviewed English.", target_status="corrected")
        self.request["context"][0].update(target_text="Earlier conversation-reviewed English.",
                                         target_status="confirmed", context_position="before",
                                         review_stage="conversation", audio="PRIVATE audio")
        self.request["first_pass"] = {"credentials": "PRIVATE state"}
        before = deepcopy(self.request)

        correct_translations(self.request)

        settings = self.client.responses.create.call_args.kwargs
        payload = json.loads(settings["input"])
        self.assertEqual(payload["review_stage"], "conversation")
        self.assertEqual(payload["segments"][0]["target_text"], "First reviewed English.")
        self.assertEqual(payload["context"][0]["target_text"], "Earlier conversation-reviewed English.")
        self.assertEqual(payload["context"][0]["target_status"], "confirmed")
        self.assertNotIn("review_stage", payload["context"][0])
        self.assertNotIn("PRIVATE", settings["input"])
        self.assertIn("first accepted review", settings["instructions"])
        self.assertIn("Accepted earlier English comes from completed conversation reviews", settings["instructions"])
        self.assertIn("or paused source context may still appear", settings["instructions"])
        self.assertIn("Do not rewrite merely to polish", settings["instructions"])
        self.assertEqual(settings["model"], self.request["config"]["model"])
        self.assertEqual(settings["reasoning"], {"effort": self.request["config"]["reasoning_effort"]})
        self.assertEqual(settings["max_output_tokens"], self.request["config"]["max_output_tokens"])
        self.assertIs(settings["store"], False)
        self.assertEqual(self.request, before)

    def test_unrecognized_review_stage_cannot_select_instructions_or_leak_to_provider(self):
        for stage in ("PRIVATE", {"stage": "PRIVATE"}, ["conversation"], None, 7):
            with self.subTest(stage=stage):
                self.request["review_stage"] = stage
                correct_translations(self.request)
                settings = self.client.responses.create.call_args.kwargs
                self.assertNotIn("review_stage", json.loads(settings["input"]))
                self.assertNotIn("PRIVATE", settings["input"])
                self.assertNotIn("first accepted review", settings["instructions"])

    def test_allowlist_excludes_private_top_level_and_nested_state(self):
        for key in ("evaluation", "audio", "reference_text", "credentials", "unrelated"):
            self.request[key] = "PRIVATE-UPLOADED-STATE"
            self.request["config"][key] = "PRIVATE-UPLOADED-STATE"
            for collection in ("segments", "context", "glossary", "learned_terms"):
                self.request[collection][0][key] = "PRIVATE-UPLOADED-STATE"

        correct_translations(self.request)

        raw = self.client.responses.create.call_args.kwargs["input"]
        payload = json.loads(raw)
        self.assertNotIn("PRIVATE-UPLOADED-STATE", raw)
        self.assertEqual(set(payload), {
            "segments", "context", "glossary", "learned_terms", "source_lang", "target_lang",
            "minimum_correction_confidence", "filter_background_speech",
        })
        self.assertEqual(payload["segments"][0]["source_text"], "檢查 ETCH-07。")
        self.assertEqual(payload["segments"][0]["dnt_hits"], ["ETCH-07"])
        self.assertEqual(payload["context"][0]["segment_id"], "previous-0")
        self.assertEqual(payload["glossary"][0]["example_tgt"], "etch chamber")
        self.assertEqual(payload["learned_terms"][0]["evidence_segment_ids"], ["previous-0"])

    def test_context_provenance_preserves_available_neighbor_sources_without_private_metadata(self):
        self.request["segments"][0]["target_status"] = "draft"
        self.request["context"][0].update(context_position="before", target_status="corrected")
        self.request["context"].append({
            "segment_id": "following-2", "base_version": 0,
            "source_text": "Next completed utterance", "target_text": None,
            "context_position": "after", "target_status": "unavailable",
            "reference_text": "PRIVATE REFERENCE", "audio": "PRIVATE AUDIO",
        })

        correct_translations(self.request)

        raw = self.client.responses.create.call_args.kwargs["input"]
        payload = json.loads(raw)
        self.assertEqual(payload["segments"][0]["target_status"], "draft")
        self.assertEqual(payload["context"][0]["target_status"], "corrected")
        self.assertEqual(payload["context"][0]["context_position"], "before")
        self.assertEqual(payload["context"][1]["context_position"], "after")
        self.assertEqual(payload["context"][1]["target_status"], "unavailable")
        self.assertEqual(payload["context"][1]["source_text"], "Next completed utterance")
        self.assertIsNone(payload["context"][1]["target_text"])
        self.assertNotIn("PRIVATE", raw)

    def test_unrecognized_provenance_values_are_not_sent(self):
        for value in ("PRIVATE", {"private": "PRIVATE"}, ["PRIVATE"], None, 7):
            with self.subTest(value=value):
                self.request["context"][0].update(context_position=value, target_status=value)
                correct_translations(self.request)
                raw = self.client.responses.create.call_args.kwargs["input"]
                row = json.loads(raw)["context"][0]
                self.assertNotIn("PRIVATE", raw)
                self.assertNotIn("context_position", row)
                self.assertNotIn("target_status", row)

    def test_incomplete_empty_malformed_and_nonobject_responses_fail(self):
        cases = [
            ("incomplete", json.dumps(self.result)), ("failed", json.dumps(self.result)),
            ("completed", "  "), ("completed", "not JSON"),
            ("completed", "[]"), ("completed", '"a string"'), ("completed", "null"),
        ]
        for status, text in cases:
            with self.subTest(status=status, text=text):
                self.client.responses.create.return_value = SimpleNamespace(status=status, output_text=text)
                with self.assertRaises(ValueError):
                    correct_translations(self.request)
        self.assertEqual(self.openai.return_value.__exit__.call_count, len(cases))

    def test_incomplete_diagnostics_only_expose_allowlisted_reasons(self):
        cases = [(reason, reason) for reason in
                 ("max_output_tokens", "max_messages", "content_filter", "steered")]
        cases += [("PRIVATE PROVIDER DETAILS", None), ({"PRIVATE": "BODY"}, None), (None, None)]
        for reason, safe_reason in cases:
            with self.subTest(reason=reason):
                self.client.responses.create.return_value = SimpleNamespace(
                    status="incomplete", output_text="PRIVATE partial output",
                    incomplete_details=SimpleNamespace(reason=reason),
                )

                self.assert_safe_provider_failure(
                    "incomplete", **({"incomplete_reason": safe_reason} if safe_reason else {}),
                )

        self.assertEqual(self.client.responses.create.call_count, len(cases))
        self.assertIsNone(self.openai.call_args.kwargs["timeout"])
        self.assertEqual(self.openai.call_args.kwargs["max_retries"], 0)

    def test_response_failure_categories_hide_raw_output_and_status(self):
        cases = [
            ("failed", "PRIVATE failure", "provider_error"),
            ("PRIVATE UNKNOWN STATUS", "PRIVATE failure", "provider_error"),
            ("completed", "  ", "empty"), ("completed", None, "empty"),
            ("completed", '{"PRIVATE BODY":', "malformed_json"),
            ("completed", '["PRIVATE BODY"]', "nonobject"),
        ]
        for status, text, category in cases:
            with self.subTest(category=category):
                self.client.responses.create.return_value = SimpleNamespace(status=status, output_text=text)

                self.assert_safe_provider_failure(category)

        self.assertEqual(self.client.responses.create.call_count, len(cases))

    def test_transport_and_http_errors_keep_only_fixed_category_and_numeric_status(self):
        request = SimpleNamespace(
            headers={"Authorization": "Bearer unit-test-key"}, content=b"PRIVATE request body",
        )
        response = SimpleNamespace(status_code=429, request=request, headers={"x-request-id": "PRIVATE"})
        cases = [
            (APIConnectionError(message="PRIVATE connection", request=request), {"category": "connection"}),
            (APITimeoutError(request), {"category": "transport_timeout"}),
            (TimeoutError("PRIVATE timeout"), {"category": "transport_timeout"}),
            (APIStatusError("PRIVATE failure", response=response, body={"PRIVATE": "BODY"}),
             {"category": "http_status", "status_code": 429}),
            (RuntimeError("PRIVATE provider failure"), {"category": "provider_error"}),
        ]
        for exception, expected in cases:
            with self.subTest(category=expected["category"]):
                self.client.responses.create.side_effect = exception

                self.assert_safe_provider_failure(**expected)

        self.assertEqual(self.client.responses.create.call_count, len(cases))
        self.assertEqual(self.openai.return_value.__exit__.call_count, len(cases))

    def test_provider_error_cannot_expose_unrecognized_metadata(self):
        error = reasoning.CorrectionProviderError(
            "PRIVATE CATEGORY", incomplete_reason="PRIVATE REASON", status_code="PRIVATE STATUS",
        )

        self.assertEqual(error.diagnostics(), {"category": "provider_error"})
        self.assertNotIn("PRIVATE", str(error))
        for status in (True, 99, 600, "429"):
            with self.subTest(status=status):
                error = reasoning.CorrectionProviderError("http_status", status_code=status)
                self.assertEqual(error.diagnostics(), {"category": "http_status"})

    def test_source_fallback_status_reaches_request_without_private_failure_details(self):
        self.request["segments"][0].update(
            source_fallback=True, error="PRIVATE-API-FAILURE",
            target_text=self.request["segments"][0]["source_text"],
        )
        self.request["context"][0]["source_fallback"] = False

        correct_translations(self.request)

        raw = self.client.responses.create.call_args.kwargs["input"]
        payload = json.loads(raw)
        self.assertIs(payload["segments"][0]["source_fallback"], True)
        self.assertIs(payload["context"][0]["source_fallback"], False)
        self.assertNotIn("PRIVATE-API-FAILURE", raw)

    def test_refusal_does_not_become_a_correction(self):
        self.client.responses.create.return_value = SimpleNamespace(
            status="completed", output_text="",
            output=[SimpleNamespace(type="message", content=[
                SimpleNamespace(type="refusal", refusal="A refusal is not a translation."),
            ])],
        )

        self.assert_safe_provider_failure("refusal")

    def test_refusal_with_output_text_is_still_not_published(self):
        self.client.responses.create.return_value = SimpleNamespace(
            status="completed", output_text=json.dumps(self.result),
            output=[SimpleNamespace(type="message", content=[
                SimpleNamespace(type="refusal", refusal="PRIVATE refusal details"),
            ])],
        )

        self.assert_safe_provider_failure("refusal")

    def test_environment_key_precedes_dotenv_and_fast_model_setting_is_ignored(self):
        def simulated_dotenv(*args, **kwargs):
            self.assertIs(kwargs["override"], False)
            os.environ.setdefault("OPENAI_API_KEY", "dotenv-test-key")

        self.dotenv.side_effect = simulated_dotenv
        correct_translations(self.request)
        self.assertEqual(self.openai.call_args.kwargs["api_key"], "unit-test-key")
        self.assertEqual(self.client.responses.create.call_args.kwargs["model"], "gpt-6-astra")

        os.environ.pop("OPENAI_API_KEY")
        correct_translations(self.request)
        self.assertEqual(self.openai.call_args.kwargs["api_key"], "dotenv-test-key")

    def test_frozen_model_and_reasoning_pass_through_despite_changed_environment(self):
        os.environ["OPENAI_CORRECTION_MODEL"] = "new-session-model"
        os.environ["OPENAI_CORRECTION_REASONING_EFFORT"] = "low"
        self.request["config"]["model"] = "frozen-session-model"
        for effort in ("high", "max"):
            with self.subTest(effort=effort):
                self.request["config"]["reasoning_effort"] = effort

                correct_translations(self.request)

                settings = self.client.responses.create.call_args.kwargs
                self.assertEqual(settings["model"], "frozen-session-model")
                self.assertEqual(settings["reasoning"], {"effort": effort})

    def test_missing_key_raises_safe_configuration_error_before_client_creation(self):
        os.environ["OPENAI_API_KEY"] = "   "

        with self.assertRaises(_MissingAPIKeyError) as raised:
            correct_translations(self.request)

        self.assertEqual(str(raised.exception), "")
        self.openai.assert_not_called()

    def test_sdk_failure_closes_client_and_is_left_for_worker_to_handle(self):
        self.client.responses.create.side_effect = TimeoutError("private request details")

        self.assert_safe_provider_failure("transport_timeout")

        self.openai.return_value.__exit__.assert_called_once()

    def test_real_slow_worker_preserves_drafts_and_hides_missing_key_or_sdk_details(self):
        for missing_key in (True, False):
            with self.subTest(missing_key=missing_key):
                if missing_key:
                    os.environ.pop("OPENAI_API_KEY", None)
                else:
                    os.environ["OPENAI_API_KEY"] = "unit-test-key"
                    self.client.responses.create.side_effect = RuntimeError("PRIVATE-API-DETAILS unit-test-key")
                self.openai.reset_mock()
                session = SlowLaneSession(correct_translations)
                try:
                    session.submit(["A source segment."], ["The usable fast draft."])
                    deadline = time.monotonic() + 2
                    while session.snapshot()["pending"] and time.monotonic() < deadline:
                        time.sleep(0.01)
                    snapshot = session.snapshot()
                    self.assertEqual(snapshot["pending"], 0, "correction review did not finish")

                    self.assertEqual(snapshot["translations"], ["The usable fast draft."])
                    self.assertEqual(snapshot["authoritative"], [None])
                    self.assertEqual(snapshot["statuses"], ["failed"])
                    expected_message = FAILED_MESSAGE if missing_key else str(reasoning.CorrectionProviderError("provider_error"))
                    self.assertEqual(snapshot["segments"][0]["error"], expected_message)
                    self.assertEqual(snapshot["segments"][0]["review_error"], {
                        "category": "error" if missing_key else "provider_error",
                    })
                    self.assertEqual(snapshot["pending"], 0)
                    self.assertNotIn("PRIVATE-API-DETAILS", json.dumps(snapshot))
                    self.assertNotIn("unit-test-key", json.dumps(snapshot))
                    if missing_key:
                        self.openai.assert_not_called()
                finally:
                    session.close()


if __name__ == "__main__":
    unittest.main()
