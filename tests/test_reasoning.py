"""Verify Astra request boundaries without loading credentials or making API calls."""

from copy import deepcopy
from dataclasses import asdict
import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.reasoning import (
    CHANGE_TYPES, DEFAULT_REASONING_MODEL, INSTRUCTIONS, RESPONSE_SCHEMA,
    correct_translations,
)
from src.translation import ENV_FILE, _MissingAPIKeyError
from src.slow_lane import FAILED_MESSAGE, SlowLaneConfig, SlowLaneSession


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

    def test_astra_medium_uses_bounded_official_request_and_strict_schema(self):
        before = deepcopy(self.request)

        result = correct_translations(self.request)

        self.assertEqual(result, self.result)
        self.assertEqual(self.request, before)
        self.assertEqual(DEFAULT_REASONING_MODEL, "gpt-6-astra")
        self.dotenv.assert_called_once_with(ENV_FILE, override=False)
        self.openai.assert_called_once_with(
            api_key="unit-test-key", base_url="https://api.openai.com/v1",
            timeout=20, max_retries=0,
        )
        settings = self.client.responses.create.call_args.kwargs
        self.assertEqual(settings["model"], "gpt-6-astra")
        self.assertEqual(settings["reasoning"], {"effort": "medium"})
        self.assertEqual(settings["max_output_tokens"], 4096)
        self.assertIs(settings["store"], False)
        self.assertEqual(settings["instructions"], INSTRUCTIONS)
        self.assertEqual(settings["prompt_cache_key"], "subtitle-unit-test-session")
        self.assertEqual(settings["text"], {"format": {
            "type": "json_schema", "name": "subtitle_corrections",
            "strict": True, "schema": RESPONSE_SCHEMA,
        }})
        self.assertNotIn("tools", settings)
        self.openai.return_value.__exit__.assert_called_once()

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
        })
        self.assertEqual(payload["segments"][0]["source_text"], "檢查 ETCH-07。")
        self.assertEqual(payload["segments"][0]["dnt_hits"], ["ETCH-07"])
        self.assertEqual(payload["context"][0]["segment_id"], "previous-0")
        self.assertEqual(payload["glossary"][0]["example_tgt"], "etch chamber")
        self.assertEqual(payload["learned_terms"][0]["evidence_segment_ids"], ["previous-0"])

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

    def test_refusal_does_not_become_a_correction(self):
        self.client.responses.create.return_value = SimpleNamespace(
            status="completed", output_text="",
            output=[SimpleNamespace(type="message", content=[
                SimpleNamespace(type="refusal", refusal="A refusal is not a translation."),
            ])],
        )

        with self.assertRaisesRegex(ValueError, "incomplete or empty"):
            correct_translations(self.request)

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

    def test_missing_key_raises_safe_configuration_error_before_client_creation(self):
        os.environ["OPENAI_API_KEY"] = "   "

        with self.assertRaises(_MissingAPIKeyError) as raised:
            correct_translations(self.request)

        self.assertEqual(str(raised.exception), "")
        self.openai.assert_not_called()

    def test_sdk_failure_closes_client_and_is_left_for_worker_to_handle(self):
        self.client.responses.create.side_effect = TimeoutError("private request details")

        with self.assertRaises(TimeoutError):
            correct_translations(self.request)

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
                    with session._lock:
                        worker = session._worker
                    if worker is not None:
                        worker.join(timeout=5)
                        self.assertFalse(worker.is_alive(), "correction worker did not finish")
                    snapshot = session.snapshot()

                    self.assertEqual(snapshot["translations"], ["The usable fast draft."])
                    self.assertEqual(snapshot["authoritative"], [None])
                    self.assertEqual(snapshot["statuses"], ["failed"])
                    self.assertEqual(snapshot["segments"][0]["error"], FAILED_MESSAGE)
                    self.assertEqual(snapshot["pending"], 0)
                    self.assertNotIn("PRIVATE-API-DETAILS", json.dumps(snapshot))
                    self.assertNotIn("unit-test-key", json.dumps(snapshot))
                    if missing_key:
                        self.openai.assert_not_called()
                finally:
                    session.close()


if __name__ == "__main__":
    unittest.main()
