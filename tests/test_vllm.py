"""Hosted model contracts with in-memory HTTP transports only."""

from email.parser import BytesParser
from email.policy import default
from io import BytesIO
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import wave

import httpx
import numpy as np

from src.translation import BACKGROUND_FILTERED_TEXT, _FilteredBackgroundTranslation
from src.vllm import (
    AsrConfig, FastProfile, VllmConfigError, VllmError, create_asr,
    create_translator, default_fast_profile, load_asr_config, load_fast_profiles,
)


def chat(text="The process is stable.", **kwargs):
    return {"choices": [{"finish_reason": "stop", "message": {
        "role": "assistant", "content": text, **kwargs,
    }}]}


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.file = patch("src.vllm.ENV_FILE", Path("/nonexistent-vllm-test.env"))
        self.file.start()
        self.addCleanup(self.file.stop)

    def configure(self):
        os.environ.update({
            "VLLM_ASR_BASE_URL": "http://asr.internal/v1/", "VLLM_ASR_MODEL": "asr-served",
            "VLLM_ASR_API_KEY": "asr-secret", "VLLM_FAST_PROFILES": "small, larger",
            "VLLM_FAST_SMALL_LABEL": "Small model", "VLLM_FAST_SMALL_BASE_URL": "http://fast.internal/v1",
            "VLLM_FAST_SMALL_MODEL": "small-served", "VLLM_FAST_SMALL_API_KEY": "small-secret",
            "VLLM_FAST_LARGER_LABEL": "Larger model", "VLLM_FAST_LARGER_BASE_URL": "https://large.internal/api/v1",
            "VLLM_FAST_LARGER_MODEL": "large-served", "VLLM_FAST_LARGER_MAX_TOKENS": "1024",
            "VLLM_FAST_LARGER_TEMPERATURE": "0", "VLLM_FAST_LARGER_REASONING_EFFORT": " low ",
            "VLLM_FAST_LARGER_CHAT_TEMPLATE_KWARGS": '{"enable_thinking":false,"nested":{"values":[1,2]}}',
            "VLLM_DEFAULT_FAST_PROFILE": "larger",
        })

    def test_profiles_load_aliases_defaults_optional_fields_and_safe_public_metadata(self):
        self.configure()
        asr, profiles = load_asr_config(), load_fast_profiles()
        self.assertEqual(asr.base_url, "http://asr.internal/v1")
        self.assertIsNone(asr.language)
        self.assertEqual(list(profiles), ["small", "larger"])
        self.assertEqual(default_fast_profile(profiles), "larger")
        self.assertEqual(profiles["small"].max_tokens, 512)
        self.assertIsNone(profiles["small"].temperature)
        self.assertIsNone(profiles["small"].reasoning_effort)
        self.assertIsNone(profiles["small"].chat_template_kwargs)
        self.assertEqual(profiles["larger"].max_tokens, 1024)
        self.assertEqual(profiles["larger"].temperature, 0)
        self.assertEqual(profiles["larger"].reasoning_effort, "low")
        exposed = repr(asr) + repr(profiles) + json.dumps({"asr": asr.public(), "fast": profiles["small"].public()})
        for private in ("asr-secret", "small-secret", "asr.internal", "fast.internal", "large.internal"):
            self.assertNotIn(private, exposed)
        with self.assertRaises(AttributeError):
            asr.model = "changed"
        with self.assertRaises(TypeError):
            profiles["larger"].chat_template_kwargs["nested"]["values"][0] = 8

    def test_direct_profile_freezes_nested_template_values(self):
        options = {"enable_thinking": False, "nested": {"values": [1]}}
        profile = FastProfile("Fast", "http://host/v1", "served", chat_template_kwargs=options)
        options["nested"]["values"].append(2)
        self.assertEqual(profile.chat_template_kwargs["nested"]["values"], (1,))

    def test_missing_or_invalid_config_never_echoes_configured_values(self):
        invalid = [
            {"VLLM_FAST_PROFILES": ""},
            {"VLLM_FAST_PROFILES": "secret alias"},
            {"VLLM_FAST_PROFILES": "small,SMALL"},
            {"VLLM_FAST_SMALL_BASE_URL": "https://private-user:private-secret@private-host/v1"},
            {"VLLM_FAST_SMALL_BASE_URL": "https://private-host/v1?key=private-secret"},
            {"VLLM_FAST_SMALL_BASE_URL": "https://private-host:999999/v1"},
            {"VLLM_FAST_SMALL_MAX_TOKENS": "private-secret"},
            {"VLLM_FAST_SMALL_MAX_TOKENS": "0"},
            {"VLLM_FAST_SMALL_TEMPERATURE": "NaN"},
            {"VLLM_FAST_SMALL_REASONING_EFFORT": "private-secret"},
            {"VLLM_FAST_SMALL_CHAT_TEMPLATE_KWARGS": "private-secret"},
            {"VLLM_FAST_SMALL_CHAT_TEMPLATE_KWARGS": '[]'},
            {"VLLM_FAST_SMALL_CHAT_TEMPLATE_KWARGS": '{"number":NaN}'},
        ]
        for changes in invalid:
            with self.subTest(changes=tuple(changes)):
                self.configure()
                os.environ.update(changes)
                with self.assertRaises(VllmConfigError) as failure:
                    load_fast_profiles()
                self.assertNotIn("private-secret", str(failure.exception))
                self.assertNotIn("private-host", str(failure.exception))
                self.assertIsNone(failure.exception.__context__)
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(VllmConfigError):
            load_asr_config()

    def test_default_alias_must_exist_or_falls_back_to_first_profile(self):
        self.configure()
        profiles = load_fast_profiles()
        os.environ["VLLM_DEFAULT_FAST_PROFILE"] = "missing-private-alias"
        with self.assertRaises(VllmConfigError) as failure:
            default_fast_profile(profiles)
        self.assertNotIn("missing-private-alias", str(failure.exception))
        os.environ["VLLM_DEFAULT_FAST_PROFILE"] = ""
        self.assertEqual(default_fast_profile(profiles), "small")

    def test_base_urls_accept_origin_or_api_root_without_duplicating_v1(self):
        for base, expected in (("http://host:8000", "http://host:8000/v1"),
                               ("http://host:8000/", "http://host:8000/v1"),
                               ("https://host/prefix", "https://host/prefix/v1"),
                               ("https://host/prefix/v1/", "https://host/prefix/v1")):
            with self.subTest(base=base):
                self.assertEqual(AsrConfig(base, "asr").base_url, expected)
                self.assertEqual(FastProfile("Fast", base, "fast").base_url, expected)

    def test_dotenv_fills_missing_values_but_keeps_existing_environment(self):
        with TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text(
                "VLLM_ASR_BASE_URL=http://from-file/v1\nVLLM_ASR_MODEL=file-model\n"
                "VLLM_ASR_API_KEY=file-key\nVLLM_ASR_LANGUAGE=zh\n"
            )
            with patch("src.vllm.ENV_FILE", env_file):
                os.environ["VLLM_ASR_MODEL"] = "existing-model"
                config = load_asr_config()
        self.assertEqual(config.model, "existing-model")
        self.assertEqual(config.api_key, "file-key")
        self.assertEqual(config.language, "zh")
        os.environ["VLLM_ASR_MODEL"] = "later-model"
        self.assertEqual(config.model, "existing-model")


class AdapterTests(unittest.TestCase):
    def asr(self, handler, **kwargs):
        adapter = create_asr(AsrConfig("http://asr.test/v1", "served-asr", **kwargs),
                             transport=httpx.MockTransport(handler))
        self.addCleanup(adapter.close)
        return adapter

    def translator(self, responses, *, profile=None, filter_background=True):
        requests = []
        def handler(request):
            requests.append(request)
            response = responses[len(requests) - 1]
            if isinstance(response, Exception):
                raise response
            return response if isinstance(response, httpx.Response) else httpx.Response(200, json=response)
        profile = profile or FastProfile("Fast", "http://fast.test/v1", "served-fast")
        adapter = create_translator(profile, filter_background=filter_background,
                                    transport=httpx.MockTransport(handler))
        self.addCleanup(adapter.close)
        return adapter, requests

    def test_asr_posts_valid_pcm_wav_and_only_configured_fields(self):
        captured = []
        def handler(request):
            captured.append(request)
            return httpx.Response(200, json={"text": " 原始文字 ", "reasoning": "private server reasoning"})
        adapter = self.asr(handler, api_key="test-api-key", language="zh")
        samples = np.array([-1, -0.5, 0, 0.5, 1, 2], dtype=np.float32)
        original = samples.copy()
        self.assertEqual(adapter(samples), "原始文字")
        np.testing.assert_array_equal(samples, original)
        request = captured[0]
        self.assertEqual(str(request.url), "http://asr.test/v1/audio/transcriptions")
        self.assertEqual(request.headers["Authorization"], "Bearer test-api-key")
        self.assertTrue(all(value is None for value in request.extensions["timeout"].values()))
        message = BytesParser(policy=default).parsebytes(
            f'Content-Type: {request.headers["content-type"]}\r\n\r\n'.encode() + request.content
        )
        parts = {part.get_param("name", header="content-disposition"): part for part in message.iter_parts()}
        self.assertEqual(set(parts), {"file", "model", "response_format", "language"})
        self.assertEqual(parts["model"].get_payload(decode=True), b"served-asr")
        self.assertEqual(parts["file"].get_filename(), "segment.wav")
        with wave.open(BytesIO(parts["file"].get_payload(decode=True)), "rb") as audio:
            self.assertEqual((audio.getnchannels(), audio.getsampwidth(), audio.getframerate()), (1, 2, 16000))
            self.assertEqual(audio.getnframes(), samples.size)
            pcm = np.frombuffer(audio.readframes(samples.size), dtype="<i2")
        np.testing.assert_array_equal(pcm, [-32768, -16384, 0, 16384, 32767, 32767])

    def test_asr_auto_language_and_optional_auth_are_omitted(self):
        requests = []
        adapter = self.asr(lambda request: requests.append(request) or httpx.Response(200, json={"text": "speech"}))
        adapter(np.zeros(20, np.float32))
        self.assertNotIn("authorization", requests[0].headers)
        self.assertNotIn(b'name="language"', requests[0].content)

    def test_asr_rejects_invalid_audio_before_request(self):
        requests = []
        adapter = self.asr(lambda request: requests.append(request) or httpx.Response(200, json={"text": "speech"}))
        for audio in ([], np.array([], np.float32), np.zeros((2, 10), np.float32),
                      np.array([float("nan")], np.float32), np.zeros(3, np.int16),
                      np.zeros(240_001, np.float32)):
            with self.subTest(shape=getattr(audio, "shape", None)), self.assertRaises(VllmError):
                adapter(audio)
        self.assertEqual(requests, [])

    def test_asr_errors_are_safe_and_never_automatically_retried(self):
        for response in (httpx.Response(503, text="private response secret"),
                         httpx.Response(200, text="private invalid JSON"),
                         httpx.Response(200, json={}),
                         httpx.Response(200, json={"text": ["private"]})):
            with self.subTest(status=response.status_code):
                requests = []
                adapter = self.asr(lambda request: requests.append(request) or response)
                with self.assertRaises(VllmError) as failure:
                    adapter(np.zeros(2, np.float32))
                self.assertEqual(len(requests), 1)
                self.assertNotIn("private", str(failure.exception))
                self.assertIsNone(failure.exception.__context__)

    def test_valid_empty_asr_text_is_successful_silence_not_a_failed_upload(self):
        for text in ("", " \n \t"):
            with self.subTest(text=repr(text)):
                adapter = self.asr(lambda request: httpx.Response(200, json={"text": text}))
                self.assertEqual(adapter(np.zeros(20, np.float32)), "")

    def test_clients_disable_timeout_redirects_and_transport_retries_and_close_idempotently(self):
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"text": "speech"}))
        with patch("src.vllm.httpx.HTTPTransport", return_value=transport) as factory:
            adapter = create_asr(AsrConfig("http://asr.test/v1", "asr"))
        factory.assert_called_once_with(retries=0)
        self.assertEqual(adapter._client.timeout, httpx.Timeout(None))
        self.assertFalse(adapter._client.follow_redirects)
        adapter.close()
        adapter.close()
        self.assertTrue(adapter._client.is_closed)
        with self.assertRaises(VllmError) as failure:
            adapter(np.zeros(2, np.float32))
        self.assertEqual(failure.exception.category, "closed")

    def test_chat_request_uses_frozen_profile_and_only_final_content(self):
        profile = FastProfile("Fast", "https://model.test/prefix/v1", "served-chat", api_key="profile-secret",
                              max_tokens=768, temperature=0.25, reasoning_effort="low",
                              chat_template_kwargs={"enable_thinking": False})
        adapter, requests = self.translator([chat(reasoning="PRIVATE REASONING", reasoning_content="PRIVATE OLD FIELD")], profile=profile)
        self.assertEqual(adapter("原文"), "The process is stable.")
        request = requests[0]
        self.assertEqual(str(request.url), "https://model.test/prefix/v1/chat/completions")
        payload = json.loads(request.content)
        self.assertEqual(payload["model"], "served-chat")
        self.assertEqual(payload["max_tokens"], 768)
        self.assertEqual(payload["temperature"], 0.25)
        self.assertEqual(payload["reasoning_effort"], "low")
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
        self.assertFalse(payload["stream"])
        self.assertEqual([message["role"] for message in payload["messages"]], ["system", "user"])
        self.assertNotIn("profile-secret", request.content.decode())
        self.assertTrue(all(value is None for value in request.extensions["timeout"].values()))

    def test_unconfigured_decoding_options_are_not_invented(self):
        adapter, requests = self.translator([chat()])
        adapter("source")
        payload = json.loads(requests[0].content)
        self.assertEqual(payload["max_tokens"], 512)
        self.assertTrue({"temperature", "reasoning_effort", "chat_template_kwargs", "store"}.isdisjoint(payload))

    def test_context_allowlist_excludes_reference_and_provenance_and_keeps_latest_two(self):
        adapter, requests = self.translator([chat("Keep ETCH-07.")])
        context = {
            "source_lang": "zh-TW+en", "target_lang": "fr", "reference": "PRIVATE REFERENCE",
            "filename": "PRIVATE FILENAME", "dnt_hits": ["ETCH-07"],
            "previous": [
                {"source_text": "old", "target_text": "Old English"},
                {"source_text": "main", "target_text": "Main English", "reference": "PRIVATE REFERENCE"},
                {"source_text": "background", "target_text": BACKGROUND_FILTERED_TEXT, "filtered": True},
                {"source_text": "recent", "target_text": "Recent English", "timing": "PRIVATE TIME"},
                {"source_text": "failed", "target_text": None},
            ],
            "glossary": [{"term_src": "腔體", "term_tgt": "chamber", "dnt": False,
                          "entry_id": "PRIVATE ENTRY", "source": "PRIVATE PATH", "reference": "PRIVATE REF"}] * 45,
        }
        adapter("維持 ETCH-07", context=context)
        body = json.loads(requests[0].content)
        data = json.loads(body["messages"][1]["content"])
        self.assertEqual(data["current_source"], "維持 ETCH-07")
        self.assertEqual(data["target_lang"], "en")
        self.assertEqual([row["target_text"] for row in data["previous"]], ["Main English", "Recent English"])
        self.assertEqual(len(data["glossary"]), 40)
        self.assertEqual(data["glossary"][0], {"term_src": "腔體", "term_tgt": "chamber", "dnt": False})
        self.assertNotIn("PRIVATE", requests[0].content.decode())

    def test_one_repair_preserves_original_context_and_cap_without_echoing_invalid_answer(self):
        adapter, requests = self.translator([chat("PRIVATE_INVALID 中文"), chat("Keep ETCH-07.")])
        self.assertEqual(adapter("維持 ETCH-07", context={"dnt_hits": ["ETCH-07"]}), "Keep ETCH-07.")
        payloads = [json.loads(request.content) for request in requests]
        self.assertEqual(len(payloads), 2)
        self.assertEqual(payloads[0]["messages"][1], payloads[1]["messages"][1])
        self.assertEqual(payloads[0]["max_tokens"], payloads[1]["max_tokens"])
        self.assertIn("English only", payloads[1]["messages"][0]["content"])
        self.assertNotIn("PRIVATE_INVALID", requests[1].content.decode())

    def test_language_and_identifier_failures_share_one_bounded_repair(self):
        cases = [
            ([chat("中文 ETCH-07"), chat("Wrong ETCH-08")], "identifiers"),
            ([chat("Wrong ETCH-08"), chat("中文 ETCH-07")], "language"),
            ([chat("Wrong ETCH-08"), chat("Wrong ETCH-08")], "identifiers"),
        ]
        for responses, category in cases:
            with self.subTest(category=category):
                adapter, requests = self.translator(responses)
                with self.assertRaises(VllmError) as failure:
                    adapter("Check ETCH-07")
                self.assertEqual(failure.exception.category, category)
                self.assertEqual(len(requests), 2)

    def test_identifier_repair_preserves_case_counts_and_array_dimensions(self):
        for invalid in ("Check etch-07 R² in 8x4 arrays.", "Check ETCH-07 R2 in 8x4 arrays.",
                        "Check ETCH-07 ETCH-07 R2 in 8x4 arrays, plus LOT-09."):
            with self.subTest(invalid=invalid):
                valid = "Check ETCH-07 ETCH-07 R2 in 8x4 arrays."
                adapter, requests = self.translator([chat(invalid), chat(valid)])
                self.assertEqual(adapter("檢查 ETCH-07 ETCH-07 R2 8x4", context={"dnt_hits": ["R2"]}), valid)
                self.assertEqual(len(requests), 2)

    def test_filter_outcomes_are_explicit_and_protected_source_cannot_be_fully_filtered(self):
        marker = "[[NO_MAIN_CONVERSATION_SPEECH]]"
        adapter, _ = self.translator([chat(marker)])
        result = adapter("Clearly unrelated announcement")
        self.assertIsInstance(result, _FilteredBackgroundTranslation)
        self.assertEqual(result, BACKGROUND_FILTERED_TEXT)
        literal, _ = self.translator([chat(BACKGROUND_FILTERED_TEXT)])
        self.assertNotIsInstance(literal("Read the displayed English marker"), _FilteredBackgroundTranslation)
        protected, requests = self.translator([chat(marker), chat("Keep ETCH-07.")])
        self.assertEqual(protected("維持 ETCH-07"), "Keep ETCH-07.")
        self.assertEqual(len(requests), 2)
        disabled, requests = self.translator([chat(marker)], filter_background=False)
        with self.assertRaises(VllmError) as failure:
            disabled("source")
        self.assertEqual(failure.exception.category, "filtered")
        self.assertEqual(len(requests), 1)

    def test_filter_setting_is_frozen_for_the_client(self):
        requests = []
        transport = httpx.MockTransport(lambda request: requests.append(request) or httpx.Response(200, json=chat()))
        with patch("src.vllm.ENV_FILE", Path("/nonexistent-vllm-test.env")), patch.dict(os.environ, {"VLLM_FILTER_BACKGROUND_SPEECH": "false"}):
            adapter = create_translator(FastProfile("Fast", "http://host/v1", "served"), transport=transport)
            self.addCleanup(adapter.close)
            os.environ["VLLM_FILTER_BACKGROUND_SPEECH"] = "true"
            adapter("source")
        self.assertNotIn("main-conversation filtering", json.loads(requests[0].content)["messages"][0]["content"])

    def test_invalid_or_partial_chat_responses_and_http_errors_are_never_repaired(self):
        cases = [
            {"choices": []},
            {"choices": [{"finish_reason": "length", "message": {"role": "assistant", "content": "partial 中文"}}]},
            {"choices": [{"finish_reason": None, "message": {"role": "assistant", "content": "partial"}}]},
            chat(None, reasoning="PRIVATE reasoning only"),
            chat("<think>PRIVATE reasoning</think>English answer"),
            chat("English", tool_calls=[{"function": "private"}]),
            chat("English", refusal="private refusal"),
            httpx.Response(401, text="PRIVATE secret server body"),
            httpx.Response(200, text="PRIVATE invalid JSON"),
            httpx.ReadError("PRIVATE transport URL or credential"),
        ]
        for response in cases:
            with self.subTest(type=type(response).__name__):
                adapter, requests = self.translator([response])
                with self.assertRaises(VllmError) as failure:
                    adapter("source")
                self.assertEqual(len(requests), 1)
                self.assertNotIn("PRIVATE", str(failure.exception))
                self.assertIsNone(failure.exception.__context__)
                self.assertIsNone(failure.exception.__cause__)


if __name__ == "__main__":
    unittest.main()
