"""Test the local Tencent adapter without loading model weights or using the Hub."""

from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from scripts import download_tencent
from src import tencent


class TokenBatch(dict):
    def to(self, device):
        self.device = device
        return self


class TencentTranslationTests(unittest.TestCase):
    def setUp(self):
        directory = self.enterContext(TemporaryDirectory())
        self.model_dir = Path(directory)
        for filename in tencent.REQUIRED_FILES:
            (self.model_dir / filename).touch()
        self.enterContext(patch.object(tencent, "MODEL_DIR", self.model_dir))
        self.enterContext(patch.object(tencent, "_model_cache", None))
        self.mps = self.enterContext(patch("torch.backends.mps.is_available", return_value=False))
        self.modern_macos = self.enterContext(patch(
            "torch.backends.mps.is_macos_or_newer", return_value=True,
        ))
        self.load_tokenizer = self.enterContext(patch("transformers.AutoTokenizer.from_pretrained"))
        self.load_model = self.enterContext(patch("transformers.AutoModelForCausalLM.from_pretrained"))
        self.batch = TokenBatch(
            input_ids=torch.tensor([[1, 2, 3]]),
            attention_mask=torch.tensor([[1, 1, 1]]),
        )
        self.tokenizer = self.load_tokenizer.return_value
        self.tokenizer.apply_chat_template.return_value = self.batch
        self.tokenizer.decode.return_value = "  The train leaves at 8.  "
        self.model = Mock()
        self.model.device = torch.device("cpu")
        self.model.generation_config = SimpleNamespace(eos_token_id=9)
        self.model.eval.return_value = self.model

        def move_model(device):
            self.model.device = torch.device(device)
            return self.model

        self.model.to.side_effect = move_model
        self.model.generate.return_value = torch.tensor([[1, 2, 3, 7, 8, 9]])
        self.load_model.return_value = self.model

    def test_cpu_load_is_strictly_local_and_cached_across_calls(self):
        self.assertIsNone(tencent._model_cache)

        self.assertEqual(tencent.translate_with_tencent("火車八點出發。"), "The train leaves at 8.")
        tencent.translate_with_tencent("明天見。")

        self.load_tokenizer.assert_called_once_with(
            self.model_dir, local_files_only=True, trust_remote_code=False,
        )
        self.load_model.assert_called_once_with(
            self.model_dir, local_files_only=True, trust_remote_code=False, dtype=torch.float32,
        )
        self.model.to.assert_called_once_with("cpu")
        self.model.eval.assert_called_once()
        self.assertEqual(self.model.generate.call_count, 2)

    def test_mps_uses_bfloat16_on_supported_macos(self):
        self.mps.return_value = True

        tencent.translate_with_tencent("你好。")

        self.model.to.assert_called_once_with("mps")
        self.assertEqual(self.load_model.call_args.kwargs["dtype"], torch.bfloat16)
        self.assertEqual(self.batch.device, torch.device("mps"))

    def test_older_mps_uses_float32(self):
        self.mps.return_value = True
        self.modern_macos.return_value = False

        tencent.translate_with_tencent("你好。")

        self.assertEqual(self.load_model.call_args.kwargs["dtype"], torch.float32)

    def test_prompt_inputs_sampling_and_decode_exclude_the_source_prefix(self):
        text = "火車八點出發。"

        result = tencent.translate_with_tencent(text)

        messages = self.tokenizer.apply_chat_template.call_args.args[0]
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        self.assertIn("into English.", messages[0]["content"])
        self.assertTrue(messages[0]["content"].endswith(text))
        self.assertEqual(self.tokenizer.apply_chat_template.call_args.kwargs, {
            "add_generation_prompt": True, "return_tensors": "pt", "return_dict": True,
        })
        settings = self.model.generate.call_args.kwargs
        self.assertIs(settings["input_ids"], self.batch["input_ids"])
        self.assertIs(settings["attention_mask"], self.batch["attention_mask"])
        self.assertEqual({key: value for key, value in settings.items() if key not in self.batch}, {
            "max_new_tokens": 512, "do_sample": True, "temperature": 0.7,
            "top_p": 0.6, "top_k": 20, "repetition_penalty": 1.05,
        })
        tokens = self.tokenizer.decode.call_args.args[0]
        torch.testing.assert_close(tokens, torch.tensor([7, 8, 9]))
        self.assertEqual(self.tokenizer.decode.call_args.kwargs, {"skip_special_tokens": True})
        self.assertEqual(result, "The train leaves at 8.")

    def test_missing_assets_raise_before_attempting_a_model_load(self):
        (self.model_dir / "model.safetensors").unlink()

        with self.assertRaisesRegex(tencent.TencentModelMissingError, "missing or incomplete"):
            tencent.translate_with_tencent("你好。")

        self.load_tokenizer.assert_not_called()
        self.load_model.assert_not_called()

    def test_failed_initial_load_does_not_poison_the_cache(self):
        self.load_model.side_effect = [RuntimeError("incompatible local weights"), self.model]

        with self.assertRaises(RuntimeError):
            tencent.translate_with_tencent("你好。")
        self.assertIsNone(tencent._model_cache)
        self.assertEqual(tencent.translate_with_tencent("你好。"), "The train leaves at 8.")

        self.assertEqual(self.load_model.call_count, 2)

    def test_blank_source_does_not_load_the_model(self):
        with self.assertRaisesRegex(ValueError, "no transcript"):
            tencent.translate_with_tencent(" \n ")

        self.load_model.assert_not_called()

    def test_output_limit_without_eos_is_an_error_instead_of_a_partial_translation(self):
        self.model.generate.return_value = torch.tensor([[1, 2, 3] + [7] * tencent.MAX_NEW_TOKENS])

        with self.assertRaisesRegex(RuntimeError, "did not finish"):
            tencent.translate_with_tencent("你好。")

        self.tokenizer.decode.assert_not_called()

    def test_empty_generated_or_decoded_output_is_rejected(self):
        self.model.generate.return_value = torch.tensor([[1, 2, 3]])
        with self.assertRaisesRegex(RuntimeError, "did not finish"):
            tencent.translate_with_tencent("你好。")

        self.model.generate.return_value = torch.tensor([[1, 2, 3, 9]])
        self.tokenizer.decode.return_value = "  "
        with self.assertRaisesRegex(RuntimeError, "empty"):
            tencent.translate_with_tencent("你好。")

    def test_multiple_eos_ids_and_eos_at_the_limit_are_valid(self):
        self.model.generation_config.eos_token_id = [9, 10]
        self.model.generate.return_value = torch.tensor([
            [1, 2, 3] + [7] * (tencent.MAX_NEW_TOKENS - 1) + [10],
        ])

        self.assertEqual(tencent.translate_with_tencent("你好。"), "The train leaves at 8.")

    def test_mixed_language_translation_gets_one_deterministic_repair_from_original_source(self):
        text = "原始 source ETCH-07"
        self.tokenizer.decode.side_effect = ["PRIVATE 工序 for ETCH-07.", "The ETCH-07 process."]

        result = tencent.translate_with_tencent(text)

        self.assertEqual(result, "The ETCH-07 process.")
        self.assertEqual(self.model.generate.call_count, 2)
        first, repair = [call.kwargs for call in self.model.generate.call_args_list]
        self.assertTrue(first["do_sample"])
        self.assertEqual(first["temperature"], 0.7)
        self.assertEqual(first["top_p"], 0.6)
        self.assertEqual(first["top_k"], 20)
        self.assertIs(repair["do_sample"], False)
        self.assertEqual(repair["max_new_tokens"], tencent.MAX_NEW_TOKENS)
        self.assertNotIn("temperature", repair)
        self.assertNotIn("top_p", repair)
        self.assertNotIn("top_k", repair)
        first_prompt, repair_prompt = [
            call.args[0][0]["content"] for call in self.tokenizer.apply_chat_template.call_args_list
        ]
        self.assertNotEqual(first_prompt, repair_prompt)
        self.assertTrue(first_prompt.endswith(text))
        self.assertTrue(repair_prompt.endswith(text))
        self.assertIn("English only", repair_prompt)
        self.assertNotIn("PRIVATE", repair_prompt)

    def test_repeated_mixed_language_translation_is_rejected_after_one_repair(self):
        self.tokenizer.decode.return_value = "PRIVATE 工序 is complete."

        with self.assertRaisesRegex(RuntimeError, "English") as caught:
            tencent.translate_with_tencent("PRIVATE SOURCE")

        self.assertNotIn("PRIVATE", str(caught.exception))
        self.assertEqual(self.model.generate.call_count, 2)

    def test_incomplete_repair_never_publishes_a_partial_translation(self):
        self.tokenizer.decode.return_value = "Mixed 工序."
        self.model.generate.side_effect = [
            torch.tensor([[1, 2, 3, 7, 8, 9]]),
            torch.tensor([[1, 2, 3, 7, 8]]),
        ]

        with self.assertRaisesRegex(RuntimeError, "did not finish"):
            tencent.translate_with_tencent("source")

        self.assertEqual(self.model.generate.call_count, 2)
        self.tokenizer.decode.assert_called_once()

    def test_generation_failure_is_not_retried(self):
        self.model.generate.side_effect = RuntimeError("Local generation failure")

        with self.assertRaises(RuntimeError):
            tencent.translate_with_tencent("source")

        self.model.generate.assert_called_once()

    def test_prompt_requires_english_and_latin_names_without_changing_english_identifiers(self):
        self.tokenizer.decode.return_value = "José checks ETCH-07 at 8."

        self.assertEqual(tencent.translate_with_tencent("檢查 ETCH-07。"), "José checks ETCH-07 at 8.")

        prompt = self.tokenizer.apply_chat_template.call_args.args[0][0]["content"]
        self.assertIn("English", prompt)
        self.assertIn("Latin", prompt)

    def test_concurrent_calls_share_one_load_and_do_not_overlap_generation(self):
        loading, release, second_started = Event(), Event(), Event()
        results, errors = [], []

        def load(*args, **kwargs):
            loading.set()
            release.wait(timeout=5)
            return self.model

        def translate(text, signal=None):
            if signal is not None:
                signal.set()
            try:
                results.append(tencent.translate_with_tencent(text))
            except Exception as exc:
                errors.append(exc)

        self.load_model.side_effect = load
        first = Thread(target=translate, args=("第一句",), daemon=True)
        second = Thread(target=translate, args=("第二句", second_started), daemon=True)
        first.start()
        try:
            self.assertTrue(loading.wait(timeout=2))
            second.start()
            self.assertTrue(second_started.wait(timeout=2))
            second.join(timeout=0.05)
            self.assertTrue(second.is_alive())
            self.load_model.assert_called_once()
        finally:
            release.set()
            first.join(timeout=5)
            if second.ident is not None:
                second.join(timeout=5)

        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 2)
        self.load_model.assert_called_once()
        self.assertEqual(self.model.generate.call_count, 2)


class TencentDownloadTests(unittest.TestCase):
    def test_download_is_pinned_public_and_limited_to_model_assets(self):
        with patch("scripts.download_tencent.snapshot_download") as download, patch("sys.stdout", new=StringIO()):
            download_tencent.main()

        download.assert_called_once_with(
            "tencent/Hy-MT2-1.8B", revision="9a341cd1b679d3efd23b46e847b01745a71ed792",
            local_dir=download_tencent.MODEL_DIR, allow_patterns=download_tencent.FILES, token=False,
        )
        self.assertIn("model.safetensors", download_tencent.FILES)
        self.assertIn("LICENSE.txt", download_tencent.FILES)
        self.assertFalse(any(name.endswith(".py") for name in download_tencent.FILES))


if __name__ == "__main__":
    unittest.main()
