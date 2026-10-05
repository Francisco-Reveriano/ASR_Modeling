"""Exercise both local model callers together without weights or GPU access."""

from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from src import pipeline, tencent


class ModelInputs(dict):
    def to(self, device):
        return self


class LocalModelConcurrencyTests(unittest.TestCase):
    def setUp(self):
        directory = Path(self.enterContext(TemporaryDirectory()))
        for filename in tencent.REQUIRED_FILES:
            (directory / filename).touch()
        self.enterContext(patch.object(tencent, "MODEL_DIR", directory))
        self.enterContext(patch.object(pipeline, "MODEL_DIR", directory / "breeze"))
        self.enterContext(patch.object(tencent, "_model_cache", None))

        processor = Mock()
        processor.return_value = ModelInputs(input_features=torch.zeros(1, 2))
        processor.batch_decode.return_value = ["Original transcript"]
        self.breeze_model = Mock()
        self.breeze_model.to.return_value = self.breeze_model
        self.breeze_model.eval.return_value = self.breeze_model
        self.breeze_model.generate.return_value = torch.tensor([[1, 2]])
        self.enterContext(patch("transformers.AutoProcessor.from_pretrained", return_value=processor))
        self.enterContext(patch(
            "transformers.AutoModelForSpeechSeq2Seq.from_pretrained", return_value=self.breeze_model,
        ))

        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = ModelInputs(input_ids=torch.tensor([[1, 2]]))
        tokenizer.decode.return_value = "English translation"
        self.tencent_model = Mock()
        self.tencent_model.device = torch.device("cpu")
        self.tencent_model.generation_config = SimpleNamespace(eos_token_id=9)
        self.tencent_model.to.return_value = self.tencent_model
        self.tencent_model.eval.return_value = self.tencent_model
        self.tencent_model.generate.return_value = torch.tensor([[1, 2, 7, 9]])
        self.enterContext(patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer))
        self.load_tencent = self.enterContext(patch(
            "transformers.AutoModelForCausalLM.from_pretrained", return_value=self.tencent_model,
        ))
        self.enterContext(patch("torch.backends.mps.is_available", return_value=False))
        self.transcribe = pipeline.load_transcriber()
        self.audio = np.zeros(800, dtype=np.float32)

    def assert_waits_for_running_model(self, model, first_call, second_call, second_entered):
        running, release, second_started = Event(), Event(), Event()
        results, errors = [], []

        def blocked_generation(*args, **kwargs):
            running.set()
            if not release.wait(timeout=5):
                raise RuntimeError("test did not release the running model")
            return model.generate.return_value

        def invoke(call, started=None):
            if started is not None:
                started.set()
            try:
                results.append(call())
            except Exception as exc:
                errors.append(exc)

        model.generate.side_effect = blocked_generation
        first = Thread(target=invoke, args=(first_call,), daemon=True)
        second = Thread(target=invoke, args=(second_call, second_started), daemon=True)
        first.start()
        try:
            self.assertTrue(running.wait(timeout=2), "first model did not start")
            second.start()
            self.assertTrue(second_started.wait(timeout=2))
            self.assertFalse(
                second_entered.wait(timeout=0.25),
                "another local model entered while generation was still running",
            )
        finally:
            release.set()
            first.join(timeout=5)
            if second.ident is not None:
                second.join(timeout=5)
            self.assertFalse(first.is_alive(), "first model thread did not exit")
            self.assertFalse(second.is_alive(), "second model thread did not exit")

        self.assertEqual(errors, [])
        self.assertTrue(second_entered.is_set(), "waiting model never resumed")
        self.assertCountEqual(results, ["Original transcript", "English translation"])

    def test_tencent_first_load_and_placement_wait_for_breeze_generation(self):
        tencent_entered = Event()

        def load_or_place(*args, **kwargs):
            tencent_entered.set()
            return self.tencent_model

        self.load_tencent.side_effect = load_or_place
        self.tencent_model.to.side_effect = load_or_place
        self.assert_waits_for_running_model(
            self.breeze_model,
            lambda: self.transcribe(self.audio),
            lambda: tencent.translate_with_tencent("原文"),
            tencent_entered,
        )

    def test_breeze_generation_waits_for_cached_tencent_generation(self):
        self.assertEqual(tencent.translate_with_tencent("載入模型"), "English translation")
        breeze_entered = Event()

        def generate(*args, **kwargs):
            breeze_entered.set()
            return self.breeze_model.generate.return_value

        self.breeze_model.generate.side_effect = generate
        self.assert_waits_for_running_model(
            self.tencent_model,
            lambda: tencent.translate_with_tencent("原文"),
            lambda: self.transcribe(self.audio),
            breeze_entered,
        )
        self.load_tencent.assert_called_once()


if __name__ == "__main__":
    unittest.main()
