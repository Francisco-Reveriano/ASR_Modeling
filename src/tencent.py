"""Translate transcript segments to English with the local Tencent model."""

from pathlib import Path

import torch

from src.model_lock import LOCAL_MODEL_LOCK

MODEL_DIR = Path(__file__).resolve().parents[1] / "Models" / "Hy-MT2-1.8B"
MAX_NEW_TOKENS = 512
TENCENT_ERROR_MESSAGE = "Local Tencent translation failed. Check the model files and retry."
REQUIRED_FILES = (
    "config.json", "generation_config.json", "tokenizer.json",
    "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
    "model.safetensors",
)
_model_cache = None


class TencentModelMissingError(RuntimeError):
    """The separately downloaded local model is not ready for inference."""


def _load_model():
    """Load once while the caller holds LOCAL_MODEL_LOCK; never access the Hub."""
    global _model_cache
    if _model_cache is None:
        if not all((MODEL_DIR / name).is_file() for name in REQUIRED_FILES):
            raise TencentModelMissingError(
                "The local Tencent model is missing or incomplete. Download it before translating."
            )
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = (
            torch.bfloat16
            if device == "mps" and torch.backends.mps.is_macos_or_newer(14, 0)
            else torch.float32
        )
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR, local_files_only=True, trust_remote_code=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR, local_files_only=True, trust_remote_code=False, dtype=dtype,
        ).to(device).eval()
        _model_cache = tokenizer, model
    return _model_cache


def translate_with_tencent(text: str) -> str:
    """Translate one completed transcript segment entirely on this computer.

    The first call loads Models/Hy-MT2-1.8B; subsequent calls reuse it. One lock
    serializes loading and generation with Breeze across sessions, preventing
    concurrent Metal command encoding. Modern Apple GPUs use
    bfloat16; CPU and older MPS devices use float32. No credentials, downloaded
    Python code, or automatic model downloads are used during inference.

    Tencent's default translation prompt targets the full language name
    "English". Only newly generated tokens are decoded. Empty or unfinished
    output raises an error rather than publishing a partial translation.
    """
    if not text.strip():
        raise ValueError("There is no transcript text to translate.")
    with LOCAL_MODEL_LOCK, torch.inference_mode():
        tokenizer, model = _load_model()
        prompt = (
            "Translate the following text into English. Note that you should only "
            "output the translated result without any additional explanation:\n\n"
            f"{text}"
        )
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, return_tensors="pt", return_dict=True,
        ).to(model.device)
        output = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True, temperature=0.7, top_p=0.6,
            top_k=20, repetition_penalty=1.05,
        )
        generated = output[0, inputs["input_ids"].shape[-1]:]
        eos = model.generation_config.eos_token_id
        eos_ids = eos if isinstance(eos, (list, tuple)) else [eos]
        if not generated.numel() or generated[-1].item() not in eos_ids:
            raise RuntimeError("The local translation did not finish within its output limit.")
        translation = tokenizer.decode(generated, skip_special_tokens=True).strip()
        if not translation:
            raise RuntimeError("The local translation was empty.")
        return translation
