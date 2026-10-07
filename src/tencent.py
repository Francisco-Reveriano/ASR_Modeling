"""Translate transcript segments to English with the local Tencent model."""

from pathlib import Path
from threading import Lock

import torch

from src.model_lock import LOCAL_MODEL_LOCK
from src.translation_validation import contains_cjk

MODEL_DIR = Path(__file__).resolve().parents[1] / "Models" / "Hy-MT2-1.8B"
MAX_NEW_TOKENS = 512
MAX_GENERATION_SECONDS = 60
TENCENT_ERROR_MESSAGE = "Local Tencent translation failed. Check the model files and retry."
REQUIRED_FILES = (
    "config.json", "generation_config.json", "tokenizer.json",
    "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
    "model.safetensors",
)
_model_cache = None
_status_lock = Lock()
_status = "not loaded"


def local_model_status():
    """Report local readiness without loading weights or contacting the Hub."""
    with _status_lock:
        return _status


class TencentModelMissingError(RuntimeError):
    """The separately downloaded local model is not ready for inference."""


def _load_model():
    """Load once while the caller holds LOCAL_MODEL_LOCK; never access the Hub."""
    global _model_cache, _status
    if _model_cache is None:
        if not all((MODEL_DIR / name).is_file() for name in REQUIRED_FILES):
            with _status_lock:
                _status = "missing model files"
            raise TencentModelMissingError(
                "The local Tencent model is missing or incomplete. Download it before translating."
            )
        with _status_lock:
            _status = "loading local weights"
        try:
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
        except Exception:
            with _status_lock:
                _status = "model loading failed"
            raise
        with _status_lock:
            _status = f"ready on {device.upper()}"
    return _model_cache


def translate_with_tencent(text: str, *, cancel_event=None) -> str:
    """Translate one completed transcript segment entirely on this computer.

    The first call loads Models/Hy-MT2-1.8B; subsequent calls reuse it. One lock
    serializes loading and generation with Breeze across sessions, preventing
    concurrent Metal command encoding. Modern Apple GPUs use
    bfloat16; CPU and older MPS devices use float32. No credentials, downloaded
    Python code, or automatic model downloads are used during inference.

    Tencent's default translation prompt targets the full language name
    "English". Only newly generated tokens are decoded. Empty or unfinished
    output raises an error rather than publishing a partial translation. Completed
    output with untranslated source script gets one deterministic English repair
    from the original source; repeated mixed-script output remains an error.
    """
    if not text.strip():
        raise ValueError("There is no transcript text to translate.")
    with LOCAL_MODEL_LOCK, torch.inference_mode():
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Local translation cancelled.")
        tokenizer, model = _load_model()
        instructions = (
            "Translate the following text into English. Note that you should only "
            "output the translated result without any additional explanation. "
            "Translate every phrase, including technical terms, into English. "
            "Use English names or Latin transliteration for names with no English form. "
            "Do not include Chinese or other untranslated source-script words. "
            "Preserve Latin identifiers and numbers exactly:"
        )
        for attempt in range(2):
            repair = (
                " Re-translate the original source below into English only. "
                "The previous attempt left untranslated words. Translate every word and phrase; "
                "render all names in English or Latin transliteration. "
                "Never copy Han characters, kana, Bopomofo, or Hangul into the output. "
                "Keep the original Latin identifiers and numbers exactly. "
                "Return only the complete English translation."
            ) if attempt else ""
            prompt = instructions + repair + "\n\n" + text
            inputs = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True, return_tensors="pt", return_dict=True,
            ).to(model.device)
            sampling = {"do_sample": False} if attempt else {
                "do_sample": True, "temperature": 0.7, "top_p": 0.6, "top_k": 20,
            }
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("Local translation cancelled.")
            limits = {"max_time": MAX_GENERATION_SECONDS}
            if cancel_event is not None:
                from transformers import StoppingCriteria, StoppingCriteriaList

                class Cancelled(StoppingCriteria):
                    def __call__(self, input_ids, scores, **kwargs):
                        return cancel_event.is_set()

                limits["stopping_criteria"] = StoppingCriteriaList([Cancelled()])
            output = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                repetition_penalty=1.05, **sampling, **limits,
            )
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("Local translation cancelled.")
            generated = output[0, inputs["input_ids"].shape[-1]:]
            eos = model.generation_config.eos_token_id
            eos_ids = eos if isinstance(eos, (list, tuple)) else [eos]
            if not generated.numel() or generated[-1].item() not in eos_ids:
                raise RuntimeError("The local translation did not finish within its output limit.")
            translation = tokenizer.decode(generated, skip_special_tokens=True).strip()
            if not translation:
                raise RuntimeError("The local translation was empty.")
            if contains_cjk(translation):
                if attempt == 0:
                    continue
                raise RuntimeError("The local translation was not entirely in English.")
            return translation
