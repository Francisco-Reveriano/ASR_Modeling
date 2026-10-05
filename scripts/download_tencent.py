"""Download the pinned public Tencent translator; no Hugging Face token is needed."""

from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_ID = "tencent/Hy-MT2-1.8B"
REVISION = "9a341cd1b679d3efd23b46e847b01745a71ed792"
MODEL_DIR = Path(__file__).resolve().parents[1] / "Models" / "Hy-MT2-1.8B"
FILES = [
    "config.json", "generation_config.json", "tokenizer.json",
    "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
    "model.safetensors", "README.md", "LICENSE.txt",
]


def main():
    snapshot_download(
        MODEL_ID, revision=REVISION, local_dir=MODEL_DIR,
        allow_patterns=FILES, token=False,
    )
    print(f"Downloaded {MODEL_ID} at revision {REVISION} to {MODEL_DIR}")


if __name__ == "__main__":
    main()
