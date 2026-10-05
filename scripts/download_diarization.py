"""Download native Transformers Nemotron diarization assets at a fixed revision."""

from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_ID = "nvidia/Nemotron-3-Diarization"
REVISION = "f667ed73aee57d40cc39428eb768b4fd87a0a29e"
MODEL_DIR = Path(__file__).resolve().parents[1] / "Models" / "Nemotron-3-Diarization"
FILES = ["config.json", "processor_config.json", "model.safetensors", "README.md", "ASR_INTEGRATION_GUIDE.md"]


def main():
    snapshot_download(
        MODEL_ID, revision=REVISION, local_dir=MODEL_DIR,
        allow_patterns=FILES, token=False,
    )
    print(f"Downloaded {MODEL_ID} at revision {REVISION} to {MODEL_DIR}")


if __name__ == "__main__":
    main()
