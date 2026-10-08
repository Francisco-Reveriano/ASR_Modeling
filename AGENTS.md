# Repository Guidelines

## Project Structure & Module Organization

This project runs a local React frontend and FastAPI server for speech recognition and English translation, with notebook workflows for preparing assets. Read `README.md` when changing recording, upload, evaluation, translation, or model setup behavior. Read `docs/react-fastapi-migration.md` when changing browser/server contracts, audio transport, or session ownership.

- `Notebooks/01 Download Assets.ipynb` downloads Breeze-ASR-26 and the Taiwanese-Minnan-Sutiau dataset from Hugging Face, then demonstrates local loading.
- `server/` owns HTTP/WebSocket contracts, session lifecycles, background orchestration, and model caching.
- `frontend/` contains the React/TypeScript workspace and separate microphone/playback audio controllers.
- `src/` contains reusable audio processing (`pipeline.py`, `uploads.py`), translation (`translation.py`, `tencent.py`), evaluation (`evaluation.py`), and rendering/export (`ui.py`).
- `frontend/src/styles.css` controls presentation; Vite proxies `/api` during development.
- `tests/` contains `unittest` tests with model and API substitutes.
- `Models/breeze-asr-26/` contains downloaded model weights, tokenizer files, and configuration.
- `scripts/download_tencent.py` downloads a pinned translator revision into `Models/Hy-MT2-1.8B/`.
- `Data/taiwanese-minnan-sutiau/` is created by the dataset download workflow.
- `requirements.txt` lists runtime and notebook dependencies.

Keep reusable processing logic in `src/` and exploratory workflows in `Notebooks/`.

## Build, Test, and Development Commands

Run these commands from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
npm --prefix frontend ci
npm --prefix frontend run build
python -m server
python -m unittest discover -s tests -v
npm --prefix frontend test
```

Prepare the local models needed by the selected mode; use the notebook for Breeze, `python scripts/download_tencent.py` for Tencent, and `python scripts/download_diarization.py` for Nemotron. Model downloads are separate from tests. Restart the Python server after backend changes. Run one Uvicorn worker: caches and model locks are process-local.

For notebook work, run `python -m jupyterlab` from `Notebooks/` and use that directory as the kernel working directory: the notebook's `../Models` and `../Data` paths depend on it. Execute cells in order. Notebooks need no separate build step.

## Coding Style & Naming Conventions

Follow the existing Python style: four-space indentation, `snake_case` variables and functions, and `UPPER_CASE` configuration constants. Use `pathlib.Path` for filesystem paths. Name sequential notebooks with a numeric prefix and descriptive title, such as `02 Transcribe Audio.ipynb`. No formatter or linter is configured; keep changes consistent with surrounding code.

## Testing Guidelines

Run the relevant `unittest` tests for Python changes and Vitest tests plus the production build for frontend changes. Tests use model/API substitutes. For UI changes, also check the affected browser flow when local assets are available. Record separately any real model/API checks and measured latency; mocked tests do not establish production targets.

For notebook changes, restart the kernel and run the affected workflow in order. Verify saved assets with the final local-loading cells. The notebook requires network access, `HF_TOKEN`, and sufficient disk space and memory. Record which checks ran and any skipped downloads or live-model checks in the pull request.

## Processing Invariants

- Keep Breeze and Tencent loading/inference under the shared `LOCAL_MODEL_LOCK` in `src/model_lock.py`; overlapping local GPU calls previously caused native Metal crashes. Keep microphone capture, VAD, and API requests outside that lock.
- Give each translation provider its own queue and the same original transcript. Preserve segment order and discard results from closed sessions when a new conversation starts.
- Keep evaluation references local and out of model prompts. OpenAI ASR/realtime modes send audio to OpenAI; Breeze keeps audio local. Disclose the active processing route. Keep response storage disabled where supported and credentials out of errors.
- Keep session progression on the server, independent of HTTP polling or WebSocket subscribers. Stop drains accepted work; cancellation suppresses late results; End and clear releases retained session data.
- Keep capture and playback AudioContexts separate. Speech uses accepted English, ordered PCM and played acknowledgements; changing speech settings never changes frozen correction settings.
- Score each provider's complete, joined English output against the full reference. Pending or failed output has no final score; reference text must never enter model prompts.

## Commit & Pull Request Guidelines

Use concise, imperative subjects, such as `Add local transcription notebook`. Describe the change, validation performed, and any dependency or asset changes in pull requests; link related issues when applicable.

## Security & Configuration

Configure `OPENAI_API_KEY` and optionally `OPENAI_DEFAULT_MODEL` in the repository's local `.env`; existing environment variables take precedence in the app. The notebook uses `HF_TOKEN` through `load_dotenv`; the app loads Breeze and Tencent from local files without a Hugging Face token. See `README.md` for setup details.

Keep tokens and credential-bearing outputs out of commits. Review staged files for `.env`, downloaded assets, caches, and local IDE files before committing; document asset sources instead of adding large downloads by default.
