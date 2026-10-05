# Repository Guidelines

This branch is the Linux Streamlit client for existing vLLM ASR and translation
instances. Read `README.md` before changing recording, upload, evaluation,
translation, endpoint configuration, or setup behavior. The previous OpenAI and
local large-model workflow is preserved on `OpenAI-Translation-Updates`.

## Structure and development

Keep reusable processing in `src/` and Streamlit state/controls in `app.py`.
`src/pipeline.py` owns local Silero VAD and microphone segmentation;
`src/uploads.py` prepares WAV uploads; `src/vllm.py` owns endpoint configuration
and transport; `src/translation.py` owns the ordered translation worker.
Rendering and downloads live in `src/ui.py` and `src/subtitle_exports.py`.
Evaluation and reference parsing remain local.

Run from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.address localhost
python -m unittest discover -s tests -v
```

Restart Streamlit after Python/configuration edits; file watching is disabled.
Tests use substitute models and HTTP transports. Run relevant tests for edits,
then the full suite before delivery. Check affected UI flows in Streamlit, and
report live endpoint checks separately from mocked tests. Endpoint URLs and model
IDs must come from the deployment configuration, not guessed defaults.

Use four-space indentation, snake_case functions/variables, UPPER_CASE constants,
and pathlib.Path. There is no configured formatter or separate build step.
The historical asset notebook has separate optional requirements in
`requirements-notebooks.txt`; its relative paths require a `Notebooks/` cwd.

## Processing invariants

- Freeze one ASR endpoint and one selected translation profile per conversation.
  A new selection applies to the next conversation; retries keep the old profile.
- Only Silero VAD runs locally. Send bounded mono 16 kHz PCM WAV speech segments
  to the configured ASR endpoint, then original transcript text and bounded
  conversation/terminology context to the selected translation endpoint.
- Run endpoint calls outside capture/UI threads and state locks. Start translation
  from ASR completion callbacks; preserve transcript order independently of polling.
- Preserve local audio timestamps, release processed audio, close owned clients,
  and discard late results after a conversation closes.
- Use only validated complete English for translation display and captions.
  Preserve explicit filtered/error/pending states; references never enter prompts.
- Score the complete joined original-language ASR transcript against the local
  source reference. Translation output remains available for visual comparison.
- No application inference deadline or automatic network retries. Invalid English
  or missing protected identifiers gets one bounded content-repair request.

## Configuration and commits

`.env.example` documents the supported environment options; process environment
variables take precedence over local `.env`. Each vLLM endpoint uses its own
optional key. Keep keys out of configuration repr, logs, errors, and exports.
Legacy OpenAI credentials must not be used for vLLM requests.

Use concise imperative commit subjects and describe behavior, validation, and any
dependency changes. Review staged files for credentials, `.env`, model/data
assets, caches, and IDE files. Keep downloaded assets excluded from commits.
