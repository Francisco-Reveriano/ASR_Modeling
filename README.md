# vLLM Voice — Linux transcription and English translation

This branch runs a Streamlit app on Linux and connects to existing vLLM model
instances. Local Silero VAD splits microphone or uploaded WAV audio into speech
segments. A hosted ASR model returns the original-language transcript, then one
selected hosted translation model produces English. ASR and translation may use
separate servers and keys. The app does not provision model servers.

The previous OpenAI, Tencent, speaker detection, and two-pass correction tool is
preserved on the `OpenAI-Translation-Updates` branch. Those workers and large
local models are absent from this branch's runtime.

## Linux setup

Create a Python environment on the Linux machine. Python 3.11 or newer is
recommended. Install the CPU build of PyTorch first, since only speech detection
runs in this process; the hosted models use their own servers' GPUs.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
cp .env.example .env  # new checkouts only; preserve an existing .env
# Edit .env with your endpoint URLs and served model IDs.
python -m streamlit run app.py --server.address localhost --server.headless true
```

The CPU wheel index is provided by [PyTorch](https://docs.pytorch.org/get-started/previous-versions/).
No CUDA toolkit, `vllm` package, OpenAI key, Hugging Face token, or downloaded ASR
and translation weights are required on the app machine. Silero ships its small
speech detector with its Python package. Existing `Models/` files are ignored.
The historical Breeze notebook has optional dependencies in
`requirements-notebooks.txt` and is not part of this app's setup.

Open <http://localhost:8501> in a browser on the Linux machine. To serve a browser
on another machine, configure your existing HTTPS reverse proxy and, if needed,
bind Streamlit with `--server.address 0.0.0.0`. Browser microphone capture needs
HTTPS or localhost. WebRTC also needs an audio route between browser and server;
an HTTP/SSH tunnel alone does not relay its media. Set `WEBRTC_ICE_SERVERS_JSON`
for your STUN/TURN service when your network requires it. The default is `[]`,
with no third-party ICE server. See the
[streamlit-webrtc deployment requirements](https://github.com/whitphx/streamlit-webrtc#https).
Upload and evaluation do not need WebRTC or microphone permission.

Restart Streamlit after Python or `.env` changes: file watching is disabled.

## Connect existing vLLM instances

[`.env.example`](.env.example) documents every supported setting. Process
environment variables take precedence over the repository's local `.env`.
Base URLs can identify the server root or end in `/v1`; use the model ID exposed
by each deployment. The app does not select a checkpoint or guess a server URL.

```dotenv
VLLM_ASR_BASE_URL=http://asr-host:8000/v1
VLLM_ASR_MODEL=your-served-asr-id
VLLM_ASR_API_KEY=
VLLM_ASR_LANGUAGE=

VLLM_FAST_PROFILES=QUICK,ALTERNATE
VLLM_DEFAULT_FAST_PROFILE=QUICK
VLLM_FAST_QUICK_LABEL=Quick English
VLLM_FAST_QUICK_BASE_URL=http://translator-one:8000/v1
VLLM_FAST_QUICK_MODEL=your-first-served-translator-id
VLLM_FAST_QUICK_API_KEY=
VLLM_FAST_QUICK_MAX_TOKENS=512
VLLM_FAST_QUICK_TEMPERATURE=
VLLM_FAST_QUICK_REASONING_EFFORT=
VLLM_FAST_QUICK_CHAT_TEMPLATE_KWARGS=

VLLM_FAST_ALTERNATE_LABEL=Alternate English
VLLM_FAST_ALTERNATE_BASE_URL=http://translator-two:8000/v1
VLLM_FAST_ALTERNATE_MODEL=your-second-served-translator-id
VLLM_FAST_ALTERNATE_API_KEY=
VLLM_FAST_ALTERNATE_MAX_TOKENS=512

VLLM_FILTER_BACKGROUND_SPEECH=true
GLOSSARY_FILE=
DNT_FILE=
WEBRTC_ICE_SERVERS_JSON=[]
```

The example hosts and model IDs above are placeholders. Every alias listed in
`VLLM_FAST_PROFILES` needs its own group. Optional temperature, reasoning effort,
and chat-template kwargs are omitted from requests when blank. Reasoning and
thinking controls are model-specific; for example, a supported deployment might
use `{"enable_thinking": false}` for its chat-template kwargs. The app displays
only final assistant content, never a reasoning field or unparsed thinking markup.
The token budget includes whatever generation the served model counts against
`max_tokens`; increase it for models that spend tokens reasoning.

The hosted ASR must support
[`POST /v1/audio/transcriptions`](https://docs.vllm.ai/en/stable/serving/online_serving/speech_to_text/)
with a multipart WAV and JSON `{ "text": "..." }` response. Each request contains
one bounded mono 16 kHz PCM segment, its model ID, and an optional language code.
Blank language leaves language selection to the served model (auto-detection
where supported). This is original-language transcription,
not the audio translation API.

The selected translator must support
[`POST /v1/chat/completions`](https://docs.vllm.ai/en/stable/serving/online_serving/openai_compatible_server/)
and return completed final text. The OpenAI-compatible protocol does not require
an OpenAI account: all requests go to the explicitly configured vLLM URLs, with
that instance's key. Legacy `OPENAI_*` environment settings are unused.

## Record, upload, and evaluate

Choose **Translation model** before starting. That selection applies to
**Microphone**, **Upload WAV**, and **Evaluate**. Each conversation freezes its
ASR and translation configuration, terminology, and model labels; a later
selection applies to the next conversation. A manual translation retry uses the
original conversation's model settings.

For microphone input, click **Start recording** and allow browser access.
Speech normally ends after 500 ms of silence; continuous speech splits at
15 seconds. **Stop recording** flushes the last speech and drains accepted work.
A new conversation closes previous clients and ignores their late results.

For uploads, choose a WAV and click **Transcribe file**. Selecting a file alone
does not send it to ASR. Mono/stereo audio is decoded in blocks, averaged to mono,
and resampled to 16 kHz. Silero detects bounded segments with original-file
timestamps. Only those segments are sent to hosted ASR, not the entire WAV in one
request. Upload jobs preserve their current task through Streamlit reruns.

ASR and translation run on separate workers. A completed transcript immediately
queues English translation without waiting for the browser to refresh. Rows stay
in speech order. The UI shows source text and the selected English model, with
explicit pending, unavailable, and background-filtered states. Completed English
is used in TXT/JSON/CSV downloads and timestamped SRT/VTT captions. Pending,
failed, and fully filtered translations are excluded from captions.

There is no application inference deadline or automatic network retry. A slow
server remains pending. Connection/provider failures show a safe error; **Retry
translation** resubmits only failed translations. Retry a failed upload by starting
it again, or start a new microphone recording after ASR failure. An upstream
proxy or vLLM server can still impose its own limits.

Fast translation receives the original transcript, up to two recent completed
main-conversation rows, and relevant configured terminology. It conservatively
filters clearly unrelated background chatter; uncertain speech, brief replies,
technical fragments, and topic changes are retained. This is text filtering,
not audio source separation. Set `VLLM_FILTER_BACKGROUND_SPEECH=false` to translate
every segment. Original transcripts remain visible even when filtered.

English output that retains source-script text or loses a protected identifier
gets one bounded repair request. Empty, truncated, malformed, refused, or tool-call
responses are failures. Source text is never substituted into an English cell.
There is no slow correction pass in this version.

### References and terminology

Evaluation accepts WAV plus TXT/SRT/VTT/XLSX/CSV/TSV references. Table controls
select a worksheet, header row, source transcript column, and optional English
comparison column. Reference previews and the separately numbered reference pane
remain local. Changing the selected files/settings does not alter an existing
evaluation; running again freezes a new set of inputs.

Only the complete joined original-language ASR transcript is scored against the
source reference. Mixed Chinese/English references use character/word **Mixed
match**; English source references use **1-wMER**. Both compute matches divided
by matches + substitutions + deletions + insertions after normalization. This is
a text-match measure, not semantic translation quality. Pending or failed ASR
has no final score. English references support visual comparison only and are
never included in endpoint requests.

`GLOSSARY_FILE` and `DNT_FILE` configure local terminology; relative paths resolve
from the repository. CSV/TSV/JSON/XLSX glossaries use `term_src`, `term_tgt`, and
optional `aliases_src` (pipe-separated), `dnt`, `domain`, `priority`,
`example_src`, and `example_tgt`. DNT files can also be plain text, one identifier
per line. Protected identifiers must use English/Latin script; use ordinary
source-to-English glossary entries for Chinese words. Only relevant terminology
and recent conversation context are sent to the selected translator.

## Memory and privacy

The Linux app keeps the small CPU VAD model, audio buffers, conversation text,
and endpoint clients. It does not load Breeze, Tencent, Nemotron, Transformers,
or OpenAI clients. Microphone capture retains at most eight waiting segments
plus one in progress; if ASR falls behind that bound, capture stops visibly and
accepted work drains. Processed speech buffers are released. Uploads still hold
decoded mono float32 audio until processing ends (about 220 MiB per hour), plus
the uploaded file and any resampling buffers. Transcript/export memory grows
with conversation length. Hosted model memory belongs to the vLLM servers.

Audio speech segments leave the app machine for the configured ASR instance.
Completed transcript text, recent English, and relevant glossary/DNT entries go
to the selected translator instance. References remain local. Endpoint keys stay
on the server and are omitted from UI errors and conversation exports. Model
servers control their own logging/retention; this client does not configure them.
Keep `.env`, model weights, datasets, and private recordings out of Git.

## Verification and code layout

```bash
python -m unittest discover -s tests -v
```

Tests exercise audio segmentation/lifecycles, mocked HTTP payloads and malformed
responses, translation context/retry/English validation, Streamlit flows, exports,
and local evaluation without real model calls. Mock tests cannot establish
hosted-model accuracy or server compatibility; verify your configured deployments
with a short upload before relying on live microphone translation.

Migration verification: 212 tests passed in a clean Linux ARM64 container with
Python 3.11 and CPU PyTorch, plus real Silero silence detection and a Streamlit
health check. The environment contained no OpenAI, Transformers, or vLLM Python
packages. Live hosted inference still requires your deployment's endpoint URLs
and served model IDs.

`src/vllm.py` owns endpoint configuration and transport. `src/pipeline.py` and
`src/uploads.py` prepare audio; `src/translation.py` owns the translation queue.
`src/ui.py` and `src/subtitle_exports.py` render/export completed English;
`src/evaluation.py` and `src/reference_tables.py` process local references.
See [the branch architecture](docs/vllm-architecture.md) for worker boundaries.
