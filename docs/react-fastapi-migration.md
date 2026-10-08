# React and FastAPI local deployment

The `OpenAI` branch runs a React/TypeScript workspace backed by a single FastAPI
process. The processing adapters in `src/` keep their existing models, prompts,
queues, locks, and English-only validation. The browser never receives API keys.

## Run and development

From the repository root, install Python requirements, run `npm --prefix frontend ci`
and `npm --prefix frontend run build`, then `python -m server`. Open
`http://localhost:8000`. For frontend development, `npm --prefix frontend run dev`
serves port 5173 and proxies `/api`, including WebSockets, to port 8000.

Run one Uvicorn worker. Session ownership, Breeze's single-flight cache, Tencent's
cache, Nemotron's cache, and the shared Breeze/Tencent GPU lock are process-local.
Multiple processes would duplicate model memory and defeat shared serialization.
Model loading and inference run on background threads, outside the ASGI event loop.

## Session ownership

`server/sessions.py` owns a conversation independently of HTTP requests and event
subscribers. It seeds source metadata, submits each new row to provider queues,
coordinates review passes, advances file jobs, and selects accepted English for
speech. A browser repaint or reconnect never schedules duplicate inference.

Each conversation freezes its model, translation workflow, and correction settings.
Corrections can be paused, failed provider rows retried, and validated terminology
reloaded. Optional speech can be enabled/disabled or reconfigured during a session;
this never changes the frozen correction reasoning effort. Enabling or changing
speech creates a new speech queue for future source segments only. Realtime mode
uses the current English-caption prefix as its boundary.

**Stop** flushes microphone audio and drains accepted work. File Stop retains
partial results and marks them incomplete. **Cancel** discards pending work and
suppresses late publication while retaining already published results. **End and
clear** deletes the session and releases retained references, transcript and audio
buffers. A valid replacement clears the old conversation; invalid input leaves the
old result available. Browser presentation preferences remain after clearing.

Disconnected sessions expire after 30 minutes. Connected sessions stay available
for review. Speech has its own consumer lease: coordinator snapshots do not renew
it. A vanished player therefore cannot keep bounded TTS buffers alive forever.
Closing a session prevents an in-flight model/API result from entering a new one;
already running native operations or provider calls may finish before releasing
their resources. There is no durable session database or raw recording archive.

## HTTP contract

The machine-readable schema is at `/api/openapi.json`; the default CDN-backed
Swagger/Redoc pages are disabled. Errors exclude credentials and raw provider bodies.

| Method and path | Behavior |
| --- | --- |
| `GET /api/health` | Service health, independent of model inference |
| `GET /api/config` | Defaults, supported model choices, local-asset presence and OpenAI configuration presence |
| `POST /api/sessions` | JSON `SessionSettings`; return a preparing microphone session |
| `POST /api/sessions/upload` | Multipart `file` and JSON-string `settings`; validate/decode WAV before allocating the job |
| `POST /api/sessions/evaluate` | Also accepts `reference` and JSON-string `reference_options`; forces the Breeze baseline and disables speech |
| `POST /api/references/preview` | Multipart reference `file` plus JSON-string `options`; no inference |
| `GET /api/sessions/{id}` | Complete current snapshot |
| `POST /api/sessions/{id}/commands` | Discriminated `type`: `stop`, `cancel`, `retry`, `corrections`, `speech`, `terminology` |
| `DELETE /api/sessions/{id}` | Idempotent End and clear |
| `GET /api/sessions/{id}/exports/{format}` | Server-generated TXT, JSON, bilingual CSV, SRT or VTT; export metadata logged without transcript text |

WAV uploads are limited to 200 MiB; references to 1 MiB. Existing table expansion,
row and column limits remain enforced. Reference preview options include format,
worksheet, 1-based header row (`0` means no header), and zero-based source/English
column indices. Ambiguous source columns require an explicit selection. References
stay in the evaluation boundary and never enter ASR, translation or correction prompts.

## Audio and event sockets

The microphone socket `/api/sessions/{id}/audio` starts with:

```json
{"type":"capture.start","sample_rate":48000,"channels":1,"format":"pcm_s16le"}
```

Use the browser's actual sample rate. The server replies `capture.ready` after
preparation. The AudioWorklet sends approximately 100 ms binary PCM16LE batches,
including silence. The persistent PyAV resampler feeds the existing 16 kHz VAD and
speaker timeline (24 kHz for standalone realtime translation). These transport
batches do not change model chunk sizes. After flushing the final partial batch,
send `{"type":"capture.finish"}`. An interrupted socket ends input; it cannot be
reconnected as the same recording. Overflow is surfaced, never silently dropped.

The event socket `/api/sessions/{id}/events` sends:

- `snapshot`: complete initial state.
- `update`: session ID/revision, changed metadata fields, and changed segment
  rows. Retain omitted metadata, merge rows by immutable ID, preserve source
  order and reject stale revisions. Progress updates do not resend unchanged
  correction audit history.
- `speech`: current speech metadata and new PCM chunks, or `null` when disabled.
- `heartbeat`: one per second, allowing the browser to detect an outage within
  five seconds. Reconnecting restores the latest snapshot without rerunning jobs.
- `cleared`: the session was deleted.

Speech uses 24 kHz PCM16 and the existing bounded queue. Only completed playback
acknowledges a chunk:

```json
{"type":"speech.ack","speech_session_id":"...","played":12}
```

Retain newly received chunks until played; reconnect resends unacknowledged chunks,
which the controller deduplicates. Playback and capture use separate AudioContexts.
The one-minute lead starts with the first accepted English passage in each speech
queue. Two-second underrun buffering and four-second browser scheduling preserve
continuous delivery; later text corrections never replay speech already queued.

## TRS-LT-001-S1 coverage and explicit exceptions

The frontend provides one-action start with deployment defaults, English-first
subtitles, matching source reveal, hide/re-show, presentation preferences, model
provenance, processing disclosure, and named connection/capability/degraded states.
All frontend assets are local; HTTP and WebSocket origins are checked. The default
is text only. Upload previews play only when the user chooses playback.

The user explicitly retained OpenAI inference and optional spoken output for this
local release. Therefore the indicator explains data leaving the computer; this
release does not claim an entirely private inference boundary or strict text-only
compliance. Standalone realtime captions remain two unaligned streams, not invented
source/English segment pairs. Unsupported-language detection is not implemented.

Authentication/RBAC, QA retention, shared-room links, private GPU deployment,
distributed session recovery, new ASR partials/stabilization and four-hour production
capacity certification remain outside this migration. Client readiness and model
latency are distinct: a ready microphone accepts audio; cold model loading and
inference may exceed the specification's targets. Automated fixture tests establish
behavior, not production p95 latency or translation quality.

## Verification

Install `requirements-dev.txt`; run `python -m unittest discover -s tests -v`,
`npm --prefix frontend test`, and `npm --prefix frontend run build`.
API tests cover PCM ordering/final flush, origins, validation, reconnection, reference
selection, expiry and deletion. Session tests use substituted providers to prove
autonomous progress, frozen settings, cancellation and reference isolation. Browser
tests cover capture cleanup, playback ordering, buffering, acknowledged chunks,
speech toggles and state updates. See [the live validation record](react-runtime-validation.md)
for the real WAV, WebSocket, local model and speech API checks and their limits.
