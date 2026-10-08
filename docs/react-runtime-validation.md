# React/FastAPI validation — 2026-10-07

Checks ran on the local `OpenAI` branch using Python 3.14.7 and Node 26.7.0.
This is functional validation of the migration, not a production latency or
translation-quality benchmark. Several live checks overlapped.

## Automated checks

- `python -m unittest discover -s tests -v`: 463 passed.
- Final API regression rerun: 17 passed, including metadata deltas and reconnect.
- `npm --prefix frontend test`: 65 passed across seven files.
- `npm --prefix frontend run build`: TypeScript and Vite production build passed.
- `git diff --check`: passed.

The running server returned its health response, built React JS/CSS assets, and
diarization guide successfully. A real events WebSocket delivered a full initial
snapshot and partial metadata updates while retaining session state. Progress-only
updates omit unchanged audit history; initial/reconnect snapshots and actual
correction-history changes still carry the full history. Long-session audit memory
and payload size have not been certified by a soak test.

## Full WAV through the API

Uploaded `/Users/francisco/Downloads/chinglish_test.wav` through
`POST /api/sessions/upload` with the default OpenAI ASR/Luna pair, full comparison,
local Tencent, both Astra review passes, Nemotron, and speech disabled.
The file contains 306.043 seconds of mono 16 kHz PCM16 audio.

| Check | Observed result |
| --- | --- |
| Job creation | 0.11 seconds |
| First completed ASR row | 15.41 seconds |
| First OpenAI English draft | 19.44 seconds |
| First local Tencent translation | 23.46 seconds |
| Complete job, including reviews and diarization | 243.85 seconds |
| Final transcript rows | 48, in source order |
| OpenAI and Tencent results | 48 each; no provider errors |
| First Astra pass | 34 confirmed, 14 corrected; no pending rows |
| Second Astra pass | 47 confirmed, 1 corrected; no pending rows |
| Nemotron | Complete without error; one anonymous channel observed |
| TXT, JSON, CSV, SRT, VTT | All returned HTTP 200 with nonempty content |

Post-run assertions checked English script guards for every provider, accepted
review states, and all 48 CSV segment IDs against the conversation IDs. Speaker
identity/diarization accuracy was not scored against a labelled reference. The
first-Astra-display timestamp was not used as a correction latency measurement:
that display may contain an unreviewed draft.

## Real microphone protocol and speech requests

The first 12 seconds of the same WAV were sent as paced 100 ms PCM frames through
the actual audio WebSocket. These were transport tests, not physical-microphone
recordings. Both connections received `capture.ready` and `capture.finished`.

| Route | Observed result |
| --- | --- |
| `gpt-live-transcribe + gpt-6-luna` | Completed in 19.09 seconds; two transcript rows and English output |
| `gpt-realtime-translate` | Completed in 18.54 seconds; source and English captions |
| `gpt-4o-mini-tts` | 67,200 PCM bytes in seven distinct chunks; first audio in 9.55 seconds |
| `tts-1-hd` | 57,600 PCM bytes in six distinct chunks; first audio in 11.04 seconds |

Both TTS checks used accepted first-pass Astra corrections. Their initial remaining
lead-in values were 59,970 ms and 59,953 ms, respectively. Tests stopped after
receiving enough PCM; they did not claim browser playback audibility. Automated
audio-controller tests cover the full delay, buffering, contiguous scheduling,
pause/resume, replay deduplication, acknowledgement, and session replacement.

Twenty empty default-ASR/Fast-English microphone session creations reached the
backend's accepting state with p95 0.050 seconds and maximum 0.053 seconds. This
excludes browser permission, microphone setup, and model inference. It does not
establish a browser click-to-readiness target.

## Evaluation, Stop, and clear

On the final restarted server, a 12-second excerpt completed local Breeze
evaluation in 17.26 seconds, producing two transcript rows, OpenAI/Tencent English,
and a whole-source-reference Mixed match result without provider errors. The CSV
fixture's source and English columns remained separate, and all five export
formats were available. Corrections and diarization were disabled for this focused
evaluation check. The fixture is a smoke-test reference, not independent accuracy
ground truth; its numerical score is not a quality claim.

A realtime WAV upload was stopped early. It finished with an explicit incomplete
flag, retained its TXT export, and rejected timed caption export. End and clear
removed both test sessions; subsequent reads returned HTTP 404. Unit tests also
cover cancellation during model preparation, provider failures and retries,
frozen settings, evaluation speech rejection, and a failed replacement preserving
the earlier conversation.

## Browser verification limit

Browser automation failed with `unsupported Codex auth method: apikey`. React
component, state, capture, playback and production-build checks ran, but visual
layout, physical microphone permissions, and actual speaker output were not
verified in a browser during this run.
