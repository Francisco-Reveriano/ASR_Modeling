# Local runtime validation — 2026-10-06

> Historical validation of the earlier Streamlit implementation. The current React/FastAPI migration is documented in [the migration notes](react-fastapi-migration.md); do not interpret the timings below as measurements of the new interface.

Validated on the `OpenAI` branch on the existing Mac, using local model assets.
Breeze and Tencent retain their shared inference lock. Nemotron runs on its own
CPU worker. No Hugging Face endpoint or remote model loading was introduced.

## Changes verified

- Upload ASR model loading and inference run in a reusable background task.
  Streamlit returns during pending work and continues displaying completed rows.
- Tencent reports loading/readiness, active row, queue length and elapsed time.
  Closing a conversation signals cancellation between generation steps. A
  60-second generation limit per attempt rejects incomplete output and lets the
  queue continue; model loading and individual device operations are not hard
  interrupted.
- Clear sequential speaker changes split audio before ASR, preserving all PCM
  samples and absolute timestamps. Short activity flicker is ignored; substantial
  simultaneous speech retains combined labels.
- Speaker timing is allowed to catch up on background workers, for up to 30
  seconds on upload or 3 seconds on microphone ASR, before falling back to the
  original segment. Capture and rendering do not wait on that condition.

## Results

| Check | Result |
| --- | --- |
| Python `unittest` discovery | 493 passed |
| Browser audio-player Node tests | 17 passed |
| Full `chinglish_test.wav`, 306.04 seconds | Local Breeze → Tencent on Apple Metal: 48/48 English results, no failed rows, speaker processing complete, 113.02 seconds including loading |
| Three-speaker 45-second excerpt, CPU | 7/7 English results, three labels, 78.33 seconds including loading |
| Same excerpt, Apple Metal | 7/7 English results, three labels, 26.06 seconds including loading |
| Full `chinglish_teams_3spk (1).wav`, 296.79 seconds | Local Nemotron completed in 94.68 seconds; 31 VAD segments became 44 speaker-aware rows; concatenating each row's audio reproduced its original VAD segment exactly |
| Deployment | Restarted `http://localhost:8502`; health endpoint returned `ok` |

The full three-speaker recording produced three dominant channels plus two brief
extra channels (about 7.3 and 4.3 seconds of activity). Speaker labels remain
imperfect; these checks establish completion and audio alignment, not a measured
diarization error rate or verified translation accuracy. Timings are individual
runs, not service-level latency guarantees.

Streamlit integration tests cover split upload rows, model loading without
blocking rendering, reruns during blocked ASR, cancellation, speaker progress,
and completed-audio release. Live microphone hardware was not exercised in this
check. The final visual browser check could not be completed: automatic approval
review blocked broad browser inspection, and tab-scoped access failed with an
authentication error. The integration and health checks passed.

Detailed local benchmark artifacts are in
`/private/tmp/asr-openai-validation/`; transcripts and audio are not included in
this repository report.
