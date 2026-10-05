# Fast and Slow Translation Architecture

## Implemented local foundation

This repository implements a usable local workflow, not the complete production TRS.

`app.py` connects microphone/WAV input to Silero VAD and local Breeze ASR. Completed speech segments feed independent OpenAI and local Tencent translation queues. Fast OpenAI drafts also enter the Astra correction worker. Four numbered model columns preserve source order; the optional reference pane keeps its own file-line numbering.

`src/pipeline.py` resamples to mono 16 kHz, processes 512-sample/32 ms VAD frames, and queues speech without waiting for ASR. Silence is fixed at 500 ms; continuous speech is capped at 15 seconds. Breeze and Tencent share a local model lock. This is utterance transcription, not streaming ASR partials.

`src/diarization.py` independently streams continuous audio through local CPU Nemotron weights. Its rolling windows and speaker cache survive VAD boundaries. Time overlap attaches anonymous speaker labels to ASR segments; translation never waits for them. The audio backlog is bounded, with an explicit failure on overflow. See the [NVIDIA model card](https://huggingface.co/nvidia/Nemotron-3-Diarization/blob/main/README.md); advertised model latency is not measured application latency.

## Session ownership and correction policy

Starting a recording/file closes previous workers and freezes `SlowLaneConfig`. Defaults are four recent segments, ten earlier context segments, a 90-second window, a conservative 2,000 UTF-8-byte source budget, queue depth two, 20-second display revision horizon, 30-second seal timeout, 20-second request timeout, 4,096 output tokens, and confidence threshold 0.6. Pause and glossary reload are explicit runtime actions.

Each session and segment receives an immutable ID. Version checks reject stale writes; validation applies complete correction responses atomically. Duplicate work is suppressed and excess queued windows drop the oldest. Errors preserve drafts; failed fast translation is explicitly labelled source fallback.

The store version and visible version are separate. Accepted corrections within the display horizon replace the visible draft. A dispatched correction accepted later can update the final record while leaving the screen stable. The first accepted correction, `no_change`, timeout, pause, or close seals available text. Sealed text never changes. This resolves the TRS conflict in favor of immutable first sealing: “at most two” screen replacements remains an upper bound, but repeated revisions after sealing would require a specification change.

State, histories, and ordered events remain in memory. Downloads provide conversation TXT, session JSON, sealed bilingual CSV, and SRT/VTT using measured segment offsets. These exports do not provide durable recovery.

## API, terminology, and reference boundaries

Fast translation defaults to `gpt-6-luna`, with the previous two segments, up to 40 local glossary hits, and exact identifier checks. Completed utterances use a 512-token output cap rather than the TRS 64-token streaming clauses.

The slow adapter explicitly uses [GPT-6 Astra](https://developers.openai.com/api/docs/models/gpt-6-astra) with `reasoning.effort="medium"`, the [Responses API](https://developers.openai.com/api/docs/guides/text), and a [strict JSON schema](https://developers.openai.com/api/docs/guides/structured-outputs). Requests disable automatic retries and response storage. Allowlisted text/context/terminology fields exclude audio, evaluation references, and unrelated metadata. Cloud text transmission means this is not an on-premises/no-egress deployment.

`src/glossary.py` loads explicit CSV/TSV/JSON/XLSX terminology and optional TXT DNT lists. Reload swaps a complete validated snapshot. Accepted evidence from two distinct segments promotes a compatible term pair for later translation; the master remains unchanged. The requested 6,011-entry glossary was not supplied. Reference uploads remain evaluation/display inputs, never terminology or model context; English translations have no numerical quality score.

## TRS coverage and remaining gates

| Requirement group | Current coverage and gaps |
| --- | --- |
| FR-IN, FR-ASR, FR-STB | Capture, preprocessing, bounded segments, segment offsets, retained speech buffers. Missing synchronized client clock, full raw-session retention, configurable 300–800 ms VAD, low-energy live splitting, 500 ms ASR partials, 500 hotwords, word timestamps/confidence, LA2 and clause stabilization. |
| FR-FT, CTX, RL | Independent workers, contextual drafts, bounded windows, schema/confidence/DNT validation, `no_change`. Missing token streaming, 50 ms dispatch guarantee, three-second batching trigger, GraphRAG and audio re-decode. |
| COR, STO, UI | IDs, versions, atomic corrections, first sealing, separate live/final views, audit/export, status badges. Missing durable persistence and reconnect transport; live UI polls every 500 ms, with 100 ms updates while waiting for WAV ASR, not a 50 ms event guarantee. |
| GL, SES, LANG | Local terminology/reload/learning and recorded session configuration. Missing ASR terminology refresh, server affinity/prefix-warming guarantees, four-hour compaction, and validated config-only language-pair portability. |
| NFR-L/Q/C/A/S/O, DEP | Production latency, quality, capacity, recovery, security, observability and deployment gates remain unvalidated or unimplemented. No dedicated GPU pools, TLS/RBAC layer, retention service, distributed tracing or production shadow rollout. |

Mocked tests verify state transitions, payload boundaries, safe rendering and failure behavior. They do not establish p95 fast ≤2 s/slow ≤15 s, interference <5%, ASR/terminology/COMET/human-quality targets, four/twenty concurrent sessions, or four-hour soak performance. Those require representative audio, supplied terminology, deployment hardware and measured acceptance runs.
