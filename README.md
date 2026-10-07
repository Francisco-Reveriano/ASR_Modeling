# Voice transcription and translation

A Streamlit app for browser microphone recording and WAV uploads. The default
model pair is **gpt-live-transcribe + gpt-6-luna**: Silero VAD splits speech locally,
OpenAI transcribes the audio, and Luna translates the completed transcript.
Both **Microphone** and **Upload WAV** have a **Transcription + translation model**
dropdown with **Breeze + OpenAI** and **gpt-realtime-translate** as alternatives.
All options produce English translations. Breeze-ASR-26 transcribes
Taiwanese Hokkien locally from `Models/breeze-asr-26` into Chinese characters.
Each completed segment is translated into English independently:
by OpenAI through its API and by [Tencent Hy-MT2-1.8B](https://huggingface.co/tencent/Hy-MT2-1.8B)
running on this computer. A fourth model column defaults to **GPT-6 Astra, medium
reasoning**, to improve the English draft and repair likely transcription errors
using conversation context. Its model
and reasoning effort are configurable in `.env`. Local
**NVIDIA Nemotron-3-Diarization** adds anonymous speaker labels. Slow corrections
and speaker detection have separate workers and never wait on one another.
After the independent Astra corrections, a second conversation review advances
through the rows in order. Each review uses the newest reviewed English from
earlier rows and updates the same Astra cell as soon as it finishes.

## Models used

| Model | Role in this app | Input → output | Where it runs |
| --- | --- | --- | --- |
| **Silero VAD** (`silero-vad`) | Detects speech boundaries before transcription. | Mono 16 kHz audio → speech segments | Locally on CPU, loaded through the Python package. |
| **GPT Live Transcribe** (`gpt-live-transcribe`) | Default speech recognition for microphone and uploads. | Speech segments → original-language transcript | OpenAI Realtime API; requires `OPENAI_API_KEY`. |
| **GPT Realtime Translate** (`gpt-realtime-translate`) | Optional direct audio-to-English streaming captions. | Continuous audio → English captions, alongside separate source captions | OpenAI's dedicated translation endpoint; source captions use `gpt-realtime-whisper`. |
| **GPT-4o Mini TTS / TTS-1 HD** (`gpt-4o-mini-tts`, `tts-1-hd`) | Optional spoken English during microphone/upload translation. | Accepted Astra corrections (or English output in modes without Astra) → streaming speech | OpenAI Speech API; browser playback with preset speaker voices. |
| **Breeze-ASR-26** (`MediaTek-Research/Breeze-ASR-26`) | Transcribes Taiwanese Hokkien speech. | Speech segments → Chinese-character transcript | Locally from `Models/breeze-asr-26/`, using Apple MPS when available or CPU. |
| **OpenAI translation model** (code default: `gpt-6-luna`) | Translates each completed transcript segment into English through the Responses API. | Transcript text → English translation | OpenAI API; configurable with `OPENAI_DEFAULT_MODEL` and `OPENAI_DEFAULT_REASONING_EFFORT`. Requires `OPENAI_API_KEY`. |
| **Tencent Hy-MT2-1.8B** (`tencent/Hy-MT2-1.8B`) | Provides an independent local English translation of the same transcript. | Transcript text → English translation | Locally from `Models/Hy-MT2-1.8B/`, using Apple MPS when available or CPU. |
| **Astra correction model** (default: `gpt-6-astra`, medium reasoning) | Repairs supported transcription/translation mistakes and produces coherent English using nearby conversation. | Noisy source transcript, draft, and neighboring utterances → reviewed English | OpenAI API; configurable with `OPENAI_CORRECTION_MODEL` and `OPENAI_CORRECTION_REASONING_EFFORT`. |

The processing order is **audio → Silero VAD → selected transcriber → OpenAI and
Tencent translations**. Both translators receive the same original transcript.
The **Original transcript** column identifies its ASR provider and model. In the
default mode it displays the final **OpenAI ASR · gpt-live-transcribe** result;
OpenAI transcription errors never fall back to Breeze. The label stays tied to
the conversation that produced the text, including in TXT downloads, even after
changing the model dropdown for a future conversation.
With the default pair, audio is sent to OpenAI. **Breeze + OpenAI** keeps audio
local and sends transcript text for translation. Evaluation references stay local
and are never model inputs. Evaluation retains its Breeze baseline.

[`src/openai_transcription.py`](src/openai_transcription.py) uses the
[Realtime transcription protocol](https://developers.openai.com/api/docs/guides/realtime-transcription),
including 24 kHz PCM conversion, client-side speech boundaries, and explicit commits.
Uploads use the same segment adapter to keep the requested `gpt-live-transcribe`
model. Each segment owns a connection with a 45-second response deadline and no
automatic retries. Only committed final transcripts are displayed and translated;
this app still waits for utterance boundaries rather than displaying partial words.

**gpt-realtime-translate** uses a separate continuous streaming path in
[`src/realtime_translation.py`](src/realtime_translation.py). It streams microphone
audio immediately, including silence, and displays English caption fragments as
they arrive. The output language is fixed to `en`. Uploads stream at playback speed
with progress and a **Stop file translation** button. The
[translation API](https://developers.openai.com/api/docs/guides/realtime-translation)
translates audio directly; Luna, Tencent, Astra, glossary filtering and local
speaker detection do not run in this mode. The microphone **Translation type**
control is disabled for it. Its native speech audio is discarded; optional spoken
English uses the separately selected text-to-speech model below.

Source and English captions appear as two continuous texts because their fragment
boundaries need not match. **Stop recording** flushes remaining audio and waits for
the service's `session.closed` event; a failed or manually stopped upload retains
available captions with an **Incomplete** label in the UI and TXT export. There is
no automatic reconnect or retry. A missing English result is marked unavailable;
source text is never substituted as a translation. The default pair above remains
selected when opening either input.

Prepare Breeze with [`Notebooks/01 Download Assets.ipynb`](Notebooks/01%20Download%20Assets.ipynb)
and Tencent with [`scripts/download_tencent.py`](scripts/download_tencent.py),
which pins its model revision. Silero is loaded by [`src/pipeline.py`](src/pipeline.py).
The OpenAI default and request settings are defined in
[`src/translation.py`](src/translation.py); an environment override can select a
different model. Model weights are excluded from Git.

The notebook also downloads `sarahwei/Taiwanese-Minnan-Sutiau`. This is a dataset,
not an inference model, and is not required to run the app.

## Run

From the repository root:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/download_tencent.py
python scripts/download_diarization.py
python -m streamlit run app.py --server.address localhost
```

Open <http://localhost:8501>, click **Start recording**, and allow microphone
access. The default pair needs OpenAI API access and no Breeze weights. Selecting
Breeze loads its local model on first use and reuses it for later recordings.
Text is appended after roughly half a second of silence plus inference time.
**Stop recording** submits any unfinished speech and finishes waiting segments.
The next Start clears the conversation. The default pair sends speech audio to
OpenAI; completed transcript and configured terminology are sent for translation
and correction. Conversation text, timings, and correction history stay in memory
until the next session; downloads save a copy. Processed audio is released as
each worker finishes using it.

The conversation view numbers each completed speech segment (`01`, `02`, …).
Use **Download conversation** to save all model results before starting
again. Before starting, open **Microphone settings → Translation type**:

| Type | Processing |
| --- | --- |
| **Compare all translations** (default) | OpenAI and local Tencent, independent Astra corrections, then sequential conversation review. |
| **Fast English** | OpenAI translation only. |
| **Corrected English** | OpenAI translation followed by independent Astra corrections. |
| **Fully reviewed English** | OpenAI translation, independent Astra corrections, then sequential conversation review. |

The model pair and translation type apply to the next recording and stay fixed
while it runs. Each input has its own model choice. Completed results and exports
retain the model pair used, even after changing the next session's dropdown.
Only selected providers run or appear in the transcript; Astra modes also show
their fast OpenAI draft. Choosing a mode without Tencent avoids loading that
translator for the recording. Upload and evaluation keep the full comparison
workflow independently of the microphone selection, except for the standalone
realtime translation option described above. All modes output English.
The **Astra corrections** switch pauses both review passes for modes that use them.
The browser device picker appears in **Microphone settings** once recording has
started. The workspace uses a light theme and stacks its panels on narrow screens.

### Upload a WAV file

Choose **Upload WAV**, select the model pair and a `.wav` file, optionally play it back, then click
**Transcribe file**. Mono and stereo WAV files are converted to mono 16 kHz.
For the transcription-plus-translation options, Silero finds speech and the selected
model transcribes each segment in order. Realtime translation instead streams the
complete audio without VAD cuts. Progress and
numbered text appear as segments finish; each translation follows independently.
Use **Download conversation** to save all four model columns and speaker labels.
Selecting a file alone does not start transcription. Each new transcription
replaces the displayed text. Stop any live recording and let its pending segments
finish before uploading. Files are decoded in memory on this Mac; the OpenAI pair
sends speech segments to OpenAI for transcription. WAV decoding
reads at most 65,536 source frames at a time, averaging channels into one mono
buffer before resampling. The upload job keeps decoded audio across UI reruns,
then releases it on completion or failure; speaker detection owns a separate
copy until it finishes. The selected WAV remains available for playback and retry.

If creating an environment from scratch, run `python3 -m venv .venv` first.
For **Breeze + OpenAI** and evaluation, the model and processor must already be
present in `Models/breeze-asr-26`; loading uses local files and needs no Hugging Face
token. The existing download notebook can prepare the model when needed. Tencent
and Nemotron assets are still needed when their comparison/speaker options are enabled.

### Evaluate against a reference

Open **Evaluate**, upload a `.wav` and a reference as `.txt`, `.srt`, `.vtt`,
`.xlsx`, `.csv`, or `.tsv`, then check the previews before clicking **Run evaluation**.
References are limited to 1 MiB each. Text and delimited files may be UTF-8
(with or without BOM) or UTF-16 with BOM.
The same Silero → Breeze → OpenAI/Tencent pipeline produces the transcript and
fast translations, with Astra review and Nemotron speaker detection in parallel.

For spreadsheets and delimited files, use the dropdowns in the **left input pane**:

- **Worksheet** selects one sheet; `transcript` is preferred over background speech or speaker metadata.
- **Transcription reference column** selects the source words used to score Breeze. `text_zh_TW` is preferred when present.
- **English reference column** selects the separate reference pane below the model columns. `translation_en` is suggested when present; otherwise it shows the source reference.
- **Reference settings → Header row** changes the header position; `0` includes all rows for files without headers. Unrecognized columns require an explicit source choice.

For the sample workbook, the defaults are `transcript`, `H: text_zh_TW`, and
`J: translation_en`. Only selected columns are read into references; other sheets,
speaker fields, timestamps, and notes do not enter the score. Previews report
blank cells skipped. Formula/error cells in a selected column require pasted text
values. Workbooks are read locally without modifying them. Tables support up to
10,000 rows and 100 columns per sheet; XLSX contents may expand to at most 10 MiB.

The reference pane has its own `R01…` reference numbering. Reference lines follow
the file, while model segments follow speech pauses; they are not paired by row
number. The English reference supports visual comparison and receives no score.

The app detects plain text, timestamped speaker transcripts, and SRT/WebVTT
captions. For example, this target line becomes `你好Teams有點lag。`:

```text
[00:01.200 --> 00:06.856] SPK2 Vivian: (overlap) 你好Teams有點lag。
```

Recognized timestamps, speaker prefixes, leading `(overlap)` / `(backchannel)`
markers, caption metadata, `#` comments, and `[BG ...]` / `[FX ...]` entries are
excluded in transcript mode. Dialogue and its order are retained. **Reference
settings → Plain text** keeps the original content when that is what you need.
The preview reports detected format, retained segments, and excluded lines.

Provide a **source transcript** containing the words spoken in the audio, in their
original language. Only Breeze receives a transcription score. OpenAI, Tencent, and Astra
translations remain visible and included in downloads, without numerical scores.
Chinese/English references use **Mixed match**: each Chinese character and each
English word is one unit. English source transcripts use **1-wMER** with word units.
References stay local and are never supplied to a model.

Here `wMER` means **word-level match error rate**, computed with
[JiWER](https://jitsi.github.io/jiwer/reference/process/):

```text
wMER = (substitutions + deletions + insertions)
       / (matches + substitutions + deletions + insertions)
score = 1 - wMER
```

Scores appear as percentages: 100% is a perfect normalized match; higher is better.
Mixed match uses the same formula with mixed units instead of words. It is a
bounded match score, distinct from conventional ASR mixed error rate, which uses
reference length as its denominator. Breeze's completed segments are joined
and aligned against the whole reference once. Reference line breaks do not need
to match speech pauses. There is no speaker/time alignment; overlapping speech
is evaluated in transcript order and can affect the score.

Scoring applies Unicode NFKC normalization and case folding, replaces punctuation
with spaces, and splits words on whitespace (plus Han character boundaries for
Mixed match). Apostrophes and hyphens split words; accents and digits remain.
Traditional and simplified characters are not converted into each other.
**This measures text matching, not semantic translation quality**; valid
paraphrases can score lower. Only outputs with a matching reference are scored.

Pending or failed transcription has no final score. Translation delays or failures
do not affect Breeze's score. Successfully processed silence against a nonempty
reference scores 0%. **Download evaluation** includes all outputs, the Breeze score, alignment counts,
cleaned reference text, original structured text, selected sheet/columns, the
English comparison reference, and filenames. Selecting new
files or changing reference settings does not change an existing evaluation;
**Run evaluation** starts a new one and freezes its inputs and settings.

### OpenAI configuration

The app reads the repository's `.env` file, with existing environment variables
taking precedence. [`.env.example`](.env.example) lists all supported environment
options without credentials; copy it to `.env` when setting up a new checkout:

```dotenv
OPENAI_API_KEY=your-api-key
OPENAI_DEFAULT_MODEL=gpt-6-luna
OPENAI_DEFAULT_REASONING_EFFORT=none
OPENAI_FILTER_BACKGROUND_SPEECH=true
OPENAI_CORRECTION_MODEL=gpt-6-astra
OPENAI_CORRECTION_REASONING_EFFORT=medium
OPENAI_CORRECTION_MAX_OUTPUT_TOKENS=16384
GLOSSARY_FILE=
DNT_FILE=
# Optional; used only by the asset-download notebook.
HF_TOKEN=
```

`OPENAI_DEFAULT_MODEL` selects the fast translator for **Breeze + OpenAI** and
evaluation. The default **gpt-live-transcribe + gpt-6-luna** pair pins translation
to `gpt-6-luna`, regardless of that environment override.
`OPENAI_DEFAULT_REASONING_EFFORT` explicitly sets fast translation's reasoning effort. With the
effort missing or blank, this app uses `none` for `gpt-6-luna` and omits the
reasoning option for other models so the provider selects its default. Luna
supports `none`, `low`, `medium`, `high`, `xhigh`, and `max`; see the
[official Luna documentation](https://developers.openai.com/api/docs/models/gpt-6-luna).

`OPENAI_FILTER_BACKGROUND_SPEECH=true` enables conservative filtering in the fast
translator. Recent main-conversation text helps it omit clearly separate background
chatter and noise. Uncertain speech, technical fragments, brief replies, protected
identifiers, and genuine topic changes are retained. This is text-based filtering;
it does not separate overlapping voices in the audio. Set the option to `false`
to translate every transcript segment. Entirely filtered segments are visibly
marked **Background filtered**, remain in the original transcript and audit
exports, and are omitted from final SRT/VTT captions. Astra does not reintroduce
those segments. The independent Tencent translation still receives the original
transcript.

If a completed Tencent translation leaves source-script words in its English
output, it gets one local repair from the original transcript using deterministic
decoding and an English-only prompt. Truncated, empty, or repeatedly invalid
output remains a visible failure rather than a partial translation.

`OPENAI_CORRECTION_MODEL` and `OPENAI_CORRECTION_REASONING_EFFORT` independently
select the model and default reasoning effort for both passes in the Astra correction column.
With speech enabled, **Live corrections** uses Low reasoning for the first
`gpt-6-astra` pass; the background conversation pass retains this configured effort.
Other correction models retain their configured effort in both passes.
`OPENAI_CORRECTION_MAX_OUTPUT_TOKENS` sets the initial combined reasoning and
translation-response token cap, including structured output. It defaults to
16,384 and accepts integers from 512 through 16,384; explicitly configured smaller
caps are honored. High reasoning can exhaust a small cap before completing its
review. Blank correction values use the defaults above. These settings are frozen
when a recording or upload starts and included in the session export. The controls
show the configured settings, and **Reasoning + output token cap** can override the
environment default for the next session.

Leave the two terminology paths empty when unused. Confidence and
speaker/correction toggles are configured in Streamlit.
Astra corrections have no automatic timeout or expiry setting.
`VOG_RECONCILIATION_REASONING_EFFORT` is not read by this app; use
`OPENAI_DEFAULT_REASONING_EFFORT` for fast translation instead.

For Astra, use `low`, `medium`, `high`, `xhigh`, or `max`, as listed in the
[official OpenAI model documentation](https://developers.openai.com/api/docs/models/gpt-6-astra).
Other models may support different efforts and must support Responses structured
outputs. Invalid effort names produce a configuration error. Model compatibility
and API access are checked by the provider when a request runs.

The key stays on the server. Transcript text, recent translations, and configured
glossary/DNT context are sent to the official OpenAI Responses API with `store=False`.
The default transcriber and realtime translation option also send audio through OpenAI's Realtime API; that
protocol has no Responses `store` parameter. Breeze and evaluation keep audio
local. Evaluation references are never uploaded to OpenAI.
Translation needs internet access and uses your API account. Restart Streamlit
after changing `.env` credentials, model, or reasoning settings.

This cloud-backed configuration does not meet the supplied TRS's on-premises-only
requirement. An approved API exception is needed for that deployment; the local
app is not a production compliance claim.

### Astra corrections and terminology

In the left pane, expand **Translation & speaker settings** to pause corrections, select
**Live subtitles** or **Final record**, and configure the next session. Astra uses
`gpt-6-astra` with medium reasoning and confidence ≥0.6 by default. Each completed
fast translation starts its own Astra review. Reviews run in parallel, and each
accepted result appears immediately even if an earlier segment is still being
reviewed. Rows remain in transcript order. All pending segments remain eligible
until reviewed. There is no request, draft revision, or sealing deadline. The fourth
column labels
drafts, corrections, confirmed text, and fallback states.

The slow translator treats both the ASR transcript and fast draft as fallible.
It uses nearby utterances, repeated terminology, grammar, and supported phonetic
clues to repair recognition mistakes and reconstruct natural English that makes
sense in the conversation. Each request contains one editable segment plus
surrounding evidence that fits the source budget, including up to two already
available following utterances,
without waiting for more speech or increasing the source budget. Reviewed and
provisional context are explicitly distinguished. The original transcript remains
unchanged for comparison.

Astra preserves actual identifiers, quantities, negation, and speaker intent.
It uses `[unclear]` only for the smallest important detail that remains
unrecoverable after considering the context; a corrupted ASR spelling alone should
not cause an otherwise understandable sentence to become fragmented or vague.
Context-supported repairs do not establish acoustic ground truth: Astra receives
text and does not listen to the audio.

Corrections use immutable segment IDs and version checks. When an accepted
correction finishes, the Astra column replaces its draft with the reviewed text,
shows **Corrected**, and gives that cell a subtle green tint. A review
that leaves the translation unchanged shows **Confirmed**. Both live and final
views show accepted reviewed text. A correction, explicit confirmation,
pause, or session close seals that pass's result. The first pass remains unchanged
in the audit history; the second pass can publish a newer reviewed version in the
same cell. Slow-lane
errors preserve available drafts and can be retried; waiting alone never seals a
draft. A rejected review or a response truncated by `max_output_tokens` gets one
fresh singleton review per affected segment, with the same configured token cap,
confidence, English, and identifier checks. These failure types share one automatic
retry allowance per segment. Truncated partial output is never accepted. Other
provider errors require manual retry. A slow or failed first review never holds back
another segment's completed first correction. Persistent failures show a specific
safe reason and remain available for manual retry. Starting a new conversation
cancels pending reviews. The **Correction
history & session export** panel provides JSON audit history, final bilingual CSV,
and SRT/VTT with actual audio offsets.

**Conversation review** is the second pass, enabled for full comparison and
fully reviewed microphone recordings. It reviews one row at a time in transcript
order after that row's first correction and all earlier eligible conversation
reviews have succeeded. It uses earlier rows' newest reviewed wording, plus
available first-reviewed following context, to check terminology, referents, and
coherence against the original source. It does not wait for recording to stop.
The first corrections remain visible while this pass runs, and each accepted
second response replaces only its own row. Later first-pass corrections still
appear immediately even when conversation review is waiting on an earlier row.

A failed conversation review keeps the last accepted English visible and pauses
the dependent sequential work until retry. Filtered background rows need no
second request. Both passes use the same configured model, token cap, and
validation; neither has a timeout. Their reasoning effort is normally the same,
except for the **Live corrections** speech option described below. The status and audit distinguish
the two passes. Downloads use the newest accepted wording and retain review
status rather than treating pending work as fully reviewed.

Set a local glossary path in the controls or `GLOSSARY_FILE`; optionally set
`DNT_FILE` for do-not-translate identifiers. CSV/TSV/JSON/XLSX glossaries use
`term_src`, `term_tgt`, and optional `aliases_src` (pipe-separated), `dnt`,
`domain`, `priority`, `example_src`, and `example_tgt`. A DNT file may also be
plain text, one identifier per line. **Reload terminology** validates and replaces
the active master snapshot. DNT terms and aliases must use English or Latin
script so they can be preserved in English output; use ordinary Chinese-to-English
glossary mappings for Chinese terms. No master glossary is bundled. Two accepted,
consistent corrections on distinct segments can teach a session term; candidate
terms and evidence IDs are exported without editing the master.

See [architecture coverage and remaining production gates](docs/fast-slow-architecture.md).
This app still transcribes completed utterances; it does not yet implement
streaming ASR partials, LocalAgreement stabilization, durable reconnects, or the
TRS latency/capacity guarantees.

### Local speaker diarization

`python scripts/download_diarization.py` downloads the pinned
[`nvidia/Nemotron-3-Diarization`](https://huggingface.co/nvidia/Nemotron-3-Diarization)
revision into `Models/Nemotron-3-Diarization/` (about 397 MB of weights).
It requires `transformers>=5.18.0` and `librosa`. Inference uses local files on CPU,
separate from Breeze/Tencent's shared Metal lock.

**Nemotron speaker detection** enables diarization for the next recording/upload.
Continuous audio, including silence, feeds a persistent streaming speaker cache;
labels such as `Speaker 1` follow audio-time overlap with each ASR segment.
Up to eight anonymous speakers are supported. Before transcription, clear sequential
speaker changes split a VAD segment into separate rows. The cuts preserve every
audio sample and its absolute timestamp; brief activity flicker does not create
tiny requests. Substantial simultaneous speech keeps a combined label. These are
session-local speaker numbers, not people identified by their voices.

The upload worker waits up to 30 seconds for speaker timing, including cold model
loading; the microphone ASR worker waits up to 3 seconds. Capture and rendering
continue during those waits. If timing is still unavailable, transcription proceeds
with the original segment and labels can arrive later. Labels are only published
when the corresponding audio interval has been processed. The UI shows processed
audio seconds and total received seconds, including silence. Missing weights or a
full speaker queue show a status message while transcription continues. Accuracy
and real-time speed depend on the audio and hardware.

### Local Tencent configuration

`python scripts/download_tencent.py` downloads approximately 4.1 GB of model files
to `Models/Hy-MT2-1.8B/`. The script pins the Tencent repository revision, includes
the model card and license, and can resume an interrupted download. This public
model requires no Hugging Face token. It uses the existing PyTorch/Transformers
stack, without a separate model server.

The first Tencent translation loads the model; later segments and recordings reuse
it. Inference uses Apple MPS when available, otherwise CPU. Tencent reads only the
downloaded files and runs entirely locally: it sends neither audio nor transcript
text to Tencent or Hugging Face. The model receives the same Chinese-character
transcript as OpenAI, rather than OpenAI's translated text.

The UI reports model loading/readiness, the active row, queued rows, and elapsed
time during uploads as well as after transcription. Generation has a 60-second
limit per attempt, checked between decoding steps, and a 512-token output cap.
Incomplete output fails that row and allows the queue to continue. A new session
cancels old local generation between steps; model loading and a running device
operation must finish before cancellation takes effect. Single-row generation is
retained: CPU tests on this Mac found two-row batching slower.

### Comparing results and retrying

Speech recognition, OpenAI translation, and Tencent translation have separate
workers. One translator can finish even if the other is slow or unavailable.
The two local models share an execution lock: overlapping Breeze and Tencent
GPU calls caused a native Metal crash during testing. Model work takes turns;
microphone capture, VAD, and OpenAI requests continue independently.
Uploads keep both ASR model loading and inference in a reusable background task.
The Streamlit fragment returns while work is pending and polls every 0.5 seconds,
so accepted translations and corrections remain visible while subsequent audio
or speaker timing is being processed. Restart Streamlit after Python changes;
file watching is disabled. Keep one app process running to avoid duplicate model
copies consuming local memory.
Each source segment is submitted once to each provider per conversation, including
across UI refreshes. **Retry OpenAI** and **Retry Tencent · Local** retry only the
failed segments for that provider; completed results remain visible.

English columns never substitute the original Chinese transcript. Prompts require
English names or Latin transliteration, and untranslated Chinese output is rejected.
OpenAI gets one bounded repair attempt for mixed-language output, using the same
source and configured model/reasoning. If fast translation fails, enabled Astra
can translate the source through its review queue. An unavailable
English result shows **Translation unavailable**; the original stays in its own
column. The same rule applies to TXT, CSV, SRT, and VTT downloads. Session JSON
retains explicitly flagged source fallbacks for audit history.

Downloads label each provider and mark pending/unavailable translations. A new
recording or file clears all model columns and cancels queued translations.
An already running translation may finish, but cannot populate the new conversation.

## How it works

- `app.py` handles microphone controls, model caching, and transcript updates.
- `src/ui.py` renders and exports aligned source/OpenAI/Tencent/Astra rows;
  `assets/style.css` styles the responsive workspace.
- `src/evaluation.py` parses reference TXT/SRT/VTT files and computes normalized
  whole-file word or mixed-unit match scores without model or network calls.
- `src/reference_tables.py` reads XLSX/CSV/TSV references and suggests source and
  English columns; `openpyxl` reads workbooks in read-only mode.
- `src/translation.py` provides the OpenAI request and reusable translation queue;
  the app creates one queue for each provider.
- `src/openai_transcription.py` provides the default cloud speech adapter for both inputs.
- `src/slow_lane.py` owns versioned subtitles, correction windows, and sealing;
  `src/reasoning.py` calls Astra with a strict correction schema.
- `src/glossary.py` handles local terminology, DNT checks, and session learning.
- `src/diarization.py` runs continuous local speaker detection;
  `src/subtitle_exports.py` exports sealed bilingual text and timed captions.
- `src/tencent.py` lazily loads Hy-MT2-1.8B and serializes local generation with a
  lock shared with Breeze in `src/model_lock.py`.
  `scripts/download_tencent.py` prepares its pinned model assets.
- `src/uploads.py` decodes WAV uploads and finds speech segments for sequential
  transcription, avoiding the live microphone queue. File VAD drops speech bursts
  shorter than 250 ms and limits ASR segments to 15 seconds.
- `src/pipeline.py` resamples audio to mono 16 kHz and feeds Silero 512-sample
  frames. Speech uses 150 ms of padding and ends after 500 ms of silence.
- Speech longer than 15 seconds is split. One background worker transcribes
  segments in order while capture continues.
- Breeze uses Apple MPS when available, otherwise CPU, in float32. Its weights
  occupy about 5.7 GiB; allow additional memory for inference. CPU transcription
  can fall behind speech. Eight segments may wait in the queue; if it fills,
  capture stops with a visible warning and accepted segments finish.
- Completed speech is not archived in session audio buffers. Releasing those
  buffers avoids about 220 MiB per hour of retained mono 16 kHz float32 audio.
  Model precision and caching are unchanged: Breeze, Tencent, and Nemotron
  weight tensors total about 9.9 GiB on a Mac using Tencent's bfloat16 path,
  with additional memory needed for inference, uploads, and pending work.

Use the browser on the same Mac as Streamlit. If microphone access fails, check
browser permissions, press Stop, then retry. Remote hosting is outside this
version's scope.

The Streamlit/WebRTC versions are pinned to the combination verified locally;
WebRTC 0.78.1 repeatedly reset the microphone connection during testing.
File watching is disabled to avoid scanning Transformers' lazy imports.
Restart Streamlit after editing Python files.

### Read English aloud while translating

In **Microphone** or **Upload WAV**, turn on **Read English aloud** before starting.
The **Text-to-speech model** dropdown offers **gpt-4o-mini-tts** (default) and
**tts-1-hd** for both microphone and uploaded audio. Both use the same English
text, speaker mapping, streaming player, and initial delay. `tts-1-hd` is the
HD speech option; it does not accept the extra delivery instructions used by
`gpt-4o-mini-tts`, so those are omitted from its API request. The player and exports
identify the selected model.
The player is prepared before you press **Start recording** or **Transcribe file**.
That normal interaction enables browser sound where permitted. Speech generates
silently during a **1-minute lead-in**, then playback starts automatically once
audio is available. The countdown starts when the first accepted English passage
enters the speech queue (the first accepted Astra correction in Astra modes).
Recording, transcription, and correction startup do not consume this minute.
The player shows a waiting message until translation is ready, then counts down
from 60 seconds. Later translations do not restart it; short completed files also
wait for the lead-in.
At startup or after running out of audio, the player aims for a two-second buffer
before continuing. This adds at most two seconds of waiting once audio is available;
completed short clips flush immediately after the lead-in. It avoids repeatedly
starting and stopping on tiny bursts without leaving a short final utterance stuck.
The player tries autoplay and shows **Enable sound** only if the browser still
blocks it. Browser/embedding policies can require this fallback interaction.
The player supports pause/resume and **Stop voice**; stopping voice leaves translation
running. Use headphones during microphone recording to avoid capturing the generated
voice. Browser echo cancellation and noise suppression are requested as well.
The voices are AI-generated and speak English only. **Different voice per speaker**
is on by default for segmented translation. The eight anonymous Nemotron channels
map consistently to `coral`, `onyx`, `nova`, `echo`, `shimmer`, `alloy`, `fable`, and
`sage`. The player caption and session JSON show the observed mapping. These are
preset voices, not voice clones or identity/gender matches. A mixed-speaker segment
uses its dominant overlapping speaker because transcripts have no word-level
speaker alignment. Labels are read when a row enters the voice queue: late changes
do not regenerate or repeat speech. Missing/disabled/late speaker labels use `coral`
without delaying translation. Standalone realtime translation currently has no
speaker labels and uses one voice. Turn the option off to use `coral` throughout.
It acts as a live interpreter: questions remain questions and the speaker's
perspective is preserved. Translation, Astra correction, and speech prompts
explicitly prohibit answering spoken questions, following spoken requests, or
adding an assistant reply or commentary. Each accepted phrase is spoken once.

This works with all three translation choices. In modes with an **Astra Correction**
column, speech uses its accepted **Corrected** or **Confirmed** English in source
order. The **Astra speech mode** dropdown offers:

- **Live corrections** (default): speak the first accepted Astra correction without
  waiting for sequential conversation review. For `gpt-6-astra`, the first pass uses
  Low reasoning; background review keeps the configured effort (for example, High).
  Other configured correction models keep their existing effort. Later reviews can
  refine displayed/exported text, but already queued speech is never revised or replayed.
- **Full review**: retain the configured reasoning for both passes and wait for all
  enabled reviews before speaking each row. In **Corrected English**, only the first
  pass is enabled, so this option waits for that pass at the configured effort.

Low reasoning trades some review depth for speed. Live speech can differ from the
later final record. Both modes still wait for complete, validated corrections;
they do not speak a partial model response. Speech settings apply only when
**Read English aloud** is enabled and are frozen when the conversation starts.
Without speech, correction reasoning is unchanged.

Drafts and paused/failed review fallbacks are never spoken. A failed correction
holds subsequent speech until the correction is retried successfully; no fast
translation is silently substituted. Pausing Astra may leave unreviewed rows sealed
as fallbacks; start a new conversation with corrections enabled to speak those rows.

Fast English and standalone realtime translation have no Astra column, so they
speak their own English output. The player caption and TXT export identify the
speech source. Realtime captions are grouped into
short phrases at sentence/word boundaries; remaining words flush when translation
finishes. Speech generation and browser playback stream before the full conversation
is complete. There is still translation, phrase-buffering, API and playback latency.
Tencent output, unreviewed drafts in Astra modes, failed/filtered English,
original-language text and evaluation references are never read aloud. Accepted
rows are queued only once across refreshes. Settings are frozen for each conversation; the next recording
or file cancels old voice work and clears its playback queue. Evaluation has no speech
option.

[`src/speech.py`](src/speech.py) uses OpenAI's
[streaming Speech API](https://developers.openai.com/api/docs/guides/text-to-speech)
with server-side credentials and 24 kHz PCM16. The persistent browser player in
`src/speech_player/` uses one Web Audio clock to schedule ready chunks contiguously,
deduplicates refreshes, and acknowledges played chunks so the server releases them.
Up to two TTS requests run ahead, but their PCM is released strictly in phrase order.
Each in-flight request buffers at most two seconds; the published audio queue is
capped at 60 seconds, and
queued text at 20,000 characters. Playback backpressure does not block transcription
or translation. Short sentences within the same accepted speaker row are grouped
into passages of up to 300 characters; separate speaker rows stay separate.
`src/speech_audio.py` removes excess near-silent padding at each request boundary,
retaining 80 ms before and 120 ms after speech. Audible samples and internal pauses
remain unchanged. Its conservative -54 dBFS threshold and bounded two-second
lookbehind limit how much silence is removed; this is not time compression.
The browser joins ready chunks on the same audio clock and rebuilds a small buffer
after an underrun. Voice failures are reported separately without automatic retries;
start a new conversation to retry. This adds speech API usage to your OpenAI account.
The initial lead-in reduces interruptions; it cannot guarantee uninterrupted speech
if recognition, correction, or TTS consistently falls behind playback.
No new Python dependency is required.

## Tests

```bash
python -m unittest discover -s tests -v
node --test tests/test_speech_player.js
```

Tests exercise audio resampling, WAV validation, recording and upload lifecycles,
interrupted-upload recovery, provider isolation, and translation ordering/retries
with lightweight VAD/inference and API substitutes. Evaluation tests cover the
score formula, normalization, transcript/table parsing, column selection, reference
routing, frozen evaluation inputs, and incomplete results.
Additional tests cover correction concurrency, delayed reviews without expiry, DNT protection, glossary
reload/learning, speaker streaming state, and final exports. They use the installed
Nemotron processor with substitute logits, without loading model weights or
calling OpenAI.
