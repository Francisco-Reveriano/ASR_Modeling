# Breeze Voice transcription and translation

A local Streamlit app for browser microphone recording and WAV uploads. It uses
Silero VAD to split speech and transcribes each segment with the existing
`Models/breeze-asr-26` model. Breeze-ASR-26 targets Taiwanese Hokkien and outputs
Chinese characters. Each completed segment is translated into English independently:
by OpenAI through its API and by [Tencent Hy-MT2-1.8B](https://huggingface.co/tencent/Hy-MT2-1.8B)
running on this computer. A fourth model column uses **GPT-6 Astra, medium
reasoning**, to review the OpenAI draft with recent conversation context. Local
**NVIDIA Nemotron-3-Diarization** adds anonymous speaker labels. Slow corrections
and speaker detection have separate workers and never wait on one another.

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
access. The first start loads the local models; later recordings reuse Breeze.
Text is appended after roughly half a second of silence plus inference time.
**Stop recording** submits any unfinished speech and finishes waiting segments.
The next Start clears the conversation. Audio stays on this Mac; completed
transcript and configured terminology are sent to OpenAI for translation and
correction. The app keeps the conversation
in memory unless you download it.

The conversation view numbers each completed speech segment (`01`, `02`, …).
Use **Download conversation** to save all model results before starting
again. **Microphone settings** contains the browser device picker once recording
has started. The workspace uses a light theme and stacks its panels on narrow screens.

### Upload a WAV file

Choose **Upload WAV**, select a `.wav` file, optionally play it back, then click
**Transcribe file**. Mono and stereo WAV files are converted to mono 16 kHz.
Silero finds speech, and Breeze transcribes each segment in order. Progress and
numbered text appear as segments finish; each translation follows independently.
Use **Download conversation** to save all four model columns and speaker labels.
Selecting a file alone does not start transcription. Each new transcription
replaces the displayed text. Stop any live recording and let its pending segments
finish before uploading. Files are processed in memory on this Mac.

If creating an environment from scratch, run `python3 -m venv .venv` first.
The model and processor must already be present in `Models/breeze-asr-26`;
the app uses local files and needs no Hugging Face token. The existing download
notebook can prepare the model when needed.

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
taking precedence:

```dotenv
OPENAI_API_KEY=your-api-key
OPENAI_DEFAULT_MODEL=gpt-6-luna
```

The key stays on the server. Transcript text, recent translations, and configured
glossary/DNT context are sent to the official OpenAI Responses API; audio and
evaluation references are never uploaded to OpenAI. Requests use `store=False`.
Translation needs internet access and uses your API account. Restart Streamlit
after changing credentials or the model setting.

This cloud-backed configuration does not meet the supplied TRS's on-premises-only
requirement. An approved API exception is needed for that deployment; the local
app is not a production compliance claim.

### Astra corrections and terminology

In the left pane, expand **Translation & speaker settings** to pause corrections, select
**Live subtitles** or **Final record**, and configure the next session. Astra uses
`gpt-6-astra` with medium reasoning. Defaults: four recent segments, a 20-second
screen revision window, 30-second sealing deadline, confidence ≥0.6, and two
pending correction requests. Fast output appears first; the fourth column labels
drafts, corrections, confirmed text, and fallback states.

Corrections use immutable segment IDs and version checks. After the display
window, an accepted correction changes only the final record. A correction,
explicit confirmation, timeout, pause, or session close seals the result; sealed
text never changes. Slow-lane errors preserve available drafts. The **Correction
history & session export** panel provides JSON audit history, final bilingual CSV,
and SRT/VTT with actual audio offsets.

Set a local glossary path in the controls or `GLOSSARY_FILE`; optionally set
`DNT_FILE` for do-not-translate identifiers. CSV/TSV/JSON/XLSX glossaries use
`term_src`, `term_tgt`, and optional `aliases_src` (pipe-separated), `dnt`,
`domain`, `priority`, `example_src`, and `example_tgt`. A DNT file may also be
plain text, one identifier per line. **Reload terminology** validates and replaces
the active master snapshot. No master glossary is bundled. Two accepted,
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
Up to eight anonymous speakers are supported. A segment spanning multiple
speakers keeps a combined label; the app does not split or identify people from
their voices. Missing weights or a full speaker queue show a status message while
transcription continues. Speaker accuracy and real-time speed depend on the audio
and hardware; they are not established by unit tests.

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

### Comparing results and retrying

Speech recognition, OpenAI translation, and Tencent translation have separate
workers. One translator can finish even if the other is slow or unavailable.
The two local models share an execution lock: overlapping Breeze and Tencent
GPU calls caused a native Metal crash during testing. Model work takes turns;
microphone capture, VAD, and OpenAI requests continue independently.
Each source segment is submitted once to each provider per conversation, including
across UI refreshes. **Retry OpenAI** and **Retry Tencent · Local** retry only the
failed segments for that provider; completed results remain visible.

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

Use the browser on the same Mac as Streamlit. If microphone access fails, check
browser permissions, press Stop, then retry. Remote hosting is outside this
version's scope.

The Streamlit/WebRTC versions are pinned to the combination verified locally;
WebRTC 0.78.1 repeatedly reset the microphone connection during testing.
File watching is disabled to avoid scanning Transformers' lazy imports.
Restart Streamlit after editing Python files.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests exercise audio resampling, WAV validation, recording and upload lifecycles,
interrupted-upload recovery, provider isolation, and translation ordering/retries
with lightweight VAD/inference and API substitutes. Evaluation tests cover the
score formula, normalization, transcript/table parsing, column selection, reference
routing, frozen evaluation inputs, and incomplete results.
Additional tests cover correction concurrency, timeouts, DNT protection, glossary
reload/learning, speaker streaming state, and final exports. They use the installed
Nemotron processor with substitute logits, without loading model weights or
calling OpenAI.
