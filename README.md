# Breeze Voice transcription and translation

A local Streamlit app for browser microphone recording and WAV uploads. It uses
Silero VAD to split speech and transcribes each segment with the existing
`Models/breeze-asr-26` model. Breeze-ASR-26 targets Taiwanese Hokkien and outputs
Chinese characters. Each completed segment is translated into English twice:
by OpenAI through its API and by [Tencent Hy-MT2-1.8B](https://huggingface.co/tencent/Hy-MT2-1.8B)
running on this computer. The original and both translations appear in separate
columns with matching line numbers.

## Run

From the repository root:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/download_tencent.py
python -m streamlit run app.py --server.address localhost
```

Open <http://localhost:8501>, click **Start recording**, and allow microphone
access. The first start loads the local models; later recordings reuse Breeze.
Text is appended after roughly half a second of silence plus inference time.
**Stop recording** submits any unfinished speech and finishes waiting segments.
The next Start clears the conversation. Audio stays on this Mac; completed
transcript text is sent to OpenAI for translation. The app keeps the conversation
in memory unless you download it.

The conversation view numbers each completed speech segment (`01`, `02`, …).
Use **Download conversation** to save the numbered original and both English results before starting
again. **Microphone settings** contains the browser device picker once recording
has started. The workspace uses a light theme and stacks its panels on narrow screens.

### Upload a WAV file

Choose **Upload WAV**, select a `.wav` file, optionally play it back, then click
**Transcribe file**. Mono and stereo WAV files are converted to mono 16 kHz.
Silero finds speech, and Breeze transcribes each segment in order. Progress and
numbered text appear as segments finish; each translation follows independently.
Use **Download conversation** to save all three columns.
Selecting a file alone does not start transcription. Each new transcription
replaces the displayed text. Stop any live recording and let its pending segments
finish before uploading. Files are processed in memory on this Mac.

If creating an environment from scratch, run `python3 -m venv .venv` first.
The model and processor must already be present in `Models/breeze-asr-26`;
the app uses local files and needs no Hugging Face token. The existing download
notebook can prepare the model when needed.

### Evaluate against a reference

Open **Evaluate**, drop a `.wav` file and its reference as `.txt`, `.srt`, or
`.vtt`, then check **Preview spoken reference** before clicking **Run evaluation**.
References may be UTF-8 (with or without BOM) or UTF-16 with BOM, up to 1 MiB each.
The same Silero → Breeze → OpenAI/Tencent pipeline always produces a transcript
and both English translations.

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

- **Source transcript:** scores Breeze against the words spoken in the audio.
  Chinese/English references use **Mixed match**: each Chinese character and each
  English word is one unit. Add an optional **English translation reference** to
  also score OpenAI and Tencent; otherwise their translations appear unscored.
- **English translation:** scores OpenAI and Tencent with **1-wMER**. Breeze's
  source transcript remains visible.

Auto-detection chooses source when the cleaned reference contains Chinese
characters, otherwise English translation. Override **Reference type** for an
English source transcript or an English reference containing Chinese names.
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
reference length as its denominator. Each model's completed segments are joined
and aligned against its whole reference once. Reference line breaks do not need
to match speech pauses. There is no speaker/time alignment; overlapping speech
is evaluated in transcript order and can affect the score.

Scoring applies Unicode NFKC normalization and case folding, replaces punctuation
with spaces, and splits words on whitespace (plus Han character boundaries for
Mixed match). Apostrophes and hyphens split words; accents and digits remain.
Traditional and simplified characters are not converted into each other.
**This measures text matching, not semantic translation quality**; valid
paraphrases can score lower. Only outputs with a matching reference are scored.

Pending or failed outputs have no final score. Retry that provider to finish its
evaluation. Successfully processed silence against a nonempty reference scores
0%. **Download evaluation** includes all outputs, scores, alignment counts,
cleaned reference text, original structured text, and filenames. Selecting new
files or changing reference settings does not change an existing evaluation;
**Run evaluation** starts a new one and freezes its inputs and settings.

### OpenAI configuration

The app reads the repository's `.env` file, with existing environment variables
taking precedence:

```dotenv
OPENAI_API_KEY=your-api-key
OPENAI_DEFAULT_MODEL=gpt-6-luna
```

The key stays on the server. Only transcript text is sent to the official OpenAI
Responses API; audio is never uploaded to OpenAI. Requests use `store=False`.
Translation needs internet access and uses your API account. Restart Streamlit
after changing credentials or the model setting.

### Local Tencent configuration

`python scripts/download_tencent.py` downloads approximately 4.1 GB of model files
to `Models/Hy-MT2-1.8B/`. The script pins the Tencent repository revision, includes
the model card and license, and can resume an interrupted download. This public
model requires no Hugging Face token. It uses the existing PyTorch/Transformers
stack (`transformers>=5.6.0`), without a separate model server.

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
recording or file clears all three columns and cancels queued translations.
An already running translation may finish, but cannot populate the new conversation.

## How it works

- `app.py` handles microphone controls, model caching, and transcript updates.
- `src/ui.py` renders and exports aligned source/OpenAI/Tencent rows;
  `assets/style.css` styles the responsive workspace.
- `src/evaluation.py` parses reference TXT/SRT/VTT files and computes normalized
  whole-file word or mixed-unit match scores without model or network calls.
- `src/translation.py` provides the OpenAI request and reusable translation queue;
  the app creates one queue for each provider.
- `src/tencent.py` lazily loads Hy-MT2-1.8B and serializes local generation with a
  lock shared with Breeze in `src/model_lock.py`.
  `scripts/download_tencent.py` prepares its pinned model assets.
- `src/uploads.py` decodes WAV uploads and finds speech segments for sequential
  transcription, avoiding the live microphone queue. File VAD drops speech bursts
  shorter than 250 ms and limits ASR segments to 25 seconds.
- `src/pipeline.py` resamples audio to mono 16 kHz and feeds Silero 512-sample
  frames. Speech uses 150 ms of padding and ends after 500 ms of silence.
- Speech longer than 25 seconds is split. One background worker transcribes
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
score formula, normalization, transcript parsing, reference routing, and incomplete results.
They do not load either large model or call OpenAI.
