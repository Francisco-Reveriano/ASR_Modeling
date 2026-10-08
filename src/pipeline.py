"""Turn a live microphone stream into completed pieces of transcript.

VAD (voice activity detection) decides where speech starts and ends. ASR
(automatic speech recognition) turns that speech into text. Silero handles
VAD here; the local Breeze model handles ASR.

The audio takes two paths, connected by a queue:

    WebRTC audio frame
        -> push: resample to mono 16 kHz float audio
        -> _feed: collect complete 512-sample VAD frames
        -> _consume: detect speech boundaries and retain the needed audio
        -> _emit: copy completed speech segments into the queue

    Background worker (_run)
        -> take one segment from the queue
        -> call Breeze
        -> store text for the UI to read through snapshot()

Keeping ASR in its own thread lets microphone capture continue during model
inference. Speech normally ends at a pause; continuous speech is also split
into bounded segments so it fits the model's input window.

Typical use: load and reuse one transcriber, then create a LiveTranscriber
with a fresh VAD for each recording. Connect push() and finish() to WebRTC's
audio-frame and audio-ended callbacks, and poll snapshot() from the UI.
This module makes no Streamlit calls and saves no recordings or transcripts
to disk; model weights are read from the local Models directory.
"""

import logging
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from time import monotonic

import av
import numpy as np
import torch

from src.model_lock import LOCAL_MODEL_LOCK
from src.diarization import prepare_speaker_turns

SAMPLE_RATE = 16_000  # Both models receive 16,000 audio samples per second.
FRAME_SIZE = 512  # One Silero inference window: 32 ms at this sample rate.
SPEECH_PAD_MS = 150  # Keep some audio around detected speech boundaries.
# Silero can report a start before the current frame. Retain the padding plus
# one frame of recent silence so the beginning of that speech is still available.
PRE_ROLL = SAMPLE_RATE * SPEECH_PAD_MS // 1000 + FRAME_SIZE
# Resolve from this file, so launching from a different directory still works.
MODEL_DIR = Path(__file__).resolve().parents[1] / "Models" / "breeze-asr-26"
logger = logging.getLogger(__name__)


def load_transcriber():
    """Load local Breeze weights and return an audio-to-text callable.

    The callable takes a one-dimensional float32 NumPy array containing mono
    16 kHz audio, normally scaled between -1 and 1. It returns a stripped string
    without model control tokens. Audio must already be resampled and segmented;
    LiveTranscriber handles that preparation.

    Loading is expensive: cache/reuse the returned callable in the caller.
    This function itself does not cache, and each call loads another model.
    Apple MPS (Metal GPU acceleration) is used when available, otherwise CPU.
    Both paths use float32. Missing or incompatible local model files raise an
    exception to the caller; local_files_only prevents a download fallback.

    Loading and inference share an execution lock with the local Tencent
    translator. Concurrent model work can crash this Mac's Metal runtime. The
    lock is separate from each recording's audio-buffer lock, so waiting for a
    model does not block microphone capture or OpenAI requests.
    """
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    with LOCAL_MODEL_LOCK:
        processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL_DIR, local_files_only=True, dtype=torch.float32
        ).to(device).eval()

    def transcribe(audio):
        """Convert one prepared speech segment into text using the loaded model."""
        # eval() above selects inference behavior; inference_mode() here also
        # disables gradient tracking. The processor prepares Whisper features
        # and a mask indicating which parts are audio rather than padding.
        with LOCAL_MODEL_LOCK, torch.inference_mode():
            inputs = processor(
                audio, sampling_rate=SAMPLE_RATE,
                return_tensors="pt", return_attention_mask=True,
            ).to(device)
            tokens = model.generate(
                **inputs, task="transcribe", return_timestamps=False,
            )
            return processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()

    return transcribe


def load_vad():
    """Create a CPU Silero iterator with fresh state for one recording.

    The iterator consumes 512-sample tensors at 16 kHz. It returns None while
    a boundary is undecided, or a dictionary such as {"start": 1600} or
    {"end": 24000}. These are sample offsets from the start of the recording,
    not seconds or positions within the latest frame. They already include
    speech padding, so the caller must not add that padding again.

    A threshold of 0.5 starts speech detection; 500 ms of sustained silence
    closes an utterance. The 150 ms padding helps retain word beginnings and
    endings. The iterator remembers previous frames and its current speech
    state: create a new one for every recording instead of sharing it.
    """
    threads = torch.get_num_threads()
    try:
        from silero_vad import VADIterator, load_silero_vad
    finally:
        # Silero's import sets this globally to one, also affecting Breeze.
        torch.set_num_threads(threads)
    return VADIterator(
        load_silero_vad(), sampling_rate=SAMPLE_RATE,
        threshold=0.5, min_silence_duration_ms=500, speech_pad_ms=SPEECH_PAD_MS,
    )


class LiveTranscriber:
    """Own the audio buffers, VAD state, and ASR queue for a single recording.

    Thread ownership
    ----------------
    WebRTC calls push(); the UI or WebRTC's ended callback calls finish().
    Both hold _lock while changing audio state. The worker runs ASR outside
    that lock and acquires it only to publish text or errors. snapshot() takes
    the same lock so the UI sees a consistent copy of the recording's state.
    Private helpers _feed(), _consume(), _emit(), and _fail() expect their
    caller to already hold _lock; they must not acquire it again.

    Audio coordinates
    -----------------
    All offsets count resampled 16 kHz samples from the recording's start.
    _buffer contains the interval [_offset, _position), with the end excluded.
    _speech_start is the first sample of speech not yet queued, or None when
    there is no active utterance. Translate an absolute offset into a buffer
    index by subtracting _offset: if _offset is 16000, sample 16800 is at
    buffer index 800. _tail holds samples still waiting for a complete VAD
    frame; they have not advanced _position yet.

    Lifecycle
    ---------
    Construction prepares the recording, but the worker starts only when the
    first audio frame arrives. finish() stops accepting audio and flushes any
    remaining speech; queued ASR may still be running after it returns.
    _input_ended means no more segments will be produced. _finished means the
    worker has exited (or never needed to start). An error also ends input.
    Create a new instance to restart; a stopped instance cannot be reused.
    """

    def __init__(self, transcribe, vad, *, max_segment_seconds=15, max_pending=8,
                 diarization=None):
        """Prepare recording state without starting a background thread.

        Args:
            transcribe: Callable accepting a mono 16 kHz float32 array and
                returning text. Normally supplied by load_transcriber().
            vad: Fresh streaming detector, normally supplied by load_vad().
            max_segment_seconds: Positive segment limit. The 15-second
                default stays below Breeze's 30-second input window.
            max_pending: Positive capacity for waiting segments. The segment
                currently being transcribed is outside this queue, so the
                default permits eight waiting segments plus one in progress.
            diarization: Optional independent speaker session with nonblocking
                push() and finish() methods. It receives all resampled audio,
                including silence, and keeps its speaker history across VAD cuts.

        Callables are supplied rather than loaded here so the large ASR model
        can be reused and tests can substitute small, predictable functions.
        """
        self._transcribe = transcribe
        self._vad = vad
        self._max_samples = int(max_segment_seconds * SAMPLE_RATE)
        # Keep one resampler for the whole recording: resampling has history
        # and may retain a few samples that finish() must later flush.
        self._resampler = av.AudioResampler(format="fltp", layout="mono", rate=SAMPLE_RATE)
        # _tail is the incomplete VAD frame; _buffer is retained speech/pre-roll.
        self._tail = np.empty(0, dtype=np.float32)
        self._buffer = np.empty(0, dtype=np.float32)
        self._offset = 0
        self._position = 0
        self._speech_start = None
        self._queue = Queue(maxsize=max_pending)
        self._lock = Lock()
        self._input_ended = Event()
        self._finished = Event()
        self._cancelled = Event()
        self._texts = []
        self._timings = []
        self._diarization = diarization
        # Count both queued and in-progress segments for the UI's status.
        self._pending = 0
        self._error = None
        self._worker = Thread(target=self._run, daemon=True, name="speech-transcription")

    def push(self, frame):
        """Accept a PyAV AudioFrame and return the original frame to WebRTC.

        Frames arrive in order, but their sample rate, channels, and lengths
        can differ from what Silero needs. PyAV converts them to mono float32
        at 16 kHz; _feed() then assembles fixed-size VAD frames. A resampler
        call can produce zero, one, or several output frames.

        This callback performs resampling and VAD only, never ASR inference.
        Calls after Stop or an error are ignored. Audio-processing exceptions
        are logged and exposed through snapshot() rather than escaping into
        WebRTC's callback thread.
        """
        with self._lock:
            if not self._input_ended.is_set():
                if self._worker.ident is None:
                    # No idle worker is left behind if microphone permission
                    # is denied and no audio ever arrives.
                    self._worker.start()
                try:
                    for audio in self._resampler.resample(frame):
                        self._feed(audio.to_ndarray().reshape(-1))
                except Exception:
                    logger.error("Audio processing failed")
                    self._fail("Audio processing failed. Start a new recording to retry.")
        return frame

    def _feed(self, samples):
        """Feed complete VAD frames, retaining fewer than 512 leftover samples.

        For example, with an empty tail, 700 samples send 512 to _consume()
        and leave 188 in _tail. The next input continues exactly where it left
        off; browser frame boundaries do not become speech boundaries.
        """
        if self._diarization is not None and len(samples):
            # A separate worker consumes the continuous timeline, including
            # silence. Never reset speaker identity at each ASR boundary.
            self._diarization.push(samples)
        samples = np.concatenate((self._tail, samples))
        complete = len(samples) // FRAME_SIZE * FRAME_SIZE
        self._tail = samples[complete:].copy()
        for offset in range(0, complete, FRAME_SIZE):
            if self._input_ended.is_set():
                break
            self._consume(samples[offset:offset + FRAME_SIZE])

    def _consume(self, samples):
        """Apply VAD to one frame and queue speech whose boundary is known.

        Normally samples has length 512. During finish(), it can be shorter:
        only the VAD tensor is padded with zeros, while buffer positions and
        the eventual ASR segment still count only real audio samples.
        """
        self._buffer = np.concatenate((self._buffer, samples))
        self._position += len(samples)
        # Only the final frame may need padding; keep padding out of ASR audio.
        frame = np.pad(samples, (0, FRAME_SIZE - len(samples)))
        event = self._vad(torch.from_numpy(frame))
        if event and "start" in event:
            # Use Silero's already-padded absolute boundary, restricted to the
            # audio we have retained. At recording start there is no pre-roll.
            self._speech_start = max(self._offset, min(event["start"], self._position))
        if event and "end" in event and self._speech_start is not None:
            self._emit(min(event["end"], self._position))
            self._speech_start = None

        # A person may speak without pausing. Emit contiguous, non-overlapping
        # pieces at the limit, advancing the unqueued start without resetting
        # Silero's recurrent state or pretending the utterance has ended.
        while (
            self._speech_start is not None
            and self._position - self._speech_start >= self._max_samples
            and not self._input_ended.is_set()
        ):
            end = self._speech_start + self._max_samples
            self._emit(end)
            self._speech_start = end

        keep = self._speech_start
        if keep is None:
            # During silence, retain only enough history for a future start
            # event. During speech, retain audio from the unqueued start.
            keep = max(self._offset, self._position - PRE_ROLL)
        self._buffer = self._buffer[keep - self._offset:].copy()
        self._offset = keep

    def _emit(self, end):
        """Queue [_speech_start, end), splitting it into bounded copies.

        end is an absolute, exclusive sample offset. _speech_start is left
        unchanged; _consume() or finish() decides how to advance speech state.
        Enforcing the limit here also covers a VAD end event arriving on the
        same frame that crosses the duration limit.

        Each copy belongs to the worker and remains valid when _buffer is
        trimmed. Enqueueing never waits for inference: a full queue stops input
        with a visible error. Accepted segments still drain, but the segment
        that could not fit and any subsequent audio are not transcribed.
        """
        start = self._speech_start
        while start < end and not self._input_ended.is_set():
            stop = min(end, start + self._max_samples)
            audio = self._buffer[start - self._offset:stop - self._offset].copy()
            try:
                timing = {
                    "start_s": start / SAMPLE_RATE, "end_s": stop / SAMPLE_RATE,
                    "server_endpoint_ms": monotonic() * 1000,
                    # WebRTC here does not expose a synchronized client clock.
                    "t_capture_ms": None,
                }
                self._queue.put_nowait((audio, timing))
                self._pending += 1
            except Full:
                self._fail(
                    "Transcription fell behind. Recording stopped; the latest segment "
                    "was not accepted. Waiting segments will finish."
                )
            start = stop

    def _fail(self, message):
        """Publish an error and stop input; this does not itself empty the queue.

        Audio errors and overload allow accepted segments to finish. An ASR
        exception is handled separately in _run(), which discards waiting work.
        """
        self._error = message
        self._input_ended.set()

    def _release_capture_audio(self):
        """Release residual audio after capture and queued ASR have ended.

        The caller holds _lock. Text and timings outlive these buffers; speaker
        detection owns its own audio copies and may still be finishing.
        """
        self._buffer = np.empty(0, dtype=np.float32)
        self._tail = np.empty(0, dtype=np.float32)
        self._offset = self._position
        self._speech_start = None

    def finish(self):
        """Close input and submit remaining speech without waiting for ASR.

        Flush in audio order: samples retained by the resampler, the incomplete
        VAD frame, then any active utterance that has not ended in silence.
        This keeps the final word when the user presses Stop while speaking.
        Audio is queued only if VAD has marked it as speech.

        Repeated calls are safe, including a UI Stop followed by WebRTC's
        ended callback. After an earlier error, this returns without accepting
        more audio. Check snapshot()["finished"] to learn when the worker is
        actually done; this method does not join or wait for that thread.
        """
        with self._lock:
            if self._input_ended.is_set():
                return
            try:
                for audio in self._resampler.resample(None):
                    self._feed(audio.to_ndarray().reshape(-1))
                if len(self._tail) and not self._input_ended.is_set():
                    self._consume(self._tail)
                    self._tail = np.empty(0, dtype=np.float32)
                if self._speech_start is not None and not self._input_ended.is_set():
                    self._emit(self._position)
                    self._speech_start = None
            except Exception:
                logger.error("Finishing audio failed")
                self._fail("Finishing audio failed. Start a new recording to retry.")
            finally:
                self._input_ended.set()
                if self._diarization is not None:
                    self._diarization.finish()
                if self._worker.ident is None:
                    self._release_capture_audio()
                    self._finished.set()

    def close(self):
        """Cancel queued audio and discard any result from in-flight inference.

        Model calls cannot safely be interrupted by killing Python threads.
        The active call may return later, but it can no longer publish text or
        start another segment. Unlike finish(), this never flushes capture.
        """
        with self._lock:
            self._cancelled.set()
            self._input_ended.set()
            while not self._queue.empty():
                self._queue.get_nowait()
            self._pending = 0
            self._release_capture_audio()
            if self._worker.ident is None:
                self._finished.set()
        if self._diarization is not None:
            self._diarization.close()

    def _run(self):
        """Transcribe queued segments in order until input ends and work drains.

        A short queue timeout lets the worker check for shutdown during silence.
        ASR runs outside _lock so incoming audio can continue to be segmented.
        Empty transcript strings count as completed work but are not displayed.

        If inference fails, preserve existing text, discard waiting segments,
        stop accepting audio, and report the error. There is no automatic retry.
        """
        try:
            while True:
                if self._cancelled.is_set():
                    return
                try:
                    audio, timing = self._queue.get(timeout=0.1)
                except Empty:
                    # finish() may enqueue its last segment just after get()
                    # times out. Once input has ended, check the queue again
                    # so the worker cannot exit with that final segment waiting.
                    if self._input_ended.is_set() and self._queue.empty():
                        return
                    continue
                try:
                    parts = prepare_speaker_turns(audio, timing, self._diarization)
                    for part in parts:
                        if self._cancelled.is_set():
                            return
                        part_timing = {key: value for key, value in part.items() if key != "audio"}
                        part_timing["asr_start_ms"] = monotonic() * 1000
                        text = self._transcribe(part["audio"])
                        part_timing["asr_final_ms"] = monotonic() * 1000
                        with self._lock:
                            if self._cancelled.is_set():
                                return
                            if text:
                                self._texts.append(text)
                                self._timings.append(part_timing)
                    del parts, part
                    with self._lock:
                        self._pending = max(0, self._pending - 1)
                except Exception:
                    logger.error("Transcription failed")
                    with self._lock:
                        if self._cancelled.is_set():
                            return
                        self._fail("Transcription failed. Check the selected ASR model and start a new recording to retry.")
                        while not self._queue.empty():
                            self._queue.get_nowait()
                        self._pending = 0
                    return
                finally:
                    # Do not keep the last segment alive while waiting for
                    # more speech. The queue owns any remaining segments.
                    del audio
        finally:
            with self._lock:
                self._release_capture_audio()
            if self._diarization is not None:
                self._diarization.finish()
            self._finished.set()

    def snapshot(self):
        """Return UI-readable state without exposing the mutable transcript list.

        Returned fields:
            texts: Nonempty completed transcripts, in segment order. The list
                is copied, so editing the returned list cannot alter this session.
            timings: Matching audio start/end offsets in seconds and server
                monotonic processing timestamps in milliseconds. Audio offsets
                attach speaker labels and captions; monotonic times measure
                processing delays. t_capture_ms is None because this browser
                integration does not provide a synchronized client clock.
            pending: Accepted segments still queued or currently being transcribed.
            finished: True once the worker exits, or Stop occurs before any audio.
                Consult error as well: finished does not imply success.
            error: The reported error message, or None.
            accepting: Whether push() still accepts audio. This describes the
                pipeline's state, not whether the browser microphone is connected.

        After a normal Stop, accepting can be False while pending remains
        positive and finished remains False: the worker is draining its queue.
        """
        with self._lock:
            return {
                "texts": self._texts.copy(),
                "timings": [dict(timing) for timing in self._timings],
                "pending": self._pending,
                "finished": self._finished.is_set(),
                "error": self._error,
                "accepting": not self._input_ended.is_set(),
            }
