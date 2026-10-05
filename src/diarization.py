"""Stream Nemotron speaker activity on CPU, independently of the ASR worker."""

from collections import deque
from pathlib import Path
from threading import Condition, Lock, Thread

import numpy as np
import torch


SAMPLE_RATE = 16_000
MODEL_ID = "nvidia/Nemotron-3-Diarization"
REVISION = "f667ed73aee57d40cc39428eb768b4fd87a0a29e"
MODEL_DIR = Path(__file__).resolve().parents[1] / "Models" / "Nemotron-3-Diarization"
REQUIRED_FILES = ("config.json", "processor_config.json", "model.safetensors")
FAILED_MESSAGE = "Speaker diarization is unavailable. Check the local Nemotron model and start a new recording."
OVERFLOW_MESSAGE = "Speaker diarization stopped because its audio queue filled. Start a new recording."
_MODEL_CACHE = None
_MODEL_LOCK = Lock()


def _load_model():
    """Load local weights once; this CPU lock never blocks the MPS ASR lock."""
    global _MODEL_CACHE
    with _MODEL_LOCK:
        if _MODEL_CACHE is None:
            if any(not (MODEL_DIR / filename).is_file() for filename in REQUIRED_FILES):
                raise FileNotFoundError("Run python scripts/download_diarization.py first.")
            from transformers import AutoModelForAudioFrameClassification, AutoProcessor

            processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
            processor.set_streaming_mode("low_latency")
            model = AutoModelForAudioFrameClassification.from_pretrained(
                MODEL_DIR, local_files_only=True, dtype=torch.float32,
            ).to("cpu").eval()
            _MODEL_CACHE = (processor, model)
        return _MODEL_CACHE


class NemotronDiarizer:
    """One continuous audio stream with its own cursor and arrival-order speaker cache.

    The processor defines exact first/subsequent chunk sizes and overlapping
    analysis windows. Only newly scored frames advance the output time; lookahead
    stays in the rolling audio buffer for the next call. No cache is reset at an
    ASR or VAD boundary. Returned segments use the original audio's absolute time.
    """

    def __init__(self, processor=None, model=None):
        self._processor, self._model = (processor, model) if processor is not None else _load_model()
        self._audio = np.empty(0, dtype=np.float32)
        self._audio_start = 0
        self._total_samples = 0
        self._mel_cursor = 0
        self._frames = 0
        self._first = True
        self._finished = False
        self._speaker_cache = None

    def __call__(self, samples, *, final=False):
        if self._finished:
            return []
        self._audio = np.concatenate((self._audio, samples))
        self._total_samples += len(samples)
        segments = []
        processor = self._processor
        while True:
            needed = (processor.num_samples_first_audio_chunk if self._first
                      else processor.num_samples_per_audio_chunk)
            if len(self._audio) < needed:
                break
            segments.extend(self._infer(self._audio[:needed], final=False))
            self._mel_cursor += processor.num_mel_frames_per_step
            next_start = processor.audio_chunk_start(self._mel_cursor)
            self._audio = self._audio[next_start - self._audio_start:]
            self._audio_start = next_start
            self._first = False
        if final:
            # Under 10 ms contains no complete model frame. Later chunks retain
            # analysis overlap plus lookahead, so their final tail is nonempty.
            if self._total_samples // processor.feature_extractor.hop_length > self._frames:
                segments.extend(self._infer(self._audio, final=True))
            self._audio = np.empty(0, dtype=np.float32)
            self._speaker_cache = None
            self._finished = True
        return segments

    def _infer(self, samples, *, final):
        processor = self._processor
        inputs = processor(
            samples, sampling_rate=SAMPLE_RATE, is_streaming=True,
            is_first_audio_chunk=self._first, is_last_audio_chunk=final, return_tensors="pt",
        ).to("cpu", dtype=torch.float32)
        if final:
            # A short first-and-last chunk also needs streaming mode, even though
            # no earlier forward has produced its speaker cache yet.
            inputs["num_lookahead_frames"] = 0
        with _MODEL_LOCK, torch.inference_mode():
            outputs = self._model(**inputs, speaker_cache=self._speaker_cache)
        self._speaker_cache = outputs.speaker_cache
        seconds_per_frame = processor.feature_extractor.hop_length / SAMPLE_RATE
        offset = self._frames * seconds_per_frame
        self._frames += outputs.logits.shape[1]
        duration = self._total_samples / SAMPLE_RATE
        return [
            {"start_s": min(duration, offset + item["Start"]),
             "end_s": min(duration, offset + item["End"]),
             "speaker_id": f"Speaker {int(item['Speaker']) + 1}"}
            for item in processor.extract_speaker_dict(outputs.logits)[0]
            if offset + item["Start"] < duration
        ]


class DiarizationSession:
    """Queue raw mono 16 kHz audio without waiting for model loading or inference.

    Audio is copied on push so capture buffers may be reused. The queue is bounded
    by unprocessed audio duration, including work in flight. Overflow fails this
    diarization attempt explicitly; ASR can continue independently. Upload callers
    may raise max_pending_seconds to their already-known, bounded file duration.
    A worker advances at most one second per call, publishing partial intervals.
    finish() flushes the processor tail once; close() cancels without joining.
    """

    def __init__(self, diarize=None, *, enabled=True, max_pending_seconds=600):
        if not np.isfinite(max_pending_seconds) or max_pending_seconds <= 0:
            raise ValueError("max_pending_seconds must be positive.")
        self._diarize = diarize
        self._limit = int(max_pending_seconds * SAMPLE_RATE)
        self._lock = Lock()
        self._condition = Condition(self._lock)
        self._queue = deque()
        self._queued_samples = 0
        self._segments = []
        self._last_by_speaker = {}
        self._status = "idle" if enabled else "disabled"
        self._error = None
        self._closed = not enabled
        self._ending = False
        self._flushed = False
        self._has_audio = False
        self._working = False
        self._worker = None

    def push(self, samples):
        with self._lock:
            if self._closed or self._ending or self._status == "failed":
                return
        audio = np.asarray(samples, dtype=np.float32)
        if audio.ndim != 1 or not np.isfinite(audio).all():
            with self._lock:
                if self._closed or self._ending or self._status == "failed":
                    return
                self._fail(FAILED_MESSAGE)
            return
        if not len(audio):
            return
        audio = audio.copy()
        with self._lock:
            if self._closed or self._ending or self._status == "failed":
                return
            if self._queued_samples + len(audio) > self._limit:
                self._fail(OVERFLOW_MESSAGE)
                return
            self._queue.append(audio)
            self._queued_samples += len(audio)
            self._has_audio = True
            self._status = "active"
            self._start_worker()
            self._condition.notify()

    def finish(self):
        with self._lock:
            if self._closed or self._ending or self._status == "failed":
                return
            self._ending = True
            if not self._has_audio:
                self._status = "complete"
            else:
                self._start_worker()
            self._condition.notify()

    def close(self):
        with self._lock:
            self._closed = True
            self._queue.clear()
            self._queued_samples = 0
            self._segments.clear()
            self._last_by_speaker.clear()
            self._status, self._error = "disabled", None
            self._condition.notify()

    def snapshot(self):
        with self._lock:
            pending = 0 if self._closed or self._status == "failed" else (
                len(self._queue) + int(self._working) + int(self._ending and not self._flushed and self._has_audio)
            )
            return {"status": self._status,
                    "segments": [dict(item) for item in sorted(self._segments, key=lambda row: (row["start_s"], row["speaker_id"]))],
                    "error": self._error, "pending": pending}

    def labels(self, timings):
        return speaker_labels(timings, self.snapshot()["segments"])

    def _start_worker(self):
        # Caller holds the state lock, including the worker's exit transition.
        if self._worker is None:
            self._worker = Thread(target=self._run, daemon=True, name="speaker-diarization")
            self._worker.start()

    def _fail(self, message):
        self._status, self._error = "failed", message
        self._queue.clear()
        self._queued_samples = 0
        self._condition.notify()

    def _publish(self, segments):
        for item in segments or []:
            start, end, speaker = float(item["start_s"]), float(item["end_s"]), str(item["speaker_id"])
            if not np.isfinite([start, end]).all() or start < 0 or end <= start or not speaker:
                raise ValueError("Invalid speaker interval.")
            previous = self._last_by_speaker.get(speaker)
            if previous is not None and start <= previous["end_s"] + 1e-6:
                previous["end_s"] = max(previous["end_s"], end)
            else:
                segment = {"start_s": start, "end_s": end, "speaker_id": speaker}
                self._segments.append(segment)
                self._last_by_speaker[speaker] = segment

    def _run(self):
        try:
            while True:
                with self._lock:
                    while not self._queue and not self._ending and not self._closed and self._status != "failed":
                        self._condition.wait()
                    if self._closed or self._status == "failed":
                        self._worker = None
                        return
                    if self._queue:
                        audio = self._queue.popleft()
                    elif self._ending and not self._flushed:
                        audio = None
                        self._flushed = True
                    else:
                        if self._ending:
                            self._status = "complete"
                        self._worker = None
                        return
                    self._working = True
                if self._diarize is None:
                    self._diarize = NemotronDiarizer()
                # Check cancellation after a potentially long model load too.
                parts = [None] if audio is None else (
                    audio[index:index + SAMPLE_RATE] for index in range(0, len(audio), SAMPLE_RATE)
                )
                for part in parts:
                    with self._lock:
                        if self._closed or self._status == "failed":
                            self._worker = None
                            return
                    segments = self._diarize(
                        np.empty(0, dtype=np.float32) if part is None else part,
                        final=part is None,
                    )
                    with self._lock:
                        if self._closed or self._status == "failed":
                            self._worker = None
                            return
                        self._publish(segments)
                        if part is not None:
                            self._queued_samples -= len(part)
                with self._lock:
                    self._working = False
        except Exception:
            # Model/processor errors can contain audio or paths. Expose only a
            # fixed message; never place exception strings in the UI snapshot.
            with self._lock:
                if not self._closed:
                    self._fail(FAILED_MESSAGE)
                self._worker = None
                self._working = False


def speaker_labels(timings, segments):
    """Assign dominant overlapping speakers; retain substantial mixed speech.

    Labels refer only to anonymous session-local model channels, never identities.
    A second speaker needs at least 80 ms and a quarter of the dominant overlap;
    tiny VAD padding overlaps do not create an extra label. No overlap stays None.
    """
    labels = []
    for timing in timings:
        overlaps = {}
        if timing is not None:
            start, end = timing["start_s"], timing["end_s"]
            for segment in segments:
                duration = min(end, segment["end_s"]) - max(start, segment["start_s"])
                if duration > 0:
                    speaker = segment["speaker_id"]
                    overlaps[speaker] = overlaps.get(speaker, 0) + duration
        ranked = sorted(overlaps, key=lambda speaker: (-overlaps[speaker], speaker))
        if ranked:
            threshold = max(0.08, overlaps[ranked[0]] * 0.25)
            labels.append(" / ".join(speaker for index, speaker in enumerate(ranked)
                                     if index == 0 or overlaps[speaker] >= threshold))
        else:
            labels.append(None)
    return labels
