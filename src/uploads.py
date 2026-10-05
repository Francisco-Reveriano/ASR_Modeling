"""Prepare uploaded WAV audio for sequential transcription with hosted ASR."""

import logging
from io import BytesIO
from math import gcd
from concurrent.futures import Future
from threading import Thread
from time import monotonic

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch

from src.pipeline import SAMPLE_RATE, SPEECH_PAD_MS

MAX_SEGMENT_SECONDS = 15
WAV_BLOCK_FRAMES = 65_536
logger = logging.getLogger(__name__)


def transcribe_in_background(transcribe, audio, *, on_result=None):
    """Return one reusable decode task while the UI polls completed translations.

    The upload job retains this Future across Streamlit reruns. Only its owner
    appends the result, so an interrupted UI run neither repeats ASR nor lets an
    old task write into a replacement session. on_result(text, start_ms, end_ms)
    runs before the Future is published, allowing translation to start without a
    UI rerun. It must capture its owning session, never Streamlit session state.
    The caller owns the endpoint client and closes it after all upload segments.
    """
    future = Future()

    def run():
        if not future.set_running_or_notify_cancel():
            return
        try:
            started = monotonic() * 1000
            text = transcribe(audio)
            finished = monotonic() * 1000
            if on_result is not None and text:
                try:
                    on_result(text, started, finished)
                except Exception:
                    logger.error("Upload transcript observer failed")
            future.set_result((text, started, finished))
        except Exception:
            # Endpoint exception bodies/URLs may contain credentials.
            future.set_exception(RuntimeError("Transcription failed. Check the ASR endpoint and retry the file."))

    Thread(target=run, daemon=True, name="upload-transcription").start()
    return future


def decode_wav(data: bytes) -> np.ndarray:
    """Decode WAV bytes to contiguous mono float32 audio at 16 kHz.

    PCM and floating-point WAV files, including WAVEX and RF64, are decoded
    in bounded blocks in memory. Channels are averaged before resampling. Invalid,
    empty, non-WAV, or nonfinite audio raises ValueError. Silence is valid.
    This helper neither loads models nor writes uploaded audio to disk.
    """
    try:
        with sf.SoundFile(BytesIO(data)) as source:
            if source.format not in {"WAV", "WAVEX", "RF64"}:
                raise ValueError("Please upload a WAV audio file.")
            sample_rate = source.samplerate
            audio = np.empty(source.frames, dtype=np.float32)
            offset = 0
            while offset < len(audio):
                samples = source.read(
                    frames=min(WAV_BLOCK_FRAMES, len(audio) - offset),
                    dtype="float32", always_2d=True,
                )
                if not len(samples):
                    del samples
                    break
                if not np.isfinite(samples).all():
                    raise ValueError("The WAV file contains nonfinite audio samples.")
                stop = offset + len(samples)
                audio[offset:stop] = samples.mean(axis=1)
                offset = stop
                del samples
            # A truncated stream can end before its advertised frame count.
            audio.resize(offset, refcheck=False)
    except sf.SoundFileError as exc:
        raise ValueError("Could not read this WAV file. It may be invalid or corrupted.") from exc

    if not len(audio):
        raise ValueError("The WAV file is empty and contains no audio samples.")
    if sample_rate != SAMPLE_RATE:
        divisor = gcd(sample_rate, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // divisor, sample_rate // divisor)
    if not np.isfinite(audio).all():
        raise ValueError("The WAV file contains audio values that cannot be processed.")
    return np.ascontiguousarray(audio, dtype=np.float32)


def speech_segments(audio: np.ndarray, vad, *, with_timestamps=False) -> list:
    """Return ordered speech views, each at most 15 seconds, from decoded audio.

    Pass the mono 16 kHz array from decode_wav and a fresh load_vad() iterator.
    Silero resets its model state and returns sample boundaries that already
    include padding. Its 250 ms minimum speech duration removes short bursts;
    500 ms of silence separates utterances. Silence returns an empty list.

    Create the VAD before calling this helper: the lazy import below preserves
    load_vad's handling of Silero's process-wide Torch thread setting. Returned
    arrays share memory with audio, so treat them as read-only while transcribing.
    """
    from silero_vad import get_speech_timestamps

    timestamps = get_speech_timestamps(
        torch.from_numpy(audio), vad.model, sampling_rate=SAMPLE_RATE,
        threshold=0.5, min_speech_duration_ms=250,
        max_speech_duration_s=MAX_SEGMENT_SECONDS,
        min_silence_duration_ms=500, speech_pad_ms=SPEECH_PAD_MS,
    )
    max_samples = MAX_SEGMENT_SECONDS * SAMPLE_RATE
    segments = []
    for speech in timestamps:
        start = max(0, speech["start"])
        end = min(len(audio), speech["end"])
        # Keep hosted ASR input bounded even if a VAD span exceeds its limit.
        for offset in range(start, end, max_samples):
            stop = min(offset + max_samples, end)
            samples = audio[offset:stop]
            segments.append({
                "audio": samples, "start_s": offset / SAMPLE_RATE, "end_s": stop / SAMPLE_RATE,
            } if with_timestamps else samples)
    return segments
