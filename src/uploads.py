"""Prepare uploaded WAV audio for sequential transcription with local Breeze."""

from io import BytesIO
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch

from src.pipeline import SAMPLE_RATE, SPEECH_PAD_MS

MAX_SEGMENT_SECONDS = 25


def decode_wav(data: bytes) -> np.ndarray:
    """Decode WAV bytes to contiguous mono float32 audio at 16 kHz.

    PCM and floating-point WAV files, including WAVEX and RF64, are read
    entirely in memory. Channels are averaged before resampling. Invalid,
    empty, non-WAV, or nonfinite audio raises ValueError. Silence is valid.
    This helper neither loads models nor writes uploaded audio to disk.
    """
    try:
        with sf.SoundFile(BytesIO(data)) as source:
            if source.format not in {"WAV", "WAVEX", "RF64"}:
                raise ValueError("Please upload a WAV audio file.")
            sample_rate = source.samplerate
            samples = source.read(dtype="float32", always_2d=True)
    except sf.SoundFileError as exc:
        raise ValueError("Could not read this WAV file. It may be invalid or corrupted.") from exc

    if not len(samples):
        raise ValueError("The WAV file is empty and contains no audio samples.")
    if not np.isfinite(samples).all():
        raise ValueError("The WAV file contains nonfinite audio samples.")

    audio = samples.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        divisor = gcd(sample_rate, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // divisor, sample_rate // divisor)
    if not np.isfinite(audio).all():
        raise ValueError("The WAV file contains audio values that cannot be processed.")
    return np.ascontiguousarray(audio, dtype=np.float32)


def speech_segments(audio: np.ndarray, vad) -> list[np.ndarray]:
    """Return ordered speech views, each at most 25 seconds, from decoded audio.

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
        # Keep Breeze's input bounded even if a VAD span exceeds its limit.
        for offset in range(start, end, max_samples):
            segments.append(audio[offset:min(offset + max_samples, end)])
    return segments
