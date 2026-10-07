"""Transcribe bounded speech segments with OpenAI's Realtime API."""

import base64
import json
import os
from time import monotonic

from dotenv import load_dotenv
import numpy as np
from scipy.signal import resample_poly

from src.translation import ENV_FILE

TRANSCRIPTION_MODEL = "gpt-live-transcribe"
REALTIME_URL = "wss://api.openai.com/v1/realtime?intent=transcription"
REQUEST_TIMEOUT = 45.0
FAILED_MESSAGE = "OpenAI transcription failed. Check your API access and connection, then retry."
MISSING_KEY_MESSAGE = "Set OPENAI_API_KEY in .env, then retry transcription."


class TranscriptionError(RuntimeError):
    """A safe error that never includes provider payloads or credentials."""


def load_openai_transcriber():
    """Freeze credentials for a session without opening a connection yet.

    The returned callable accepts mono 16 kHz float audio, like Breeze. Each
    utterance owns one connection and one committed item, so abandoned sessions
    cannot mix transcripts with new work. Silero still detects speech locally;
    only final transcripts enter the existing ordered translation queues.
    """
    load_dotenv(ENV_FILE, override=False)
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise TranscriptionError(MISSING_KEY_MESSAGE)

    def transcribe(audio):
        try:
            return _transcribe_segment(audio, api_key)
        except Exception:
            # WebSocket/HTTP errors can contain headers and server payloads.
            # LiveTranscriber logs exceptions, so suppress their chains too.
            raise TranscriptionError(FAILED_MESSAGE) from None

    return transcribe


def _receive(connection, deadline):
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise TimeoutError()
    event = json.loads(connection.recv(timeout=remaining))
    if event["type"] in {"error", "conversation.item.input_audio_transcription.failed"}:
        raise TranscriptionError(FAILED_MESSAGE)
    return event


def _transcribe_segment(audio, api_key):
    from websockets.sync.client import connect

    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim != 1 or not np.isfinite(samples).all() or len(samples) > 15 * 16000:
        raise ValueError("Expected at most 15 seconds of finite mono 16 kHz audio.")
    if not len(samples):
        return ""
    # Realtime expects little-endian signed PCM16 at 24 kHz. A short final
    # microphone tail needs padding to satisfy the minimum 100 ms commit.
    samples = resample_poly(samples, 3, 2)
    if len(samples) < 2400:
        samples = np.pad(samples, (0, 2400 - len(samples)))
    pcm = (np.clip(samples, -1, 32767 / 32768) * 32768).astype("<i2").tobytes()
    deadline = monotonic() + REQUEST_TIMEOUT
    with connect(
        REALTIME_URL, additional_headers={"Authorization": f"Bearer {api_key}"},
        open_timeout=10, close_timeout=2, max_size=1024 * 1024,
    ) as connection:
        connection.send(json.dumps({
            "type": "session.update",
            "session": {
                "type": "transcription",
                "audio": {"input": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "transcription": {
                        "model": TRANSCRIPTION_MODEL,
                        "prompt": "Conversation in Taiwanese Hokkien, Mandarin, English, or mixed speech.",
                    },
                    "turn_detection": None,
                }},
            },
        }))
        while _receive(connection, deadline)["type"] != "session.updated":
            pass
        # Send bounded frames rather than a file transcription request: this
        # exact model uses the Realtime transcription protocol for both inputs.
        for offset in range(0, len(pcm), 48000):
            connection.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm[offset:offset + 48000]).decode("ascii"),
            }))
        connection.send(json.dumps({"type": "input_audio_buffer.commit"}))
        item_id = None
        while True:
            event = _receive(connection, deadline)
            if event["type"] == "input_audio_buffer.committed":
                item_id = event["item_id"]
            elif event["type"] == "conversation.item.input_audio_transcription.completed":
                if item_id is not None and event.get("item_id") == item_id:
                    return event["transcript"].strip()
