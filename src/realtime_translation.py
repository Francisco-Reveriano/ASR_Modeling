"""Continuous audio-to-English translation with session-owned streaming state."""

import asyncio
import base64
from collections import deque
import json
import os
from threading import Event, Lock, Thread
from time import monotonic

import av
from dotenv import load_dotenv
import numpy as np

from src.translation import ENV_FILE
from src.translation_validation import contains_cjk

REALTIME_TRANSLATION_MODEL = "gpt-realtime-translate"
TRANSLATION_URL = "wss://api.openai.com/v1/realtime/translations?model=gpt-realtime-translate"
MISSING_KEY_MESSAGE = "Set OPENAI_API_KEY in .env, then retry realtime translation."
FAILED_MESSAGE = "Realtime translation failed. Check your OpenAI access and connection, then start again."
EMPTY_TRANSLATION_MESSAGE = "No English translation was returned. Try another recording or model option."
DRAIN_TIMEOUT = 45.0
SAMPLE_RATE = 24000


class RealtimeTranslationSession:
    """Stream microphone frames or a WAV through one translation connection.

    Capture only resamples and queues audio. A background event loop sends audio
    and receives captions concurrently, including while the person is speaking.
    Source and translation deltas have independent boundaries, so they are shown
    as two continuous texts instead of claiming word/utterance alignment.
    finish() flushes the service through session.closed; close() abandons work.
    """

    def __init__(self, *, max_pending_seconds=30):
        load_dotenv(ENV_FILE, override=False)
        self._api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not self._api_key:
            raise ValueError(MISSING_KEY_MESSAGE)
        self._lock = Lock()
        self._cancelled = Event()
        self._finished = Event()
        self._input_ended = False
        self._queue = deque()
        self._queued_bytes = 0
        self._max_pending_bytes = int(max_pending_seconds * SAMPLE_RATE * 2)
        self._resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
        self._source = []
        self._english = []
        self._error = None
        self._duration = 0.0
        self._sent_seconds = 0.0
        self._file_audio = None
        self._file_duration = None
        self._worker = Thread(target=self._run, daemon=True, name="realtime-translation")

    def _pcm(self, frame):
        for converted in self._resampler.resample(frame):
            yield converted.to_ndarray().astype("<i2", copy=False).tobytes()

    def _enqueue(self, pcm):
        if self._queued_bytes + len(pcm) > self._max_pending_bytes:
            self._error = "Realtime translation fell behind. Recording stopped; accepted audio will finish."
            self._input_ended = True
            return
        self._queue.append(pcm)
        self._queued_bytes += len(pcm)
        self._duration += len(pcm) / (SAMPLE_RATE * 2)

    def push(self, frame):
        """WebRTC callback; never performs network I/O or model inference."""
        with self._lock:
            if self._input_ended or self._cancelled.is_set() or self._finished.is_set():
                return frame
            try:
                for pcm in self._pcm(frame):
                    self._enqueue(pcm)
                    if self._input_ended:
                        break
                if self._worker.ident is None:
                    self._worker.start()
            except Exception:
                self._error = "Could not process microphone audio. Start a new recording to retry."
                self._input_ended = True
                if self._worker.ident is None:
                    self._finished.set()
        return frame

    def start_file(self, audio):
        """Stream decoded mono 16 kHz audio at playback speed, retaining silence."""
        with self._lock:
            if self._worker.ident is not None or self._input_ended:
                raise ValueError("This translation session has already started.")
            self._file_audio = audio
            self._file_duration = len(audio) / 16000
            self._input_ended = True
            self._worker.start()

    def finish(self):
        """Stop capture and let queued audio and the final English captions drain."""
        with self._lock:
            if self._input_ended:
                return
            try:
                for pcm in self._pcm(None):
                    self._enqueue(pcm)
            except Exception:
                self._error = "Could not finish microphone audio. Start a new recording to retry."
            self._input_ended = True
            if self._worker.ident is None:
                self._finished.set()

    def close(self):
        """Cancel without waiting on network I/O; suppress all later deltas."""
        with self._lock:
            self._cancelled.set()
            self._input_ended = True
            self._queue.clear()
            self._queued_bytes = 0
            if self._worker.ident is None:
                self._finished.set()

    def snapshot(self):
        with self._lock:
            source, english = "".join(self._source).strip(), "".join(self._english).strip()
            has_text = bool(source or english)
            finished = self._finished.is_set()
            # Never label a non-English provider response as an English result.
            invalid = bool(english and contains_cjk(english))
            error = self._error or ("English translation unavailable." if invalid else None)
            return {
                "texts": [source or "[Source transcript unavailable]"] if has_text else [],
                "timings": [], "pending": 0,
                "accepting": not self._input_ended and not finished,
                "finished": finished, "error": error, "realtime": True,
                "sent_seconds": self._sent_seconds, "duration": self._file_duration or self._duration,
                "direct_translation": {
                    "translations": [english if english and not invalid else None] if has_text else [],
                    "errors": [error] if has_text else [],
                    "pending": int(not finished),
                },
            }

    async def _send_event(self, connection, event):
        await asyncio.wait_for(connection.send(json.dumps(event)), timeout=10)

    async def _send_audio(self, connection, pcm):
        await self._send_event(connection, {
            "type": "session.input_audio_buffer.append",
            "audio": base64.b64encode(pcm).decode("ascii"),
        })
        with self._lock:
            self._sent_seconds += len(pcm) / (SAMPLE_RATE * 2)

    async def _send(self, connection):
        if self._file_audio is not None:
            # Bound conversion to 100 ms at a time. Pacing avoids an unbounded
            # service backlog while receiving translated audio and text.
            for start in range(0, len(self._file_audio), 1600):
                if self._cancelled.is_set():
                    return
                frame = av.AudioFrame.from_ndarray(
                    self._file_audio[start:start + 1600].reshape(1, -1), format="fltp", layout="mono",
                )
                frame.sample_rate = 16000
                for pcm in self._pcm(frame):
                    await self._send_audio(connection, pcm)
                    await asyncio.sleep(len(pcm) / (SAMPLE_RATE * 2))
            for pcm in self._pcm(None):
                await self._send_audio(connection, pcm)
        else:
            while not self._cancelled.is_set():
                with self._lock:
                    pcm = self._queue.popleft() if self._queue else None
                    if pcm is not None:
                        self._queued_bytes -= len(pcm)
                    ended = self._input_ended
                if pcm is not None:
                    await self._send_audio(connection, pcm)
                elif ended:
                    break
                else:
                    await asyncio.sleep(0.01)
        if not self._cancelled.is_set():
            await self._send_event(connection, {"type": "session.close"})

    async def _stream(self):
        from websockets.asyncio.client import connect

        async with connect(
            TRANSLATION_URL, additional_headers={"Authorization": f"Bearer {self._api_key}"},
            open_timeout=10, close_timeout=2, max_size=1024 * 1024,
        ) as connection:
            await self._send_event(connection, {
                "type": "session.update",
                "session": {"audio": {
                    "input": {"transcription": {"model": "gpt-realtime-whisper"}},
                    "output": {"language": "en"},
                }},
            })
            deadline = monotonic() + 15
            while True:
                event = json.loads(await asyncio.wait_for(connection.recv(), max(0, deadline - monotonic())))
                if event["type"] == "error":
                    raise ValueError()
                if event["type"] == "session.updated":
                    break
            sender = asyncio.create_task(self._send(connection))
            draining_since = None
            try:
                while not self._cancelled.is_set():
                    if sender.done():
                        sender.result()
                        draining_since = draining_since or monotonic()
                        if monotonic() - draining_since > DRAIN_TIMEOUT:
                            raise TimeoutError()
                    try:
                        event = json.loads(await asyncio.wait_for(connection.recv(), timeout=0.2))
                    except TimeoutError:
                        continue
                    kind = event["type"]
                    if kind == "error":
                        raise ValueError()
                    if kind == "session.closed":
                        if not sender.done():
                            raise ValueError()
                        sender.result()
                        return
                    with self._lock:
                        if self._cancelled.is_set():
                            return
                        if kind == "session.input_transcript.delta":
                            self._source.append(event["delta"])
                        elif kind == "session.output_transcript.delta":
                            self._english.append(event["delta"])
                    # Audio output is intentionally not retained or played: this
                    # app displays English captions, avoiding microphone feedback.
            finally:
                sender.cancel()
                await asyncio.gather(sender, return_exceptions=True)

    def _run(self):
        try:
            asyncio.run(self._stream())
            with self._lock:
                if self._source and not "".join(self._english).strip() and not self._cancelled.is_set():
                    self._error = self._error or EMPTY_TRANSLATION_MESSAGE
        except Exception:
            # No provider exceptions, payloads, or credential-bearing headers
            # escape this worker, including through chained tracebacks or logs.
            with self._lock:
                if not self._cancelled.is_set():
                    self._error = self._error or FAILED_MESSAGE
        finally:
            with self._lock:
                self._input_ended = True
                self._queue.clear()
                self._queued_bytes = 0
                self._file_audio = None
                self._resampler = None
                self._api_key = ""
                self._finished.set()
