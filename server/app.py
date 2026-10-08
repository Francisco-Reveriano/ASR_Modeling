"""Same-origin local API. All inference is owned by background sessions."""

import asyncio
import base64
from contextlib import asynccontextmanager, suppress
from io import BytesIO
import json
import hashlib
import logging
import os
import re
from pathlib import Path
from time import monotonic

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import ValidationError
import soundfile as sf
from starlette.middleware.trustedhost import TrustedHostMiddleware

from server.models import (
    CaptureStart, Command, LEGACY_MODEL_PAIR, MODEL_PAIRS, ReferenceOptions,
    SessionSettings, SessionSnapshot, SpeechAck, TRANSLATION_TYPES,
)
from server.references import MAX_REFERENCE_BYTES, preview_reference
from server.registry import SessionRegistry
from src.speech import ASTRA_SPEECH_MODES, SPEECH_MODELS
from src.glossary import load_glossary
from src.translation import ENV_FILE
from src.uploads import decode_wav

ROOT = Path(__file__).resolve().parents[1]
MAX_WAV_BYTES = 200 * 1024 * 1024
MAX_BODY_BYTES = MAX_WAV_BYTES + MAX_REFERENCE_BYTES + 1024 * 1024
LOGGER = logging.getLogger(__name__)


class LocalBoundaryMiddleware:
    """Reject foreign browser origins and bound uploads before form parsing."""

    def __init__(self, app, *, origins):
        self.app = app
        self.origins = origins
        guide = ROOT / "docs" / "diarization-explainer.html"
        scripts = re.findall(r"<script>(.*?)</script>", guide.read_text(), flags=re.S) if guide.is_file() else []
        self.guide_hashes = " ".join(
            "'sha256-" + base64.b64encode(hashlib.sha256(script.encode()).digest()).decode() + "'"
            for script in scripts
        )

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        origin = headers.get(b"origin", b"").decode("latin1")
        if origin and origin not in self.origins:
            return await JSONResponse({"detail": "This app accepts requests from its local frontend only."}, 403)(scope, receive, send)
        try:
            length = int(headers.get(b"content-length", b"0"))
        except ValueError:
            length = -1
        if length < 0 or length > MAX_BODY_BYTES:
            return await JSONResponse({"detail": "The upload exceeds the 200 MiB audio limit."}, 413)(scope, receive, send)
        received = 0

        async def bounded_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > MAX_BODY_BYTES:
                    raise HTTPException(413, "The upload exceeds the 200 MiB audio limit.")
            return message

        async def protected_send(message):
            if message["type"] == "http.response.start":
                scripts = "'self'" + (" " + self.guide_hashes if scope["path"] == "/guide/diarization" else "")
                policy = ("default-src 'self'; script-src " + scripts + "; style-src 'self' 'unsafe-inline'; "
                          "img-src 'self' data:; media-src 'self' blob:; connect-src 'self'; "
                          "worker-src 'self'; frame-ancestors 'none'")
                message.setdefault("headers", []).extend([
                    (b"x-content-type-options", b"nosniff"),
                    (b"referrer-policy", b"no-referrer"),
                    (b"content-security-policy", policy.encode()),
                ])
                if scope["path"].startswith("/api/"):
                    message["headers"].append((b"cache-control", b"no-store"))
            await send(message)

        await self.app(scope, bounded_receive, protected_send)


async def _read_file(file, limit):
    data = bytearray()
    try:
        while chunk := await file.read(65536):
            data.extend(chunk)
            if len(data) > limit:
                raise HTTPException(413, "Audio must be 200 MiB or smaller; references must be 1 MiB or smaller.")
    finally:
        await file.close()
    return bytes(data)


def _wav(data):
    # Bound malicious or inconsistent file headers before allocating a waveform.
    try:
        with sf.SoundFile(BytesIO(data)) as file:
            if file.frames > 150_000_000 or file.frames / file.samplerate > 14400:
                raise ValueError("This WAV is too large to decode. Use a shorter recording.")
    except sf.SoundFileError as exc:
        raise ValueError("Could not read this WAV file. It may be invalid or corrupted.") from exc
    return decode_wav(data)


def _parse_json(model, value):
    try:
        return model.model_validate_json(value)
    except (ValueError, ValidationError):
        raise HTTPException(422, "Invalid settings. Check the selected options and try again.") from None


def _configuration():
    load_dotenv(ENV_FILE, override=False)
    defaults = SessionSettings(
        glossary_path=os.getenv("GLOSSARY_FILE", ""), dnt_path=os.getenv("DNT_FILE", ""),
    )
    models = {}
    for name, directory in (("breeze", "breeze-asr-26"), ("tencent", "Hy-MT2-1.8B"),
                            ("nemotron", "Nemotron-3-Diarization")):
        path = ROOT / "Models" / directory
        models[name] = {"available": path.is_dir() and (path / "config.json").is_file(),
                        "location": "This computer", "loaded": False}
    models["openai"] = {"available": bool(os.getenv("OPENAI_API_KEY", "").strip()), "location": "OpenAI API"}
    return {
        "defaults": defaults.model_dump(), "model_pairs": list(MODEL_PAIRS),
        "translation_types": list(TRANSLATION_TYPES), "speech_models": list(SPEECH_MODELS),
        "speech_modes": list(ASTRA_SPEECH_MODES), "models": models,
        "source_language": "zh-TW+en", "target_language": "en",
        "capabilities": {"microphone": True, "upload": True, "evaluation": True, "speech": True},
        "processing_disclosure": "Runs on this computer. Selected OpenAI models receive audio and/or text. Sessions stay in memory until cleared; no recording archive is kept.",
        "limits": {"audio_bytes": MAX_WAV_BYTES, "reference_bytes": MAX_REFERENCE_BYTES},
    }


def _validate_settings(settings):
    """Reject configuration failures before replacing a usable conversation."""
    load_dotenv(ENV_FILE, override=False)
    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise ValueError("Set OPENAI_API_KEY in the server's .env before starting translation.")
    if settings.model_pair == LEGACY_MODEL_PAIR and not (ROOT / "Models" / "breeze-asr-26" / "config.json").is_file():
        raise ValueError("Prepare the local Breeze model before using Breeze or evaluation.")
    if settings.model_pair != "gpt-realtime-translate":
        try:
            load_glossary(settings.glossary_path or None, settings.dnt_path or None)
        except Exception:
            raise ValueError("Could not load terminology. Check the configured glossary and do-not-translate files.") from None


def create_app(*, session_factory=None, allowed_origins=None, registry=None, frontend_dir=None,
               settings_validator=None):
    if session_factory is None:
        from server.sessions import Conversation
        session_factory = Conversation
        settings_validator = settings_validator or _validate_settings
    validate_settings = settings_validator or (lambda settings: None)
    origins = set(allowed_origins or (
        "http://localhost:8000", "http://127.0.0.1:8000", "http://localhost:5173", "http://127.0.0.1:5173",
    ))
    sessions = registry or SessionRegistry()
    frontend = Path(frontend_dir) if frontend_dir is not None else ROOT / "frontend" / "dist"

    @asynccontextmanager
    async def lifespan(app):
        async def cleanup():
            while True:
                await asyncio.sleep(5)
                await asyncio.to_thread(sessions.expire)
        task = asyncio.create_task(cleanup())
        try:
            yield
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            await asyncio.to_thread(sessions.close)

    # Swagger's default assets use a CDN. Keep the machine-readable OpenAPI
    # schema, and omit that externally hosted page from this local application.
    app = FastAPI(title="Live Translation", version="1.0.0", lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url="/api/openapi.json")
    app.state.sessions = sessions
    hosts = {"localhost", "127.0.0.1", "[::1]"}
    from urllib.parse import urlsplit
    hosts.update(urlsplit(origin).hostname for origin in origins)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=sorted(host for host in hosts if host))
    app.add_middleware(LocalBoundaryMiddleware, origins=origins)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # Do not echo submitted settings, filenames, or file contents in errors.
        return JSONResponse({"detail": "Invalid request. Check the selected options and try again."}, 422)

    @app.exception_handler(ValueError)
    async def value_error(request, exc):
        return JSONResponse({"detail": str(exc)}, 400)

    def find(session_id):
        try:
            return sessions.get(session_id)
        except KeyError:
            raise HTTPException(404, "This session is unavailable or has been cleared.") from None

    @app.get("/api/health")
    async def health():
        return {"status": "ok", "service": "Live Translation", "storage": "memory"}

    @app.get("/api/config")
    async def config():
        try:
            return await asyncio.to_thread(_configuration)
        except ValueError:
            raise HTTPException(503, "Check the server's translation model configuration.") from None

    @app.post("/api/sessions", status_code=201, response_model=SessionSnapshot)
    async def create(settings: SessionSettings):
        await asyncio.to_thread(validate_settings, settings)
        conversation = await asyncio.to_thread(sessions.create, session_factory, settings)
        return await asyncio.to_thread(conversation.snapshot)

    @app.post("/api/references/preview")
    async def reference_preview(file: UploadFile = File(...), options: str = Form("{}")):
        selected = _parse_json(ReferenceOptions, options)
        data = await _read_file(file, MAX_REFERENCE_BYTES)
        return await asyncio.to_thread(preview_reference, data, file.filename or "reference.txt", selected)

    async def file_session(file, settings, *, reference=None, reference_options="{}"):
        selected = _parse_json(SessionSettings, settings)
        evaluation = None
        if reference is not None:
            options = _parse_json(ReferenceOptions, reference_options)
            parsed = await asyncio.to_thread(preview_reference, await _read_file(reference, MAX_REFERENCE_BYTES),
                                             reference.filename or "reference.txt", options)
            if parsed["needs_source_selection"]:
                raise HTTPException(422, "Choose a transcription reference column before evaluating.")
            evaluation = {"reference": parsed["reference"], "reference_view": parsed["reference_view"],
                          "reference_kind": "source"}
            selected = selected.model_copy(update={"model_pair": LEGACY_MODEL_PAIR, "speech_enabled": False})
        selected = selected.model_copy(update={"translation_type": "Compare all translations"})
        await asyncio.to_thread(validate_settings, selected)
        data = await _read_file(file, MAX_WAV_BYTES)
        audio = await asyncio.to_thread(_wav, data)
        del data
        conversation = await asyncio.to_thread(
            sessions.create, session_factory, selected, kind="evaluation" if evaluation else "upload",
            name=Path(file.filename or "audio.wav").name, audio=audio, evaluation=evaluation,
        )
        return await asyncio.to_thread(conversation.snapshot)

    @app.post("/api/sessions/upload", status_code=201, response_model=SessionSnapshot)
    async def upload(file: UploadFile = File(...), settings: str = Form("{}")):
        return await file_session(file, settings)

    @app.post("/api/sessions/evaluate", status_code=201, response_model=SessionSnapshot)
    async def evaluate(file: UploadFile = File(...), reference: UploadFile = File(...),
                       settings: str = Form("{}"), reference_options: str = Form("{}")):
        return await file_session(file, settings, reference=reference, reference_options=reference_options)

    @app.get("/api/sessions/{session_id}", response_model=SessionSnapshot)
    async def snapshot(session_id: str):
        return await asyncio.to_thread(find(session_id).snapshot)

    @app.post("/api/sessions/{session_id}/commands", response_model=SessionSnapshot)
    async def command(session_id: str, command: Command):
        conversation = find(session_id)
        await asyncio.to_thread(conversation.command, command.model_dump(exclude_none=True))
        return await asyncio.to_thread(conversation.snapshot)

    @app.delete("/api/sessions/{session_id}", status_code=204)
    async def delete(session_id: str):
        await asyncio.to_thread(sessions.remove, session_id)
        return Response(status_code=204)

    @app.get("/api/sessions/{session_id}/exports/{format}")
    async def export(session_id: str, format: str):
        if format not in {"txt", "json", "csv", "srt", "vtt"}:
            raise HTTPException(400, "Choose TXT, JSON, CSV, SRT, or VTT.")
        text, mime, filename = await asyncio.to_thread(find(session_id).export, format)
        LOGGER.info("Session export: session=%s format=%s", session_id, format)
        return Response(text, media_type=mime, headers={"Content-Disposition": f'attachment; filename="{filename}"'})

    async def socket_session(websocket, session_id, *, audio=False):
        if websocket.headers.get("origin") not in origins:
            await websocket.close(code=1008)
            return None
        try:
            conversation = sessions.attach(session_id, audio=audio)
        except (KeyError, ValueError):
            await websocket.close(code=1008)
            return None
        if audio and conversation.snapshot().get("kind") != "microphone":
            sessions.detach(session_id, audio=True)
            await websocket.close(code=1008)
            return None
        try:
            await websocket.accept()
        except (WebSocketDisconnect, RuntimeError, OSError):
            sessions.detach(session_id, audio=audio)
            if audio:
                await asyncio.to_thread(conversation.finish)
            return None
        return conversation

    @app.websocket("/api/sessions/{session_id}/audio")
    async def audio_socket(websocket: WebSocket, session_id: str):
        conversation = await socket_session(websocket, session_id, audio=True)
        if conversation is None:
            return
        try:
            initial = await asyncio.wait_for(websocket.receive_json(), 10)
            capture = CaptureStart.model_validate(initial)
            deadline = monotonic() + 120
            while True:
                state = await asyncio.to_thread(conversation.snapshot)
                if state["kind"] != "microphone":
                    raise ValueError("This session does not accept microphone audio.")
                if state["status"] in {"failed", "cancelled", "complete"}:
                    raise ValueError("Recording is unavailable. Start a new session.")
                if state["accepting"]:
                    break
                if monotonic() >= deadline:
                    raise ValueError("The microphone pipeline is still loading. Try again when models are ready.")
                await asyncio.sleep(0.1)
            await websocket.send_json({"type": "capture.ready"})
            while True:
                message = await asyncio.wait_for(websocket.receive(), 10)
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    pcm = message["bytes"]
                    if not pcm or len(pcm) % 2 or len(pcm) > capture.sample_rate * 2:
                        raise ValueError("Invalid microphone audio frame. Recording stopped.")
                    await asyncio.to_thread(conversation.push_pcm, pcm, capture.sample_rate)
                elif message.get("text"):
                    control = json.loads(message["text"])
                    if control != {"type": "capture.finish"}:
                        raise ValueError("Invalid microphone control message.")
                    await asyncio.to_thread(conversation.finish)
                    await websocket.send_json({"type": "capture.finished"})
                    break
        except (ValueError, ValidationError, asyncio.TimeoutError, TypeError, RuntimeError) as exc:
            message = (str(exc) if isinstance(exc, ValueError) and not isinstance(exc, ValidationError)
                       else "Microphone connection failed. Check your device and start again.")
            with suppress(WebSocketDisconnect, RuntimeError):
                await websocket.send_json({"type": "error", "message": message})
        except WebSocketDisconnect:
            pass
        finally:
            await asyncio.to_thread(conversation.finish)
            sessions.detach(session_id, audio=True)
            with suppress(WebSocketDisconnect, RuntimeError):
                await websocket.close()

    @app.websocket("/api/sessions/{session_id}/events")
    async def event_socket(websocket: WebSocket, session_id: str):
        conversation = await socket_session(websocket, session_id)
        if conversation is None:
            return

        async def receive_controls():
            while True:
                value = await websocket.receive_json()
                if value == {"type": "ping"}:
                    continue
                ack = SpeechAck.model_validate(value)
                await asyncio.to_thread(conversation.acknowledge, ack.speech_session_id, ack.played)

        receiver = asyncio.create_task(receive_controls())
        previous = None
        previous_rows = {}
        speech_id, sent_chunks = None, set()
        last_heartbeat = 0
        try:
            while not receiver.done():
                try:
                    sessions.get(session_id)
                except KeyError:
                    await websocket.send_json({"type": "cleared"})
                    break
                state = await asyncio.to_thread(conversation.snapshot)
                if previous is None:
                    await asyncio.wait_for(websocket.send_json({"type": "snapshot", "snapshot": state}), 3)
                elif state != previous:
                    rows = state.get("segments", [])
                    changed = [row for row in rows if previous_rows.get(row["id"]) != row]
                    # Audit history can be much larger than the visible rows.
                    # Progress-only changes must not resend unchanged history.
                    metadata = {key: value for key, value in state.items()
                                if key != "segments" and
                                (key in {"id", "revision"} or previous.get(key) != value)}
                    await asyncio.wait_for(websocket.send_json({"type": "update", **metadata, "segments": changed}), 3)
                previous = state
                previous_rows = {row["id"]: row for row in state.get("segments", [])}
                speech = await asyncio.to_thread(conversation.speech_snapshot)
                if speech is not None:
                    if speech_id != speech["session_id"]:
                        speech_id, sent_chunks = speech["session_id"], set()
                    chunks = [chunk for chunk in speech.get("chunks", []) if chunk["id"] not in sent_chunks]
                    sent_chunks.update(chunk["id"] for chunk in chunks)
                    sent_chunks = {value for value in sent_chunks if value > speech.get("acked", 0)}
                    await asyncio.wait_for(websocket.send_json({"type": "speech", "speech": dict(speech, chunks=chunks)}), 3)
                elif speech_id is not None:
                    await websocket.send_json({"type": "speech", "speech": None})
                    speech_id, sent_chunks = None, set()
                if monotonic() - last_heartbeat >= 1:
                    await asyncio.wait_for(websocket.send_json({"type": "heartbeat"}), 3)
                    last_heartbeat = monotonic()
                await asyncio.sleep(0.1)
            if receiver.done():
                receiver.result()
        except (WebSocketDisconnect, RuntimeError, ValueError, asyncio.TimeoutError):
            pass
        finally:
            receiver.cancel()
            with suppress(asyncio.CancelledError, WebSocketDisconnect, RuntimeError, ValueError):
                await receiver
            sessions.detach(session_id)
            with suppress(WebSocketDisconnect, RuntimeError):
                await websocket.close()

    @app.get("/guide/diarization")
    async def guide():
        return FileResponse(ROOT / "docs" / "diarization-explainer.html")

    @app.get("/{path:path}")
    async def frontend_file(path: str):
        if path.startswith("api/"):
            raise HTTPException(404, "Unknown API route.")
        candidate = (frontend / path).resolve()
        if candidate.is_relative_to(frontend.resolve()) and candidate.is_file():
            return FileResponse(candidate)
        index = frontend / "index.html"
        if not index.is_file():
            return JSONResponse({"detail": "Build the frontend first: cd frontend && npm ci && npm run build"}, 503)
        return FileResponse(index, headers={"Cache-Control": "no-cache"})

    return app
