"""Transport tests exercise the real API without weights, OpenAI, or audio devices."""

from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest
from uuid import uuid4
import wave

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from server.app import create_app
from server.models import ReferenceOptions, SessionSettings
from server.references import preview_reference
from server.registry import SessionRegistry


def wav_bytes():
    output = BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(bytes(3200))
    return output.getvalue()


class FakeConversation:
    def __init__(self, settings, *, kind="microphone", name=None, audio=None, evaluation=None):
        self.id = uuid4().hex
        self.settings = settings.model_dump()
        self.evaluation = evaluation
        self.frames = []
        self.acks = []
        self.closed = False
        self.ended = Event()
        self.speech = None
        self.state = {"id": self.id, "revision": 1, "kind": kind, "name": name,
                      "status": "ready", "accepting": True, "input_finished": False,
                      "finished": False, "settings": self.settings, "segments": [],
                      "error": None, "speech": None}

    def snapshot(self):
        return deepcopy(self.state)

    def push_pcm(self, pcm, rate):
        self.frames.append((pcm, rate))

    def finish(self):
        self.state.update(accepting=False, input_finished=True, finished=True, status="complete", revision=2)
        self.ended.set()

    def cancel(self):
        self.finish()

    def close(self):
        self.closed = True
        self.finish()

    def command(self, command):
        if command["type"] == "stop":
            self.finish()

    def speech_snapshot(self):
        return deepcopy(self.speech)

    def acknowledge(self, session_id, played):
        self.acks.append((session_id, played))

    def export(self, format):
        return "Example conversation", "text/plain", "conversation.txt"


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.registry = SessionRegistry()
        self.app = create_app(session_factory=FakeConversation, registry=self.registry,
                              allowed_origins={"http://testserver"})
        self.client = TestClient(self.app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)

    def create(self, **settings):
        response = self.client.post("/api/sessions", json=settings)
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()["id"]

    def websocket(self, session_id, name):
        return self.client.websocket_connect(f"/api/sessions/{session_id}/{name}",
                                             headers={"origin": "http://testserver"})

    def test_config_has_models_and_defaults_without_credentials(self):
        response = self.client.get("/api/config")
        self.assertEqual(response.status_code, 200)
        value = response.json()
        self.assertEqual(value["defaults"]["model_pair"], "gpt-live-transcribe + gpt-6-luna")
        self.assertFalse(value["defaults"]["speech_enabled"])
        self.assertNotIn("OPENAI_API_KEY", response.text)
        self.assertIn("no-store", response.headers["cache-control"])

    def test_stop_keeps_results_delete_clears_and_is_idempotent(self):
        sid = self.create()
        session = self.registry.get(sid)
        response = self.client.post(f"/api/sessions/{sid}/commands", json={"type": "stop"})
        self.assertTrue(response.json()["finished"])
        self.assertEqual(self.client.get(f"/api/sessions/{sid}/exports/txt").status_code, 200)
        self.assertEqual(self.client.delete(f"/api/sessions/{sid}").status_code, 204)
        self.assertTrue(session.closed)
        self.assertEqual(self.client.get(f"/api/sessions/{sid}").status_code, 404)
        self.assertEqual(self.client.delete(f"/api/sessions/{sid}").status_code, 204)

    def test_invalid_settings_and_foreign_origins_rejected_without_echo(self):
        value = self.client.post("/api/sessions", json={"model_pair": "sk-secret-do-not-echo"})
        self.assertEqual(value.status_code, 422)
        self.assertNotIn("sk-secret", value.text)
        response = self.client.post("/api/sessions", json={}, headers={"origin": "https://elsewhere.invalid"})
        self.assertEqual(response.status_code, 403)
        self.assertEqual(self.client.get("/api/health", headers={"host": "elsewhere.invalid"}).status_code, 400)

    def test_upload_validates_before_allocating_and_evaluation_freezes_reference(self):
        old = self.create()
        invalid = self.client.post("/api/sessions/upload", files={"file": ("bad.wav", b"bad")})
        self.assertEqual(invalid.status_code, 400)
        self.assertFalse(self.registry.get(old).closed)
        result = self.client.post("/api/sessions/evaluate", files={
            "file": ("meeting.wav", wav_bytes(), "audio/wav"),
            "reference": ("reference.txt", "你好 Teams".encode(), "text/plain"),
        }, data={"settings": json.dumps({"speech_enabled": True, "translation_type": "Fast English"})})
        self.assertEqual(result.status_code, 201, result.text)
        session = self.registry.get(result.json()["id"])
        self.assertEqual(session.settings["model_pair"], "Breeze + OpenAI")
        self.assertFalse(session.settings["speech_enabled"])
        self.assertEqual(session.settings["translation_type"], "Compare all translations")
        self.assertEqual(session.evaluation["reference"]["text"], "你好 Teams")

    def test_reference_preview_requires_explicit_ambiguous_column(self):
        files = {"file": ("test.csv", b"a,b\nhello,world\n", "text/csv")}
        response = self.client.post("/api/references/preview", files=files)
        self.assertTrue(response.json()["needs_source_selection"])
        response = self.client.post("/api/references/preview", files=files,
                                    data={"options": '{"source_column":0,"header_row":1}'})
        self.assertEqual(response.json()["reference"]["text"], "hello")

    def test_declared_oversize_upload_rejected_before_reading(self):
        response = self.client.post("/api/sessions/upload", headers={"content-length": str(300 * 1024 * 1024)})
        self.assertEqual(response.status_code, 413)

    def test_audio_binary_frames_and_final_partial_buffer_are_preserved(self):
        sid = self.create()
        with self.websocket(sid, "audio") as ws:
            ws.send_json({"type": "capture.start", "sample_rate": 48000, "channels": 1, "format": "pcm_s16le"})
            self.assertEqual(ws.receive_json()["type"], "capture.ready")
            ws.send_bytes(bytes(9600))
            ws.send_bytes(b"\x01\x00\x02\x00")
            ws.send_json({"type": "capture.finish"})
            self.assertEqual(ws.receive_json()["type"], "capture.finished")
        session = self.registry.get(sid)
        self.assertEqual([len(pcm) for pcm, _ in session.frames], [9600, 4])
        self.assertEqual([rate for _, rate in session.frames], [48000, 48000])
        self.assertTrue(session.ended.is_set())
        with self.assertRaises(WebSocketDisconnect):
            with self.websocket(sid, "audio"):
                pass

    def test_bad_pcm_and_abrupt_disconnect_finish_input(self):
        sid = self.create()
        with self.websocket(sid, "audio") as ws:
            ws.send_json({"type": "capture.start", "sample_rate": 44100, "channels": 1, "format": "pcm_s16le"})
            ws.receive_json()
            ws.send_bytes(b"odd")
            self.assertEqual(ws.receive_json()["type"], "error")
        self.assertTrue(self.registry.get(sid).ended.is_set())
        sid = self.create()
        with self.websocket(sid, "audio") as ws:
            ws.send_json({"type": "capture.start", "sample_rate": 48000, "channels": 1, "format": "pcm_s16le"})
            ws.receive_json()
        self.assertTrue(self.registry.get(sid).ended.wait(1))

    def test_events_reconnect_snapshot_and_only_changed_rows(self):
        sid = self.create()
        session = self.registry.get(sid)
        row = {"id": sid + ":0", "source": "你好", "english": "Hello", "index": 0}
        session.state["correction"] = {"events": [{"text": "retained audit history"}]}
        with self.websocket(sid, "events") as ws:
            self.assertEqual(ws.receive_json()["type"], "snapshot")
            session.state.update(revision=2, segments=[row])
            for _ in range(10):
                update = ws.receive_json()
                if update["type"] == "update":
                    break
            self.assertEqual(update["segments"], [row])
            self.assertNotIn("correction", update)
            self.assertNotIn("settings", update)
            self.assertEqual(update["id"], sid)
            self.assertEqual(update["revision"], 2)
        with self.websocket(sid, "events") as ws:
            snapshot = ws.receive_json()["snapshot"]
            self.assertEqual(snapshot["segments"], [row])
            self.assertEqual(snapshot["correction"], session.state["correction"])
        self.assertFalse(session.closed)

    def test_speech_chunks_are_not_resent_until_reconnect(self):
        sid = self.create()
        session = self.registry.get(sid)
        session.speech = {"session_id": "speech-one", "chunks": [{"id": 1, "pcm": "AAA="}], "acked": 0}
        with self.websocket(sid, "events") as ws:
            first = None
            while first is None:
                event = ws.receive_json()
                if event["type"] == "speech": first = event["speech"]
            self.assertEqual(len(first["chunks"]), 1)
            while True:
                event = ws.receive_json()
                if event["type"] == "speech": break
            self.assertEqual(event["speech"]["chunks"], [])
            ws.send_json({"type": "speech.ack", "speech_session_id": "speech-one", "played": 1})
            while not session.acks:
                ws.receive_json()
            self.assertEqual(session.acks, [("speech-one", 1)])

    def test_foreign_websocket_origin_rejected(self):
        sid = self.create()
        with self.assertRaises(WebSocketDisconnect):
            with self.client.websocket_connect(f"/api/sessions/{sid}/events", headers={"origin": "https://other.invalid"}):
                pass

    def test_audio_socket_cannot_stop_a_file_job(self):
        response = self.client.post("/api/sessions/upload", files={"file": ("a.wav", wav_bytes())})
        sid = response.json()["id"]
        with self.assertRaises(WebSocketDisconnect):
            with self.websocket(sid, "audio"):
                pass
        self.assertFalse(self.registry.get(sid).ended.is_set())

    def test_guide_script_hash_is_scoped_to_guide(self):
        guide = self.client.get("/guide/diarization")
        self.assertEqual(guide.status_code, 200)
        self.assertIn("'sha256-", guide.headers["content-security-policy"])
        self.assertNotIn("'sha256-", self.client.get("/api/health").headers["content-security-policy"])

    def test_openapi_exposes_snapshot_and_command_schemas(self):
        response = self.client.get("/api/openapi.json")
        self.assertEqual(response.status_code, 200)
        schemas = response.json()["components"]["schemas"]
        self.assertIn("SessionSnapshot", schemas)
        self.assertIn("SpeechCommand", schemas)

    def test_frontend_deep_links_serve_local_build_only(self):
        with TemporaryDirectory() as directory:
            Path(directory, "index.html").write_text("<h1>Local app</h1>")
            app = create_app(session_factory=FakeConversation, frontend_dir=directory,
                             allowed_origins={"http://testserver"})
            with TestClient(app) as client:
                self.assertIn("Local app", client.get("/conversation").text)
                self.assertEqual(client.get("/api/nonexistent").status_code, 404)


class RegistryTests(unittest.TestCase):
    def test_expiry_respects_connections_and_shutdown_closes_sessions(self):
        now = [0]
        registry = SessionRegistry(ttl=30, capacity=2, clock=lambda: now[0])
        first = registry.create(FakeConversation, SessionSettings())
        second = registry.create(FakeConversation, SessionSettings())
        registry.attach(first.id)
        now[0] = 31
        self.assertEqual(registry.expire(), 1)
        self.assertTrue(second.closed)
        self.assertFalse(first.closed)
        registry.detach(first.id)
        now[0] = 62
        self.assertEqual(registry.expire(), 1)
        self.assertTrue(first.closed)
        registry.close()
        with self.assertRaisesRegex(ValueError, "shutting down"):
            registry.create(FakeConversation, SessionSettings())

    def test_no_header_keeps_first_reference_row(self):
        result = preview_reference(b"hello,world\nnext,row\n", "a.csv",
                                   ReferenceOptions(header_row=0, source_column=0, english_column=1))
        self.assertEqual(result["reference"]["text"], "hello\nnext")
        self.assertEqual(result["reference_view"]["text"], "world\nrow")
