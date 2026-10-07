"""Persistent browser PCM player; credentials and API calls remain server-side."""

from pathlib import Path

from streamlit.components.v1 import declare_component

_player = declare_component("english_speech_player", path=Path(__file__).parent / "speech_player")


def render_speech_player(session=None, *, armed=False):
    # Mount before Start/Transcribe so that gesture can resume Web Audio before
    # Streamlit's server round-trip. Reuse this iframe for the resulting session.
    state = dict(session.snapshot(), armed=True) if session is not None else {
        "session_id": "standby", "armed": armed, "acked": 0, "chunks": [],
        "sample_rate": 24000, "closed": False, "complete": False, "pending": 0,
    }
    result = _player(state=state, key="english_speech_player", default=None)
    if session is not None and isinstance(result, dict) and result.get("session_id") == session.session_id:
        session.acknowledge(result["session_id"], result.get("played"))
        if result.get("stopped"):
            session.close()
