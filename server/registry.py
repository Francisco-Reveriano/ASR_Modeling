"""Process-local session ownership; one Uvicorn worker shares the model locks."""

from threading import RLock
from time import monotonic


class SessionRegistry:
    def __init__(self, *, ttl=1800, capacity=16, clock=monotonic):
        self.ttl = ttl
        self.capacity = capacity
        self._clock = clock
        self._lock = RLock()
        self._entries = {}
        self._reserved = 0
        self._closed = False

    def create(self, factory, *args, **kwargs):
        with self._lock:
            if self._closed:
                raise ValueError("The server is shutting down.")
            if len(self._entries) + self._reserved >= self.capacity:
                raise ValueError("Close an existing conversation before starting another.")
            self._reserved += 1
        try:
            session = factory(*args, **kwargs)
            with self._lock:
                if self._closed:
                    session.close()
                    raise ValueError("The server is shutting down.")
                self._entries[session.id] = {"session": session, "seen": self._clock(), "connections": 0,
                                             "audio_attached": False, "audio_started": False}
            return session
        finally:
            with self._lock:
                self._reserved -= 1

    def get(self, session_id, *, touch=True):
        with self._lock:
            entry = self._entries.get(session_id)
            if entry is None:
                raise KeyError(session_id)
            if touch:
                entry["seen"] = self._clock()
            return entry["session"]

    def attach(self, session_id, *, audio=False):
        with self._lock:
            session = self.get(session_id)
            entry = self._entries[session_id]
            if audio:
                if entry["audio_started"]:
                    raise ValueError("This recording already has an audio connection. Start a new recording.")
                entry["audio_attached"] = entry["audio_started"] = True
            entry["connections"] += 1
            return session

    def detach(self, session_id, *, audio=False):
        with self._lock:
            entry = self._entries.get(session_id)
            if entry is not None:
                entry["connections"] = max(0, entry["connections"] - 1)
                entry["seen"] = self._clock()
                if audio:
                    entry["audio_attached"] = False

    def remove(self, session_id):
        with self._lock:
            entry = self._entries.pop(session_id, None)
        if entry is not None:
            entry["session"].close()

    def expire(self):
        with self._lock:
            expired = [self._entries.pop(key) for key, entry in list(self._entries.items())
                       if not entry["connections"] and self._clock() - entry["seen"] >= self.ttl]
        for entry in expired:
            entry["session"].close()
        return len(expired)

    def close(self):
        with self._lock:
            self._closed = True
            keys = list(self._entries)
        for key in keys:
            self.remove(key)
