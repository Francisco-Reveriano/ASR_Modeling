"""Serialize local model work across recording and translation threads."""

from threading import Lock

# Breeze and Tencent must not encode Metal commands simultaneously: this MPS
# runtime can abort the process. Keep capture, VAD, and OpenAI outside this lock.
LOCAL_MODEL_LOCK = Lock()
