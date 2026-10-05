"""Asynchronous, session-scoped subtitle corrections with immutable sealed results.

The injected correct(request) callable owns network access. This module owns
bounded scheduling, validation, versions, deadlines, and replayable audit events.
"""

from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass
import math
from threading import Event, Lock, Thread
import time
from uuid import uuid4

SCHEMA_VERSION = 1
CHANGE_TYPES = frozenset({"terminology", "asr_fix", "word_order", "number_or_id", "omission", "style"})
FAILED_MESSAGE = "Correction failed. The fast translation remains available."
TIMEOUT_MESSAGE = "Correction timed out. The fast translation remains available."


@dataclass(frozen=True)
class SlowLaneConfig:
    model: str = "gpt-6-astra"
    reasoning_effort: str = "medium"
    window_n: int = 4
    earlier_context_n: int = 10
    max_source_tokens: int = 2000
    max_window_seconds: float = 90
    queue_depth: int = 2
    revision_horizon_s: float = 20
    seal_timeout_s: float = 30
    request_timeout_s: float = 20
    max_output_tokens: int = 4096
    confidence_threshold: float = 0.6
    source_lang: str = "zh-TW+en"
    target_lang: str = "en"
    enabled: bool = True

    def __post_init__(self):
        for name, maximum, minimum in (
            ("window_n", 8, 1), ("earlier_context_n", 100, 0),
            ("max_source_tokens", 100_000, 1), ("queue_depth", 2, 1),
            ("max_output_tokens", 16_384, 512),
        ):
            value = getattr(self, name)
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be an integer between {minimum} and {maximum}.")
        for name, minimum, maximum in (("max_window_seconds", 0.001, 3600), ("revision_horizon_s", 0, 120),
                                       ("seal_timeout_s", 5, 120), ("request_timeout_s", 1, 60)):
            value = getattr(self, name)
            if not _finite_number(value) or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be between {minimum} and {maximum}.")
        if not _finite_number(self.confidence_threshold) or not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1.")
        for name in ("model", "source_lang", "target_lang"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonempty string.")
        if self.reasoning_effort not in {"none", "low", "medium", "high", "xhigh"}:
            raise ValueError("Unsupported reasoning_effort.")
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be a boolean.")


def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _source_cost(text):
    # UTF-8 bytes are a conservative token upper bound, without loading a tokenizer.
    return len(text.encode("utf-8"))


class _Rejected(Exception):
    def __init__(self, reason):
        self.reason = reason


def _validated_actions(response, request, confidence_threshold, glossary):
    """Validate the whole response without touching subtitle state."""
    if not isinstance(response, dict) or set(response) != {"corrections", "no_change"}:
        raise _Rejected("schema")
    requested = {segment["segment_id"]: segment for segment in request["segments"]}
    seen, actions = set(), []
    for kind in ("corrections", "no_change"):
        entries = response[kind]
        if not isinstance(entries, list) or len(entries) > len(requested):
            raise _Rejected("schema")
        for entry in entries:
            required = {"segment_id", "base_version"}
            if kind == "corrections":
                required |= {"target_text", "change_type", "confidence", "rationale", "term_pairs"}
            if not isinstance(entry, dict) or set(entry) != required:
                raise _Rejected("schema")
            segment_id = entry["segment_id"]
            if not isinstance(segment_id, str) or type(entry["base_version"]) is not int:
                raise _Rejected("schema")
            if segment_id in seen:
                raise _Rejected("schema")
            seen.add(segment_id)
            if segment_id not in requested or entry["base_version"] != requested[segment_id]["base_version"]:
                raise _Rejected("stale")
            original = requested[segment_id]
            if kind == "corrections":
                if not isinstance(entry["target_text"], str) or not entry["target_text"].strip():
                    raise _Rejected("schema")
                changes = entry["change_type"]
                if (not isinstance(changes, list) or not changes or
                    any(not isinstance(item, str) or item not in CHANGE_TYPES for item in changes) or
                    len(set(changes)) != len(changes)):
                    raise _Rejected("schema")
                confidence = entry["confidence"]
                if not _finite_number(confidence) or not 0 <= confidence <= 1:
                    raise _Rejected("schema")
                if confidence < confidence_threshold:
                    raise _Rejected("low_confidence")
                if not isinstance(entry["rationale"], str) or not isinstance(entry["term_pairs"], list):
                    raise _Rejected("schema")
                for pair in entry["term_pairs"]:
                    if (not isinstance(pair, dict) or set(pair) != {"source", "target"} or
                        any(not isinstance(value, str) or not value.strip() for value in pair.values())):
                        raise _Rejected("schema")
                target = entry["target_text"].strip()
            else:
                target = original["target_text"]
            if not glossary.compare_dnt(original["source_text"], target)["ok"]:
                raise _Rejected("dnt")
            actions.append((kind, deepcopy(entry), target))
    return actions


class SlowLaneSession:
    """Own one append-only transcript and its speculative/authoritative subtitles.

    submit() receives cumulative source/draft lists; delayed drafts may fill None
    positions, but existing source and first drafts are immutable. submit, snapshot,
    close, and events_after never wait on correction requests. A drain worker exits
    when idle. At most two queued windows and two outstanding daemon request calls
    exist; a callable ignoring its timeout cannot spawn unbounded replacement calls.

    Each accepted correction or no-change seals immediately. Otherwise polling
    commits after seal_timeout_s from the live endpoint or source creation. Sealed
    subtitle text and versions never change. Speaker metadata may arrive later and
    is audited separately. A correction must match its dispatched request and exact
    base version. After the display horizon or two replacements, accepted corrections
    still seal the authoritative record while the screen pointer stays unchanged.
    Validation and application are all-or-nothing for every response.
    """

    def __init__(self, correct, *, config=SlowLaneConfig(), glossary=None, clock=time.monotonic):
        if not callable(correct) or not isinstance(config, SlowLaneConfig) or not callable(clock):
            raise ValueError("A correction callable, SlowLaneConfig, and clock callable are required.")
        if glossary is None:
            from src.glossary import Glossary
            glossary = Glossary()
        self.config = config
        self.session_id = uuid4().hex
        self._correct, self._glossary, self._clock = correct, glossary, clock
        self._lock = Lock()
        self._segments, self._events = [], []
        self._queue = deque()
        self._worker = None
        self._active = None
        self._calls = []
        self._last_key = None
        self._epoch = 0
        self._closed = False
        self._enabled = config.enabled
        self._degraded = False
        self._learned_terms = []
        self._metrics = {key: 0 for key in (
            "drafts", "requests", "completed", "queue_dropped", "corrections", "no_change",
            "schema_rejections", "low_confidence_rejections", "stale_rejections", "dnt_rejections",
            "timeouts", "errors", "cancelled", "sealed", "screen_replacements", "learned_terms",
        )}
        self._metrics.update(last_latency_s=0.0, total_latency_s=0.0)
        self._event_locked("SessionStatus", status=self._status_locked())

    def submit(self, texts, drafts, *, draft_errors=None, timings=None):
        """Append sources/reveal fast drafts; enrich timing and speaker metadata.

        Speaker labels may update sealed records' metadata, but never their text,
        versions, or fixed deadline. Existing timestamps cannot drift. Such updates
        are recorded in SessionStatus events under metadata_updates for replay.
        """
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Source segments must be nonempty strings.")
        if len(drafts) > len(texts) or any(value is not None and not isinstance(value, str) for value in drafts):
            raise ValueError("Drafts must be aligned strings or None.")
        with self._lock:
            if self._closed:
                return
            now = self._clock()
            self._expire_locked(now)
            if len(texts) < len(self._segments) or any(
                texts[index] != record["source_text"] for index, record in enumerate(self._segments)
            ):
                raise ValueError("Source segments are append-only; start a new session to replace them.")
            for index, record in enumerate(self._segments):
                draft = drafts[index] if index < len(drafts) else None
                if draft and record["draft_text"] is not None and not record["source_fallback"] and draft.strip() != record["draft_text"]:
                    raise ValueError("An existing fast draft cannot be replaced; start a new session.")
            first_new = len(self._segments)
            timing_updates = [
                self._merge_timing(record["timing"], self._timing(timings[index]))
                if timings and index < len(timings) else record["timing"]
                for index, record in enumerate(self._segments)
            ]
            new_timings = [self._timing(timings[index] if timings and index < len(timings) else None)
                           for index in range(len(self._segments), len(texts))]
            metadata_updates = []
            for record, timing in zip(self._segments, timing_updates):
                if timing != record["timing"]:
                    record["timing"] = timing
                    metadata_updates.append({"segment_id": record["segment_id"], "timing": timing})
            if metadata_updates:
                self._event_locked("SessionStatus", status=self._status_locked(), metadata_updates=metadata_updates)
            for index in range(len(self._segments), len(texts)):
                timing = new_timings[index - first_new]
                endpoint_ms = (timing or {}).get("server_endpoint_ms")
                self._segments.append({
                    "segment_id": f"{self.session_id}:{index + 1}", "index": index,
                    "source_text": texts[index], "draft_text": None, "target_text": None,
                    "screen_text": None, "screen_version": 0,
                    "version": 0, "status": "waiting", "sealed": False,
                    "seal_reason": None, "created_at": now, "draft_at": None,
                    "updated_at": now, "sealed_at": None, "replacements": 0,
                    "error": None, "source_fallback": False, "screen_source_fallback": False,
                    "timing": timing, "deadline_origin_at": endpoint_ms / 1000 if endpoint_ms is not None else now,
                    "history": [],
                })
            changed = False
            for index, record in enumerate(self._segments):
                if record["sealed"] or (record["draft_text"] is not None and not record["source_fallback"]):
                    continue
                draft = drafts[index] if index < len(drafts) else None
                if draft is not None and draft.strip():
                    show = record["screen_text"] is None or (
                        now - self._origin(record) < self.config.revision_horizon_s and record["replacements"] < 2
                    )
                    replaced = show and record["screen_text"] is not None and draft.strip() != record["screen_text"]
                    record.update(draft_text=draft.strip(), target_text=draft.strip(), version=record["version"] + 1,
                                  status="draft", draft_at=record["draft_at"] if record["draft_at"] is not None else now,
                                  updated_at=now, error=None, source_fallback=False,
                                  replacements=record["replacements"] + int(replaced))
                    if show:
                        record.update(screen_text=record["target_text"], screen_version=record["version"], screen_source_fallback=False)
                    record["history"].append({"version": record["version"], "target_text": draft.strip(), "kind": "Draft", "at": now})
                    self._metrics["drafts"] += 1
                    self._metrics["screen_replacements"] += int(replaced)
                    self._event_locked("Draft", segment=record)
                    changed = True
                    if not self._enabled:
                        self._seal_locked(record, "paused", now)
                elif (draft is not None) or (draft_errors and index < len(draft_errors) and draft_errors[index]):
                    if record["target_text"] is None:
                        self._fallback_locked(record, now)
                        if not self._enabled:
                            self._seal_locked(record, "paused", now)
            self._expire_locked(now)
            if changed and self._enabled:
                self._enqueue_locked(now)

    @staticmethod
    def _timing(value):
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValueError("Segment timing must be a mapping or None.")
        result = {key: deepcopy(value[key]) for key in (
            "start", "end", "start_s", "end_s", "speaker_id", "server_endpoint_ms",
            "asr_start_ms", "asr_final_ms", "t_capture_ms",
        ) if key in value}
        for key, number in result.items():
            if key != "speaker_id" and number is not None and (not _finite_number(number) or number < 0):
                raise ValueError("Segment timestamps must be finite, nonnegative numbers.")
        start = result.get("start") if result.get("start") is not None else result.get("start_s")
        end = result.get("end") if result.get("end") is not None else result.get("end_s")
        for name, alias in (("start", "start_s"), ("end", "end_s")):
            if result.get(name) is not None and result.get(alias) is not None and result[name] != result[alias]:
                raise ValueError("Segment timestamp aliases must agree.")
        if start is not None and end is not None and end < start:
            raise ValueError("Segment end cannot precede its start.")
        if result.get("speaker_id") is not None and not isinstance(result["speaker_id"], (str, int)):
            raise ValueError("speaker_id must be a string or integer.")
        return result

    @staticmethod
    def _merge_timing(previous, incoming):
        """Enrich metadata, permitting speaker relabeling but never timestamp drift."""
        if not incoming:
            return previous
        merged = deepcopy(previous or {})
        for name, alias in (("start", "start_s"), ("end", "end_s")):
            old = merged.get(name) if merged.get(name) is not None else merged.get(alias)
            new = incoming.get(name) if incoming.get(name) is not None else incoming.get(alias)
            if old is not None and new is not None and old != new:
                raise ValueError("Existing segment start/end timestamps cannot change.")
        for key, value in incoming.items():
            if key != "speaker_id" and value is not None and merged.get(key) is not None and merged[key] != value:
                raise ValueError("Existing segment timestamps cannot change.")
            if value is not None or key not in merged:
                merged[key] = value
        return merged

    def snapshot(self):
        """Return independent aligned results, immutable IDs/versions, and audit state."""
        with self._lock:
            self._expire_locked(self._clock())
            return deepcopy({
                "schema_version": SCHEMA_VERSION, "session_id": self.session_id,
                "model": self.config.model, "config": {**asdict(self.config), "enabled": self._enabled},
                "translations": [record["screen_text"] for record in self._segments],
                "statuses": [record["status"] for record in self._segments],
                "authoritative": [record["target_text"] if record["sealed"] else None for record in self._segments],
                "segments": self._segments, "pending": len(self._queue) + int(self._active is not None),
                "status": self._status_locked(), "metrics": self._metrics,
                "learned_terms": self._learned_terms, "events": self._events,
                "last_sequence": len(self._events),
            })

    def events_after(self, sequence):
        """Return ordered versioned events after an exclusive sequence cursor."""
        if type(sequence) is not int or sequence < 0:
            raise ValueError("The event sequence must be a nonnegative integer.")
        with self._lock:
            self._expire_locked(self._clock())
            return deepcopy(self._events[sequence:])

    def set_enabled(self, enabled):
        if type(enabled) is not bool:
            raise ValueError("enabled must be a boolean.")
        with self._lock:
            if self._closed or enabled == self._enabled:
                return
            self._enabled = enabled
            self._epoch += 1
            self._queue.clear()
            self._active = None
            if not enabled:
                for record in self._segments:
                    if record["target_text"] is not None:
                        self._seal_locked(record, "paused", self._clock())
            else:
                self._last_key = None
                self._enqueue_locked(self._clock())
            self._event_locked("SessionStatus", status=self._status_locked())

    def retry_failed(self):
        """Retry eligible correction failures; never retry or change sealed records."""
        with self._lock:
            if self._closed or not self._enabled:
                return
            now = self._clock()
            self._expire_locked(now)
            changed = False
            for record in self._segments:
                if not record["sealed"] and record["target_text"] is not None and record["error"]:
                    if now - self._origin(record) < self.config.seal_timeout_s:
                        record.update(error=None, status="draft")
                        changed = True
            if changed:
                self._degraded = False
                self._last_key = None
                self._enqueue_locked(now)

    def close(self):
        """Seal current values and discard queued/late results without waiting."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._epoch += 1
            self._queue.clear()
            self._active = None
            for record in self._segments:
                if record["target_text"] is None:
                    self._fallback_locked(record, self._clock())
                self._seal_locked(record, "closed", self._clock())
            self._event_locked("SessionStatus", status="closed")

    def _event_locked(self, kind, **payload):
        self._events.append(deepcopy({
            "schema_version": SCHEMA_VERSION, "sequence": len(self._events) + 1,
            "session_id": self.session_id, "type": kind, "at": self._clock(), **payload,
        }))

    def _status_locked(self):
        if self._closed:
            return "closed"
        if not self._enabled:
            return "paused"
        if self._degraded:
            return "degraded"
        return "active" if self._active is not None or self._queue else "idle"

    def _seal_locked(self, record, reason, now):
        if record["sealed"]:
            return
        statuses = {"correction": "corrected", "no_change": "confirmed", "timeout": "timeout",
                    "paused": "paused", "closed": "paused"}
        status = "failed" if record["source_fallback"] else statuses[reason]
        record.update(sealed=True, seal_reason=reason, sealed_at=now, updated_at=now, status=status)
        self._metrics["sealed"] += 1
        self._event_locked("Seal", segment=record)

    def _expire_locked(self, now):
        for record in self._segments:
            if not record["sealed"] and now - self._origin(record) >= self.config.seal_timeout_s:
                if record["target_text"] is None:
                    self._fallback_locked(record, now)
                self._seal_locked(record, "timeout", now)

    @staticmethod
    def _origin(record):
        return record["deadline_origin_at"]

    def _fallback_locked(self, record, now):
        record.update(draft_text=record["source_text"], target_text=record["source_text"], version=1,
                      screen_text=record["source_text"], screen_version=1, screen_source_fallback=True, draft_at=now,
                      updated_at=now, source_fallback=True, status="failed",
                      error="Fast translation is unavailable; showing the source text.")
        record["history"].append({"version": 1, "target_text": record["source_text"],
                                  "kind": "Draft", "at": now, "source_fallback": True})
        self._metrics["drafts"] += 1
        self._event_locked("Draft", segment=record)

    def _window_locked(self, now):
        selected, budget = [], self.config.max_source_tokens
        recent = self._segments[-self.config.window_n:]
        if not recent:
            return []
        newest = recent[-1]
        last_timing = newest["timing"] or {}
        end = last_timing.get("end") if last_timing.get("end") is not None else last_timing.get("end_s")
        for record in reversed(recent):
            timing = record["timing"] or {}
            start = timing.get("start") if timing.get("start") is not None else timing.get("start_s")
            span = end - start if end is not None and start is not None else newest["created_at"] - record["created_at"]
            cost = _source_cost(record["source_text"])
            if span > self.config.max_window_seconds or cost > budget:
                continue
            budget -= cost
            if (record["target_text"] is not None and not record["sealed"] and not record["error"] and
                now - self._origin(record) < self.config.seal_timeout_s):
                selected.append(record)
        return list(reversed(selected))

    def _enqueue_locked(self, now):
        records = self._window_locked(now)
        key = tuple((record["segment_id"], record["version"]) for record in records)
        if not key or key == self._last_key:
            return
        self._last_key = key
        if len(self._queue) >= self.config.queue_depth:
            self._queue.popleft()
            self._metrics["queue_dropped"] += 1
        self._queue.append(key)
        if self._worker is None:
            self._worker = Thread(target=self._run, daemon=True, name="subtitle-corrections")
            self._worker.start()

    def _request_locked(self, key, now):
        allowed = dict(key)
        records = [record for record in self._window_locked(now)
                   if allowed.get(record["segment_id"]) == record["version"]]
        if not records:
            return None
        def item(record):
            return {"segment_id": record["segment_id"], "base_version": record["version"],
                    "source_text": record["source_text"], "target_text": record["target_text"],
                    "speaker_id": (record["timing"] or {}).get("speaker_id")}
        remaining = self.config.max_source_tokens - sum(_source_cost(record["source_text"]) for record in records)
        context = []
        first = records[0]["index"]
        for record in reversed(self._segments[max(0, first - self.config.earlier_context_n):first]):
            cost = _source_cost(record["source_text"])
            if cost <= remaining:
                context.append({**item(record), "read_only": True})
                remaining -= cost
        return {"session_id": self.session_id, "config": {**asdict(self.config), "enabled": self._enabled},
                "segments": [item(record) for record in records], "context": list(reversed(context)),
                "glossary": [], "learned_terms": []}

    def _call(self, request, epoch):
        """Bound waiting and abandoned calls even if the injected callable hangs."""
        self._calls = [call for call in self._calls if not call[0].is_set()]
        if len(self._calls) >= 2:
            return "error", None
        done, result = Event(), {}
        def invoke():
            try:
                result["value"] = self._correct(deepcopy(request))
            except Exception:
                result["failed"] = True
            finally:
                done.set()
        thread = Thread(target=invoke, daemon=True, name="subtitle-correction-request")
        self._calls.append((done, thread))
        started = self._clock()
        thread.start()
        while not done.wait(0.02):
            with self._lock:
                if self._closed or epoch != self._epoch:
                    return "cancelled", None
            if self._clock() - started >= self.config.request_timeout_s:
                return "timeout", None
        if self._clock() - started >= self.config.request_timeout_s:
            return "timeout", None
        return ("error", None) if result.get("failed") else ("ok", result.get("value"))

    def _run(self):
        while True:
            with self._lock:
                now = self._clock()
                self._expire_locked(now)
                if self._closed or not self._enabled or not self._queue:
                    self._worker = None
                    return
                request = self._request_locked(self._queue.popleft(), now)
                if request is None:
                    continue
                epoch = self._epoch
                self._active = request
                self._metrics["requests"] += 1
                self._event_locked("SessionStatus", status="active")
            started = self._clock()
            try:
                combined = "\n".join(item["source_text"] for item in request["segments"] + request["context"])
                request["glossary"] = self._glossary.retrieve(combined, limit=40)
                request["learned_terms"] = self._glossary.learned_terms()[-40:]
                for item in request["segments"]:
                    item["dnt_hits"] = self._glossary.dnt_hits(item["source_text"])
                outcome, response = self._call(request, epoch)
                actions = _validated_actions(response, request, self.config.confidence_threshold, self._glossary) if outcome == "ok" else []
            except _Rejected as rejected:
                outcome, actions = rejected.reason, []
            except Exception:
                outcome, actions = "error", []
            observations = []
            with self._lock:
                now = self._clock()
                self._expire_locked(now)
                if self._closed or epoch != self._epoch:
                    self._metrics["cancelled"] += 1
                    continue
                self._active = None
                elapsed = max(0.0, now - started)
                self._metrics["last_latency_s"] = elapsed
                self._metrics["total_latency_s"] += elapsed
                current = {record["segment_id"]: record for record in self._segments if not record["sealed"]}
                if outcome == "ok" and any(
                    entry["segment_id"] not in current or current[entry["segment_id"]]["version"] != entry["base_version"]
                    for _, entry, _ in actions
                ):
                    outcome = "stale"
                if outcome != "ok":
                    metric = {"schema": "schema_rejections", "low_confidence": "low_confidence_rejections",
                              "stale": "stale_rejections", "dnt": "dnt_rejections", "timeout": "timeouts",
                              "cancelled": "cancelled"}.get(outcome, "errors")
                    self._metrics[metric] += 1
                    self._degraded = True
                    requested_ids = {item["segment_id"] for item in request["segments"]}
                    for record in self._segments:
                        if record["segment_id"] in requested_ids and not record["sealed"]:
                            record.update(status="timeout" if outcome == "timeout" else "failed",
                                          error=TIMEOUT_MESSAGE if outcome == "timeout" else FAILED_MESSAGE)
                else:
                    self._metrics["completed"] += 1
                    self._degraded = False
                    for kind, entry, target in actions:
                        record = current[entry["segment_id"]]
                        if kind == "corrections":
                            show = (now - self._origin(record) < self.config.revision_horizon_s and record["replacements"] < 2)
                            replaced = show and target != record["screen_text"]
                            record.update(target_text=target, version=record["version"] + 1, updated_at=now,
                                          replacements=record["replacements"] + int(replaced), error=None, source_fallback=False)
                            if show:
                                record.update(screen_text=target, screen_version=record["version"], screen_source_fallback=False)
                            record["history"].append({**entry, "version": record["version"], "target_text": target,
                                                      "kind": "Correction", "at": now, "screen_updated": show})
                            self._metrics["corrections"] += 1
                            self._metrics["screen_replacements"] += int(replaced)
                            self._event_locked("Correction", segment=record)
                            self._seal_locked(record, "correction", now)
                            if "terminology" in entry["change_type"]:
                                for pair in entry["term_pairs"]:
                                    if pair["source"].casefold() in record["source_text"].casefold() and pair["target"].casefold() in target.casefold():
                                        observations.append((pair["source"], pair["target"], record["segment_id"]))
                        else:
                            self._metrics["no_change"] += 1
                            self._seal_locked(record, "no_change", now)
                self._event_locked("SessionStatus", status=self._status_locked())
            try:
                promoted = sum(bool(self._glossary.observe(*pair)) for pair in observations)
                learned = self._glossary.learned_terms()
                with self._lock:
                    if not self._closed and epoch == self._epoch:
                        self._metrics["learned_terms"] += promoted
                        self._learned_terms = deepcopy(learned)
            except Exception:
                # Glossary promotion is optional; accepted subtitle versions remain valid.
                pass
