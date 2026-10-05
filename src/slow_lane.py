"""Asynchronous, session-scoped subtitle corrections with immutable sealed results.

The injected correct(request) callable owns network access. This module owns
bounded requests, lossless scheduling, validation, versions, and replayable audit events.
"""

from collections import OrderedDict
from copy import deepcopy
from dataclasses import asdict, dataclass
import math
from threading import Lock, Thread
import time
from uuid import uuid4
import weakref

from src.translation_validation import contains_cjk
from src.reasoning import CorrectionProviderError, DEFAULT_CORRECTION_MAX_OUTPUT_TOKENS
from src.translation import BACKGROUND_FILTERED_TEXT

SCHEMA_VERSION = 1
CHANGE_TYPES = frozenset({"terminology", "asr_fix", "word_order", "number_or_id", "omission", "style"})
FAILED_MESSAGE = "Correction failed. Any available English draft is unchanged."
BUDGET_MESSAGE = "The source segment exceeds the correction context budget."
_FOLLOWING_CONTEXT_N = 2
_VALIDATION_MESSAGES = {
    "schema": "The correction response is invalid or incomplete.",
    "low_confidence": "The correction response did not meet the confidence threshold.",
    "stale": "The correction response no longer matches the current subtitle version.",
    "dnt": "The correction response changed a protected identifier.",
    "language": "The correction response did not provide English for every segment.",
}
_PROVIDER_FAILURES = frozenset({
    "incomplete", "empty", "refusal", "malformed_json", "nonobject",
    "connection", "transport_timeout", "http_status", "provider_error",
})


def _failure_details(category, details=None):
    """Keep diagnostics fixed and allowlisted, never raw exception/response text."""
    if category in _PROVIDER_FAILURES:
        details = details or {}
        error = CorrectionProviderError(
            category, incomplete_reason=details.get("incomplete_reason"),
            status_code=details.get("status_code"),
        )
        if error.category == "incomplete" and error.incomplete_reason == "max_output_tokens":
            return error.diagnostics(), (
                "The correction reached its output token limit, which also includes reasoning. "
                "Any available English draft is unchanged."
            )
        return error.diagnostics(), str(error)
    if category in _VALIDATION_MESSAGES:
        return {"category": category}, _VALIDATION_MESSAGES[category]
    if category == "budget":
        return {"category": category}, BUDGET_MESSAGE
    if category == "capacity":
        return {"category": category}, "Earlier canceled correction requests are still finishing. Retry this review later."
    return {"category": "error"}, FAILED_MESSAGE


# Slots include canceled calls until the underlying provider returns. Waiting
# sessions are weak references; queued rows remain in their existing transcript.
_MAX_ACTIVE_REVIEWS = 64
_REVIEW_POOL_LOCK = Lock()
_ACTIVE_REVIEW_CALLS = 0
_REVIEW_WAITERS = OrderedDict()


def _prune_review_waiters_locked():
    for reference in list(_REVIEW_WAITERS):
        if reference() is None:
            _REVIEW_WAITERS.pop(reference, None)


def _claim_review_slot(session):
    global _ACTIVE_REVIEW_CALLS
    reference = weakref.ref(session)
    with _REVIEW_POOL_LOCK:
        _prune_review_waiters_locked()
        first = next(iter(_REVIEW_WAITERS), None)
        if _ACTIVE_REVIEW_CALLS >= _MAX_ACTIVE_REVIEWS or (first is not None and first != reference):
            _REVIEW_WAITERS.setdefault(reference, None)
            return False
        _REVIEW_WAITERS.pop(reference, None)
        _ACTIVE_REVIEW_CALLS += 1
        return True


def _forget_review_waiter(session):
    with _REVIEW_POOL_LOCK:
        _REVIEW_WAITERS.pop(weakref.ref(session), None)


def _release_review_slot():
    global _ACTIVE_REVIEW_CALLS
    with _REVIEW_POOL_LOCK:
        _ACTIVE_REVIEW_CALLS -= 1


def _wake_review_waiters():
    while True:
        with _REVIEW_POOL_LOCK:
            _prune_review_waiters_locked()
            if _ACTIVE_REVIEW_CALLS >= _MAX_ACTIVE_REVIEWS or not _REVIEW_WAITERS:
                return
            session = next(iter(_REVIEW_WAITERS))()
        if session is not None:
            session._dispatch_pending()


@dataclass(frozen=True)
class SlowLaneConfig:
    model: str = "gpt-6-astra"
    reasoning_effort: str = "medium"
    earlier_context_n: int = 10
    max_source_tokens: int = 2000
    max_output_tokens: int = DEFAULT_CORRECTION_MAX_OUTPUT_TOKENS
    confidence_threshold: float = 0.6
    source_lang: str = "zh-TW+en"
    target_lang: str = "en"
    filter_background_speech: bool = True
    enabled: bool = True

    def __post_init__(self):
        for name, maximum, minimum in (
            ("earlier_context_n", 100, 0),
            ("max_source_tokens", 100_000, 1),
            ("max_output_tokens", 16_384, 512),
        ):
            value = getattr(self, name)
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be an integer between {minimum} and {maximum}.")
        if not _finite_number(self.confidence_threshold) or not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1.")
        for name in ("model", "source_lang", "target_lang"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonempty string.")
        if self.reasoning_effort not in {"none", "minimal", "low", "medium", "high", "xhigh", "max"}:
            raise ValueError("Unsupported reasoning_effort.")
        for name in ("enabled", "filter_background_speech"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean.")


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
                if original.get("source_fallback"):
                    raise _Rejected("language")
            if contains_cjk(target):
                raise _Rejected("language")
            if not glossary.compare_dnt(original["source_text"], target)["ok"]:
                raise _Rejected("dnt")
            actions.append((kind, deepcopy(entry), target))
    if seen != set(requested):
        raise _Rejected("schema")
    return actions


class SlowLaneSession:
    """Own one append-only transcript and its speculative/authoritative subtitles.

    submit() receives cumulative source/draft lists; delayed drafts may fill None
    positions, but existing source and first drafts are immutable. Each ready row
    starts an independent daemon review and publishes as soon as it finishes.
    A process-wide slot limit bounds active and canceled-but-unfinished calls.
    Pending work stays in transcript records, without growing copied queues.

    An accepted correction or no-change, explicit pause, or close seals a result.
    Elapsed time never seals or invalidates a draft or review. Sealed subtitle text
    and versions never change. Speaker metadata may arrive later and is audited
    separately. A correction must match its dispatched request and exact base
    version. Accepted reviews always publish the reviewed version.
    Validation and application are all-or-nothing for every response.
    """

    def __init__(self, correct, *, config=SlowLaneConfig(), glossary=None, clock=time.monotonic,
                 on_update=None, session_id=None):
        if not callable(correct) or not isinstance(config, SlowLaneConfig) or not callable(clock):
            raise ValueError("A correction callable, SlowLaneConfig, and clock callable are required.")
        if on_update is not None and not callable(on_update):
            raise ValueError("on_update must be a callable or None.")
        if session_id is not None and (not isinstance(session_id, str) or not session_id.strip()):
            raise ValueError("session_id must be a nonempty string or None.")
        if glossary is None:
            from src.glossary import Glossary
            glossary = Glossary()
        self.config = config
        self.session_id = session_id or uuid4().hex
        self._correct, self._glossary, self._clock = correct, glossary, clock
        self._on_update = on_update
        self._lock = Lock()
        self._segments, self._events = [], []
        self._calls = {}
        self._epoch = 0
        self._closed = False
        self._enabled = config.enabled
        self._learned_terms = []
        self._metrics = {key: 0 for key in (
            "drafts", "requests", "completed", "corrections", "no_change",
            "schema_rejections", "low_confidence_rejections", "stale_rejections", "dnt_rejections", "language_rejections",
            "budget_rejections", "errors", "cancelled", "sealed", "screen_replacements", "learned_terms",
        )}
        self._metrics.update(last_latency_s=0.0, total_latency_s=0.0, automatic_retries=0,
                             filtered=0, failure_reasons={})
        self._event_locked("SessionStatus", status=self._status_locked())

    def submit(self, texts, drafts, *, draft_errors=None, draft_filtered=None, timings=None, allow_prefix=False):
        """Append sources/reveal fast drafts; enrich timing and speaker metadata.

        Speaker labels may update sealed records' metadata, but never their text
        or versions. Existing timestamps cannot drift. Such updates
        are recorded in SessionStatus events under metadata_updates for replay.
        allow_prefix accepts an older completed-fast snapshot without removing
        later sources already submitted by the UI or another completion callback.
        """
        if type(allow_prefix) is not bool:
            raise ValueError("allow_prefix must be a boolean.")
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Source segments must be nonempty strings.")
        if len(drafts) > len(texts) or any(value is not None and not isinstance(value, str) for value in drafts):
            raise ValueError("Drafts must be aligned strings or None.")
        if draft_filtered is not None and (
            len(draft_filtered) > len(texts) or any(type(value) is not bool for value in draft_filtered)
        ):
            raise ValueError("Filtered flags must be aligned booleans.")
        with self._lock:
            if self._closed:
                return
            now = self._clock()
            if (len(texts) < len(self._segments) and not allow_prefix) or any(
                texts[index] != self._segments[index]["source_text"]
                for index in range(min(len(texts), len(self._segments)))
            ):
                raise ValueError("Source segments are append-only; start a new session to replace them.")
            for index, record in enumerate(self._segments):
                draft = drafts[index] if index < len(drafts) else None
                if draft and record["draft_text"] is not None and not record["source_fallback"] and draft.strip() != record["draft_text"]:
                    raise ValueError("An existing fast draft cannot be replaced; start a new session.")
            for index, filtered in enumerate(draft_filtered or []):
                if not filtered or (index < len(self._segments) and self._segments[index]["sealed"]):
                    continue
                # Validate only a new filtering decision. Later glossary reloads
                # cannot invalidate a sealed result during an idempotent UI poll.
                if (
                    index >= len(drafts) or drafts[index] is None
                    or drafts[index].strip() != BACKGROUND_FILTERED_TEXT
                    or (draft_errors and index < len(draft_errors) and draft_errors[index])
                    or self._glossary.dnt_hits(texts[index])
                ):
                    raise ValueError(
                        "Filtered segments require the background marker, no translation error, "
                        "and no protected identifiers."
                    )
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
                self._segments.append({
                    "segment_id": f"{self.session_id}:{index + 1}", "index": index,
                    "source_text": texts[index], "draft_text": None, "target_text": None,
                    "screen_text": None, "screen_version": 0,
                    "version": 0, "status": "waiting", "sealed": False,
                    "seal_reason": None, "created_at": now, "draft_at": None,
                    "updated_at": now, "sealed_at": None, "replacements": 0,
                    "error": None, "review_error": None, "repair_count": 0, "review_feedback": None,
                    "filtered": False, "source_fallback": False, "screen_source_fallback": False,
                    "timing": timing,
                    "history": [],
                })
            changed = False
            for index, record in enumerate(self._segments):
                if record["sealed"] or (record["draft_text"] is not None and not record["source_fallback"]):
                    continue
                if draft_filtered and index < len(draft_filtered) and draft_filtered[index]:
                    self._filter_locked(record, now)
                    continue
                draft = drafts[index] if index < len(drafts) else None
                if draft is not None and draft.strip():
                    replaced = record["screen_text"] is not None and draft.strip() != record["screen_text"]
                    record.update(draft_text=draft.strip(), target_text=draft.strip(), version=record["version"] + 1,
                                  status="draft", draft_at=record["draft_at"] if record["draft_at"] is not None else now,
                                  updated_at=now, error=None, review_error=None, repair_count=0,
                                  review_feedback=None, source_fallback=False,
                                  replacements=record["replacements"] + int(replaced))
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
                        else:
                            # A failed fast call still leaves original source for
                            # one bounded Astra translation attempt. Later failures
                            # require explicit retry, just like draft review failures.
                            record["error"] = None
                            changed = True
            if changed and self._enabled:
                self._dispatch_ready_locked()
        _wake_review_waiters()

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
            return deepcopy({
                "schema_version": SCHEMA_VERSION, "session_id": self.session_id,
                "model": self.config.model, "config": {**asdict(self.config), "enabled": self._enabled},
                "translations": [record["screen_text"] for record in self._segments],
                "statuses": [record["status"] for record in self._segments],
                "authoritative": [record["target_text"] if record["sealed"] else None for record in self._segments],
                "segments": self._segments, "pending": self._pending_locked(),
                "active_reviews": self._active_reviews_locked(),
                "status": self._status_locked(), "metrics": self._metrics,
                "learned_terms": self._learned_terms, "events": self._events,
                "last_sequence": len(self._events),
            })

    def events_after(self, sequence):
        """Return ordered versioned events after an exclusive sequence cursor."""
        if type(sequence) is not int or sequence < 0:
            raise ValueError("The event sequence must be a nonnegative integer.")
        with self._lock:
            return deepcopy(self._events[sequence:])

    def set_enabled(self, enabled):
        if type(enabled) is not bool:
            raise ValueError("enabled must be a boolean.")
        with self._lock:
            if self._closed or enabled == self._enabled:
                return
            self._enabled = enabled
            self._epoch += 1
            if not enabled:
                for record in self._segments:
                    if record["target_text"] is not None:
                        self._seal_locked(record, "paused", self._clock())
                _forget_review_waiter(self)
            else:
                self._dispatch_ready_locked()
            self._event_locked("SessionStatus", status=self._status_locked())
        _wake_review_waiters()

    def retry_failed(self):
        """Retry eligible correction failures; never retry or change sealed records."""
        with self._lock:
            if self._closed or not self._enabled:
                return
            changed = False
            for record in self._segments:
                if not record["sealed"] and record["target_text"] is not None and record["error"]:
                    record.update(error=None, review_error=None, repair_count=0, review_feedback=None, status="draft")
                    changed = True
            if changed:
                self._dispatch_ready_locked()
        _wake_review_waiters()

    def close(self):
        """Seal current values and discard queued/late results without waiting."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._epoch += 1
            for record in self._segments:
                if record["target_text"] is None:
                    self._fallback_locked(record, self._clock())
                self._seal_locked(record, "closed", self._clock())
            _forget_review_waiter(self)
            self._event_locked("SessionStatus", status="closed")
        _wake_review_waiters()

    def _event_locked(self, kind, **payload):
        self._events.append(deepcopy({
            "schema_version": SCHEMA_VERSION, "sequence": len(self._events) + 1,
            "session_id": self.session_id, "type": kind, "at": self._clock(), **payload,
        }))

    def _count_failure_locked(self, category):
        counts = self._metrics["failure_reasons"]
        counts[category] = counts.get(category, 0) + 1

    def _fail_record_locked(self, record, details, message):
        record.update(status="failed", error=message, review_error=deepcopy(details), review_feedback=None)
        self._event_locked("ReviewFailed", segment_id=record["segment_id"],
                           base_version=record["version"], review_error=details)

    def _status_locked(self):
        if self._closed:
            return "closed"
        if not self._enabled:
            return "paused"
        if any(record["error"] and not record["sealed"] for record in self._segments):
            return "degraded"
        return "active" if self._pending_locked() else "idle"

    def _seal_locked(self, record, reason, now):
        if record["sealed"]:
            return
        statuses = {"correction": "corrected", "no_change": "confirmed", "filtered": "filtered",
                    "paused": "paused", "closed": "paused"}
        status = "failed" if record["source_fallback"] else statuses[reason]
        record.update(sealed=True, seal_reason=reason, sealed_at=now, updated_at=now, status=status)
        self._metrics["sealed"] += 1
        self._event_locked("Seal", segment=record)

    def _publish_review_locked(self, record):
        """Show every accepted review without an elapsed-time display limit."""
        replaced = record["target_text"] != record["screen_text"]
        record.update(screen_text=record["target_text"], screen_version=record["version"],
                      screen_source_fallback=record["source_fallback"],
                      replacements=record["replacements"] + int(replaced))
        self._metrics["screen_replacements"] += int(replaced)

    def _fallback_locked(self, record, now):
        record.update(draft_text=None, target_text=record["source_text"], version=1,
                      screen_text=record["source_text"], screen_version=1, screen_source_fallback=True, draft_at=now,
                      updated_at=now, source_fallback=True, status="failed",
                      error="English translation is unavailable.")
        record["history"].append({"version": 1, "target_text": record["source_text"],
                                  "kind": "Draft", "at": now, "source_fallback": True})
        self._metrics["drafts"] += 1
        self._event_locked("Draft", segment=record)

    def _filter_locked(self, record, now):
        replaced = record["screen_text"] is not None and record["screen_text"] != BACKGROUND_FILTERED_TEXT
        record.update(draft_text=BACKGROUND_FILTERED_TEXT, target_text=BACKGROUND_FILTERED_TEXT,
                      screen_text=BACKGROUND_FILTERED_TEXT, version=record["version"] + 1,
                      status="filtered",
                      filtered=True, source_fallback=False, screen_source_fallback=False,
                      error=None, review_error=None, review_feedback=None,
                      draft_at=now, updated_at=now, replacements=record["replacements"] + int(replaced))
        record["screen_version"] = record["version"]
        record["history"].append({"version": record["version"], "target_text": BACKGROUND_FILTERED_TEXT,
                                  "kind": "Filtered", "at": now})
        self._metrics["filtered"] += 1
        self._metrics["screen_replacements"] += int(replaced)
        self._event_locked("Filtered", segment=record)
        self._seal_locked(record, "filtered", now)

    @staticmethod
    def _ready(record):
        return record["target_text"] is not None and not record["sealed"] and not record["error"]

    def _pending_locked(self):
        """Count queued and active rows without copying a queue."""
        if self._closed or not self._enabled:
            return 0
        return sum(self._ready(record) for record in self._segments)

    def _active_reviews_locked(self):
        return sum(
            epoch == self._epoch and self._segments[index]["version"] == version
            and self._ready(self._segments[index])
            for index, version, epoch in self._calls
        )

    def _dispatch_pending(self):
        # Called by the global pool only after releasing its lock.
        with self._lock:
            self._dispatch_ready_locked()

    def _dispatchable_locked(self, record):
        """Allow independent dispatch; ordered review layers override this gate."""
        return True

    def _dispatch_ready_locked(self):
        if self._closed or not self._enabled:
            _forget_review_waiter(self)
            return
        for record in self._segments:
            key = (record["index"], record["version"], self._epoch)
            if not self._ready(record) or key in self._calls or not self._dispatchable_locked(record):
                continue
            if _source_cost(record["source_text"]) > self.config.max_source_tokens:
                details, message = _failure_details("budget")
                self._fail_record_locked(record, details, message)
                self._metrics["budget_rejections"] += 1
                self._count_failure_locked("budget")
                continue
            if not _claim_review_slot(self):
                return
            try:
                request = self._request_locked(record)
                thread = Thread(target=self._run_review, args=(request, key), daemon=True,
                                name="subtitle-correction-request")
                self._calls[key] = thread
                thread.start()
            except Exception:
                self._calls.pop(key, None)
                _release_review_slot()
                details, message = _failure_details("error")
                self._fail_record_locked(record, details, message)
                self._metrics["errors"] += 1
                self._count_failure_locked("error")
                continue
            self._metrics["requests"] += 1
            self._event_locked("SessionStatus", status="active")
        _forget_review_waiter(self)

    def _context_neighbors_locked(self, records):
        """Return bounded nearest neighbors without waiting for future speech."""
        before, after = [], []
        if self.config.earlier_context_n:
            for index in range(records[0]["index"] - 1, -1, -1):
                record = self._segments[index]
                if not record["filtered"]:
                    before.append(record)
                    if len(before) >= self.config.earlier_context_n:
                        break
        for index in range(records[-1]["index"] + 1, len(self._segments)):
            record = self._segments[index]
            if not record["filtered"]:
                after.append(record)
                if len(after) >= _FOLLOWING_CONTEXT_N:
                    break
        return before, after

    @staticmethod
    def _target_status(record):
        target = record["target_text"]
        if record["source_fallback"] or not target or contains_cjk(target):
            return "unavailable"
        if record["sealed"] and record["status"] in {"corrected", "confirmed"}:
            return record["status"]
        return "draft"

    def _request_locked(self, record):
        """Freeze one editable row and bounded chronological neighbor evidence."""
        records = [record]
        remaining = self.config.max_source_tokens - _source_cost(record["source_text"])

        def item(record, position=None):
            status = self._target_status(record)
            result = {"segment_id": record["segment_id"], "base_version": record["version"],
                      "source_text": record["source_text"], "target_text": record["target_text"],
                      "source_fallback": record["source_fallback"], "target_status": status,
                      "speaker_id": (record["timing"] or {}).get("speaker_id")}
            if position is not None:
                result.update(read_only=True, context_position=position)
                if status == "unavailable":
                    result["target_text"] = None
            return result

        before, after = self._context_neighbors_locked(records)
        context = []
        for distance in range(max(len(before), len(after))):
            for neighbors, position in ((before, "before"), (after, "after")):
                if distance < len(neighbors):
                    record = neighbors[distance]
                    cost = _source_cost(record["source_text"])
                    if cost <= remaining:
                        context.append((record, position))
                        remaining -= cost
        context.sort(key=lambda entry: entry[0]["index"])
        request = {"session_id": self.session_id, "config": {**asdict(self.config), "enabled": self._enabled},
                   "segments": [item(record) for record in records],
                   "context": [item(record, position) for record, position in context],
                   "glossary": [], "learned_terms": []}
        if records[0]["review_feedback"]:
            request["review_feedback"] = deepcopy(records[0]["review_feedback"])
        return request

    def _run_review(self, request, key):
        try:
            self._review(request, key)
        finally:
            try:
                if self._on_update is not None:
                    try:
                        # No session/pool lock is held. Keep the slot until the
                        # observer returns so finished worker tails stay bounded.
                        self._on_update()
                    except Exception:
                        # Observers cannot invalidate accepted subtitles.
                        pass
            finally:
                with self._lock:
                    self._calls.pop(key, None)
                _release_review_slot()
                # Wake older waiting sessions before this session claims more slots.
                _wake_review_waiters()
                self._dispatch_pending()
                _wake_review_waiters()

    def _review(self, request, key):
        index, version, epoch = key
        started = self._clock()
        failure = None
        try:
            combined = "\n".join(item["source_text"] for item in request["segments"] + request["context"])
            request["glossary"] = self._glossary.retrieve(combined, limit=40)
            request["learned_terms"] = self._glossary.learned_terms()[-40:]
            for item in request["segments"]:
                item["dnt_hits"] = self._glossary.dnt_hits(item["source_text"])
            response = self._correct(deepcopy(request))
            actions = _validated_actions(response, request, self.config.confidence_threshold, self._glossary)
            outcome = "ok"
        except CorrectionProviderError as exc:
            outcome, actions, failure = "error", [], exc.diagnostics()
        except _Rejected as rejected:
            outcome, actions = rejected.reason, []
        except Exception:
            outcome, actions = "error", []
        with self._lock:
            now = self._clock()
            if self._closed or epoch != self._epoch:
                self._metrics["cancelled"] += 1
                return
            record = self._segments[index]
            if record["sealed"] or record["version"] != version:
                self._metrics["stale_rejections"] += 1
                return
            elapsed = max(0.0, now - started)
            self._metrics["last_latency_s"] = elapsed
            self._metrics["total_latency_s"] += elapsed
            if outcome != "ok":
                category = (failure or {}).get("category", outcome)
                details, message = _failure_details(category, failure)
                repair_category = category if category in _VALIDATION_MESSAGES else None
                if (details["category"] == "incomplete"
                    and details.get("incomplete_reason") == "max_output_tokens"):
                    repair_category = "output_budget"
                metric = {"schema": "schema_rejections", "low_confidence": "low_confidence_rejections",
                          "stale": "stale_rejections", "dnt": "dnt_rejections", "language": "language_rejections",
                          "cancelled": "cancelled"}.get(outcome, "errors")
                self._metrics[metric] += 1
                self._count_failure_locked(details["category"])
                if repair_category is not None and record["repair_count"] == 0:
                    record.update(status="draft", error=None, review_error=None,
                                  repair_count=1, review_feedback={"category": repair_category})
                    self._metrics["automatic_retries"] += 1
                    self._event_locked("ReviewRetry", segment_id=record["segment_id"],
                                       base_version=record["version"], review_feedback=record["review_feedback"])
                else:
                    self._fail_record_locked(record, details, message)
            else:
                self._metrics["completed"] += 1
                observations = []
                for kind, entry, target in actions:
                    if kind == "corrections":
                        record.update(target_text=target, version=record["version"] + 1, updated_at=now,
                                      error=None, review_error=None, review_feedback=None, source_fallback=False)
                        self._publish_review_locked(record)
                        record["history"].append({**entry, "version": record["version"], "target_text": target,
                                                  "kind": "Correction", "at": now, "screen_updated": True})
                        self._metrics["corrections"] += 1
                        self._event_locked("Correction", segment=record)
                        self._seal_locked(record, "correction", now)
                        if "terminology" in entry["change_type"]:
                            for pair in entry["term_pairs"]:
                                if pair["source"].casefold() in record["source_text"].casefold() and pair["target"].casefold() in target.casefold():
                                    observations.append((pair["source"], pair["target"], record["segment_id"]))
                    else:
                        record.update(error=None, review_error=None, review_feedback=None)
                        self._publish_review_locked(record)
                        self._metrics["no_change"] += 1
                        self._seal_locked(record, "no_change", now)
                try:
                    # Serialize observation and snapshot publication so concurrent
                    # completions cannot overwrite a newer glossary snapshot.
                    promoted = sum(bool(self._glossary.observe(*pair)) for pair in observations)
                    self._learned_terms = deepcopy(self._glossary.learned_terms())
                    self._metrics["learned_terms"] += promoted
                except Exception:
                    # Optional terminology learning never invalidates accepted text.
                    pass
            self._event_locked("SessionStatus", status=self._status_locked())
