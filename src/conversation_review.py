"""Ordered conversation review layered over independent immutable corrections."""

from copy import deepcopy
from threading import Lock
import time

from src.slow_lane import SlowLaneConfig, SlowLaneSession, _wake_review_waiters


_ACCEPTED = frozenset({"corrected", "confirmed"})
_SKIPPED_REASONS = frozenset({"filtered", "paused", "closed"})


def _accepted(record):
    return record["sealed"] and record["status"] in _ACCEPTED


class _SequentialReviewSession(SlowLaneSession):
    """Reuse validation, cancellation, budgets, and the global request slots."""

    def __init__(self, *args, **kwargs):
        self._first_statuses = []
        self._syncing = False
        super().__init__(*args, **kwargs)

    def _dispatchable_locked(self, record):
        # Missing first reviews and failed conversation reviews both hold the
        # frontier. Explicitly skipped rows are sealed without being reviewed.
        return not self._syncing and all(
            previous["sealed"] for previous in self._segments[:record["index"]]
        )

    def _target_status(self, record):
        status = super()._target_status(record)
        if status == "draft" and record["index"] < len(self._first_statuses):
            first_status = self._first_statuses[record["index"]]
            if first_status in _ACCEPTED:
                return first_status
        return status

    def _request_locked(self, record):
        request = super()._request_locked(record)
        request["review_stage"] = "conversation"
        return request

    def sync_first_pass(self, first):
        records = first["segments"]
        with self._lock:
            self._syncing = True
            self._first_statuses = [record["status"] for record in records]
        try:
            self.submit(
                [record["source_text"] for record in records],
                [record["target_text"] if _accepted(record) else None for record in records],
                timings=[record["timing"] for record in records],
            )
            with self._lock:
                if self._closed:
                    return
                for original, record in zip(records, self._segments):
                    if record["sealed"] or original["seal_reason"] not in _SKIPPED_REASONS:
                        continue
                    if original["seal_reason"] == "filtered":
                        # The first layer already validated this immutable filtering
                        # decision; a glossary reload cannot revoke it in this layer.
                        self._filter_locked(record, self._clock())
                    else:
                        self._seal_locked(record, original["seal_reason"], self._clock())
        finally:
            with self._lock:
                self._syncing = False
        with self._lock:
            self._dispatch_ready_locked()
        _wake_review_waiters()


class ConversationReviewSession:
    """Display the best accepted English while reviewing the conversation in order.

    First reviews remain independent and immutable. Each second review waits for
    its own accepted first review and all earlier non-skipped second reviews.
    Worker completion callbacks advance this layer without UI polling. The two
    stores share source IDs so terminology evidence counts an utterance once.
    """

    def __init__(self, correct, *, config=SlowLaneConfig(), glossary=None,
                 second_pass_enabled=True, clock=time.monotonic):
        if type(second_pass_enabled) is not bool:
            raise ValueError("second_pass_enabled must be a boolean.")
        if glossary is None:
            from src.glossary import Glossary
            glossary = Glossary()
        self.config = config
        self._lock = Lock()
        self._closed = False
        self._enabled = config.enabled
        self._second_pass_enabled = second_pass_enabled
        self._glossary = glossary
        self._events = []
        self._event_cursors = {"first": 0, "conversation": 0}
        self._first = SlowLaneSession(correct, config=config, glossary=glossary,
                                      clock=clock, on_update=self._on_update)
        self.session_id = self._first.session_id
        self._second = (
            _SequentialReviewSession(correct, config=config, glossary=glossary,
                                     clock=clock, on_update=self._on_update,
                                     session_id=self.session_id)
            if second_pass_enabled else None
        )

    def submit(self, texts, drafts, *, draft_errors=None, draft_filtered=None,
               timings=None, allow_prefix=False):
        with self._lock:
            self._first.submit(texts, drafts, draft_errors=draft_errors,
                               draft_filtered=draft_filtered, timings=timings,
                               allow_prefix=allow_prefix)
            self._sync_locked()

    def set_enabled(self, enabled):
        if type(enabled) is not bool:
            raise ValueError("enabled must be a boolean.")
        with self._lock:
            if self._closed:
                return
            self._enabled = enabled
            if self._second is not None:
                self._second.set_enabled(enabled)
            self._first.set_enabled(enabled)
            self._sync_locked()

    def retry_failed(self):
        with self._lock:
            self._first.retry_failed()
            if self._second is not None:
                self._second.retry_failed()
            self._sync_locked()

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._second is not None:
                self._second.close()
            self._first.close()
            self._sync_locked()

    def _on_update(self):
        with self._lock:
            self._sync_locked()

    @staticmethod
    def _map_second_record(record, first):
        """Map the second store's local versions onto combined version history."""
        result = deepcopy(record)
        # A skipped row can seal its untouched version-zero placeholder. It
        # still represents the unchanged first-layer version in combined audit.
        offset = first["version"] - (1 if record["version"] else 0)
        for field in ("version", "screen_version", "base_version"):
            if field in result:
                result[field] += offset
        for entry in result["history"]:
            for field in ("version", "base_version"):
                if field in entry:
                    entry[field] += offset
            entry["review_stage"] = "conversation"
        return result

    def _capture_events_locked(self, first, second):
        by_id = {record["segment_id"]: record for record in first["segments"]}
        for stage, snapshot in (("first", first), ("conversation", second)):
            if snapshot is None:
                continue
            for event in snapshot["events"][self._event_cursors[stage]:]:
                value = deepcopy(event)
                value["layer_sequence"] = value["sequence"]
                value.update(sequence=len(self._events) + 1, review_stage=stage)
                if stage == "conversation":
                    if "segment" in value:
                        original = by_id[value["segment"]["segment_id"]]
                        value["segment"] = self._map_second_record(value["segment"], original)
                    elif "base_version" in value:
                        value["base_version"] += by_id[value["segment_id"]]["version"] - 1
                self._events.append(value)
            self._event_cursors[stage] = snapshot["last_sequence"]

    def _sync_locked(self):
        first = self._first.snapshot()
        first["review_stage"] = "first"
        second = None
        if self._second is not None:
            self._second.sync_first_pass(first)
            second = self._second.snapshot()
            second["review_stage"] = "conversation"
        self._capture_events_locked(first, second)
        return first, second

    def events_after(self, sequence):
        if type(sequence) is not int or sequence < 0:
            raise ValueError("The event sequence must be a nonnegative integer.")
        with self._lock:
            self._sync_locked()
            return deepcopy(self._events[sequence:])

    def snapshot(self):
        with self._lock:
            first, second = self._sync_locked()
            combined = deepcopy(first)
            combined["review_stage"] = "combined"
            combined["config"]["second_pass_enabled"] = self._second_pass_enabled
            combined["first_pass"] = first
            combined["second_pass"] = second
            combined["events"] = deepcopy(self._events)
            combined["last_sequence"] = len(self._events)
            pending = reviewed = total = 0
            earlier_unreviewed = False
            for index, record in enumerate(combined["segments"]):
                original = first["segments"][index]
                review = second["segments"][index] if second is not None else None
                skipped = original["seal_reason"] in _SKIPPED_REASONS
                if review is not None and review["seal_reason"] in {"paused", "closed"}:
                    skipped = True
                if self._second_pass_enabled and not skipped:
                    total += 1
                if not self._second_pass_enabled or skipped:
                    stage = "disabled"
                elif _accepted(review):
                    stage = review["status"]
                    reviewed += 1
                    mapped = self._map_second_record(review, original)
                    for field in ("target_text", "screen_text", "version", "screen_version",
                                  "updated_at", "sealed_at", "seal_reason"):
                        record[field] = mapped[field]
                    record["replacements"] += mapped["replacements"]
                    record["status"] = "corrected" if "corrected" in {original["status"], stage} else "confirmed"
                    record["history"].extend(entry for entry in mapped["history"] if entry["kind"] != "Draft")
                    record["review_stage"] = "conversation"
                    combined["translations"][index] = record["screen_text"]
                    combined["authoritative"][index] = record["target_text"]
                    combined["statuses"][index] = record["status"]
                elif review["error"]:
                    stage = "failed"
                elif not _accepted(original):
                    stage = "blocked" if original["error"] else "waiting"
                elif earlier_unreviewed:
                    stage = "blocked"
                    pending += 1
                else:
                    # Only the frontier can run; a ready frontier with no active
                    # call is queued at the shared process capacity limit.
                    stage = "reviewing" if second["active_reviews"] else "waiting"
                    pending += 1
                if stage not in {"disabled", "corrected", "confirmed"}:
                    earlier_unreviewed = True
                record["first_pass_status"] = original["status"]
                record["conversation_review_status"] = stage
                record["conversation_review_error"] = review["error"] if review is not None else None
                record["conversation_review_error_details"] = deepcopy(review["review_error"]) if review is not None else None

            active = second["active_reviews"] if second is not None else 0
            if not self._second_pass_enabled:
                stage_status = "disabled"
            elif self._closed:
                stage_status = "closed"
            elif not self._enabled:
                stage_status = "paused"
            elif first["status"] == "degraded" or (second is not None and second["status"] == "degraded"):
                stage_status = "degraded"
            elif pending:
                stage_status = "active"
            elif reviewed < total:
                stage_status = "waiting"
            else:
                stage_status = "idle"
            combined["conversation_review"] = {
                "enabled": self._second_pass_enabled, "status": stage_status,
                "pending": pending, "active_reviews": active, "reviewed": reviewed, "total": total,
            }
            combined["pending"] = first["pending"] + pending
            combined["active_reviews"] = first["active_reviews"] + active
            if self._closed:
                combined["status"] = "closed"
            elif not self._enabled:
                combined["status"] = "paused"
            elif first["status"] == "degraded" or stage_status == "degraded":
                combined["status"] = "degraded"
            else:
                combined["status"] = "active" if combined["pending"] else "idle"
            if second is not None:
                for key, value in second["metrics"].items():
                    if key in {"drafts", "sealed", "filtered"}:
                        continue
                    if key == "failure_reasons":
                        for reason, count in value.items():
                            previous = combined["metrics"][key].get(reason, 0)
                            combined["metrics"][key][reason] = previous + count
                    elif key == "last_latency_s":
                        if second["metrics"]["completed"]:
                            combined["metrics"][key] = value
                    else:
                        combined["metrics"][key] += value
            combined["learned_terms"] = deepcopy(self._glossary.learned_terms())
            return combined
