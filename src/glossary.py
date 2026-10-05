"""Local terminology snapshots, exact DNT checks, and conservative session learning."""

from collections import Counter, OrderedDict
import csv
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
import re
from threading import Lock
from types import MappingProxyType
import unicodedata


MAX_RETRIEVED = 40
MAX_LEARNED = 200
MAX_CANDIDATES = 1000
MAX_FILE_BYTES = 10 * 1024 * 1024
_FIELDS = ("entry_id", "term_src", "term_tgt", "aliases_src", "dnt", "domain",
           "priority", "source", "example_src", "example_tgt")
_IDENTIFIER = re.compile(
    r"(?<![A-Za-z0-9_])(?:[A-Za-z][A-Za-z0-9]{1,15}(?:[-_][A-Za-z0-9]+)+"
    r"|(?:LOT|WAFER|TOOL)[A-Za-z]?\d[A-Za-z0-9]*)(?![A-Za-z0-9_])", re.IGNORECASE,
)


def _normal(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _pattern(term):
    left = r"(?<![A-Za-z0-9_])" if term[0].isascii() and term[0].isalnum() else ""
    right = r"(?![A-Za-z0-9_])" if term[-1].isascii() and term[-1].isalnum() else ""
    return re.compile(left + re.escape(term) + right, re.IGNORECASE)


def _copy(entry):
    result = dict(entry)
    result["aliases_src"] = list(entry["aliases_src"])
    return result


def _dnt_hits(text, patterns):
    spans = [(match.start(), match.end(), match.group())
             for pattern in patterns for match in pattern.finditer(text)]
    spans.extend((match.start(), match.end(), match.group())
                 for match in _IDENTIFIER.finditer(text) if any(char.isdigit() for char in match.group()))
    spans.sort(key=lambda item: (item[0], -item[1]))
    hits, end = [], -1
    for start, stop, value in spans:
        if start >= end:
            hits.append(value)
            end = stop
    return hits


def _dnt_comparison(source_hits, output_hits):
    missing = list((Counter(source_hits) - Counter(output_hits)).elements())
    added = list((Counter(output_hits) - Counter(source_hits)).elements())
    return {"source": source_hits, "output": output_hits, "missing": missing,
            "added": added, "ok": not missing and not added}


def _entry(value):
    if not isinstance(value, dict):
        raise ValueError("Each glossary entry must be an object with term_src and term_tgt.")
    result = {key: value.get(key, "") for key in _FIELDS}
    for key in _FIELDS:
        if key not in {"aliases_src", "dnt", "priority"}:
            result[key] = str(result[key] or "").strip()
    source = result["term_src"]
    if not source:
        raise ValueError("Glossary entries need a nonempty term_src.")
    flag = result["dnt"]
    if isinstance(flag, str):
        if flag.strip().casefold() not in {"", "false", "0", "no", "true", "1", "yes"}:
            raise ValueError("Glossary dnt must be true or false.")
        flag = flag.strip().casefold() in {"true", "1", "yes"}
    elif flag not in (True, False, 0, 1, None):
        raise ValueError("Glossary dnt must be true or false.")
    result["dnt"] = bool(flag)
    if result["dnt"]:
        if result["term_tgt"] and result["term_tgt"] != source:
            raise ValueError("DNT entries must preserve term_src exactly in term_tgt.")
        result["term_tgt"] = source
    elif not result["term_tgt"]:
        raise ValueError("Non-DNT glossary entries need a nonempty term_tgt.")
    aliases = result["aliases_src"] or ()
    if isinstance(aliases, str):
        aliases = aliases.split("|")
    if not isinstance(aliases, (list, tuple)) or any(not isinstance(item, str) for item in aliases):
        raise ValueError("Glossary aliases_src must be a pipe-separated string or a string list.")
    result["aliases_src"] = tuple(dict.fromkeys(item.strip() for item in aliases if item.strip()))
    try:
        result["priority"] = int(str(result["priority"] or 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("Glossary priority must be an integer.") from exc
    result["source"] = result["source"] or "master"
    result["entry_id"] = result["entry_id"] or "term-" + sha256(
        f"{_normal(source)}\0{_normal(result['term_tgt'])}".encode()
    ).hexdigest()[:16]
    return MappingProxyType(result)


class Glossary:
    """Keep configured entries immutable and learned terms local to one session.

    Retrieval uses whole Latin terms or literal Han phrases and aliases, ordered
    by priority, then matched length. It never guesses fuzzy translations.
    observe() accepts a proposed term pair only after its correction was accepted
    by the caller. Two distinct segment IDs are required for promotion; duplicate
    observations do not count. Nothing is written back to configured files.
    """

    def __init__(self, entries=()):
        self._entries = tuple(_entry(item) for item in entries)
        self._master = {}
        entry_ids = {}
        for entry in self._entries:
            previous = entry_ids.setdefault(entry["entry_id"], entry)
            if previous != entry:
                raise ValueError("Glossary entry_id values must identify a single entry.")
            for term in (entry["term_src"], *entry["aliases_src"]):
                key, target = _normal(term), _normal(entry["term_tgt"])
                if key in self._master and self._master[key] != target:
                    raise ValueError("Conflicting glossary translations share a source term or alias.")
                self._master[key] = target
        self._indexed = tuple(
            (entry, tuple(_pattern(_normal(term)) for term in (entry["term_src"], *entry["aliases_src"])))
            for entry in self._entries
        )
        self._dnt_patterns = tuple(
            _pattern(term) for entry in self._entries if entry["dnt"]
            for term in (entry["term_src"], *entry["aliases_src"])
        )
        self._lock = Lock()
        self._learned = {}
        self._evidence = {}
        self._candidates = OrderedDict()

    @property
    def entries(self):
        """An immutable tuple of configured mappings; aliases are tuples too."""
        return self._entries

    def retrieve(self, text, limit=MAX_RETRIEVED):
        """Return at most 40 matching copied entries; scan outside the state lock."""
        limit = min(MAX_RETRIEVED, max(0, int(limit)))
        if not limit or not text.strip():
            return []
        with self._lock:
            indexed = self._indexed
            learned = tuple(self._learned.values())
        indexed = indexed + tuple(
            (entry, (_pattern(_normal(entry["term_src"])),)) for entry in learned
        )
        text = _normal(text)
        matches = []
        for position, (entry, patterns) in enumerate(indexed):
            lengths = [len(match.group()) for pattern in patterns if (match := pattern.search(text))]
            if lengths:
                matches.append((-entry["priority"], -max(lengths), position, entry))
        matches.sort(key=lambda item: item[:3])
        results, seen = [], set()
        for _, _, _, entry in matches:
            pair = (_normal(entry["term_src"]), _normal(entry["term_tgt"]))
            if pair not in seen:
                results.append(_copy(entry))
                seen.add(pair)
            if len(results) == limit:
                break
        return results

    def dnt_hits(self, text):
        """Return exact source spellings in order, including repeated tool/lot IDs."""
        with self._lock:
            patterns = self._dnt_patterns
        return _dnt_hits(text, patterns)

    def compare_dnt(self, source, output):
        with self._lock:
            patterns = self._dnt_patterns
        return _dnt_comparison(_dnt_hits(source, patterns), _dnt_hits(output, patterns))

    def replace_master(self, other):
        """Swap a preloaded master/DNT snapshot; preserve compatible learned evidence.

        File loading occurs before this call. Each request sees a complete old or
        new snapshot, never partially replaced entries. Pending proposals restart
        their evidence count; promoted terms survive unless the new master or DNT
        rules conflict. Replacing this object does not copy the other's learning.
        """
        if not isinstance(other, Glossary):
            raise TypeError("The replacement master must be a Glossary.")
        with other._lock:
            entries, indexed, master, patterns = other._entries, other._indexed, other._master, other._dnt_patterns
        with self._lock:
            retained = {}
            for key, entry in self._learned.items():
                target = _normal(entry["term_tgt"])
                protected = _dnt_comparison(
                    _dnt_hits(entry["term_src"], patterns), _dnt_hits(entry["term_tgt"], patterns),
                )["ok"]
                if master.get(key, target) == target and protected:
                    retained[key] = entry
            self._entries, self._indexed, self._master, self._dnt_patterns = entries, indexed, master, patterns
            self._learned = retained
            self._evidence = {key: self._evidence[key] for key in retained}
            self._candidates.clear()

    def observe(self, source, target, segment_id):
        """Promote a term pair after two accepted segments; return True only then.

        Reject master/alias conflicts, changes to DNT tokens, empty or oversized
        phrases, and conflicting learned translations. Keep at most 200 learned
        entries and 1,000 pending pairs; older pending evidence is discarded first.
        Promoted evidence remains available through learned_terms().
        """
        if not isinstance(source, str) or not isinstance(target, str) or segment_id is None:
            return False
        source, target, segment_id = source.strip(), target.strip(), str(segment_id).strip()
        key, translated = _normal(source), _normal(target)
        if not key or not translated or not segment_id or max(len(source), len(target)) > 160 or key == translated:
            return False
        pair = (key, translated)
        with self._lock:
            if key in self._master or not _dnt_comparison(
                _dnt_hits(source, self._dnt_patterns), _dnt_hits(target, self._dnt_patterns),
            )["ok"]:
                return False
            if key in self._learned:
                if _normal(self._learned[key]["term_tgt"]) == translated:
                    self._evidence[key].add(segment_id)
                return False
            if len(self._learned) >= MAX_LEARNED:
                return False
            if pair not in self._candidates:
                self._candidates[pair] = {"source": source, "target": target, "segments": set()}
                if len(self._candidates) > MAX_CANDIDATES:
                    self._candidates.popitem(last=False)
            candidate = self._candidates[pair]
            candidate["segments"].add(segment_id)
            if len(candidate["segments"]) < 2:
                return False
            self._learned[key] = _entry({
                "term_src": candidate["source"], "term_tgt": candidate["target"], "source": "learned",
            })
            self._evidence[key] = candidate["segments"].copy()
            for other in [item for item in self._candidates if item[0] == key]:
                del self._candidates[other]
            return True

    def learned_terms(self):
        """Return independent canonical entries and their accepted segment evidence."""
        with self._lock:
            return [dict(_copy(entry), evidence_segment_ids=sorted(self._evidence[key]))
                    for key, entry in self._learned.items()]


def compare_dnt(source, output, glossary=None):
    """Compare exact DNT spellings and counts, including unexpected output IDs."""
    glossary = glossary if glossary is not None else Glossary()
    return glossary.compare_dnt(source, output)


def _decode(data):
    try:
        return data.decode("utf-16" if data.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Glossary files must use UTF-8 or BOM-marked UTF-16 text.") from exc


def _load_entries(path, *, dnt=False):
    path = Path(path).expanduser()
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("Glossary files must be 10 MiB or smaller.")
    data, suffix = path.read_bytes(), path.suffix.lower()
    if suffix == ".json":
        try:
            rows = json.loads(_decode(data))
        except json.JSONDecodeError as exc:
            raise ValueError("The glossary JSON is invalid.") from exc
        if isinstance(rows, dict):
            rows = rows.get("entries", rows.get("terms"))
        if not isinstance(rows, list):
            raise ValueError("Glossary JSON must contain an entries array.")
    elif suffix == ".txt" and dnt:
        rows = [line.strip() for line in _decode(data).splitlines()
                if line.strip() and not line.lstrip().startswith("#")]
    elif suffix in {".csv", ".tsv", ".xlsx"}:
        if suffix == ".xlsx":
            from src.reference_tables import read_reference_tables
            grids = list(read_reference_tables(data, path.name).values())
        else:
            try:
                grids = [list(csv.reader(StringIO(_decode(data)),
                                         delimiter="\t" if suffix == ".tsv" else ",", strict=True))]
            except csv.Error as exc:
                raise ValueError("The glossary table has invalid quoted fields.") from exc
        rows = []
        for grid in grids:
            populated = [row for row in grid if any(value != "" for value in row)]
            if not populated:
                continue
            if any(value is None for row in populated for value in row):
                raise ValueError("Glossary tables cannot contain formula or error cells; paste text values.")
            header = [str(value or "").strip().casefold() for value in populated[0]]
            if "term_src" not in header:
                raise ValueError("Glossary tables need a term_src header.")
            for cells in populated[1:]:
                rows.append(dict(zip(header, cells)))
    else:
        raise ValueError("Use a CSV, TSV, JSON, or XLSX glossary (or a TXT DNT list).")
    result = []
    for row in rows:
        if dnt:
            if not isinstance(row, (str, dict)):
                raise ValueError("DNT entries must be strings or glossary objects.")
            row = {"term_src": row} if isinstance(row, str) else dict(row)
            row.update(dnt=True, term_tgt=row.get("term_src", ""))
        if not isinstance(row, dict):
            raise ValueError("Each glossary entry must be an object.")
        result.append(dict(row, source=row.get("source") or path.name))
    return result


def load_glossary(path=None, dnt_path=None):
    """Load configured local files only; unset paths produce an empty glossary.

    CSV/TSV/JSON use canonical field names and support files up to 10 MiB. XLSX
    reuses the reference-table reader's 1 MiB upload and 10 MiB expanded limits.
    A DNT file can additionally be a UTF text list with one token per line.
    Missing or malformed configured files raise errors rather than silently
    disabling terminology. This function never reads evaluation references.
    """
    entries = _load_entries(path) if path else []
    if dnt_path:
        entries.extend(_load_entries(dnt_path, dnt=True))
    return Glossary(entries)
