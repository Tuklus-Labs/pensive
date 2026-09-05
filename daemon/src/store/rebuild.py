"""Rebuild a fresh canonical store from a JSONL export -- the decades guarantee.

Given the JSONL snapshot :func:`store.export.exportJSONL` writes, ``rebuild``
reconstructs the canonical tables and retained history. Format 2 and legacy
four-file dumps remain readable; format 3 also restores task and feedback
history. The ``embeddings`` and ``fts`` tables are absent from the export on
purpose -- the fts index regenerates because atoms are inserted through the
normal triggers, and embeddings are re-derived later by the Phase 2 embedder.

Three rules make it safe to lean on for decades:

  * The export must be COMPLETE. ``exportJSONL`` always writes all six files (an
    empty table yields an empty file), so a missing file is a truncated or
    damaged export, never a legitimately empty table. ``rebuild`` refuses it up
    front. Format-2 file digests are checked through the same descriptors used to
    parse rows, so concurrent per-file replacement cannot mix generations.
  * Rows are inserted in FK order so every reference resolves at insert time.
    ``PRAGMA foreign_keys`` is on, so malformed data fails before a trailing
    digest check can hide the specific SQLite error.
  * ``rebuild`` constructs and validates a private database, then publishes its
    closed, checkpointed main file with an atomic no-overwrite hard link. Failed
    builds never create or clean up through the requested public path.
"""
import hashlib
import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from store.store import openStore, CURRENT_SCHEMA_VERSION
from store.export import (
    _TABLES,
    _TABLES_V2,
    FORMAT_VERSION,
    LEGACY_FORMAT_VERSION,
    MANIFEST_NAME,
    PUBLICATION_MARKER,
    _fsyncDirectory,
)

__all__ = ["rebuild"]

# (filename, table, columns) in FK-safe insertion order: when each table loads,
# every foreign key it carries already points at a loaded row.
_LOAD_ORDER_V2 = tuple((name, table, cols) for name, table, cols, _ in _TABLES_V2)
_LOAD_ORDER = tuple((name, table, cols) for name, table, cols, _ in _TABLES)
_V4_TABLES = frozenset(row[1] for row in _TABLES[6:])


def _loadOrder(directory):
    if (directory / PUBLICATION_MARKER).exists():
        raise ValueError(f"incomplete export publication at {directory}")
    path = directory / MANIFEST_NAME
    if not path.exists():
        if any((directory / name).exists() for name, _, _ in _LOAD_ORDER_V2[4:]):
            raise ValueError("incomplete export: history files require a format manifest")
        # An older four-file dump may live beside empty files left by a failed
        # format-3 attempt. Empty files carry no rows and are harmless; any
        # nonempty v4 file means durable history would be silently discarded.
        v4Files = [directory / name for name, _, _ in _LOAD_ORDER[6:]
                   if (directory / name).exists()]
        if any(path.stat().st_size for path in v4Files):
            raise ValueError("incomplete export: history files require a format manifest")
        return _LOAD_ORDER_V2[:4], None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("invalid export format manifest")
    version = manifest.get("formatVersion")
    if type(version) is not int or version not in (LEGACY_FORMAT_VERSION, FORMAT_VERSION):
        raise ValueError(f"unsupported export format version: {version!r}")
    if manifest.get("format") != "pensive-jsonl":
        raise ValueError("invalid export format name")
    schema = manifest.get("schemaVersion")
    maxSchema = 3 if version == LEGACY_FORMAT_VERSION else CURRENT_SCHEMA_VERSION
    if type(schema) is not int or not 1 <= schema <= maxSchema:
        raise ValueError(f"unsupported export schema version: {schema!r}")
    loadOrder = _LOAD_ORDER_V2 if version == LEGACY_FORMAT_VERSION else _LOAD_ORDER
    if manifest.get("files") != [row[0] for row in loadOrder]:
        raise ValueError("export format manifest has an unexpected table set")
    hashes = manifest.get("sha256")
    expectedNames = [row[0] for row in loadOrder]
    if not isinstance(hashes, dict) or set(hashes) != set(expectedNames):
        raise ValueError("export format manifest has an invalid digest set")
    for filename in expectedNames:
        digest = hashes[filename]
        if (type(digest) is not str or len(digest) != 64
                or any(ch not in "0123456789abcdef" for ch in digest)):
            raise ValueError(
                f"export format manifest has an invalid digest for {filename}")
    return loadOrder, hashes


def _readRows(path, expectedDigest=None):
    """Yield each JSONL line of ``path`` as a dict.

    The caller guarantees the file exists. Format 2 hashes the exact bytes through
    this descriptor and checks them at EOF. An empty file yields no rows.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for rawLine in fh:
            digest.update(rawLine)
            line = rawLine.strip()
            if not line:
                continue
            yield json.loads(line.decode("utf-8"))
    if expectedDigest is not None:
        actual = digest.hexdigest()
        if actual != expectedDigest:
            raise ValueError(
                f"export digest mismatch for {path.name}: "
                f"expected {expectedDigest}, got {actual}")


def _validateExportText(value, field, maximum=None, *, optional=False, allowEmpty=False):
    if optional and value is None:
        return
    if not isinstance(value, str):
        raise ValueError(f"format 3 {field} must be text")
    if not allowEmpty and not value.strip():
        raise ValueError(f"format 3 {field} must be nonblank")
    if maximum is not None and len(value) > maximum:
        raise ValueError(f"format 3 {field} exceeds maximum length {maximum}")


def _validateExportInteger(value, field, *, minimum=0, optional=False):
    if optional and value is None:
        return
    if type(value) is not int or value < minimum:
        raise ValueError(
            f"format 3 {field} must be a nonnegative integer"
            if minimum == 0
            else f"format 3 {field} must be an integer >= {minimum}"
        )


def _validateFormat3Row(table, row):
    if not isinstance(row, dict):
        raise ValueError(f"format 3 {table} row must be an object")
    if table == "task_checkpoints":
        _validateExportText(row.get("id"), "task checkpoint id", 256)
        _validateExportText(row.get("project"), "checkpoint project", 256)
        _validateExportText(row.get("agent"), "checkpoint agent", 64)
        _validateExportText(row.get("task_id"), "checkpoint task_id", 256)
        _validateExportInteger(row.get("revision"), "checkpoint revision", minimum=1)
        _validateExportText(row.get("request_id"), "checkpoint request_id", 256)
        if row.get("state") not in ("active", "blocked", "completed", "abandoned"):
            raise ValueError(f"format 3 checkpoint state is invalid: {row.get('state')!r}")
        _validateExportText(row.get("body"), "checkpoint body", 32000, allowEmpty=True)
        _validateExportText(row.get("source"), "checkpoint source")
        _validateExportText(row.get("writer_session"), "checkpoint writer_session", optional=True, allowEmpty=True)
        _validateExportText(row.get("source_ref"), "checkpoint source_ref", optional=True, allowEmpty=True)
        _validateExportInteger(row.get("recorded_at"), "checkpoint recorded_at")
        return
    if table == "recall_receipts":
        _validateExportText(row.get("id"), "receipt id", 256)
        _validateExportText(row.get("query"), "receipt query", 8192)
        _validateExportText(row.get("project"), "receipt project", 256, optional=True)
        _validateExportText(row.get("agent"), "receipt agent", 64)
        _validateExportText(row.get("task_id"), "receipt task_id", 256)
        _validateExportText(row.get("source_ref"), "receipt source_ref", 256)
        _validateExportInteger(row.get("recorded_at"), "receipt recorded_at")
        return
    if table == "recall_exposures":
        _validateExportText(row.get("receipt_id"), "exposure receipt_id", 256)
        _validateExportText(row.get("atom_id"), "exposure atom_id", 256)
        _validateExportInteger(row.get("rank"), "exposure rank", minimum=1)
        score = row.get("score")
        if score is not None and (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
        ):
            raise ValueError("format 3 exposure score must be finite or null")
        if row.get("delivery") not in ("body", "handle"):
            raise ValueError(f"format 3 exposure delivery is invalid: {row.get('delivery')!r}")
        return
    if table == "recall_feedback":
        for field in ("event_id", "receipt_id", "atom_id", "source", "task_id"):
            _validateExportText(row.get(field), f"feedback {field}", 256)
        _validateExportText(row.get("agent"), "feedback agent", 64)
        if row.get("feedback_type") not in ("shown", "used", "helpful", "irrelevant", "outdated"):
            raise ValueError(f"format 3 feedback_type is invalid: {row.get('feedback_type')!r}")
        _validateExportText(row.get("session_id"), "feedback session_id", 256, optional=True)
        _validateExportText(row.get("source_ref"), "feedback source_ref", 2048, optional=True)
        _validateExportText(row.get("note"), "feedback note", 2048, optional=True)
        if row.get("feedback_type") != "shown" and not (row.get("note") or "").strip():
            raise ValueError("format 3 feedback note is required for non-shown feedback")
        _validateExportInteger(row.get("recorded_at"), "feedback recorded_at")
        _validateExportInteger(row.get("processed_at"), "feedback processed_at", optional=True)
        return
    if table == "memory_credits":
        _validateExportText(row.get("atom_id"), "credit atom_id", 256)
        _validateExportText(row.get("agent"), "credit agent", 64)
        _validateExportText(row.get("task_id"), "credit task_id", 256)
        _validateExportText(row.get("feedback_id"), "credit feedback_id", 256)
        _validateExportInteger(row.get("awarded_at"), "credit awarded_at")


def _checkedRows(table, columns, rows):
    for row in rows:
        _validateFormat3Row(table, row)
        if set(row) != set(columns):
            raise ValueError(f"format 3 {table} row has an unexpected column set")
        yield row


def _insertRows(conn, table, cols, rows):
    sql = (
        f"INSERT INTO {table}({', '.join(cols)}) "
        f"VALUES ({', '.join('?' for _ in cols)})"
    )
    for obj in rows:
        conn.execute(sql, tuple(obj[c] for c in cols))


def _validateFormat3Relations(conn, loadOrder):
    """Reject cross-table scope violations before publishing a v4 restore."""
    if loadOrder != _LOAD_ORDER:
        return
    feedbackMismatch = conn.execute(
        "SELECT f.event_id FROM recall_feedback AS f "
        "LEFT JOIN recall_receipts AS r ON r.id = f.receipt_id "
        "WHERE r.id IS NULL OR f.agent != r.agent OR f.task_id != r.task_id "
        "LIMIT 1"
    ).fetchone()
    if feedbackMismatch is not None:
        raise ValueError(
            "format 3 feedback scope does not match its receipt: "
            f"event_id={feedbackMismatch[0]!r}"
        )
    invalidNote = conn.execute(
        "SELECT event_id FROM recall_feedback "
        "WHERE feedback_type != 'shown' AND (note IS NULL OR trim(note) = '') "
        "LIMIT 1"
    ).fetchone()
    if invalidNote is not None:
        raise ValueError(
            "format 3 feedback note is required for non-shown feedback: "
            f"event_id={invalidNote[0]!r}"
        )
    oversizedReceipt = conn.execute(
        "SELECT receipt_id, COUNT(*) FROM recall_exposures "
        "GROUP BY receipt_id HAVING COUNT(*) > 32 LIMIT 1"
    ).fetchone()
    if oversizedReceipt is not None:
        raise ValueError(
            "format 3 receipt has too many exposures: "
            f"receipt_id={oversizedReceipt[0]!r} count={oversizedReceipt[1]}"
        )
    creditMismatch = conn.execute(
        "SELECT c.atom_id, c.agent, c.task_id, c.feedback_id "
        "FROM memory_credits AS c "
        "LEFT JOIN recall_feedback AS f ON f.event_id = c.feedback_id "
        "WHERE c.feedback_id IS NULL OR ("
        "f.event_id IS NULL OR f.feedback_type != 'helpful' OR f.processed_at IS NULL "
        "OR f.atom_id != c.atom_id OR f.agent != c.agent OR f.task_id != c.task_id) "
        "LIMIT 1"
    ).fetchone()
    if creditMismatch is not None:
        raise ValueError(
            "format 3 memory credit does not reference matching helpful feedback: "
            f"atom_id={creditMismatch[0]!r} agent={creditMismatch[1]!r} "
            f"task_id={creditMismatch[2]!r} feedback_id={creditMismatch[3]!r}"
        )
    missingCredit = conn.execute(
        "SELECT f.event_id, f.atom_id, f.agent, f.task_id "
        "FROM recall_feedback AS f "
        "WHERE f.feedback_type = 'helpful' AND f.processed_at IS NOT NULL "
        "AND NOT EXISTS ("
        "SELECT 1 FROM memory_credits AS c "
        "WHERE c.atom_id = f.atom_id AND c.agent = f.agent AND c.task_id = f.task_id) "
        "LIMIT 1"
    ).fetchone()
    if missingCredit is not None:
        raise ValueError(
            "format 3 processed helpful feedback requires a scope credit: "
            f"event_id={missingCredit[0]!r} atom_id={missingCredit[1]!r} "
            f"agent={missingCredit[2]!r} task_id={missingCredit[3]!r}"
        )


def _targetExistsError(target):
    return FileExistsError(
        f"rebuild target already exists: {target}; refusing to overwrite "
        "(canonical data must never be clobbered)"
    )


def _checkpointAndClose(store):
    """Make a staged WAL database self-contained before publishing its main file."""
    conn = store._conn
    try:
        checkpoint = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if checkpoint is None or checkpoint[0] != 0:
            raise RuntimeError(
                f"rebuild staging checkpoint did not complete: {checkpoint!r}")
    finally:
        store.close()


def _fsyncFile(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish(staged, target):
    """Atomically add ``target`` without replacing any existing directory entry."""
    try:
        os.link(staged, target)
    except FileExistsError:
        raise _targetExistsError(target) from None
    except OSError as exc:
        raise OSError(
            f"rebuild cannot publish {target} safely: atomic no-overwrite "
            f"hard link failed: {exc}"
        ) from exc
    _fsyncDirectory(target.parent)


def _checkLegacySourceUnchanged(directory):
    if ((directory / PUBLICATION_MARKER).exists()
            or (directory / MANIFEST_NAME).exists()):
        raise ValueError(
            "legacy export changed during rebuild; retry from a stable snapshot")


def rebuild(fromDir, newDbPath):
    """Build a fresh store at ``newDbPath`` from the JSONL export in ``fromDir``.

    Returns the open :class:`store.store.Store`. The database is created by
    :func:`store.store.openStore` (schema installed, version stamped), then the
    declared tables are re-inserted in FK-safe order via plain INSERTs --
    which fire the atoms fts triggers and regenerate the full-text index. Every
    column value is preserved verbatim from the export (ids, timestamps,
    statuses, weights, per-row ``schema_version``).

    Preconditions and publication, so the target path is never left in a bad state:

    * If ``newDbPath`` already exists, raises ``FileExistsError`` rather than
      clobbering a possibly-live store.
    * Every file required by the export version must exist (empty files are
      valid). Missing files, unsupported manifests and interrupted publication
      are refused before any database is created.
    * Inserts happen in a private temporary directory under the target's parent.
      Failure removes only that private directory; the requested target does not
      exist until the complete, checkpointed database is linked into place.
    """
    srcDir = Path(fromDir)
    target = Path(newDbPath)
    # Fast refusal for a common error. _publish repeats the no-overwrite decision
    # atomically after the private build, closing the concurrent-creator gap.
    if target.exists():
        raise _targetExistsError(target)

    # Require a COMPLETE export before creating anything, so a truncated dump can
    # never produce a store silently missing a whole canonical table.
    loadOrder, expectedDigests = _loadOrder(srcDir)
    missing = [fn for fn, _t, _c in loadOrder if not (srcDir / fn).exists()]
    if missing:
        raise FileNotFoundError(
            f"incomplete export in {srcDir}: missing {', '.join(missing)}; "
            "refusing to rebuild a partial store"
        )

    with TemporaryDirectory(
            prefix=f".{target.name}.pensive-rebuild-", dir=target.parent) as temporary:
        staged = Path(temporary) / "store.db"
        store = openStore(staged)
        try:
            conn = store._conn
            for filename, table, cols in loadOrder:
                expected = None if expectedDigests is None else expectedDigests[filename]
                _insertRows(
                    conn, table, cols,
                    _checkedRows(
                        table,
                        cols,
                        _readRows(srcDir / filename, expectedDigest=expected),
                    )
                    if loadOrder == _LOAD_ORDER and table in _V4_TABLES
                    else _readRows(srcDir / filename, expectedDigest=expected),
                )
            _validateFormat3Relations(conn, loadOrder)
            if expectedDigests is None:
                _checkLegacySourceUnchanged(srcDir)
            conn.commit()
        except BaseException:
            store._conn.rollback()
            store.close()
            raise

        _checkpointAndClose(store)
        os.chmod(staged, 0o600)
        _fsyncFile(staged)
        _publish(staged, target)

    # Publication transfers ownership to the public path. A reopen failure must
    # be reported without deleting or replacing the complete database now there.
    return openStore(target)
