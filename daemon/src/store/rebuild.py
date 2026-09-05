"""Rebuild a fresh canonical store from a JSONL export -- the decades guarantee.

Given the JSONL snapshot :func:`store.export.exportJSONL` writes, ``rebuild``
reconstructs the canonical tables and retained history. Legacy four-file dumps
remain readable; format 2 requires the usage and review-history tables too. The
``embeddings`` and ``fts`` tables are absent from the export on purpose -- the
fts index regenerates because atoms are inserted through the normal triggers, and
embeddings are re-derived later by the Phase 2 embedder.

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
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from store.store import openStore, CURRENT_SCHEMA_VERSION
from store.export import (
    _TABLES,
    FORMAT_VERSION,
    MANIFEST_NAME,
    PUBLICATION_MARKER,
    _fsyncDirectory,
)

__all__ = ["rebuild"]

# (filename, table, columns) in FK-safe insertion order: when each table loads,
# every foreign key it carries already points at a loaded row.
_LOAD_ORDER = tuple((name, table, cols) for name, table, cols, _ in _TABLES)


def _loadOrder(directory):
    if (directory / PUBLICATION_MARKER).exists():
        raise ValueError(f"incomplete export publication at {directory}")
    path = directory / MANIFEST_NAME
    if not path.exists():
        if any((directory / name).exists() for name, _, _ in _LOAD_ORDER[4:]):
            raise ValueError("incomplete export: history files require a format manifest")
        return _LOAD_ORDER[:4], None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("invalid export format manifest")
    version = manifest.get("formatVersion")
    if type(version) is not int or version != FORMAT_VERSION:
        raise ValueError(f"unsupported export format version: {version!r}")
    if manifest.get("format") != "pensive-jsonl":
        raise ValueError("invalid export format name")
    schema = manifest.get("schemaVersion")
    if type(schema) is not int or not 1 <= schema <= CURRENT_SCHEMA_VERSION:
        raise ValueError(f"unsupported export schema version: {schema!r}")
    if manifest.get("files") != [row[0] for row in _LOAD_ORDER]:
        raise ValueError("export format manifest has an unexpected table set")
    hashes = manifest.get("sha256")
    expectedNames = [row[0] for row in _LOAD_ORDER]
    if not isinstance(hashes, dict) or set(hashes) != set(expectedNames):
        raise ValueError("export format manifest has an invalid digest set")
    for filename in expectedNames:
        digest = hashes[filename]
        if (type(digest) is not str or len(digest) != 64
                or any(ch not in "0123456789abcdef" for ch in digest)):
            raise ValueError(
                f"export format manifest has an invalid digest for {filename}")
    return _LOAD_ORDER, hashes


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


def _insertRows(conn, table, cols, rows):
    sql = (
        f"INSERT INTO {table}({', '.join(cols)}) "
        f"VALUES ({', '.join('?' for _ in cols)})"
    )
    for obj in rows:
        conn.execute(sql, tuple(obj[c] for c in cols))


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
                    _readRows(srcDir / filename, expectedDigest=expected),
                )
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
