"""Operator durability helpers for SQLite snapshots and restore staging."""
import hashlib
import re
import shutil
import sqlite3
import stat
import time
from datetime import datetime, UTC
from pathlib import Path

from store.store import CURRENT_SCHEMA_VERSION

__all__ = ["SNAPSHOT_KEEP", "snapshot", "restoreSnapshot"]

# Keep enough generations to survive several bad operator runs or interrupted
# rotations, while keeping a decades-scale store from silently filling disks.
SNAPSHOT_KEEP = 5


_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")
_SNAPSHOT_SUFFIX = ".sqlite3"


def snapshot(store, dir):
    """Create a complete standalone SQLite snapshot using ``VACUUM INTO``.

    The live store remains the canonical asset. This function does not change
    journal mode, run checkpoint operations, or copy sidecar files; SQLite's own
    ``VACUUM INTO`` reads a consistent view from the open connection and writes a
    new database file. Older snapshots are pruned only when they match this
    store's snapshot prefix.
    """
    snapshot_dir = Path(dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    identity = _storeIdentity(store)
    prefix = _snapshotPrefix(store, identity)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    snapshot_at = time.time_ns()
    target = snapshot_dir / f"{prefix}-{stamp}-{snapshot_at}{_SNAPSHOT_SUFFIX}"
    if target.exists():
        raise FileExistsError(target)

    store._conn.execute("VACUUM main INTO ?", (str(target),))
    _stampSnapshot(target, identity, snapshot_at)
    _assertIntegrity(target)
    _pruneSnapshots(snapshot_dir, prefix, identity)
    return target


def restoreSnapshot(snapshotPath, targetPath):
    """Copy a snapshot into a new target path for operator-approved restore.

    Automatic overwrite of the canonical live database is intentionally out of
    scope. Operators must choose the target path and perform any ledgered swap
    themselves; this helper only stages a verified snapshot at an unused path.
    """
    source = Path(snapshotPath)
    target = Path(targetPath)
    if not source.is_file():
        raise FileNotFoundError(source)
    if target.exists():
        raise FileExistsError(target)
    for sidecar in _sidecars(target):
        if sidecar.exists():
            raise FileExistsError(sidecar)

    _assertSnapshotSource(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    _assertSnapshotSource(target)
    return target


def _snapshotPrefix(store, identity=None):
    source = _mainDatabasePath(store)
    stem = _SAFE_NAME.sub("-", source.stem).strip("-") or "store"
    return f"{stem}-{identity or _storeIdentity(store)}"


def _storeIdentity(store):
    source = _mainDatabasePath(store)
    return hashlib.sha256(str(source.resolve()).encode("utf-8")).hexdigest()[:12]


def _mainDatabasePath(store):
    for _, name, path in store._conn.execute("PRAGMA database_list"):
        if name == "main":
            if not path:
                raise RuntimeError("cannot snapshot an in-memory SQLite store")
            return Path(path)
    raise RuntimeError("SQLite connection has no main database")


def _pruneSnapshots(snapshot_dir, prefix, identity):
    candidates = sorted(
        (
            path
            for path in snapshot_dir.iterdir()
            if _isOwnedSnapshotCandidate(path, prefix, identity)
        ),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )
    for stale in candidates[:-SNAPSHOT_KEEP]:
        stale.unlink()


def _isOwnedSnapshotCandidate(path, prefix, identity):
    if not (path.name.startswith(f"{prefix}-") and path.name.endswith(_SNAPSHOT_SUFFIX)):
        return False
    try:
        if not stat.S_ISREG(path.lstat().st_mode):
            return False
        conn = _connectReadOnly(path)
        try:
            if not _integrityOk(conn):
                return False
            return _metaValue(conn, "snapshot_of") == identity
        finally:
            conn.close()
    except sqlite3.Error:
        return False
    except OSError:
        return False


def _stampSnapshot(path, identity, snapshot_at):
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            "INSERT INTO meta(key, value) VALUES ('snapshot_of', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (identity,),
        )
        conn.execute(
            "INSERT INTO meta(key, value) VALUES ('snapshot_at', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (str(snapshot_at),),
        )
        conn.commit()
    finally:
        conn.close()


def _assertSnapshotSource(path):
    conn = _connectReadOnly(path)
    try:
        _assertIntegrityConn(conn, path)
        version = _metaValue(conn, "schema_version")
        snapshot_of = _metaValue(conn, "snapshot_of")
        if version is None or snapshot_of is None:
            raise RuntimeError(f"not a Pensive snapshot: {path}")
        if int(version) > CURRENT_SCHEMA_VERSION:
            raise RuntimeError(
                f"snapshot schema_version {version} is newer than this build "
                f"supports (max {CURRENT_SCHEMA_VERSION}); refusing to restore"
            )
    except sqlite3.Error as exc:
        raise RuntimeError(f"not a Pensive snapshot: {path}") from exc
    finally:
        conn.close()


def _assertIntegrity(path):
    conn = _connectReadOnly(path)
    try:
        _assertIntegrityConn(conn, path)
    finally:
        conn.close()


def _connectReadOnly(path):
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    return sqlite3.connect(uri, uri=True)


def _integrityOk(conn):
    result = conn.execute("PRAGMA integrity_check").fetchone()
    return result is not None and result[0] == "ok"


def _assertIntegrityConn(conn, path):
    result = conn.execute("PRAGMA integrity_check").fetchone()
    if result is None or result[0] != "ok":
        detail = None if result is None else result[0]
        raise RuntimeError(f"SQLite integrity_check failed for {path}: {detail}")


def _metaValue(conn, key):
    table = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'meta'"
    ).fetchone()
    if table is None:
        return None
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    if row is None:
        return None
    return row[0]


def _sidecars(path):
    return (Path(f"{path}-wal"), Path(f"{path}-shm"))
