"""Operator durability helpers for SQLite snapshots and restore staging."""
import hashlib
import re
import shutil
import time
from datetime import datetime, UTC
from pathlib import Path

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

    prefix = _snapshotPrefix(store)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    target = snapshot_dir / f"{prefix}-{stamp}-{time.time_ns()}{_SNAPSHOT_SUFFIX}"
    if target.exists():
        raise FileExistsError(target)

    store._conn.execute("VACUUM main INTO ?", (str(target),))
    _assertIntegrity(target)
    _pruneSnapshots(snapshot_dir, prefix)
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

    _assertIntegrity(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    _assertIntegrity(target)
    return target


def _snapshotPrefix(store):
    source = _mainDatabasePath(store)
    stem = _SAFE_NAME.sub("-", source.stem).strip("-") or "store"
    digest = hashlib.sha256(str(source.resolve()).encode("utf-8")).hexdigest()[:12]
    return f"{stem}-{digest}"


def _mainDatabasePath(store):
    for _, name, path in store._conn.execute("PRAGMA database_list"):
        if name == "main":
            if not path:
                raise RuntimeError("cannot snapshot an in-memory SQLite store")
            return Path(path)
    raise RuntimeError("SQLite connection has no main database")


def _pruneSnapshots(snapshot_dir, prefix):
    candidates = sorted(
        (
            path
            for path in snapshot_dir.iterdir()
            if path.is_file()
            and path.name.startswith(f"{prefix}-")
            and path.name.endswith(_SNAPSHOT_SUFFIX)
        ),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )
    for stale in candidates[:-SNAPSHOT_KEEP]:
        stale.unlink()


def _assertIntegrity(path):
    import sqlite3

    conn = sqlite3.connect(str(path))
    try:
        result = conn.execute("PRAGMA integrity_check").fetchone()
        if result is None or result[0] != "ok":
            detail = None if result is None else result[0]
            raise RuntimeError(f"SQLite integrity_check failed for {path}: {detail}")
    finally:
        conn.close()


def _sidecars(path):
    return (Path(f"{path}-wal"), Path(f"{path}-shm"))
