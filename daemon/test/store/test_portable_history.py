"""Regressions for RISK_MODEL_PORTABLE_HISTORY.md."""
import errno
import json
from pathlib import Path
import stat
import threading

import pytest

import store.export as exporter
import store.rebuild as rebuilder
from lifecycle.importance import accrueImportance
from store.rebuild import rebuild
from store.store import getAtom, logRecall, openStore, putAtom


def seed(store):
    old = putAtom(store, {"text": "remember why", "kind": "atom",
                         "importance": 0.3, "provenance": {"source": "explicit-emit"}})
    new = putAtom(store, {"text": "a proposed revision", "kind": "atom",
                         "provenance": {"source": "explicit-emit"}})
    logRecall(store, [old], query="why was this chosen?", weight=2.0)
    accrueImportance(store)
    logRecall(store, [old], query="記憶の理由", sourceRef="session:α", weight=3.0)
    store._conn.execute(
        "INSERT INTO supersession_proposals VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("proposal-1", old, new, 0.91, "operator reviewed", "rejected", 123))
    store._conn.commit()
    return old


def test_restore_preserves_usage_and_review_history_legacy_usage_is_inert(tmp_path):
    # E1/E2: processed_at is state, not disposable index data. Legacy exposure
    # logs remain durable history, but no longer earn importance; only explicit
    # recall_feedback can create a memory credit in schema v4.
    src = openStore(tmp_path / "source.db")
    dst = None
    try:
        atom = seed(src)
        expected = {table: src._conn.execute(f"SELECT * FROM {table} ORDER BY id").fetchall()
                    for table in ("recall_log", "supersession_proposals")}
        exporter.exportJSONL(src, tmp_path / "dump")
        dst = rebuild(tmp_path / "dump", tmp_path / "restored.db")
        actual = {table: dst._conn.execute(f"SELECT * FROM {table} ORDER BY id").fetchall()
                  for table in expected}
        assert actual == expected, (
            f"E1 durable history must round trip: expected={expected!r}, actual={actual!r}")
        before = getAtom(dst, atom)["importance"]
        first = accrueImportance(dst)
        second = accrueImportance(dst)
        assert first["processed"] == 0 and second["processed"] == 0, (
            f"E2 legacy usage stays inert: first={first!r}, second={second!r}")
        assert getAtom(dst, atom)["importance"] == pytest.approx(before), (
            f"E2 legacy exposure does not earn importance: before={before}, "
            f"after={getAtom(dst, atom)['importance']}")
        assert dst._conn.execute(
            "SELECT COUNT(*) FROM recall_log"
        ).fetchone()[0] == len(expected["recall_log"]), (
            f"E2 legacy exposure history remains durable: expected={expected['recall_log']!r}"
        )
    finally:
        src.close()
        if dst:
            dst.close()


def test_legacy_four_file_dump_remains_readable(tmp_path):
    # E3: old exports cannot contain history fields that did not exist in them.
    src = openStore(tmp_path / "source.db")
    try:
        atom = seed(src)
        exporter.exportJSONL(src, tmp_path / "dump")
    finally:
        src.close()
    for name in ("manifest.json", "recall_log.jsonl", "supersession_proposals.jsonl"):
        (tmp_path / "dump" / name).unlink(missing_ok=True)
    dst = rebuild(tmp_path / "dump", tmp_path / "legacy.db")
    try:
        assert getAtom(dst, atom)["text"] == "remember why", (
            "E3 legacy canonical text remains readable")
        assert dst._conn.execute("SELECT count(*) FROM recall_log").fetchone()[0] == 0, (
            "E3 legacy restores must not invent usage history")
    finally:
        dst.close()


@pytest.mark.parametrize("missing", ["recall_log.jsonl", "supersession_proposals.jsonl"])
def test_current_dump_requires_each_declared_history_file(tmp_path, missing):
    # E3/E4: an empty table still has a file, so absence means incomplete.
    src = openStore(tmp_path / "source.db")
    try:
        exporter.exportJSONL(src, tmp_path / "dump")
    finally:
        src.close()
    (tmp_path / "dump" / missing).unlink(missing_ok=True)
    target = tmp_path / "restored.db"
    with pytest.raises(FileNotFoundError, match=missing):
        rebuild(tmp_path / "dump", target)
    assert not target.exists(), f"E4 incomplete export must not create {target}"


def test_unknown_export_version_is_refused_before_target_creation(tmp_path):
    # E4: silently treating a future format as legacy loses fields.
    src = openStore(tmp_path / "source.db")
    try:
        exporter.exportJSONL(src, tmp_path / "dump")
    finally:
        src.close()
    (tmp_path / "dump/manifest.json").write_text(json.dumps({"formatVersion": 999}))
    target = tmp_path / "future.db"
    with pytest.raises(ValueError, match="format"):
        rebuild(tmp_path / "dump", target)
    assert not target.exists(), f"E4 unknown format must not create {target}"


def test_export_reads_all_tables_from_one_snapshot(tmp_path, monkeypatch):
    # E5: insert a new atom/provenance pair between table reads from another DB connection.
    src = openStore(tmp_path / "source.db")
    other = openStore(tmp_path / "source.db")
    original = exporter._writeTable
    late = []

    def interleaved(conn, directory, filename, table, cols, order_by):
        digest = original(conn, directory, filename, table, cols, order_by)
        if table == "atoms":
            late.append(putAtom(other, {"text": "arrived during export", "kind": "atom",
                                       "provenance": {"source": "explicit-emit"}}))
        return digest

    monkeypatch.setattr(exporter, "_writeTable", interleaved)
    dst = None
    try:
        seed(src)
        exporter.exportJSONL(src, tmp_path / "dump")
        dst = rebuild(tmp_path / "dump", tmp_path / "snapshot.db")
        assert len(late) == 1 and getAtom(src, late[0]) is not None, (
            f"E5 interleaving control must perform a real write: late={late!r}")
        assert getAtom(dst, late[0]) is None, (
            "E5 later tables must not contain data newer than the atoms snapshot")
        assert dst._conn.execute("PRAGMA foreign_key_check").fetchall() == [], (
            "E5 a consistent snapshot preserves cross-table foreign keys")
    finally:
        src.close()
        other.close()
        if dst:
            dst.close()


def test_staging_failure_preserves_previous_complete_export(tmp_path, monkeypatch):
    # E6: a failed replacement must not overwrite part of the only good dump.
    src = openStore(tmp_path / "source.db")
    try:
        seed(src)
        directory = tmp_path / "dump"
        exporter.exportJSONL(src, directory)
        before = {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}
        putAtom(src, {"text": "new unsaved change", "kind": "atom",
                      "provenance": {"source": "explicit-emit"}})
        original = exporter._writeTable

        def fail(conn, directory, filename, table, cols, order_by):
            if table == "facets":
                raise OSError("injected staging failure")
            original(conn, directory, filename, table, cols, order_by)

        monkeypatch.setattr(exporter, "_writeTable", fail)
        with pytest.raises(OSError, match="staging failure"):
            exporter.exportJSONL(src, directory)
        after = {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}
        assert after == before, (
            f"E6 failed staging must preserve old export files: changed="
            f"{[key for key in before if before[key] != after.get(key)]!r}")
        assert not src._conn.in_transaction, "E6 failed export must release its read snapshot"
    finally:
        src.close()


def test_interrupted_publication_is_refused_as_incomplete(tmp_path, monkeypatch):
    # E6: files partially replaced during publication cannot masquerade as a dump.
    src = openStore(tmp_path / "source.db")
    try:
        seed(src)
        directory = tmp_path / "dump"
        exporter.exportJSONL(src, directory)
        putAtom(src, {"text": "new snapshot", "kind": "atom",
                      "provenance": {"source": "explicit-emit"}})
        original = exporter.os.replace
        calls = []

        def fail_mid_publication(source, target):
            calls.append(str(target))
            if len(calls) == 3:
                raise OSError("injected publication failure")
            return original(source, target)

        monkeypatch.setattr(exporter.os, "replace", fail_mid_publication)
        with pytest.raises(OSError, match="publication failure"):
            exporter.exportJSONL(src, directory)
        assert len(calls) == 3, (
            f"E6 interruption must occur after real replacements: calls={calls!r}")
        target = tmp_path / "partial.db"
        with pytest.raises(ValueError, match="incomplete export"):
            rebuild(directory, target)
        assert not target.exists(), f"E6 partial publication must not create {target}"
    finally:
        src.close()


def test_rebuild_rejects_generation_changed_after_manifest_read(
        tmp_path, monkeypatch):
    # E7: publication can begin after rebuild's initial marker/manifest check.
    src = openStore(tmp_path / "source.db")
    old = putAtom(src, {"text": "old", "kind": "atom",
                        "provenance": {"source": "explicit-emit"}})
    directory = tmp_path / "dump"
    exporter.exportJSONL(src, directory)
    new = putAtom(src, {"text": "new", "kind": "atom",
                        "provenance": {"source": "explicit-emit"}})

    checked = threading.Event()
    firstReplaced = threading.Event()
    rebuildDone = threading.Event()
    originalLoadOrder = rebuilder._loadOrder
    originalReplace = exporter.os.replace
    replaced = [0]
    observation = {}

    def pauseAfterManifest(directory):
        value = originalLoadOrder(directory)
        checked.set()
        if not firstReplaced.wait(5):
            raise RuntimeError("E7 publisher never replaced its first file")
        return value

    def pauseAfterFirstReplace(source, target):
        replaced[0] += 1
        value = originalReplace(source, target)
        if replaced[0] == 1:
            firstReplaced.set()
            if not rebuildDone.wait(5):
                raise RuntimeError("E7 rebuild never completed")
        return value

    monkeypatch.setattr(rebuilder, "_loadOrder", pauseAfterManifest)
    monkeypatch.setattr(exporter.os, "replace", pauseAfterFirstReplace)
    target = tmp_path / "racing.db"

    def runRebuild():
        try:
            restored = rebuilder.rebuild(directory, target)
            restored.close()
            observation["completed"] = True
        except BaseException as exc:
            observation["error"] = exc
        finally:
            rebuildDone.set()

    thread = threading.Thread(target=runRebuild)
    thread.start()
    try:
        assert checked.wait(5), "E7 rebuild must finish its initial manifest read"
        exporter.exportJSONL(src, directory)
    finally:
        rebuildDone.set()
        thread.join(5)
        src.close()

    error = observation.get("error")
    assert isinstance(error, ValueError) and "digest mismatch" in str(error), (
        "E7 mixed export generations must fail a file digest check: "
        f"old={old} new={new} observation={observation!r}")
    assert not target.exists(), (
        f"E7 rejected mixed generation must remove owned target: target={target}")


@pytest.mark.parametrize("transition", ["marker", "manifest"])
def test_legacy_rebuild_rechecks_publication_transition_before_commit(
        tmp_path, monkeypatch, transition):
    # E7: legacy has no digests, so publication state must stay legacy through read.
    src = openStore(tmp_path / "source.db")
    try:
        seed(src)
        directory = tmp_path / "dump"
        exporter.exportJSONL(src, directory)
    finally:
        src.close()
    manifest = (directory / exporter.MANIFEST_NAME).read_bytes()
    for name in (exporter.MANIFEST_NAME, "recall_log.jsonl",
                 "supersession_proposals.jsonl"):
        (directory / name).unlink()

    originalInsertRows = rebuilder._insertRows

    def startPublication(conn, table, cols, rows):
        originalInsertRows(conn, table, cols, rows)
        if table == "facets":
            if transition == "marker":
                (directory / exporter.PUBLICATION_MARKER).write_text("incomplete\n")
            else:
                (directory / exporter.MANIFEST_NAME).write_bytes(manifest)

    monkeypatch.setattr(rebuilder, "_insertRows", startPublication)
    target = tmp_path / f"legacy-{transition}.db"
    with pytest.raises(ValueError, match="legacy export changed"):
        rebuilder.rebuild(directory, target)
    assert not target.exists(), (
        "E7 transitioning legacy source must remove owned target: "
        f"transition={transition} target={target}")


def test_rebuild_atomically_refuses_competing_target_creator(
        tmp_path, monkeypatch):
    # E8: a creator can win after the initial target check but before openStore.
    src = openStore(tmp_path / "source.db")
    try:
        seed(src)
        directory = tmp_path / "dump"
        exporter.exportJSONL(src, directory)
    finally:
        src.close()

    target = tmp_path / "claimed.db"
    originalLoadOrder = rebuilder._loadOrder
    competitor = {}

    def createCompetingTarget(directory):
        store = openStore(target)
        try:
            competitor["atom"] = putAtom(
                store, {"text": "competitor", "kind": "atom",
                        "provenance": {"source": "explicit-emit"}})
        finally:
            store.close()
        competitor["bytes"] = target.read_bytes()
        return originalLoadOrder(directory)

    monkeypatch.setattr(rebuilder, "_loadOrder", createCompetingTarget)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        rebuilder.rebuild(directory, target)
    assert target.read_bytes() == competitor["bytes"], (
        "E8 competing target must remain byte-identical after refusal: "
        f"target={target}")
    existing = openStore(target)
    try:
        rows = existing._conn.execute("SELECT id FROM atoms").fetchall()
        assert rows == [(competitor["atom"],)], (
            "E8 refused rebuild must not merge rows into competing target: "
            f"rows={rows!r} competitor={competitor!r}")
    finally:
        existing.close()


def test_rebuild_keeps_target_unclaimed_until_atomic_publication(
        tmp_path, monkeypatch):
    # E8: a legitimate creator can claim the destination while the private build
    # is in progress. Publication must refuse it without changing its bytes.
    src = openStore(tmp_path / "source.db")
    try:
        seed(src)
        directory = tmp_path / "dump"
        exporter.exportJSONL(src, directory)
    finally:
        src.close()

    target = tmp_path / "claimed-during-build.db"
    competitor = b"independent creator\n"
    originalInsertRows = rebuilder._insertRows
    claimed = []

    def claimDuringBuild(conn, table, cols, rows):
        if not claimed:
            with target.open("xb") as handle:
                handle.write(competitor)
            claimed.append(True)
        return originalInsertRows(conn, table, cols, rows)

    monkeypatch.setattr(rebuilder, "_insertRows", claimDuringBuild)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        rebuilder.rebuild(directory, target)

    assert claimed == [True], (
        f"E8 competing creator must claim an initially free target: claimed={claimed!r}")
    actual = target.read_bytes()
    assert actual == competitor, (
        "E8 failed publication must not alter a competing creator's file: "
        f"actual={actual!r} expected={competitor!r}")


def test_rebuild_failure_never_exposes_or_cleans_up_the_public_target(
        tmp_path, monkeypatch):
    # E8: invalid source rows fail inside private staging. The public path and
    # SQLite sidecars must never appear, and TemporaryDirectory removes staging.
    src = openStore(tmp_path / "source.db")
    try:
        seed(src)
        directory = tmp_path / "dump"
        exporter.exportJSONL(src, directory)
    finally:
        src.close()

    target = tmp_path / "failed.db"
    observations = []

    def failDuringBuild(conn, table, cols, rows):
        observations.append(target.exists())
        raise ValueError("injected private build failure")

    monkeypatch.setattr(rebuilder, "_insertRows", failDuringBuild)
    with pytest.raises(ValueError, match="private build failure"):
        rebuilder.rebuild(directory, target)

    assert observations == [False], (
        f"E8 public target must remain absent during build: observations={observations!r}")
    assert not target.exists() and not Path(f"{target}-wal").exists(), (
        f"E8 failed private build must not expose a target or WAL: target={target}")
    assert not Path(f"{target}-shm").exists(), (
        f"E8 failed private build must not expose shared-memory state: target={target}")
    staging = list(tmp_path.glob(f".{target.name}.pensive-rebuild-*"))
    assert staging == [], (
        f"E8 failed private build must clean private staging: staging={staging!r}")


def test_rebuild_postpublication_open_failure_keeps_published_database(
        tmp_path, monkeypatch):
    # E8: after an atomic publication succeeds, a caller-side reopen failure must
    # never be treated as permission to delete the now-public canonical file.
    src = openStore(tmp_path / "source.db")
    try:
        atom = seed(src)
        directory = tmp_path / "dump"
        exporter.exportJSONL(src, directory)
    finally:
        src.close()

    target = tmp_path / "published.db"
    originalOpenStore = rebuilder.openStore

    def failPublicOpen(path):
        if Path(path) == target:
            raise OSError("injected postpublication open failure")
        return originalOpenStore(path)

    monkeypatch.setattr(rebuilder, "openStore", failPublicOpen)
    with pytest.raises(OSError, match="postpublication open failure"):
        rebuilder.rebuild(directory, target)

    assert target.exists(), (
        f"E8 a published database must survive reopen failure: target={target}")
    restored = originalOpenStore(target)
    try:
        body = getAtom(restored, atom)["text"]
        assert body == "remember why", (
            f"E8 published database must be complete before public reopen: body={body!r}")
    finally:
        restored.close()


def test_rebuild_refuses_unsafe_fallback_when_hard_links_are_unavailable(
        tmp_path, monkeypatch):
    # E8: filesystems without hard-link publication fail clearly. Rebuild must
    # not substitute a replacing rename or copy into the public destination.
    src = openStore(tmp_path / "source.db")
    try:
        seed(src)
        directory = tmp_path / "dump"
        exporter.exportJSONL(src, directory)
    finally:
        src.close()

    target = tmp_path / "unsupported-link.db"

    def unsupportedLink(_source, _target):
        raise OSError(errno.EOPNOTSUPP, "hard links unavailable")

    monkeypatch.setattr(rebuilder.os, "link", unsupportedLink)
    with pytest.raises(OSError, match="atomic no-overwrite hard link failed"):
        rebuilder.rebuild(directory, target)

    assert not target.exists(), (
        f"E8 unsupported publication must leave target absent: target={target}")
    staging = list(tmp_path.glob(f".{target.name}.pensive-rebuild-*"))
    assert staging == [], (
        f"E8 unsupported publication must clean private staging: staging={staging!r}")


def test_format2_rebuild_reads_from_read_only_source(tmp_path):
    # Integration: integrity validation must not require a writable export medium.
    src = openStore(tmp_path / "source.db")
    try:
        atom = seed(src)
        directory = tmp_path / "dump"
        exporter.exportJSONL(src, directory)
    finally:
        src.close()

    directory.chmod(0o555)
    try:
        restored = rebuilder.rebuild(directory, tmp_path / "readonly.db")
        try:
            assert getAtom(restored, atom)["text"] == "remember why", (
                f"format-2 read-only source must rebuild atom={atom}")
        finally:
            restored.close()
    finally:
        directory.chmod(0o755)


def test_export_fsyncs_staged_files_marker_and_directories(
        tmp_path, monkeypatch):
    # E9: data and the fail-closed marker must reach storage before publication.
    src = openStore(tmp_path / "source.db")
    calls = {"files": 0, "directories": 0}
    originalFsync = exporter.os.fsync

    def recordFsync(fd):
        mode = exporter.os.fstat(fd).st_mode
        key = "directories" if stat.S_ISDIR(mode) else "files"
        calls[key] += 1
        return originalFsync(fd)

    monkeypatch.setattr(exporter.os, "fsync", recordFsync)
    try:
        exporter.exportJSONL(src, tmp_path / "dump")
    finally:
        src.close()

    assert calls["files"] >= 8, (
        "E9 six tables, manifest, and marker must each be fsynced: "
        f"calls={calls!r}")
    if exporter.os.name == "posix":
        assert calls["directories"] >= 4, (
            "E9 staging and publication directory transitions must be fsynced: "
            f"calls={calls!r}")
