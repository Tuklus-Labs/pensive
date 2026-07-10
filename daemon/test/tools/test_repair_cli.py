"""CLI behavior: dry-run mutates nothing; --apply backs up first."""
import json
import subprocess
import sys
from pathlib import Path

_DAEMON = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_DAEMON / "tools"))
sys.path.insert(0, str(_DAEMON / "src"))

from store.store import openStore, putAtom


def _seed(tmp_path):
    p = tmp_path / "mem.db"
    s = openStore(p)
    putAtom(s, {
        "text": "b", "kind": "document_chunk", "project": None,
        "importance": 0.0,
        "provenance": {"source": "bulk-import",
                       "sourceRef": "projects/obol/api/x.go#c0"},
    })
    s.close()
    return p


def _run(args):
    return subprocess.run(
        [sys.executable, str(_DAEMON / "tools" / "repair_corpus.py"), *args],
        capture_output=True, text=True)


def test_dry_run_default_mutates_nothing(tmp_path):
    p = _seed(tmp_path)
    r = _run(["--store", str(p)])
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["applied"] is False
    s = openStore(p)
    assert s._conn.execute(
        "SELECT project FROM atoms").fetchone()[0] is None  # untouched
    s.close()
    assert not list(tmp_path.glob("*.bak-pre-repair-*"))


def test_apply_backs_up_then_repairs(tmp_path):
    p = _seed(tmp_path)
    r = _run(["--store", str(p), "--apply"])
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["applied"] is True
    assert out["backfillProjects"]["backfilled"] == 1
    assert out["verify"]["ok"] is True
    backups = list(tmp_path.glob("*.bak-pre-repair-*"))
    assert len(backups) == 1
    s = openStore(p)
    assert s._conn.execute(
        "SELECT project FROM atoms").fetchone()[0] == "obol"
    s.close()
