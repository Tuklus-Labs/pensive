"""Regression tests for the Pass 3 fixes."""
import math
import pytest

from pensive.parallel_hybrid import ParallelHybrid
from pensive.spreading import SpreadingActivation
from pensive.ingestion.pipeline import _default_key_path, _load_or_create_key


# ------------- NaN bypass in rank fusion invariant -------------

def test_rank_fusion_rejects_nan():
    with pytest.raises(ValueError, match="NaN"):
        ParallelHybrid(rank_fusion_agreement=float("nan"))
    with pytest.raises(ValueError, match="NaN"):
        ParallelHybrid(rank_fusion_l2_only=float("nan"))
    with pytest.raises(ValueError, match="NaN"):
        ParallelHybrid(rank_fusion_sa_only=float("nan"))


# ------------- XDG_CONFIG_HOME empty-string -------------

def test_empty_xdg_config_home_falls_back_to_home(monkeypatch, tmp_path):
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("XDG_CONFIG_HOME", "")
    p = _default_key_path()
    assert str(p).startswith(str(fake_home)), (
        f"expected key path under HOME when XDG_CONFIG_HOME is empty, got {p}"
    )


def test_whitespace_xdg_config_home_falls_back(monkeypatch, tmp_path):
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("XDG_CONFIG_HOME", "   ")
    p = _default_key_path()
    assert str(p).startswith(str(fake_home))


# ------------- Trailing whitespace in key file -------------

def test_key_file_trailing_newline_stripped(monkeypatch, tmp_path):
    monkeypatch.delenv("PENSIVE_PICKLE_KEY", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    keydir = tmp_path / "pensive"
    keydir.mkdir()
    keyfile = keydir / "pickle.key"
    keyfile.write_bytes(b"my-key-value\n")
    k = _load_or_create_key()
    assert k == b"my-key-value"


# ------------- build() exception safety + lock acquisition -------------

def test_build_exception_resets_state():
    """If build raises partway through, _built stays False."""
    sa = SpreadingActivation()

    # First a successful build so the object has state
    sa.add_documents([{"id": "a", "content": "hello world", "value": "hi"}])
    assert sa._built

    # Now a build that will blow up (missing 'content' key)
    with pytest.raises(KeyError):
        sa.build([{"id": "bad"}])  # no 'content' -> KeyError during extract

    # After failure, the graph should be reset and _built False
    assert sa._built is False


def test_build_parallel_exception_resets_state():
    """Same guarantee for build_parallel."""
    sa = SpreadingActivation()
    sa.add_documents([{"id": "a", "content": "hello", "value": "hi"}])
    assert sa._built

    # build_parallel falls through to build() when workers <= 1 or docs <
    # 2000. To actually exercise the parallel path we need >=2000 docs.
    # Craft a batch where one doc is malformed and triggers KeyError.
    docs = [{"id": f"d{i}", "content": "ok", "value": str(i)} for i in range(2500)]
    docs[1000] = {"id": "boom"}  # missing content

    with pytest.raises(KeyError):
        sa.build_parallel(docs, workers=2)

    assert sa._built is False


# ------------- build_lock exists -------------

def test_build_lock_present():
    sa = SpreadingActivation()
    assert hasattr(sa, "_build_lock")
    # RLock is reentrant; acquire + acquire + release + release should not deadlock
    with sa._build_lock:
        with sa._build_lock:
            pass
