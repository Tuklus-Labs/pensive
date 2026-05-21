"""Regression tests for PENPY-DEP-1 / PENPY-P6-IMP-1.

The pass-4 fix (commit 2fcb719) added a `getattr` fallback in
`L2Handler.__init__` to cope with the sentence-transformers >=5 rename
of `get_sentence_embedding_dimension` -> `get_embedding_dimension`.

The pass-6 audit found that the fix was code-only with no regression
test: when the fallback was reverted, the suite still passed because
the installed sentence-transformers version still aliases the old name.
When upstream removes the alias entirely, the unguarded code path will
AttributeError and no test would catch it.

These tests close that gap by monkey-patching the
`sentence_transformers.SentenceTransformer` import target with a fake
class that exposes ONLY one of the two method names. The L2Handler
must succeed under both shapes.

Sabotage gate: reverting l2.py:65-73 to a bare
``self._dim = self._model.get_sentence_embedding_dimension()`` makes
``test_dep1_only_new_name_works`` fail with AttributeError. Reverting
to a bare ``self._dim = self._model.get_embedding_dimension()`` makes
``test_dep1_only_legacy_name_works`` fail.

The tests intentionally bypass the real sentence-transformers model
load by monkey-patching the module's ``SentenceTransformer`` symbol,
so they run in milliseconds without downloading any weights.
"""
import numpy as np
import pytest

from pensive import l2 as l2_module


pytest.importorskip("sentence_transformers")
pytest.importorskip("faiss")


class _FakeModelOnlyNewName:
    """Fake SentenceTransformer exposing only get_embedding_dimension."""

    def __init__(self, *_args, **_kwargs):
        pass

    def get_embedding_dimension(self) -> int:
        return 384

    def encode(self, texts, **_kwargs):
        if isinstance(texts, str):
            texts = [texts]
        return np.zeros((len(texts), 384), dtype=np.float32)


class _FakeModelOnlyLegacyName:
    """Fake SentenceTransformer exposing only get_sentence_embedding_dimension."""

    def __init__(self, *_args, **_kwargs):
        pass

    def get_sentence_embedding_dimension(self) -> int:
        return 384

    def encode(self, texts, **_kwargs):
        if isinstance(texts, str):
            texts = [texts]
        return np.zeros((len(texts), 384), dtype=np.float32)


@pytest.fixture
def patch_sentence_transformer_new_name(monkeypatch):
    """Replace the sentence_transformers.SentenceTransformer import target.

    L2Handler.__init__ does ``from sentence_transformers import SentenceTransformer``
    inside the method body, so we patch the source module's attribute.
    """
    import sentence_transformers
    monkeypatch.setattr(
        sentence_transformers, "SentenceTransformer", _FakeModelOnlyNewName
    )
    yield


@pytest.fixture
def patch_sentence_transformer_legacy_name(monkeypatch):
    import sentence_transformers
    monkeypatch.setattr(
        sentence_transformers, "SentenceTransformer", _FakeModelOnlyLegacyName
    )
    yield


def test_dep1_only_new_name_works(patch_sentence_transformer_new_name):
    """L2Handler must accept a model that only has get_embedding_dimension.

    This is the sentence-transformers >=5 shape after the legacy alias
    is eventually removed. The current `getattr` fallback in l2.py:65-73
    must NOT pre-evaluate ``self._model.get_sentence_embedding_dimension``
    eagerly (Python eagerly evaluates the default arg of getattr), or
    this test will fail with AttributeError before getattr can return.

    Sabotage: revert l2.py:65-73 to a bare
    ``self._dim = self._model.get_sentence_embedding_dimension()``
    and this test will fail with AttributeError.
    """
    handler = l2_module.L2Handler()
    assert handler._dim == 384, (
        f"PENPY-DEP-1: L2Handler._dim should be 384 from fake model's "
        f"get_embedding_dimension(), got {handler._dim}"
    )


def test_dep1_only_legacy_name_works(patch_sentence_transformer_legacy_name):
    """L2Handler must accept a model that only has get_sentence_embedding_dimension.

    This is the sentence-transformers <5 shape. The `getattr` fallback
    in l2.py:65-73 must fall back to the legacy name when the new one
    is absent.

    Sabotage: revert l2.py:65-73 to a bare
    ``self._dim = self._model.get_embedding_dimension()``
    and this test will fail with AttributeError.
    """
    handler = l2_module.L2Handler()
    assert handler._dim == 384, (
        f"PENPY-DEP-1: L2Handler._dim should be 384 from fake model's "
        f"get_sentence_embedding_dimension(), got {handler._dim}"
    )
