"""Gate for the ONNX serve-path embedder.

THE CLAIM UNDER TEST, stated so a drifted question reads as a wrong sentence:

    Running bge-small under onnxruntime produces the SAME embedding space as the
    sentence-transformers path, under the SAME model id, so the 340,722 stored
    vectors stay valid and no re-embed is triggered.

The failure this guards against is not a slow answer, it is a silently wrong
one. Two ways it goes wrong:

  1. The vectors drift. A query embedded in a slightly different space still
     returns ten results, ranked wrongly, with no error anywhere. The int8
     candidate rejected in this same campaign looked fine on cosine (0.9701)
     and changed the top-1 answer for 11 of 32 real queries.
  2. The model id changes. Embeddings are keyed (atom_id, model_id), so a new
     id strands every stored vector and silently re-embeds the whole store on
     next startup. Cheap to do by accident, expensive to notice.

The cosine assertions carry a NEGATIVE CONTROL. A cosine near 1.0 proves
nothing alone: it is exactly what comparing a thing to itself yields, and a
broken test that embedded the same text twice would pass without the control.
"""
import os

import numpy as np
import pytest

from recall.embedder import Embedder, OnnxEmbedder, makeEmbedder

MODEL = "BAAI/bge-small-en-v1.5"
ONNX = os.path.expanduser("~/.local/share/pensive-v3/models/bge-small-en-v1.5.onnx")

TEXTS = [
    "the daemon refuses unrecognized Host values to close DNS rebinding",
    "spreading activation over a sparse bipartite entity graph",
    "def _compile(self): snapshot the COO buffers under the build lock",
    "Gary asked for semantically adjacent retrievals",
    "importance accrual drains the recall_log and caps at 1.0",
    "short",
]

onnxOnly = pytest.mark.skipif(
    not os.path.exists(ONNX), reason=f"no exported graph at {ONNX}")


@pytest.fixture(scope="module")
def onnxEmb():
    return OnnxEmbedder(MODEL, ONNX)


@pytest.fixture(scope="module")
def torchEmb():
    return Embedder(MODEL)


# --------------------------------------------------------------------------- #
# the identity that keeps 340k vectors valid
# --------------------------------------------------------------------------- #


@onnxOnly
def test_the_model_id_is_unchanged(onnxEmb):
    """Embeddings are keyed (atom_id, model_id). A different id here strands
    every stored vector and re-embeds the store on next startup. The id names
    the embedding SPACE, not the inference library."""
    assert onnxEmb.modelId == MODEL


@onnxOnly
def test_the_dimension_matches_the_stored_blobs(onnxEmb):
    """schema.sql fixes 384 float32 for this model; a mismatch would write
    blobs the flat index cannot stack."""
    assert onnxEmb.dim == 384


# --------------------------------------------------------------------------- #
# same space, with a control
# --------------------------------------------------------------------------- #


@onnxOnly
def test_vectors_match_the_torch_path_with_a_negative_control(onnxEmb, torchEmb):
    ref = np.vstack(torchEmb.embed(TEXTS))
    got = np.vstack(onnxEmb.embed(TEXTS))
    assert got.shape == ref.shape

    paired = np.sum(ref * got, axis=1)
    assert paired.min() > 0.9999, f"embedding space drifted: min {paired.min()}"

    # CONTROL: compare each ONNX vector against a DIFFERENT text's reference.
    # Without this, a test that embedded the same text twice would also pass.
    rolled = np.sum(ref * np.roll(got, 1, axis=0), axis=1)
    assert np.median(rolled) < 0.9, (
        "negative control did not separate; the cosine assertion above proves "
        f"nothing (control median {np.median(rolled)})")


@onnxOnly
def test_vectors_are_unit_norm(onnxEmb):
    """A flat cosine search reduces to a dot product only if rows are unit
    length. Normalization is folded into the exported graph, so this asserts
    the graph, not a helper here."""
    got = np.vstack(onnxEmb.embed(TEXTS))
    norms = np.linalg.norm(got, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), norms


@onnxOnly
def test_empty_input_yields_empty_output_without_touching_the_model(onnxEmb):
    assert onnxEmb.embed([]) == []


@onnxOnly
def test_a_batch_larger_than_the_batch_size_still_returns_one_vector_each(onnxEmb):
    """_BATCH_SIZE is 64; the loop must not drop or duplicate a tail."""
    many = TEXTS * 20  # 120 > 64
    got = onnxEmb.embed(many)
    assert len(got) == len(many)


@onnxOnly
def test_each_returned_vector_owns_its_buffer(onnxEmb):
    """A row view would alias the batch matrix, so a consumer's in-place op
    would silently mutate its siblings."""
    got = onnxEmb.embed(TEXTS)
    before = got[1].copy()
    got[0][:] = 0.0
    assert np.array_equal(got[1], before)


# --------------------------------------------------------------------------- #
# the factory degrades loudly, never silently
# --------------------------------------------------------------------------- #


def test_no_env_var_means_the_torch_path(monkeypatch):
    monkeypatch.delenv("PENSIVE_V3_ONNX_MODEL", raising=False)
    assert type(makeEmbedder(MODEL)).__name__ == "Embedder"


def test_a_bad_path_falls_back_rather_than_killing_the_daemon(monkeypatch, capfd):
    """The feature is an optimization over a working encoder. A typo in the env
    var must not take recall down -- but it must SAY so, because an operator who
    set it deserves to know it did not take."""
    monkeypatch.setenv("PENSIVE_V3_ONNX_MODEL", "/nonexistent/graph.onnx")
    emb = makeEmbedder(MODEL)
    assert type(emb).__name__ == "Embedder"
    out = capfd.readouterr().out
    assert "ONNX embedder unavailable" in out, "fell back SILENTLY"
