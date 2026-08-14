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
    the graph, not a helper here.

    THIS TEST CARRIES MORE WEIGHT THAN IT LOOKS. An adversarial review of a
    sibling candidate (2026-08-13) showed that COSINE CANNOT SEE A DROPPED
    NORMALIZE: cosine is scale-invariant, so an encoder returning correctly
    directed but unnormalized vectors scores exactly 1.000000000000 against the
    reference on every sample. A fidelity gate built only on cosine would pass
    an encoder that silently breaks the store's flat-cosine-equals-dot-product
    assumption.

    This suite escapes that hole for two reasons, one deliberate and one lucky.
    Deliberate: this test exists. Lucky: the paired check above uses a raw dot
    product (`np.sum(ref * got)`) rather than a normalized cosine, so it is
    scale-SENSITIVE by construction. Writing `cosine_similarity()` there would
    have reintroduced the blind spot.

    Verified by planting an encoder that preserves direction exactly and scales
    each row by a different factor: 3 tests fail, this one among them."""
    got = np.vstack(onnxEmb.embed(TEXTS))
    norms = np.linalg.norm(got, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), norms


# --------------------------------------------------------------------------- #
# guards derived from an adversarial refutation (2026-08-13)                    #
# --------------------------------------------------------------------------- #
#
# A verifier REFUTED the drop-in claim as originally reported, against an export
# made with torch.onnx.export(dynamo=True). Three defects, all of which produce
# a class that CONSTRUCTS FINE and is silently wrong:
#
#   1. the graph emits last_hidden_state, so a runner that does no pooling
#      returns one (seq, 384) object per call instead of a (384,) vector
#   2. dynamo baked the output batch axis to a literal 1 despite dynamic_axes,
#      so any batch > 1 raises at LayerNormalization
#   3. dynamo writes external data: a 0.1MB graph plus a 133MB .data sidecar
#
# The shipped export avoids all three (pooling folded in, dynamo=False,
# self-contained). These tests exist so a future re-export cannot reintroduce
# them quietly. Defect 2 was already covered by the batch test below, which is
# the only reason it was never a live risk here.


@onnxOnly
def test_a_vector_is_one_dimensional_and_blob_sized(onnxEmb):
    """The pooling defect does not raise, it returns the wrong SHAPE.

    An unpooled graph hands back (tokens, 384). That still indexes, still
    serializes, and writes a blob whose length depends on the input's token
    count instead of the schema's fixed 1536 bytes."""
    import numpy as np

    from recall.embedder import vecToBlob

    vec = onnxEmb.embed(["why did the flat index stall? "])[0]
    assert vec.ndim == 1, f"expected a vector, got shape {vec.shape} (unpooled graph?)"
    assert len(vecToBlob(vec)) == 1536, (
        f"blob is {len(vecToBlob(vec))} bytes; schema.sql fixes 1536 "
        "(384 float32). A token-count-dependent length means no pooling.")
    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-5


@onnxOnly
def test_the_dim_probe_cannot_report_a_token_count(onnxEmb):
    """`dim` is derived from a forward pass, which is only safe if that pass
    yields a vector. On the unpooled graph the same probe reported 4, the token
    count of the probe string, and every downstream size check inherited it."""
    assert onnxEmb.dim == 384


@onnxOnly
def test_the_artifact_is_self_contained():
    """External-data exports split into a stub plus a sidecar. Pointing
    PENSIVE_V3_ONNX_MODEL at the stub works only while the sidecar travels with
    it, which is a deployment failure waiting for the first copy."""
    import onnx

    model = onnx.load(ONNX, load_external_data=False)
    external = [
        i.name for i in model.graph.initializer
        if i.HasField("data_location") and i.data_location == onnx.TensorProto.EXTERNAL
    ]
    assert not external, f"{len(external)} initializers live in a sidecar file"
    assert os.path.getsize(ONNX) > 50e6, "a weightless graph, not a model"


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
    """Each vector must own its data rather than view the batch matrix.

    The first version of this test wrote zeros into got[0] and asserted got[1]
    was unchanged. That assertion CANNOT FAIL: distinct rows of a matrix never
    alias each other, view or copy, so it passed against a planted defect that
    removed the .copy() entirely. It was a test shaped like a guard.

    What actually distinguishes the two is ownership: a view carries a .base
    pointing at the parent matrix and keeps that whole (batch x 384) array alive
    for as long as any one vector survives. Asserting on .base is a question the
    defect can answer wrongly."""
    got = onnxEmb.embed(TEXTS)
    offenders = [i for i, v in enumerate(got) if v.base is not None]
    assert not offenders, (
        f"vectors {offenders} are views into the batch matrix, not owned copies")


# --------------------------------------------------------------------------- #
# the factory degrades loudly, never silently
# --------------------------------------------------------------------------- #


def test_the_daemon_actually_builds_its_embedder_through_the_factory():
    """The factory must be ON the daemon's startup path, not merely importable.

    This exists because it was not. Every test above passed against a build in
    which `serve/daemon.py` still called `Embedder(modelId)` directly, so
    PENSIVE_V3_ONNX_MODEL could be set, the systemd drop-in could be installed,
    the whole feature could be 'shipped', and production would have kept running
    the torch path with nothing anywhere reporting a problem.

    That is the same defect class as a gate nothing invokes: the code is correct
    and unreached. Unit tests cannot see it, because they call the unit
    directly. Source inspection is a blunt instrument, but it asks the one
    question the rest of this file cannot: does the caller call it.
    """
    import inspect

    import serve.daemon as daemon

    src = inspect.getsource(daemon)
    assert "makeEmbedder(modelId)" in src, (
        "serve/daemon.py does not build its embedder through makeEmbedder; "
        "the ONNX path is unreachable in production")
    assert "embedder = Embedder(" not in src, (
        "serve/daemon.py still constructs Embedder directly, which bypasses "
        "the factory and pins production to the torch path")


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
