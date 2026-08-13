"""Gate for provenance integrity on the MCP write edge.

THE CLAIM UNDER TEST, stated so a drifted question reads as a wrong sentence:

    A client that has declared an identity on its connection CANNOT write
    provenance rows attributed to some other agent; no write path can put a
    filesystem path or a control byte into the agent column; and no caller can
    set a reserved system ``source`` value that another subsystem reads as a
    trust signal.

Why these are unit tests over pure functions, where ``test_transport_agent.py``
insists on booting a real daemon: the open question there is what the MCP SDK
DOES with ``RequestContext`` under ``stateless=True``, which only the SDK can
answer. The open question HERE is our own precedence and validation logic,
which is a pure function of its inputs. Asking a real daemon would not make
the answer truer, only slower.

Born 2026-08-12 from a glasswing pass. The gap was recorded IN WRITING in
``_sanitizeAgent``'s own docstring -- validation applied to the transport value
and deliberately skipped on the caller-supplied argument, for backward
compatibility -- four lines under the sentence "a wrong stamp is counterfeit
provenance". The lesson worth keeping is that a guard documented as skipped at
one entry point is not a boundary anywhere, so grep every caller of a guard
rather than trusting that the guard exists.

Blast radius was measured before precedence was inverted: no caller anywhere on
this box passes an ``agent`` tool argument, and ``aegis-pensive-who``'s canary
is a pure read that never emits. Inverting is therefore safe, and the
argument-only path is kept working for CLI callers that have no transport.
"""
import pytest

from serve import mcp as M
from serve.mcp import ServeContext
from recall.embedder import Embedder
from store.store import openStore

MODEL_ID = "BAAI/bge-small-en-v1.5"

# The GPU stack emits two SwigPy DeprecationWarnings on first import under
# CPython 3.14; filter exactly those, mirroring the other serve tests.
pytestmark = pytest.mark.filterwarnings(
    "ignore:builtin type SwigPy.* has no __module__ attribute:DeprecationWarning"
)


@pytest.fixture(scope="session")
def embedder():
    return Embedder(MODEL_ID)


@pytest.fixture
def store(tmp_path):
    s = openStore(tmp_path / "mem.db")
    try:
        yield s
    finally:
        s.close()


@pytest.fixture
def ctx(store, embedder):
    return ServeContext(store, embedder, MODEL_ID, agent="heph")


# --------------------------------------------------------------------------- #
# agent precedence and sanitization
# --------------------------------------------------------------------------- #


class _Ctx:
    """Minimal stand-in: _resolveAgent only reads ``.agent`` off the context."""

    def __init__(self, agent=None):
        self.agent = agent


@pytest.fixture
def noTransport(monkeypatch):
    monkeypatch.setattr(M, "_transportAgent", lambda: None)


@pytest.fixture
def transportIsGrok(monkeypatch):
    monkeypatch.setattr(M, "_transportAgent", lambda: "grok")


def test_connection_identity_beats_a_caller_claim(transportIsGrok):
    """A connection declared as grok cannot write rows stamped heph.

    This is the forgery case. The argument is a CLAIM about one emit; the
    connection is what the client actually is.
    """
    assert M._resolveAgent(_Ctx("daemon-default"), {"agent": "heph"}) == "grok"


def test_caller_claim_still_used_when_no_transport_declared(noTransport):
    """Regression guard: the CLI/stdio path has no transport identity, so an
    explicit argument is the only signal there and must keep working."""
    assert M._resolveAgent(_Ctx(None), {"agent": "codex"}) == "codex"


def test_caller_claim_is_sanitized_like_the_transport_value(noTransport):
    """A filesystem path is not an identity, whichever door it arrives through.

    Eight rows of ``/root/v3_publication_controller`` already exist in the live
    store, which is why the transport sanitizer was written; the argument path
    could still write more.
    """
    assert M._resolveAgent(_Ctx(None), {"agent": "/root/not_an_identity"}) is None


def test_caller_claim_with_control_bytes_is_rejected(noTransport):
    assert M._resolveAgent(_Ctx(None), {"agent": "he\x00ph"}) is None
    assert M._resolveAgent(_Ctx(None), {"agent": "heph\nclaude"}) is None


def test_overlong_caller_claim_is_rejected(noTransport):
    assert M._resolveAgent(_Ctx(None), {"agent": "x" * 5000}) is None


def test_process_default_fills_in_last(noTransport):
    assert M._resolveAgent(_Ctx("heph"), {}) == "heph"


# --------------------------------------------------------------------------- #
# reserved provenance sources
# --------------------------------------------------------------------------- #


def _seedAtom(ctx):
    return M.handle_emit_atom(ctx, {
        "project": "gatecheck", "shape": "s", "approach": "a",
        "outcome": "succeeded", "reason": "r", "principle": "p",
    })


@pytest.mark.parametrize("reserved", ["bulk-import", "repair-tool", "edge-campaign"])
def test_correct_refuses_reserved_system_source(ctx, reserved):
    """``source`` distinguishes an agent's own writes from corpus imports and
    operator tooling. A caller that can set it can disguise a hand-written atom
    as 299k-row bulk corpus, or as the repair tool's output."""
    _seedAtom(ctx)
    oldId = ctx.store._conn.execute(
        "SELECT id FROM atoms ORDER BY created_at DESC LIMIT 1").fetchone()[0]
    with pytest.raises(ValueError):
        M.handle_correct(ctx, {
            "oldAtomId": oldId, "newText": "x",
            "provenance": {"source": reserved},
        })


def test_correct_refuses_person_prefixed_source(ctx):
    """``lifecycle/supersede_detect.py`` EXCLUDES ``person-*`` rows from
    supersession candidates. A caller that can forge that prefix makes its atom
    invisible to that job, which is a durability claim it has no right to make.
    """
    _seedAtom(ctx)
    oldId = ctx.store._conn.execute(
        "SELECT id FROM atoms ORDER BY created_at DESC LIMIT 1").fetchone()[0]
    with pytest.raises(ValueError):
        M.handle_correct(ctx, {
            "oldAtomId": oldId, "newText": "x",
            "provenance": {"source": "person-gary"},
        })


def test_correct_still_accepts_the_ordinary_source(ctx):
    """Regression guard: the legitimate value must keep working."""
    _seedAtom(ctx)
    oldId = ctx.store._conn.execute(
        "SELECT id FROM atoms ORDER BY created_at DESC LIMIT 1").fetchone()[0]
    out = M.handle_correct(ctx, {
        "oldAtomId": oldId, "newText": "a corrected body",
        "provenance": {"source": "explicit-emit"},
    })
    assert "corrected" in out


def test_correct_refuses_a_sourceRef_that_escapes_its_root(ctx):
    """``enrich.locateChunk`` read_text()s a resolved sourceRef during recall,
    so a ref that leaves its import root turns recall into a local file read."""
    _seedAtom(ctx)
    oldId = ctx.store._conn.execute(
        "SELECT id FROM atoms ORDER BY created_at DESC LIMIT 1").fetchone()[0]
    with pytest.raises(ValueError):
        M.handle_correct(ctx, {
            "oldAtomId": oldId, "newText": "x",
            "provenance": {"sourceRef": "projects//etc/passwd"},
        })


# --------------------------------------------------------------------------- #
# resource bounds (a fast wrong answer is still wrong)
# --------------------------------------------------------------------------- #


def test_recall_rejects_negative_k(ctx):
    """``engine.py`` ends with ``assessed[:k]``. Python slicing makes ``k=-1``
    mean "everything except the last", so a negative k does not error, it
    quietly returns the entire fused candidate set at full body length."""
    with pytest.raises(ValueError):
        M.handle_recall(ctx, {"query": "anything", "k": -1})


def test_pensive_recall_rejects_negative_limit(ctx):
    with pytest.raises(ValueError):
        M.handle_pensive_recall(ctx, {"query": "anything", "limit": -1})


def test_recall_rejects_absurd_k(ctx):
    with pytest.raises(ValueError):
        M.handle_recall(ctx, {"query": "anything", "k": 10_000})


def test_recall_rejects_absurd_token_budget(ctx):
    with pytest.raises(ValueError):
        M.handle_recall(ctx, {"query": "anything", "tokenBudget": 10_000_000})


def test_recall_rejects_an_unbounded_query(ctx):
    """With the aux dense signal enabled this string is POSTed verbatim to a
    third party, so an unbounded query is an egress size hole as well as a
    compute one."""
    with pytest.raises(ValueError):
        M.handle_recall(ctx, {"query": "q" * 50_000})


def test_emit_rejects_a_body_over_the_byte_cap(ctx):
    """``_truncateWords`` splits on whitespace, so a single 200k-character token
    is "one word" and was stored whole, then handed to the embedder."""
    with pytest.raises(ValueError):
        M.handle_emit_narrative(ctx, {
            "project": "gatecheck", "narrative": "W" * 200_000,
        })


def test_emit_still_accepts_an_ordinary_body(ctx):
    """Regression guard: the cap must not be so tight it refuses real writes."""
    out = M.handle_emit_narrative(ctx, {
        "project": "gatecheck", "narrative": "an ordinary narrative fragment.",
    })
    assert "emitted" in out
