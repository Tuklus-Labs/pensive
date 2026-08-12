"""Gate for connection-declared agent attribution (``?agent=`` on the MCP URL).

THE CLAIM UNDER TEST, stated so a drifted question reads as a wrong sentence:

    An MCP client that declares an identity in its connection URL produces
    provenance rows stamped with that identity, without any caller passing an
    ``agent`` argument -- and a client that declares nothing produces NULL,
    not an accidental default.

Why this is an END-TO-END test against a real daemon rather than a unit test
with a stubbed context: the whole mechanism depends on the MCP SDK populating
``RequestContext.request`` under ``stateless=True``. A unit test that sets the
contextvar by hand would prove only that our code reads a contextvar we
ourselves set, which is the question we already know the answer to. The open
question is what the SDK DOES, so the gate asks the SDK (STYLE.md, "ask the
system what it DOES, not what it was handed").

All four live probes share one daemon boot; results are collected and asserted
at the end so a failure reports every probe rather than only the first
(STYLE.md, "accumulate failures").
"""
import asyncio
import json
import os
import socket
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

from store.store import openStore

SRC_ROOT = str(Path(__file__).resolve().parents[2] / "src")


def _free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _subprocess_output(logFile):
    offset = logFile.tell()
    logFile.seek(0)
    output = logFile.read().decode(errors="replace")
    logFile.seek(offset)
    return output


def _wait_for_port(host, port, proc, logFile, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise AssertionError(
                f"daemon exited early (rc={proc.returncode}):\n"
                f"{_subprocess_output(logFile)}")
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError:
            time.sleep(0.5)
    raise AssertionError(
        f"daemon did not bind the port in time:\n{_subprocess_output(logFile)}")


def _shutdown(proc):
    """SIGINT first, per the house rule; SIGKILL only as a last resort."""
    proc.send_signal(2)
    try:
        return proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=30)
        return -9


async def _emit_via_url(url, principle, agentArg=None):
    """One emit over the real streamable-HTTP transport at ``url``."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    args = {"project": "gate", "principle": principle}
    if agentArg is not None:
        args["agent"] = agentArg
    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await session.call_tool("engram_emit_discovery", args)


def _emit(url, principle, agentArg=None):
    return asyncio.run(_emit_via_url(url, principle, agentArg))


def _agent_for(dbPath, principle):
    """Ask the store who authored the atom carrying this principle.

    Returns the sentinel ``"<MISSING>"`` when the atom never landed, so a failed
    emit cannot be misread as an unattributed one: those are different bugs and
    an empty result must never be quiet (STYLE.md).
    """
    conn = sqlite3.connect(f"file:{dbPath}?mode=ro", uri=True)
    try:
        row = conn.execute(
            "SELECT p.agent FROM provenance p JOIN atoms a ON a.id = p.atom_id "
            "WHERE a.text LIKE ? ORDER BY p.recorded_at DESC LIMIT 1",
            (f"%{principle}%",),
        ).fetchone()
        if row is None:
            return "<MISSING>"
        return row[0]
    finally:
        conn.close()


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_connection_declared_agent_is_stamped_without_caller_cooperation(tmp_path):
    dbPath = tmp_path / "attrib.db"
    openStore(dbPath).close()

    port = _free_port()
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    env["PENSIVE_V3_STORE"] = str(dbPath)
    env["PENSIVE_V3_PORT"] = str(port)
    # No PENSIVE_V3_AGENT: the process-wide default must stay out of the way so
    # the undeclared probe below is a true negative rather than an env artifact.
    env.pop("PENSIVE_V3_AGENT", None)
    env["HIP_VISIBLE_DEVICES"] = ""
    env["CUDA_VISIBLE_DEVICES"] = ""

    base = f"http://127.0.0.1:{port}/mcp"
    results = {}

    logPath = tmp_path / "daemon.log"
    with logPath.open("w+b") as daemonLog:
        proc = subprocess.Popen(
            [sys.executable, "-m", "serve.daemon"],
            cwd=SRC_ROOT, env=env, stdout=daemonLog, stderr=subprocess.STDOUT)
        try:
            _wait_for_port("127.0.0.1", port, proc, daemonLog, timeout=300)

            # P1 the load-bearing claim: connection declares, nobody passes an arg
            _emit(f"{base}?agent=probe-grok", "PROBE_DECLARED alpha")
            # P2 true negative: no declaration anywhere -> honestly unattributed
            _emit(base, "PROBE_UNDECLARED bravo")
            # P3 precedence: an explicit per-emit arg beats the connection
            _emit(f"{base}?agent=probe-grok", "PROBE_PRECEDENCE charlie",
                  agentArg="probe-explicit")
            # P4 the column is guarded: a path is not an identity
            _emit(f"{base}?agent=/root/not_an_identity", "PROBE_MALFORMED delta")
            # P5 an over-long declaration is refused rather than truncated
            _emit(f"{base}?agent={'x' * 200}", "PROBE_TOOLONG echo")
        finally:
            rc = _shutdown(proc)

    for key, needle in (("declared", "PROBE_DECLARED alpha"),
                        ("undeclared", "PROBE_UNDECLARED bravo"),
                        ("precedence", "PROBE_PRECEDENCE charlie"),
                        ("malformed", "PROBE_MALFORMED delta"),
                        ("toolong", "PROBE_TOOLONG echo")):
        results[key] = _agent_for(dbPath, needle)

    failures = []

    def expect(name, got, want, why):
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r} -- {why}")

    expect("P1 connection-declared", results["declared"], "probe-grok",
           "a client declaring ?agent= must be stamped with no caller cooperation; "
           "if this is None the SDK is not exposing RequestContext.request under "
           "stateless=True and the whole mechanism is inert")
    expect("P2 undeclared", results["undeclared"], None,
           "a client declaring nothing must stay honestly unattributed, never "
           "inherit a neighbouring connection's identity")
    expect("P3 precedence", results["precedence"], "probe-explicit",
           "an explicit per-emit argument outranks the per-connection default")
    expect("P4 malformed", results["malformed"], None,
           "a filesystem path is not an identity and must be refused at the write "
           "boundary, not stored and cleaned up later")
    expect("P5 over-long", results["toolong"], None,
           "an over-long declaration is refused whole, never silently truncated "
           "into a different identity than the one declared")

    if rc != 0:
        failures.append(f"clean-SIGINT rule violated: returnCode={rc}")

    assert not failures, (
        "connection-attribution gate failed:\n  " + "\n  ".join(failures)
        + f"\n\nobserved: {json.dumps(results, default=str)}"
    )


# --------------------------------------------------------------------------- #
# Unit half: the sanitizer's own discrimination, both directions.              #
# Cheap, and it re-proves itself on every run, but it does NOT subsume the      #
# live gate above: it cannot tell whether the SDK hands us a request at all.    #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("raw,want", [
    ("heph", "heph"),
    ("  grok  ", "grok"),                       # trimmed
    ("gpt-5.6-sol", "gpt-5.6-sol"),
    ("Muse (muse-glimmer-30b)", "Muse (muse-glimmer-30b)"),
    ("", None),                                  # empty is not an answer
    ("   ", None),
    (None, None),
    (123, None),                                 # wrong type
    ("/root/v3_publication_controller", None),   # the real leaked shape
    ("./relative_runner", None),
    ("windows\\path", None),
    ("has\nnewline", None),                      # control byte
    ("x" * 65, None),                            # over the cap
    ("x" * 64, "x" * 64),                        # exactly at the cap is fine
])
def test_sanitize_agent_discriminates(raw, want):
    from serve.mcp import _sanitizeAgent
    assert _sanitizeAgent(raw) == want, f"_sanitizeAgent({raw!r})"


def test_resolve_agent_precedence_without_a_transport():
    """Off-transport (stdio, tee forward), behavior is exactly as before.

    _transportAgent returns None when there is no request context, so this
    pins the pre-existing contract: caller arg, else ctx.agent, else None.
    """
    from serve.mcp import _resolveAgent

    class Ctx:
        def __init__(self, agent):
            self.agent = agent

    assert _resolveAgent(Ctx(None), None) is None
    assert _resolveAgent(Ctx("daemon-wide"), None) == "daemon-wide"
    assert _resolveAgent(Ctx("daemon-wide"), {"agent": "caller"}) == "caller"
    assert _resolveAgent(Ctx("daemon-wide"), {"agent": "   "}) == "daemon-wide"
    assert _resolveAgent(Ctx(None), {"agent": "caller"}) == "caller"
