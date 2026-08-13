"""Double-write tee receiver: land every OLD-path emit in the v3 store too.

Phase 3's "make the parallel run real." The production MCP server
(``~/Projects/Engram/tools/pensive-mcp-server``) stays authoritative: it writes
to the old store first and always completes its own response. Behind the
``PENSIVE_V3_TEE`` flag it then fire-and-forgets the same emit payload to this
daemon's ``POST /tee/emit`` endpoint, which replays it through the Task 12 emit
handlers into the v3 store. The old write is authoritative; this is a shadow copy
for the A/B cutover evidence, nothing depends on it, and its failure must never
bleed back into the old path.

This module is the ONE place in v3 where failures are contained by design rather
than raised: the v3 components fail LOUD, but the tee boundary contains, counts,
and logs, because the old serving path must stay byte-identical whether or not the
tee succeeds. Every outcome updates a :class:`Counters` the daemon exposes at
``GET /status`` for the Phase 3 gate check.

Contract notes:

- Only the legacy EMIT tools are accepted (``_TEE_EMIT_TOOLS``); a recall belongs
  on ``/shadow/recall``, not here, so a non-emit tool name is a 400.
- A duplicate emit tee'd twice writes TWO rows, and that is CORRECT: the old path
  is authoritative and the distiller (a later task) owns dedup. The tee never
  dedups -- it faithfully mirrors whatever the old path accepted.
- The tee holds only the v3 ``store`` (through ``ctx``); it has no handle to the
  legacy socket/store, so by construction a v3 write failure cannot touch the old
  path. Containment here just means the old server's fire-and-forget POST never
  sees an exception or a hang.
- Every request must clear :func:`checkLocalWriteRequest` BEFORE it can write.
  See that function for why loopback binding is not, by itself, a defense.
"""
import hmac
import json
import os
import secrets
import threading
from pathlib import Path
from urllib.parse import urlsplit

from serve.mcp import dispatch

__all__ = [
    "Counters",
    "handleTeeEmit",
    "TEE_EMIT_TOOLS",
    "checkLocalWriteRequest",
    "checkAllowedHost",
    "allowedHostsFromEnv",
    "DEFAULT_ALLOWED_HOSTS",
    "checkRequestOrigin",
    "loadTeeSecret",
    "defaultTeeSecretPath",
    "LOCAL_WRITE_HEADER",
    "TEE_SECRET_FILE_ENV",
]

# The legacy emit tool names the tee accepts. These map 1:1 to the v3 compat emit
# handlers (serve.mcp.HANDLERS). pensive_recall / pensive_analytics / the v3
# natives are deliberately absent -- a recall is shadowed via /shadow/recall.
TEE_EMIT_TOOLS = frozenset({
    "engram_emit_atom",
    "engram_emit_discovery",
    "engram_emit_failure",
    "engram_emit_narrative",
    "engram_emit_snapshot",
})


class Counters:
    """Thread-safe tallies for the tee/shadow boundary, read at ``GET /status``.

    Four counters back the Phase 3 gate check:

    - ``teeReceived``  -- every ACCEPTED POST that reached ``/tee/emit``.
    - ``teeFailed``    -- of those, the ones that failed (malformed payload or a
      v3 store error). Successful writes = ``teeReceived - teeFailed``.
    - ``shadowLogged`` -- shadow recalls whose JSONL line was appended.
    - ``shadowFailed`` -- shadow recalls that failed (bad payload, recall error,
      or log-write error).

    Two more count requests refused by :func:`checkLocalWriteRequest` before they
    were allowed to mean anything:

    - ``teeRejected``    -- forged/unauthorized POSTs to ``/tee/emit``.
    - ``shadowRejected`` -- the same for ``/shadow/recall``.

    Rejections are deliberately kept OUT of the four gate counters. Those four
    answer "did every old-path emit land in v3?", and folding hostile traffic into
    them would let an outsider move the cutover gate's number at will, with a
    forged POST reading as a tee bug. An instrument an attacker can drive is not
    an instrument.

    Increments take a lock because the daemon may serve concurrent requests;
    reads of a single int are atomic enough for assertions/reporting, and
    :meth:`snapshot` takes the lock for a consistent view.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.teeReceived = 0
        self.teeFailed = 0
        self.shadowLogged = 0
        self.shadowFailed = 0
        self.teeRejected = 0
        self.shadowRejected = 0

    def _inc(self, name):
        with self._lock:
            setattr(self, name, getattr(self, name) + 1)

    def snapshot(self):
        with self._lock:
            return {
                "teeReceived": self.teeReceived,
                "teeFailed": self.teeFailed,
                "shadowLogged": self.shadowLogged,
                "shadowFailed": self.shadowFailed,
                "teeRejected": self.teeRejected,
                "shadowRejected": self.shadowRejected,
            }


# --------------------------------------------------------------------------- #
# The local-write guard                                                        #
# --------------------------------------------------------------------------- #
#
# Binding 127.0.0.1 keeps the daemon off the network. It does NOT make the write
# routes safe, and conflating the two is the failure this guard exists to prevent:
# a browser the operator is already running can reach loopback, and a "simple"
# cross-origin POST is transmitted with no preflight at all. CORS only stops the
# hostile page from READING the reply -- the write still lands. Since these routes
# feed the memory every agent on this box recalls as its own, a forged emit does
# not stay data; it becomes another agent's belief.
#
# Three independent guards, because each covers what the others cannot:
#
#   1. A shared secret in a CUSTOM header. Custom headers are not on the CORS
#      safelist, so a page that tries to set one triggers a preflight this server
#      never answers -- the request dies in the browser. The secret itself lives in
#      a 0600 file, which a page cannot read at all, so the guard still holds if a
#      permissive CORS middleware is ever added upstream of us. This is a loopback
#      capability check between processes of the same user, NOT an authentication
#      system (no identities, no sessions) -- the house "no DIY auth" rule points
#      web perimeters at Authelia, and this is not a web perimeter.
#   2. Content-Type must be application/json. text/plain,
#      x-www-form-urlencoded and multipart/form-data are exactly the three types a
#      no-preflight request is allowed to set, so refusing them removes the shape
#      of the attack even before the secret is considered.
#   3. Origin/Referer, when present, must be loopback. This is the DNS-rebinding
#      case: the attacker's hostname resolves to 127.0.0.1, the browser therefore
#      believes it is same-origin, and guard 2 stops applying.
#
# Every rejection returns a machine-readable ``reason``. A status code says an
# outcome, not a mechanism, and three guards can all produce a 4xx -- without the
# reason, a test asserting "403" cannot tell which guard fired, or whether the one
# it meant to exercise is even wired up.

# The custom header carrying the loopback secret. Renaming this is a cross-repo
# wire-contract change: the Engram-side forward has to match it.
LOCAL_WRITE_HEADER = "x-pensive-tee-secret"

# Env override naming the secret FILE (never the secret value). Keeping the value
# out of the environment keeps it out of /proc/<pid>/environ and out of every child
# process the daemon spawns; a path is not a credential.
TEE_SECRET_FILE_ENV = "PENSIVE_V3_TEE_SECRET_FILE"

# Hosts that count as "this machine" for an Origin/Referer.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})

# The only content type a write route accepts.
_JSON_CONTENT_TYPE = "application/json"


def defaultTeeSecretPath():
    """Resolve the secret file: ``$PENSIVE_V3_TEE_SECRET_FILE`` or the store dir.

    Defaults beside the v3 store (``~/.local/share/pensive-v3/tee.secret``) so it
    shares the store's lifetime and backup story. Tests point the env at tmp_path
    so they never read, create, or rotate the live daemon's secret.
    """
    override = os.environ.get(TEE_SECRET_FILE_ENV)
    if override:
        return Path(override)
    return Path.home() / ".local" / "share" / "pensive-v3" / "tee.secret"


def loadTeeSecret(path=None, create=False):
    """Read the loopback secret -> ``str`` or ``None``. Creates it only if asked.

    ``create`` defaults to FALSE, and that default is load-bearing. Generating the
    secret is a filesystem side effect, and :func:`serve.daemon.buildApp` is called
    by a dozen tests across three test files -- with creation on by default, merely
    building an app object wrote a credential into the real
    ``~/.local/share/pensive-v3/`` (observed exactly once, which is why this
    argument exists). Only the daemon's ``main()`` entry point provisions; every
    other caller reads what is already there.

    Self-provisioning at ``main()`` is still deliberate: a guard that needs a manual
    setup step is a guard someone disables the first time it blocks them. The file
    is created with mode 0600 BEFORE anything is written to it (the mode is passed
    to ``os.open`` alongside ``O_CREAT|O_EXCL``), so the secret is never briefly
    world-readable -- a chmod after the write would leave exactly that window.

    Returns ``None`` if the secret cannot be read, or is absent and ``create`` is
    False. Callers must treat ``None`` as "refuse writes", never as "skip the
    check": see :func:`checkLocalWriteRequest`.
    """
    path = Path(path) if path is not None else defaultTeeSecretPath()
    try:
        existing = path.read_text().strip()
        if existing:
            return existing
    except FileNotFoundError:
        pass
    except OSError:
        # Unreadable (bad perms, bad mount). Fail closed rather than regenerate:
        # silently replacing a secret we merely cannot read would lock out every
        # legitimate caller that already holds the real one.
        return None

    if not create:
        return None

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        generated = secrets.token_urlsafe(32)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(fd, generated.encode("utf-8"))
        finally:
            os.close(fd)
        return generated
    except FileExistsError:
        # Another process created it between our read and our create. Theirs wins.
        try:
            return path.read_text().strip() or None
        except OSError:
            return None
    except OSError:
        return None


def _normalizeHeaders(headers):
    """Lowercase-key a header mapping -> ``dict`` or ``None`` if unusable.

    Accepts Starlette's case-insensitive ``Headers`` and a plain dict alike. A
    caller that passes ``None`` (or anything without ``items()``) gets ``None``,
    which every guard below treats as "no request metadata" and refuses -- a write
    path must never be reachable by simply omitting the headers argument.
    """
    if headers is None:
        return None
    try:
        return {str(k).lower(): v for k, v in headers.items()}
    except AttributeError:
        return None


def _isLoopbackUrl(value):
    """True only for an ``http(s)`` URL whose host is this machine.

    ``Origin: null`` -- what a sandboxed iframe or a ``file://`` page sends -- has
    no scheme and no hostname, so it lands here as False. Treating an unparseable
    origin as "absent, therefore fine" is the classic bypass for this check.
    """
    parts = urlsplit(value or "")
    if parts.scheme not in ("http", "https"):
        return False
    return (parts.hostname or "").lower() in _LOOPBACK_HOSTS


# Hostnames this daemon will answer to. Loopback ONLY by default: a daemon that
# ships trusting a public hostname trusts it in every deployment it ever lands
# in. A deployment that fronts this with a reverse proxy adds its own hostname
# through PENSIVE_V3_ALLOWED_HOSTS, which EXTENDS this set rather than replacing
# it -- replacing would let a config typo drop loopback and lock every local
# agent out of its own memory.
DEFAULT_ALLOWED_HOSTS = frozenset({"127.0.0.1", "localhost", "[::1]", "::1"})

ALLOWED_HOSTS_ENV = "PENSIVE_V3_ALLOWED_HOSTS"


def allowedHostsFromEnv():
    """The configured host allowlist: the loopback defaults plus any extras."""
    hosts = set(DEFAULT_ALLOWED_HOSTS)
    for entry in (os.environ.get(ALLOWED_HOSTS_ENV) or "").split(","):
        entry = entry.strip().lower()
        if entry:
            hosts.add(entry)
    return hosts


def _hostnameOf(hostHeader):
    """Host header -> bare hostname, port removed, IPv6 brackets preserved."""
    value = hostHeader.strip().lower()
    if value.startswith("["):
        end = value.find("]")
        return value[:end + 1] if end != -1 else value
    return value.split(":", 1)[0]


def checkAllowedHost(headers, allowed):
    """Host half of the guard -> ``None`` to allow, else ``(status, body)``.

    Closes the DNS-rebinding path the Origin check cannot: the attacker points a
    hostname they control at 127.0.0.1, the browser then treats the response as
    same-origin so CORS never applies, and a GET is permitted to carry no Origin
    at all. Host is what the browser fills in from the URL it thinks it is
    addressing, so an unknown value there IS the attack, and unlike Origin its
    ABSENCE is refused rather than allowed -- HTTP/1.1 requires it, so a missing
    Host is a malformed request, not a local process being terse.

    Matched on the bare hostname: the port is deployment detail, and pinning it
    would break the moment anyone moved the daemon.
    """
    norm = _normalizeHeaders(headers)
    if norm is None:
        return 403, {"error": "request metadata unavailable",
                     "reason": "no-request-headers"}
    host = norm.get("host")
    if not host:
        return 403, {"error": "missing Host header", "reason": "no-host"}
    name = _hostnameOf(host)
    if name not in allowed:
        return 403, {
            "error": f"unrecognized Host refused: {host!r}",
            "reason": "unknown-host",
        }
    return None


def checkRequestOrigin(headers):
    """Origin/Referer half of the guard -> ``None`` to allow, else ``(status, body)``.

    Split out from :func:`checkLocalWriteRequest` because ``/mcp`` needs THIS check
    and only this one: the MCP SDK already requires ``application/json`` (which a
    no-preflight request cannot set), and every live agent on this box talks to
    ``/mcp`` without a secret. Non-browser clients send no Origin at all, so this is
    invisible to them while still closing the DNS-rebinding path.

    Absent Origin AND Referer is allowed: that is what a local process sends. The
    check is "if the browser told us where this came from, it had better be here".

    LIMIT, corrected 2026-08-13: this does NOT by itself close DNS rebinding, as
    an earlier version of this docstring claimed. A rebound same-origin GET may
    omit Origin entirely and suppress Referer, and the permitted absence above is
    exactly the gap it walks through. :func:`checkAllowedHost` is the half that
    closes it, because Host is the one header the browser sets from the URL it
    believes it is talking to and the page cannot forge away.
    """
    norm = _normalizeHeaders(headers)
    if norm is None:
        return 403, {"error": "request metadata unavailable",
                     "reason": "no-request-headers"}

    for name in ("origin", "referer"):
        value = norm.get(name)
        if value and not _isLoopbackUrl(value):
            return 403, {
                "error": f"cross-origin {name} refused: {value!r}",
                "reason": "cross-origin",
            }
    return None


def checkLocalWriteRequest(headers, secret):
    """Full guard for a state-changing route -> ``None`` to allow, else ``(status, body)``.

    Order is origin -> content-type -> secret, and it is load-bearing for
    diagnosis: the broadest, most obviously-hostile signal is reported first, so a
    forged request is labelled ``cross-origin`` rather than the incidental
    ``missing-local-secret`` it would also have triggered.

    ``secret`` of ``None`` means the daemon could not load its own secret. That
    fails CLOSED (503). A guard that waves requests through when its configuration
    is missing is worse than no guard at all, because from the outside it reads as
    armed.
    """
    rejection = checkRequestOrigin(headers)
    if rejection is not None:
        return rejection
    norm = _normalizeHeaders(headers)

    # Only application/json. This is the guard that removes the SHAPE of the
    # attack: the three types a cross-origin POST can set without a preflight are
    # all refused here, whatever else the request claims.
    contentType = (norm.get("content-type") or "").split(";", 1)[0].strip().lower()
    if contentType != _JSON_CONTENT_TYPE:
        return 415, {
            "error": f"writes require Content-Type {_JSON_CONTENT_TYPE}, got "
                     f"{contentType!r}",
            "reason": "bad-content-type",
        }

    if not secret:
        return 503, {"error": "loopback write secret unavailable; refusing writes",
                     "reason": "local-secret-unavailable"}

    provided = norm.get(LOCAL_WRITE_HEADER)
    if not provided:
        return 403, {
            "error": f"missing {LOCAL_WRITE_HEADER} header",
            "reason": "missing-local-secret",
        }
    # Constant-time: a plain == leaks the shared prefix length through timing, and
    # a local attacker is exactly who is positioned to measure that.
    if not hmac.compare_digest(str(provided), str(secret)):
        return 403, {
            "error": f"invalid {LOCAL_WRITE_HEADER} header",
            "reason": "bad-local-secret",
        }
    return None


def handleTeeEmit(ctx, counters, rawBody, headers, secret):
    """Replay one tee'd emit into the v3 store -> ``(httpStatus, bodyDict)``.

    ``rawBody`` is the request body bytes: a JSON object ``{"tool": <emit name>,
    "args": <the legacy tool arguments>}`` -- exactly the shape the Engram-side
    forward sends. ``headers`` is the request's header mapping and ``secret`` the
    daemon's loopback secret; both are REQUIRED, and both are consulted before
    anything is written.

    The guard lives here, in the handler, rather than only in the Starlette route,
    so a future route that also replays emits cannot reintroduce the hole by
    forgetting to call it. Refused requests increment ``teeRejected`` and stop --
    they never touch ``teeReceived``/``teeFailed``, which belong to the Phase 3
    gate (see :class:`Counters`).

    Past the guard: every call increments ``teeReceived``; any failure also
    increments ``teeFailed``. Never raises -- a malformed payload is a 400, a v3
    store error is a 500, and both come back as a normal return so the old
    server's fire-and-forget POST is never disturbed.
    """
    rejection = checkLocalWriteRequest(headers, secret)
    if rejection is not None:
        counters._inc("teeRejected")
        return rejection

    counters._inc("teeReceived")

    try:
        parsed = json.loads(rawBody)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        counters._inc("teeFailed")
        return 400, {"error": f"malformed tee payload: {exc}"}
    if not isinstance(parsed, dict):
        counters._inc("teeFailed")
        return 400, {"error": "tee payload must be a JSON object"}

    tool = parsed.get("tool")
    args = parsed.get("args")
    if tool not in TEE_EMIT_TOOLS:
        counters._inc("teeFailed")
        return 400, {"error": f"tee: unknown or non-emit tool {tool!r}"}
    if not isinstance(args, dict):
        counters._inc("teeFailed")
        return 400, {"error": "tee: 'args' must be a JSON object"}

    # dispatch() already contains handler exceptions and returns (text, isError),
    # so a v3 store error surfaces as isError. The try/except is belt-and-suspenders
    # for an UNMODELED raise (a bug in dispatch, an error before it returns): the
    # boundary must still count it as teeFailed and return a clean 500, never let a
    # raw exception escape to Starlette (which would 500 with the counter stuck at
    # "received but not failed").
    try:
        text, isError = dispatch(ctx, tool, args)
    except Exception as exc:  # noqa: BLE001 -- contained boundary, must not bleed
        counters._inc("teeFailed")
        return 500, {"error": f"tee dispatch failed: {exc}"}
    if isError:
        counters._inc("teeFailed")
        return 500, {"error": text}
    return 200, {"ok": True, "result": text}
