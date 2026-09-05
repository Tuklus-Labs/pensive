"""The resident MCP daemon: load the models once, serve over localhost.

Phase 3's "make it servable" host. One long-lived process loads the store, the
embedder, the cross-encoder reranker, and the dense index ONCE at startup, then
serves the :mod:`serve.mcp` tool surface over MCP Streamable-HTTP bound to
127.0.0.1 only. This is the same SDK transport the production pensive server uses
for its network mode, on a DIFFERENT port and WITHOUT the tailnet bearer gate --
localhost-only means no external reachability, so there is no DIY auth to get
wrong (a web exposure, which this is not, would sit behind Authelia forward-auth).

It is deliberately a NEW process on a NEW port, off to the side: it never touches
the running production MCP server or any live session config. Wiring an agent to
it is the Gary-gated cutover (Task 22), not this task. Emits land in the v3 store
ONLY.

Configuration (all env, documented so nothing is a mystery):

- ``PENSIVE_V3_STORE``  -- store path. Default ``~/.local/share/pensive-v3/shadow.db``
  (a daemon-local DEV path; NEVER a production store). Parent dirs are created.
- ``PENSIVE_V3_PORT``   -- localhost port. Default 5999 (adjacent to the legacy
  net server's 5998, clearly the v3 shadow sibling).
- ``PENSIVE_V3_MODEL``  -- embedding model id. Default ``BAAI/bge-small-en-v1.5``.
- ``PENSIVE_V3_AGENT``  -- agent name stamped into emit provenance. Default unset
  (NULL agent).
- ``PENSIVE_V3_OPENAI_MODEL`` -- aux dense model id (e.g.
  ``text-embedding-3-large``). Default unset = feature off, recall byte-identical
  to the two-signal engine. When set, its backfilled vectors (see
  ``~/Projects/pensive-embeddings/``) fuse as a third recall signal over the
  memory class; a missing key/SDK logs and disables rather than failing startup.

Shutdown: uvicorn installs SIGINT/SIGTERM handlers and drains cleanly, so a
SIGINT stops the daemon with exit 0 (the house "SIGINT first" rule).
"""
import contextlib
import os
import sys
from pathlib import Path

# daemon/src on sys.path so `serve`, `recall`, `store` import both under `-m
# serve.daemon` (cwd=src) and when launched from elsewhere.
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from serve.mcp import ServeContext, buildServer, SERVER_NAME  # noqa: E402
from serve.tee import (  # noqa: E402
    Counters, handleTeeEmit, checkLocalWriteRequest, checkRequestOrigin,
    checkAllowedHost, allowedHostsFromEnv,
    checkDeclaredBodyLength, readBoundedBody, bodyTooLarge,
    loadTeeSecret, defaultTeeSecretPath, LOCAL_WRITE_HEADER,
)
from serve.shadow import runShadow, defaultShadowLogPath  # noqa: E402
from serve.l1 import handleGet, handleLookup  # noqa: E402
from recall.engine import recall as recallEngine, TIERS  # noqa: E402
from store.store import logRecall  # noqa: E402
from serve import viz  # noqa: E402
from ambient.briefer import brief, DEFAULT_BUDGET  # noqa: E402
from store.store import openStore  # noqa: E402
from recall.embedder import makeEmbedder  # noqa: E402

_DEFAULT_STORE = Path.home() / ".local" / "share" / "pensive-v3" / "shadow.db"
_DEFAULT_PORT = 5999
_DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
# Bind localhost ONLY -- never an external interface (the brief's hard rule).
_HOST = "127.0.0.1"


def _log(msg):
    print(f"[{SERVER_NAME}] {msg}", file=sys.stderr, flush=True)


def _setProcTitle():
    """Name the process for btop/htop per the house daemon-naming rule."""
    try:
        import setproctitle

        setproctitle.setproctitle(SERVER_NAME)
    except Exception as exc:  # pragma: no cover - cosmetic only
        _log(f"setproctitle unavailable ({exc}); process name unchanged")


def _warmReranker():
    """Load the cross-encoder if it will import. Never take the daemon down.

    Serve-path L2/L3 do not run the reranker (CPU, 3164ms for 50 pairs). Startup
    still used to import it unconditionally, so a torchcodec/FFmpeg mismatch
    crash-looped pensive-v3 on every restart while the already-running process
    kept answering. Fail open: log, serve without it.
    """
    try:
        from recall.rerank import _getReranker
        _getReranker()
        return True
    except Exception as exc:
        _log(f"cross-encoder warmup failed; serving without rerank ({exc})")
        return False


def buildContext():
    """Open the store, load the models, warm the reranker, build the index."""
    storePath = Path(os.environ.get("PENSIVE_V3_STORE", str(_DEFAULT_STORE)))
    storePath.parent.mkdir(parents=True, exist_ok=True)
    modelId = os.environ.get("PENSIVE_V3_MODEL", _DEFAULT_MODEL)
    agent = os.environ.get("PENSIVE_V3_AGENT") or None

    _log(f"opening store {storePath}")
    store = openStore(storePath)

    _log(f"loading embedder {modelId}")
    # makeEmbedder, not Embedder: it returns the ONNX runtime when
    # PENSIVE_V3_ONNX_MODEL points at an exported graph of this same model, and
    # the torch path otherwise. Same weights, same embedding space, same model
    # id, so stored vectors stay valid either way (see OnnxEmbedder).
    embedder = makeEmbedder(modelId)
    # Which runtime actually loaded is not inferable from the line above, and an
    # operator who set the env var needs to see whether it took. The type name
    # is the honest answer; the model id is identical in both cases by design.
    _log(f"embedder runtime: {type(embedder).__name__} on {embedder.device}")

    _log("warming cross-encoder reranker")
    _warmReranker()

    # Aux dense signal (optional): construction failure means feature-off, never
    # a dead daemon -- recall must come up on base signals no matter what.
    aux = None
    openaiModel = os.environ.get("PENSIVE_V3_OPENAI_MODEL")
    if openaiModel:
        try:
            from recall.aux_dense import AuxDense, OpenAIEmbedder

            aux = AuxDense(OpenAIEmbedder(openaiModel))
            _log(f"aux dense signal enabled: {openaiModel}")
        except Exception as exc:
            _log(f"aux dense signal DISABLED ({exc}); serving base signals only")
            aux = None

    _log("embedding backlog + building dense index")
    ctx = ServeContext(store, embedder, modelId, agent=agent, aux=aux)
    _log("context ready")
    return ctx


def buildApp(ctx):
    """A Starlette ASGI app hosting the low-level MCP server over Streamable-HTTP.

    Stateless + JSON responses keep the localhost transport simple (no session-id
    ceremony, plain JSON round-trips). The session manager's ``run()`` context is
    driven by the app lifespan so it is active for the life of the server.

    Alongside the ``/mcp`` mount the app carries the Phase 3 double-write surface,
    all localhost-only (same bind): ``POST /tee/emit`` replays a tee'd emit into
    the v3 store, ``POST /shadow/recall`` logs a v3 answer beside the old one, and
    ``GET /status`` exposes the tee/shadow counters for the gate check.

    Both POST routes sit behind :func:`serve.tee.checkLocalWriteRequest`, which
    runs BEFORE their body is read, and ``/tee/emit`` additionally bounds that
    body at :data:`serve.tee.MAX_TEE_BODY_BYTES`. The ``/mcp`` mount sits behind
    the guard's origin half. Loopback binding keeps the daemon off
    the network but is NOT a defense against a browser the operator is already
    running: a page can POST cross-origin to 127.0.0.1 with no preflight, and CORS
    only stops it from reading the reply. See the guard's own comments in
    ``serve/tee.py`` for the full model. The read routes (``/status``, ``/brief``,
    ``/viz*``) are deliberately ungated -- they change nothing, and gating ``/brief``
    would break the SessionStart hook that curls it for no security gain.

    The tee/shadow boundary handlers are synchronous and run inline on the
    event-loop thread -- the SAME thread that created the sqlite store (and the
    same path the MCP ``call_tool`` handler already takes) -- so the store's
    thread affinity is preserved; they do not go through a threadpool.
    """
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from starlette.applications import Starlette
    from starlette.datastructures import Headers
    from starlette.middleware import Middleware
    from starlette.responses import JSONResponse
    from starlette.routing import Mount, Route

    server = buildServer(ctx)
    manager = StreamableHTTPSessionManager(
        app=server, json_response=True, stateless=True,
    )

    counters = Counters()
    shadowLogPath = defaultShadowLogPath()

    # The loopback write secret, loaded ONCE at app build. Re-reading it per
    # request would put a filesystem call on the hot path and, worse, would make
    # the daemon silently adopt a rotated secret mid-flight while its callers still
    # held the old one. A None here is not fatal to startup: the guard fails closed
    # on it, so /mcp and the read routes keep serving while the write routes refuse.
    teeSecret = loadTeeSecret()
    if teeSecret is None:
        _log(f"WARNING: no loopback write secret ({defaultTeeSecretPath()}); "
             "POST /tee/emit and POST /shadow/recall will refuse every request")

    async def handle_mcp(scope, receive, send):
        # /mcp dispatches the SAME emit tools as the tee, so it is a write surface
        # too. It is already hard to forge from a page (the SDK requires
        # application/json, which a no-preflight request cannot set), but that
        # leaves DNS rebinding, where the browser thinks it is same-origin and the
        # content-type barrier stops applying. An Origin check is the documented
        # MCP defense and is invisible to every live agent here: non-browser
        # clients send no Origin. Deliberately origin-ONLY -- requiring the tee
        # secret would break every wired agent on this box.
        rejection = checkRequestOrigin(Headers(scope=scope))
        if rejection is not None:
            status, payload = rejection
            await JSONResponse(payload, status_code=status)(scope, receive, send)
            return
        await manager.handle_request(scope, receive, send)

    async def l1_get(request):
        # L1: identity retrieval, budget P95 <= 1ms. Deliberately NOT an MCP
        # tool -- the tools/call envelope alone measured 2.486ms P95 for a
        # 142-byte error touching no store, so the transport choice IS the
        # design, not an optimization detail.
        #
        # Origin-checked but not secret-guarded: this is a READ, and the read
        # routes (/brief, /status) already work this way. The origin check is
        # the same DNS-rebinding defense /mcp applies and is invisible to every
        # non-browser client, which sends no Origin at all.
        rejection = checkRequestOrigin(request.headers)
        if rejection is not None:
            status, payload = rejection
            return JSONResponse(payload, status_code=status)
        status, payload = handleGet(
            ctx.store,
            request.query_params.get("id", ""),
            withProvenance=request.query_params.get("full") == "1",
        )
        return JSONResponse(payload, status_code=status)

    async def l1_lookup(request):
        # The operation Charon needed and did not have: exact set membership by
        # facet. A similarity engine cannot report absence, so it returned the
        # nearest thing instead, which was its own source code.
        rejection = checkRequestOrigin(request.headers)
        if rejection is not None:
            status, payload = rejection
            return JSONResponse(payload, status_code=status)
        status, payload = handleLookup(
            ctx.store,
            request.query_params.get("key", ""),
            request.query_params.get("value", ""),
            request.query_params.get("limit", 100),
        )
        return JSONResponse(payload, status_code=status)

    async def lean_recall(request):
        # L2/L3 over a plain route, for the same reason L1 is not an MCP tool.
        #
        # Measured on this daemon, 95-char query, 4KB response: the same recall
        # costs 16.66ms in-process and 24.62ms through MCP tools/call, so the
        # envelope is ~7.6ms of pydantic and JSON-RPC. Against a 20ms L2 budget
        # that is not an optimization detail, it is most of the gap. The lean
        # route serves a 947-byte response in 0.13ms.
        #
        # /mcp stays exactly as it is: every wired agent on this box speaks it,
        # and this is an ADDITIONAL door for callers that need the budget, not a
        # replacement. Origin- and Host-checked like every other read here.
        rejection = checkRequestOrigin(request.headers)
        if rejection is not None:
            status, payload = rejection
            return JSONResponse(payload, status_code=status)
        qp = request.query_params
        query = qp.get("q", "")
        if not query.strip():
            return JSONResponse({"error": "q is required"}, status_code=400)
        tier = qp.get("tier", "L2")
        if tier not in TIERS:
            return JSONResponse(
                {"error": f"tier must be one of {list(TIERS)}"}, status_code=400)
        try:
            k = int(qp.get("k", 10))
            budget = int(qp.get("budget", 1500))
        except ValueError:
            return JSONResponse({"error": "k and budget must be integers"},
                                status_code=400)
        if not 1 <= k <= 200 or not 1 <= budget <= 8000:
            return JSONResponse({"error": "k must be 1..200, budget 1..8000"},
                                status_code=400)
        if len(query) > 8192:
            return JSONResponse({"error": "q exceeds 8192 characters"},
                                status_code=400)
        out = recallEngine(
            ctx.store, ctx.indexes, ctx.embedder, query,
            k=k, tokenBudget=budget, aux=ctx.aux, tier=tier,
        )
        try:
            logRecall(ctx.store, [r["atomId"] for r in out["results"]],
                      query=query, sourceRef=f"lean.recall.{tier}")
        except Exception:  # noqa: BLE001 -- serving wins over telemetry
            pass
        return JSONResponse({"tier": tier, "payload": out["payload"],
                             "count": len(out["results"])})

    async def tee_emit(request):
        # ORDER IS THE FIX. This used to `await request.body()` first and call
        # handleTeeEmit (where the guard lives) second, so an unauthenticated
        # POST -- one already destined to be refused for its Origin, its
        # Content-Type or a missing secret -- got to allocate whatever it cared
        # to send BEFORE anything checked it: Request.body() concatenates every
        # chunk with no cap. The guard was armed and the attack had already run.
        # /shadow/recall below has always had this order; the tee had not.
        #
        # handleTeeEmit still runs the same guard itself. That is deliberate
        # defense in depth, not a leftover: it keeps a future route that also
        # replays emits from reintroducing the hole by forgetting to guard. The
        # rejection COUNTER stays here at the route, because a request refused
        # here never reaches the handler and must be counted exactly once.
        rejection = checkLocalWriteRequest(request.headers, teeSecret)
        if rejection is not None:
            counters._inc("teeRejected")
            status, payload = rejection
            return JSONResponse(payload, status_code=status)

        # Past the guard the caller is authenticated -- it read a 0600 file --
        # but authenticated is not unlimited. This handler runs INLINE on the
        # event-loop thread that also serves /mcp and every read route, so an
        # oversized body is a stall for every other caller on this box.
        rejection = checkDeclaredBodyLength(request.headers)
        body = None
        if rejection is None:
            body, overflowed = await readBoundedBody(request.stream())
            if overflowed:
                rejection = bodyTooLarge()
        if rejection is not None:
            # teeReceived/teeFailed ARE the Phase 3 gate ("did every old-path
            # emit land in v3?"). This request cleared the guard, so it came
            # from the real forwarder and is a genuine emit that did not land:
            # the gate has to see it. teeRejected is reserved for traffic that
            # never proved it was local, and putting a real dropped emit there
            # would hide the gap the gate exists to find.
            counters._inc("teeReceived")
            counters._inc("teeFailed")
            status, payload = rejection
            return JSONResponse(payload, status_code=status)

        status, payload = handleTeeEmit(
            ctx, counters, body, request.headers, teeSecret)
        return JSONResponse(payload, status_code=status)

    async def shadow_recall(request):
        # runShadow lives in shadow.py and takes no request metadata, so unlike the
        # tee its guard has to sit here at the route. Same three checks, same
        # reasons: /shadow/recall writes the A/B corpus the cutover decision is read
        # off, and a forged line there is evidence tampering.
        rejection = checkLocalWriteRequest(request.headers, teeSecret)
        if rejection is not None:
            counters._inc("shadowRejected")
            status, payload = rejection
            return JSONResponse(payload, status_code=status)
        body = await request.body()
        status, payload = runShadow(ctx, counters, body, shadowLogPath)
        return JSONResponse(payload, status_code=status)

    async def status(request):
        return JSONResponse({"server": SERVER_NAME, "counters": counters.snapshot()})

    async def brief_endpoint(request):
        # The Phase 4 session-start working set as a VIEW over the store. GET so a
        # SessionStart hook can curl it; agent, project, taskId and budget are
        # query params. brief() performs zero writes, so this handler is read-only
        # like /status.
        agent = request.query_params.get("agent")
        if isinstance(agent, str):
            agent = agent.strip() or None
        taskId = request.query_params.get("taskId")
        if isinstance(taskId, str):
            taskId = taskId.strip() or None
        project = request.query_params.get("project")
        if isinstance(project, str):
            project = project.strip() or None
        if taskId is not None and len(taskId) > 256:
            return JSONResponse(
                {"error": "taskId exceeds maximum length 256"}, status_code=400)
        if project is not None and len(project) > 256:
            return JSONResponse(
                {"error": "project exceeds maximum length 256"}, status_code=400)
        budgetRaw = request.query_params.get("budget")
        if budgetRaw in (None, ""):
            budget = DEFAULT_BUDGET
        else:
            try:
                budget = int(budgetRaw)
            except (TypeError, ValueError):
                return JSONResponse(
                    {"error": "budget must be an integer"}, status_code=400)
            if budget < 1:
                return JSONResponse(
                    {"error": "budget must be >= 1"}, status_code=400)
        options = {"agent": agent, "budget": budget}
        if taskId is not None:
            options.update({"project": project, "taskId": taskId})
        text = brief(ctx.store, options)
        payload = {"brief": text, "agent": agent, "budget": budget}
        if taskId is not None:
            payload.update({"project": project, "taskId": taskId})
        return JSONResponse(payload)

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        async with manager.run():
            yield

    # Host allowlist, applied as MIDDLEWARE rather than per route.
    #
    # A per-route check is a hand-maintained coverage list, and those fail in the
    # same direction as the defect they exist to catch: whoever adds the next
    # route is the same person who forgets to guard it. Middleware derives the
    # coverage from the app instead, so a route cannot be added unguarded.
    #
    # Closes the DNS-rebinding path that the Origin check structurally cannot:
    # a rebound same-origin GET may omit Origin and suppress Referer, and that
    # absence is deliberately ALLOWED because every non-browser client on this
    # box sends neither. Host is the header the page cannot forge away.
    allowedHosts = allowedHostsFromEnv()
    _log(f"host allowlist: {sorted(allowedHosts)}")

    class HostGuard:
        def __init__(self, app):
            self.app = app

        async def __call__(self, scope, receive, send):
            if scope["type"] != "http":
                await self.app(scope, receive, send)
                return
            rejection = checkAllowedHost(Headers(scope=scope), allowedHosts)
            if rejection is not None:
                status, payload = rejection
                await JSONResponse(payload, status_code=status)(scope, receive, send)
                return
            await self.app(scope, receive, send)

    app = Starlette(
        middleware=[Middleware(HostGuard)],
        routes=[
            Route("/tee/emit", tee_emit, methods=["POST"]),
            Route("/shadow/recall", shadow_recall, methods=["POST"]),
            Route("/recall", lean_recall, methods=["GET"]),
            Route("/get", l1_get, methods=["GET"]),
            Route("/lookup", l1_lookup, methods=["GET"]),
            Route("/status", status, methods=["GET"]),
            Route("/brief", brief_endpoint, methods=["GET"]),
            Route("/viz", viz.staticPage, methods=["GET"]),
            Route("/viz/events", viz.eventsEndpoint, methods=["GET"]),
            Route("/viz/graph", viz.graphEndpoint, methods=["GET"]),
            Route("/viz/history", viz.historyEndpoint, methods=["GET"]),
            Mount("/mcp", app=handle_mcp),
        ],
        lifespan=lifespan,
    )
    app.state.ctx = ctx
    return app


def main():
    _setProcTitle()
    port = int(os.environ.get("PENSIVE_V3_PORT", _DEFAULT_PORT))

    # Provision the loopback write secret HERE, in the process entry point, and
    # nowhere else. buildApp() only reads it: that function is called by tests in
    # three different files, and a create-by-default read would have them writing a
    # credential into the real data dir just by constructing an app object.
    secretPath = defaultTeeSecretPath()
    if loadTeeSecret(create=True) is None:
        _log(f"WARNING: could not read or create {secretPath}; "
             "POST /tee/emit and POST /shadow/recall will refuse every request")
    else:
        _log(f"loopback write secret at {secretPath} "
             f"(callers send it as {LOCAL_WRITE_HEADER})")

    ctx = buildContext()
    app = buildApp(ctx)

    import uvicorn

    _log(f"serving MCP (Streamable-HTTP) on http://{_HOST}:{port}/mcp")
    uvicorn.run(app, host=_HOST, port=port, log_level="warning")


if __name__ == "__main__":
    main()
