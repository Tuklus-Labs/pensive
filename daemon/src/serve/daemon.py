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
from serve.tee import Counters, handleTeeEmit  # noqa: E402
from serve.shadow import runShadow, defaultShadowLogPath  # noqa: E402
from serve import viz  # noqa: E402
from ambient.briefer import brief, DEFAULT_BUDGET  # noqa: E402
from store.store import openStore  # noqa: E402
from recall.embedder import Embedder  # noqa: E402

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


def buildContext():
    """Open the store, load the models, warm the reranker, build the index."""
    storePath = Path(os.environ.get("PENSIVE_V3_STORE", str(_DEFAULT_STORE)))
    storePath.parent.mkdir(parents=True, exist_ok=True)
    modelId = os.environ.get("PENSIVE_V3_MODEL", _DEFAULT_MODEL)
    agent = os.environ.get("PENSIVE_V3_AGENT") or None

    _log(f"opening store {storePath}")
    store = openStore(storePath)

    _log(f"loading embedder {modelId}")
    embedder = Embedder(modelId)

    _log("warming cross-encoder reranker")
    from recall.rerank import _getReranker

    _getReranker()

    _log("embedding backlog + building dense index")
    ctx = ServeContext(store, embedder, modelId, agent=agent)
    _log("context ready")
    return ctx


def buildApp(ctx):
    """A Starlette ASGI app hosting the low-level MCP server over Streamable-HTTP.

    Stateless + JSON responses keep the localhost transport simple (no session-id
    ceremony, plain JSON round-trips). The session manager's ``run()`` context is
    driven by the app lifespan so it is active for the life of the server.

    Alongside the ``/mcp`` mount the app carries the Phase 3 double-write surface,
    all localhost-only (no auth, same bind): ``POST /tee/emit`` replays a tee'd
    emit into the v3 store, ``POST /shadow/recall`` logs a v3 answer beside the old
    one, and ``GET /status`` exposes the four tee/shadow counters for the gate
    check. The tee/shadow boundary handlers are synchronous and run inline on the
    event-loop thread -- the SAME thread that created the sqlite store (and the
    same path the MCP ``call_tool`` handler already takes) -- so the store's
    thread affinity is preserved; they do not go through a threadpool.
    """
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Mount, Route

    server = buildServer(ctx)
    manager = StreamableHTTPSessionManager(
        app=server, json_response=True, stateless=True,
    )

    counters = Counters()
    shadowLogPath = defaultShadowLogPath()

    async def handle_mcp(scope, receive, send):
        await manager.handle_request(scope, receive, send)

    async def tee_emit(request):
        body = await request.body()
        status, payload = handleTeeEmit(ctx, counters, body)
        return JSONResponse(payload, status_code=status)

    async def shadow_recall(request):
        body = await request.body()
        status, payload = runShadow(ctx, counters, body, shadowLogPath)
        return JSONResponse(payload, status_code=status)

    async def status(request):
        return JSONResponse({"server": SERVER_NAME, "counters": counters.snapshot()})

    async def brief_endpoint(request):
        # The Phase 4 session-start working set as a VIEW over the store. GET so a
        # SessionStart hook can curl it; agent + budget are query params. brief()
        # performs zero writes, so this handler is read-only like /status.
        agent = request.query_params.get("agent")
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
        text = brief(ctx.store, {"agent": agent, "budget": budget})
        return JSONResponse({"brief": text, "agent": agent, "budget": budget})

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        async with manager.run():
            yield

    app = Starlette(
        routes=[
            Route("/tee/emit", tee_emit, methods=["POST"]),
            Route("/shadow/recall", shadow_recall, methods=["POST"]),
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
    ctx = buildContext()
    app = buildApp(ctx)

    import uvicorn

    _log(f"serving MCP (Streamable-HTTP) on http://{_HOST}:{port}/mcp")
    uvicorn.run(app, host=_HOST, port=port, log_level="warning")


if __name__ == "__main__":
    main()
