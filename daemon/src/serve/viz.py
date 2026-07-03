"""Activation visualization surface for the resident daemon."""
import asyncio
import json
import time
from collections import deque
from pathlib import Path

from starlette.responses import HTMLResponse, JSONResponse, Response

INPUT_ATOM_CAP = 30
HUB_CAP = 128
NEIGHBOR_CAP = 40
HISTORY_LIMIT_CAP = 200
EVENT_RING_MAX = 256
SSE_SEND_TIMEOUT = 0.25
SSE_PING_INTERVAL = 15
_STATIC_PATH = Path(__file__).resolve().parent / "static" / "viz.html"
_STATIC_HTML = None


def ensureVizState(ctx, maxlen=EVENT_RING_MAX):
    if not hasattr(ctx, "vizEvents"):
        ctx.vizEvents = deque(maxlen=maxlen)
    if not hasattr(ctx, "vizSeq"):
        ctx.vizSeq = 0
    if not hasattr(ctx, "vizEventErrors"):
        ctx.vizEventErrors = 0


def emitRecallEvent(ctx, query, atomIds, sourceRef):
    try:
        ensureVizState(ctx)
        ctx.vizSeq += 1
        ctx.vizEvents.append({
            "seq": ctx.vizSeq,
            "ts": time.time(),
            "query": query,
            "atomIds": list(atomIds),
            "sourceRef": sourceRef,
        })
    except Exception:  # noqa: BLE001 -- visualization telemetry is best-effort
        ctx.vizEventErrors = getattr(ctx, "vizEventErrors", 0) + 1


def _snippet(text):
    return " ".join((text or "").split())[:120]


def _rowToNode(row):
    return {
        "id": row[0],
        "kind": row[1],
        "textSnippet": _snippet(row[2]),
        "importance": row[3],
        "status": row[4],
    }


def _liveAtomRows(ctx, atomIds):
    if not atomIds:
        return []
    placeholders = ",".join("?" for _ in atomIds)
    rows = ctx.store._conn.execute(
        "SELECT id, kind, text, importance, status FROM atoms "
        f"WHERE status = 'live' AND id IN ({placeholders})",
        atomIds,
    ).fetchall()
    byId = {row[0]: row for row in rows}
    return [byId[atomId] for atomId in atomIds if atomId in byId]


def _facetNeighborRows(ctx, key, value):
    degree = ctx.store._conn.execute(
        "SELECT COUNT(*) FROM facets WHERE key = ? AND value = ?",
        (key, value),
    ).fetchone()[0]
    if degree <= 1 or degree > HUB_CAP:
        return [], degree
    return ctx.store._conn.execute(
        "SELECT a.id, a.kind, a.text, a.importance, a.status "
        "FROM facets f JOIN atoms a ON a.id = f.atom_id "
        "WHERE f.key = ? AND f.value = ? AND a.status = 'live' "
        "ORDER BY a.id LIMIT ?",
        (key, value, HUB_CAP),
    ).fetchall(), degree


def buildGraphPayload(ctx, atomIds):
    seen = []
    for atomId in atomIds:
        if atomId not in seen:
            seen.append(atomId)
    truncated = len(seen) > INPUT_ATOM_CAP
    seedIds = seen[:INPUT_ATOM_CAP]
    nodes = {}
    edges = {}

    for row in _liveAtomRows(ctx, seedIds):
        seedId = row[0]
        nodes[seedId] = _rowToNode(row)
        neighborCount = 0
        facets = ctx.store._conn.execute(
            "SELECT key, value FROM facets WHERE atom_id = ? "
            "AND key IN ('entity', 'tag') ORDER BY key, value",
            (seedId,),
        ).fetchall()
        for key, value in facets:
            rows, degree = _facetNeighborRows(ctx, key, value)
            if not rows:
                continue
            weight = 1.0 / degree
            for neighbor in rows:
                neighborId = neighbor[0]
                if neighborId == seedId:
                    continue
                nodes.setdefault(neighborId, _rowToNode(neighbor))
                edgeKey = (seedId, neighborId, key, value)
                edges[edgeKey] = {
                    "from": seedId,
                    "to": neighborId,
                    "weight": weight,
                    "degree": degree,
                    "facet": {"key": key, "value": value},
                }
                neighborCount += 1
                if neighborCount >= NEIGHBOR_CAP:
                    break
            if neighborCount >= NEIGHBOR_CAP:
                break

    return {
        "seedIds": seedIds,
        "truncatedInput": truncated,
        "nodes": list(nodes.values()),
        "edges": list(edges.values()),
    }


def parseAtomsParam(raw):
    if raw is None:
        raise ValueError("atoms query param is required")
    atomIds = [part.strip() for part in raw.split(",") if part.strip()]
    if not atomIds:
        raise ValueError("atoms query param must contain at least one atom id")
    if len(atomIds) > INPUT_ATOM_CAP:
        raise ValueError(f"atoms query param is capped at {INPUT_ATOM_CAP}")
    return atomIds


def _parseLimit(raw):
    if raw in (None, ""):
        return 50
    try:
        limit = int(raw)
    except (TypeError, ValueError):
        raise ValueError("limit must be an integer") from None
    if limit < 1:
        raise ValueError("limit must be >= 1")
    return min(limit, HISTORY_LIMIT_CAP)


def historyPayload(ctx, limitRaw=None):
    limit = _parseLimit(limitRaw)
    rows = ctx.store._conn.execute(
        "SELECT id, atom_id, query, recorded_at FROM recall_log "
        "ORDER BY rowid DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return {
        "limit": limit,
        "events": [
            {"id": row[0], "atom_id": row[1], "query": row[2], "recorded_at": row[3]}
            for row in rows
        ],
    }


def _error(message):
    return JSONResponse({"error": message}, status_code=400)


async def staticPage(_request):
    global _STATIC_HTML
    if _STATIC_HTML is None:
        _STATIC_HTML = _STATIC_PATH.read_text(encoding="utf-8")
    return HTMLResponse(_STATIC_HTML)


async def graphEndpoint(request):
    try:
        atomIds = parseAtomsParam(request.query_params.get("atoms"))
    except ValueError as exc:
        return _error(str(exc))
    return JSONResponse(buildGraphPayload(request.app.state.ctx, atomIds))


async def historyEndpoint(request):
    try:
        payload = historyPayload(request.app.state.ctx, request.query_params.get("limit"))
    except ValueError as exc:
        return _error(str(exc))
    return JSONResponse(payload)


def _sseFrame(payload):
    data = json.dumps(payload, separators=(",", ":"))
    return f"data: {data}\n\n".encode("utf-8")


async def sendSsePayload(send, payload, timeout=None):
    if timeout is None:
        timeout = SSE_SEND_TIMEOUT
    try:
        await asyncio.wait_for(send({
            "type": "http.response.body",
            "body": payload,
            "more_body": True,
        }), timeout=timeout)
        return True
    except (asyncio.TimeoutError, OSError, RuntimeError):
        return False


class VizEventResponse(Response):
    media_type = "text/event-stream"

    def __init__(self, ctx):
        super().__init__(content=b"", media_type=self.media_type)
        self.ctx = ctx

    async def __call__(self, scope, receive, send):
        ensureVizState(self.ctx)
        headers = [
            (b"content-type", b"text/event-stream; charset=utf-8"),
            (b"cache-control", b"no-cache"),
            (b"connection", b"keep-alive"),
        ]
        await send({"type": "http.response.start", "status": 200, "headers": headers})
        lastSeq = self.ctx.vizSeq
        lastPing = time.monotonic()

        while True:
            if await self._disconnected(receive):
                break
            events = [event for event in list(self.ctx.vizEvents) if event["seq"] > lastSeq]
            if events:
                for event in events:
                    ok = await sendSsePayload(send, _sseFrame(event))
                    if not ok:
                        return
                    lastSeq = event["seq"]
                lastPing = time.monotonic()
            elif time.monotonic() - lastPing > SSE_PING_INTERVAL:
                ok = await sendSsePayload(send, b": ping\n\n")
                if not ok:
                    return
                lastPing = time.monotonic()
            await asyncio.sleep(0.25)
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    async def _disconnected(self, receive):
        try:
            message = await asyncio.wait_for(receive(), timeout=0)
        except asyncio.TimeoutError:
            return False
        return message.get("type") == "http.disconnect"


async def eventsEndpoint(request):
    return VizEventResponse(request.app.state.ctx)
