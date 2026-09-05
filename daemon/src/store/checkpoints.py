"""Append-only task checkpoint records outside the semantic atom corpus."""

import time

from util.ulid import ulid

__all__ = [
    "putTaskCheckpoint",
    "getTaskStates",
    "listTaskHistory",
    "listRecentTaskStates",
]

_STATES = frozenset(("active", "blocked", "completed", "abandoned"))
_PROJECT_LIMIT = 256
_AGENT_LIMIT = 64
_TASK_ID_LIMIT = 256
_REQUEST_ID_LIMIT = 256
_BODY_LIMIT = 32_000

_CHECKPOINT_COLUMNS = (
    "id", "project", "agent", "task_id", "revision", "request_id", "state",
    "body", "source", "writer_session", "source_ref", "recorded_at",
)
_CHECKPOINT_COLS = ", ".join(_CHECKPOINT_COLUMNS)


def _selectColumns(alias=None):
    if alias is None:
        return _CHECKPOINT_COLS
    return ", ".join(f"{alias}.{column}" for column in _CHECKPOINT_COLUMNS)


def _text(name, value, maximum=None, *, nonempty=True):
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if nonempty and not value.strip():
        raise ValueError(f"{name} must be nonempty")
    if maximum is not None and len(value) > maximum:
        raise ValueError(f"{name} exceeds maximum length {maximum}")
    return value


def _optional_text(name, value, maximum=None):
    if value is not None:
        _text(name, value, maximum, nonempty=False)
    return value


def _integer(name, value, *, minimum=None):
    if type(value) is not int:
        raise TypeError(f"{name} must be a real integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _validateCheckpointInput(
    project,
    agent,
    taskId,
    expectedRevision,
    requestId,
    state,
    body,
    source,
    sessionId,
    sourceRef,
):
    _text("project", project, _PROJECT_LIMIT)
    _text("agent", agent, _AGENT_LIMIT)
    _text("taskId", taskId, _TASK_ID_LIMIT)
    _integer("expectedRevision", expectedRevision, minimum=0)
    _text("requestId", requestId, _REQUEST_ID_LIMIT)
    if not isinstance(state, str) or state not in _STATES:
        raise ValueError(f"state must be one of {sorted(_STATES)!r}")
    _text("body", body, _BODY_LIMIT, nonempty=False)
    _text("source", source)
    _optional_text("sessionId", sessionId)
    _optional_text("sourceRef", sourceRef)


def _validateReadScope(taskId, project, agent):
    _text("taskId", taskId, _TASK_ID_LIMIT)
    if project is not None:
        _text("project", project, _PROJECT_LIMIT)
    if agent is not None:
        _text("agent", agent, _AGENT_LIMIT)


def _rowToCheckpoint(row):
    return {
        "id": row[0],
        "project": row[1],
        "agent": row[2],
        "taskId": row[3],
        "revision": row[4],
        "requestId": row[5],
        "state": row[6],
        "body": row[7],
        "source": row[8],
        "sessionId": row[9],
        "sourceRef": row[10],
        "recordedAt": row[11],
    }


def _matchesRequest(
    row,
    project,
    agent,
    taskId,
    expectedRevision,
    state,
    body,
    source,
    sessionId,
    sourceRef,
):
    return (
        row[1] == project
        and row[2] == agent
        and row[3] == taskId
        and row[4] - 1 == expectedRevision
        and row[6] == state
        and row[7] == body
        and row[8] == source
        and row[9] == sessionId
        and row[10] == sourceRef
    )


def _selectByRequestId(conn, requestId):
    return conn.execute(
        f"SELECT {_CHECKPOINT_COLS} FROM task_checkpoints WHERE request_id = ?",
        (requestId,),
    ).fetchone()


def putTaskCheckpoint(
    store,
    *,
    project,
    agent,
    taskId,
    expectedRevision,
    requestId,
    state,
    body,
    source,
    sessionId=None,
    sourceRef=None,
):
    """Append one checkpoint after a compare-and-swap revision check."""
    _validateCheckpointInput(
        project,
        agent,
        taskId,
        expectedRevision,
        requestId,
        state,
        body,
        source,
        sessionId,
        sourceRef,
    )
    conn = store._conn
    if conn.in_transaction:
        raise RuntimeError("checkpoint requires its own transaction")
    try:
        # The lock must cover both request replay and the head read. Otherwise
        # two writers can observe the same head and both append its successor.
        conn.execute("BEGIN IMMEDIATE")
        existing = _selectByRequestId(conn, requestId)
        if existing is not None:
            if not _matchesRequest(
                existing,
                project,
                agent,
                taskId,
                expectedRevision,
                state,
                body,
                source,
                sessionId,
                sourceRef,
            ):
                raise ValueError(
                    f"requestId {requestId!r} was already used with different input"
                )
            result = _rowToCheckpoint(existing)
            conn.commit()
            return result

        head = conn.execute(
            "SELECT MAX(revision) FROM task_checkpoints "
            "WHERE project = ? AND agent = ? AND task_id = ?",
            (project, agent, taskId),
        ).fetchone()[0]
        currentRevision = 0 if head is None else head
        if expectedRevision != currentRevision:
            raise ValueError(
                "checkpoint CAS failed: "
                f"project={project!r} agent={agent!r} taskId={taskId!r} "
                f"expectedRevision={expectedRevision} currentRevision={currentRevision}"
            )
        revision = currentRevision + 1
        recordedAt = _now()
        checkpointId = ulid()
        conn.execute(
            f"INSERT INTO task_checkpoints({_CHECKPOINT_COLS}) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                checkpointId,
                project,
                agent,
                taskId,
                revision,
                requestId,
                state,
                body,
                source,
                sessionId,
                sourceRef,
                recordedAt,
            ),
        )
        conn.commit()
        return {
            "id": checkpointId,
            "project": project,
            "agent": agent,
            "taskId": taskId,
            "revision": revision,
            "requestId": requestId,
            "state": state,
            "body": body,
            "source": source,
            "sessionId": sessionId,
            "sourceRef": sourceRef,
            "recordedAt": recordedAt,
        }
    except Exception:
        conn.rollback()
        raise


def _now():
    return int(time.time())


def _queryCheckpoints(conn, sql, params):
    return [_rowToCheckpoint(row) for row in conn.execute(sql, params).fetchall()]


def getTaskStates(
    store,
    *,
    taskId,
    project=None,
    agent=None,
    revision=None,
    asOf=None,
    limit=8,
):
    """Return current heads or one exact stream's historical state."""
    _validateReadScope(taskId, project, agent)
    _integer("limit", limit, minimum=1)
    if revision is not None:
        _integer("revision", revision, minimum=0)
    if asOf is not None:
        _integer("asOf", asOf, minimum=0)
    if revision is not None and asOf is not None:
        raise ValueError("revision and asOf are mutually exclusive")
    if (revision is not None or asOf is not None) and (project is None or agent is None):
        raise ValueError("revision/asOf requires exact project and agent")

    conn = store._conn
    if revision is not None or asOf is not None:
        predicates = ["project = ?", "agent = ?", "task_id = ?"]
        params = [project, agent, taskId]
        if revision is not None:
            predicates.append("revision <= ?")
            params.append(revision)
            order = "revision DESC"
        else:
            predicates.append("recorded_at <= ?")
            params.append(asOf)
            order = "recorded_at DESC, revision DESC"
        sql = (
            f"SELECT {_CHECKPOINT_COLS} FROM task_checkpoints WHERE "
            + " AND ".join(predicates)
            + f" ORDER BY {order} LIMIT 1"
        )
        rows = _queryCheckpoints(conn, sql, params)
        return {"checkpoints": rows, "truncated": False}

    predicates = ["c.task_id = ?"]
    params = [taskId]
    if project is not None:
        predicates.append("c.project = ?")
        params.append(project)
    if agent is not None:
        predicates.append("c.agent = ?")
        params.append(agent)
    scope = " AND ".join(predicates)
    sql = (
        f"SELECT {_selectColumns('c')} "
        "FROM task_checkpoints AS c "
        f"WHERE {scope} AND c.revision = ("
        "SELECT MAX(h.revision) FROM task_checkpoints AS h "
        "WHERE h.project = c.project AND h.agent = c.agent "
        "AND h.task_id = c.task_id) "
        "ORDER BY c.project, c.agent, c.task_id LIMIT ?"
    )
    params.append(limit + 1)
    rows = _queryCheckpoints(conn, sql, params)
    return {"checkpoints": rows[:limit], "truncated": len(rows) > limit}


def listTaskHistory(store, *, project, agent, taskId, afterRevision=0, limit=20):
    """Return an exact task stream in ascending revision order."""
    _text("project", project, _PROJECT_LIMIT)
    _text("agent", agent, _AGENT_LIMIT)
    _text("taskId", taskId, _TASK_ID_LIMIT)
    _integer("afterRevision", afterRevision, minimum=0)
    _integer("limit", limit, minimum=1)
    rows = _queryCheckpoints(
        store._conn,
        f"SELECT {_CHECKPOINT_COLS} FROM task_checkpoints "
        "WHERE project = ? AND agent = ? AND task_id = ? AND revision > ? "
        "ORDER BY revision ASC LIMIT ?",
        (project, agent, taskId, afterRevision, limit + 1),
    )
    truncated = len(rows) > limit
    # afterRevision is exclusive. Return the last delivered revision so feeding
    # this cursor back cannot skip the first row of the following page.
    nextRevision = rows[limit - 1]["revision"] if truncated else None
    return {"checkpoints": rows[:limit], "nextRevision": nextRevision,
            "truncated": truncated}


def listRecentTaskStates(store, *, project, agent=None, limit=20):
    """Return the newest head for each task in a project."""
    _text("project", project, _PROJECT_LIMIT)
    if agent is not None:
        _text("agent", agent, _AGENT_LIMIT)
    _integer("limit", limit, minimum=1)
    predicates = ["c.project = ?"]
    params = [project]
    if agent is not None:
        predicates.append("c.agent = ?")
        params.append(agent)
    sql = (
        f"SELECT {_selectColumns('c')} "
        "FROM task_checkpoints AS c WHERE "
        + " AND ".join(predicates)
        + " AND c.revision = (SELECT MAX(h.revision) FROM task_checkpoints AS h "
          "WHERE h.project = c.project AND h.agent = c.agent "
          "AND h.task_id = c.task_id) "
          "ORDER BY c.recorded_at DESC, c.id DESC LIMIT ?"
    )
    params.append(limit + 1)
    rows = _queryCheckpoints(store._conn, sql, params)
    return {"checkpoints": rows[:limit], "truncated": len(rows) > limit}
