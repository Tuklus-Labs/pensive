"""L1: identity retrieval. Answer by NAME, never by similarity.

``Aegis/CLAUDE.md:113`` specifies a hot L1 tier and the v3 rewrite dropped it,
leaving one monolithic recall path where every caller pays for a cross-encoder
whether or not it asked a question that needs one. This restores the tier.

The contract, and every clause is load-bearing:

* **No ranking, no scoring, no model.** An L1 answer is a row, not a judgement.
  Returning a score would invite a caller to threshold on it, which recreates
  the ranking contest this tier exists to avoid.
* **Present and absent are DISTINGUISHABLE.** A tier that answers the same way
  for a row that exists and one that cannot is unusable for lookup, and its
  absence is indistinguishable from its presence.
* **Bounded before the store is touched.** Length and limit caps are checked
  first, so a malformed request costs a comparison rather than a query.

Why this is a plain route rather than an MCP tool, measured 2026-08-12: the MCP
``tools/call`` envelope alone costs 2.486ms P95 for a 142-byte error that never
reaches the store, against a 1ms budget, while the bare HTTP route floor is
0.297ms. No handler optimization reaches the budget through MCP, so the
transport choice IS the design.

The two operations are the two that were being answered by semantic search:

* ``get(atomId)`` -- what is this id
* ``lookup(key, value)`` -- which atoms carry this exact facet

The second is the one Charon needed. It asked "which fragments belong to session
<uuid>" through a hybrid retrieval engine, which cannot report absence, so an
empty session came back as nearest neighbours: its own source code, because the
best lexical match for "narrative_fragment session" in a corpus containing
Charon's source is Charon's source. 91.4% of every recall this store has ever
served came from that one mistake.

Stdlib plus the store's public accessors only.
"""

__all__ = ["handleGet", "handleLookup", "MAX_ID_LEN", "MAX_LOOKUP_LIMIT",
           "DEFAULT_LOOKUP_LIMIT", "MAX_FACET_LEN", "MAX_PROVENANCE_ROWS"]

# A ULID is 26 characters. The cap is generous rather than exact so a future id
# scheme does not silently start 400ing, but it is still a cap: an unbounded id
# is a free SQL parameter of arbitrary size on an unauthenticated read route.
MAX_ID_LEN = 256

# Facet key/value bounds, same reasoning.
MAX_FACET_LEN = 512

# Lookup is an exact-match set read, not a search, so the cap exists to bound the
# response rather than to rank anything. A caller wanting more than this wants a
# different operation.
DEFAULT_LOOKUP_LIMIT = 100
MAX_LOOKUP_LIMIT = 1000

# Provenance rows returned under ?full=1. One atom in the live store carries 674
# of them (a duplicated memory re-captured by the distiller), so an uncapped list
# turns a single read into an unbounded response.
MAX_PROVENANCE_ROWS = 64


def _bad(message):
    return 400, {"error": message}


def handleGet(store, atomId, withProvenance=False):
    """``(status, payload)`` for an atom by id. 200 with the row, or 404.

    A SUPERSEDED atom is returned, with its status. Identity retrieval answers
    "what is this id", which is a different question from "what should you
    believe": the trust layer withholds belief, and making this route also judge
    would put two mechanisms on one question, which is how one of them quietly
    stops being load-bearing.

    ``withProvenance`` is off by default, and the default path genuinely does not
    read the provenance table. An earlier version called ``getAtom``, which
    ALWAYS fetches every provenance row, so the docstring's claim that a caller
    "pays explicitly" for provenance was false: every caller paid, and an atom
    with many provenance rows made an unauthenticated read arbitrarily expensive.
    Caught by a blue-team review. The lean path is now one indexed row read.

    ``full=1`` returns provenance capped at :data:`MAX_PROVENANCE_ROWS`. One atom
    in this store carries 674 provenance rows, so an uncapped list is an
    unbounded response on a public read route; truncation is REPORTED rather than
    silent, because a short list that looks complete is its own kind of lie.
    """
    if not isinstance(atomId, str) or not atomId.strip():
        return _bad("id is required")
    if len(atomId) > MAX_ID_LEN:
        return _bad(f"id exceeds {MAX_ID_LEN} characters")

    row = store._conn.execute(
        "SELECT id, text, kind, project, created_at, occurred_at, status "
        "FROM atoms WHERE id = ?",
        (atomId,),
    ).fetchone()
    if row is None:
        # An explicit found:false rather than a bare 404 body, so a caller that
        # only reads the body can still tell absence from a transport failure.
        return 404, {"id": atomId, "found": False}

    payload = {
        "id": row[0],
        "text": row[1],
        "kind": row[2],
        "project": row[3],
        "createdAt": row[4],
        "occurredAt": row[5],
        "status": row[6],
        "found": True,
    }
    if withProvenance:
        prov = store._conn.execute(
            "SELECT id, source, session_id, agent, source_ref, recorded_at "
            "FROM provenance WHERE atom_id = ? ORDER BY id LIMIT ?",
            (atomId, MAX_PROVENANCE_ROWS + 1),
        ).fetchall()
        payload["provenance"] = [
            {"id": p[0], "source": p[1], "sessionId": p[2], "agent": p[3],
             "sourceRef": p[4], "recordedAt": p[5]}
            for p in prov[:MAX_PROVENANCE_ROWS]
        ]
        payload["provenanceTruncated"] = len(prov) > MAX_PROVENANCE_ROWS
    return 200, payload


def handleLookup(store, key, value, limit=DEFAULT_LOOKUP_LIMIT):
    """``(status, payload)`` for atoms carrying an exact ``(key, value)`` facet.

    Returns ``{key, value, ids, limit, truncated}``. An absent value returns an
    EMPTY id list, which is the property that distinguishes this from search: a
    similarity engine cannot report absence and returns the nearest thing
    instead, and that substitution is what poisoned this store's traffic for
    forty days.

    Live atoms only. A caller asking "which atoms are in this set" means the set
    as it stands; a superseded row is reachable by :func:`handleGet` when the
    caller wants the history.
    """
    if not isinstance(key, str) or not key.strip():
        return _bad("key is required")
    if not isinstance(value, str) or not value.strip():
        return _bad("value is required")
    if len(key) > MAX_FACET_LEN or len(value) > MAX_FACET_LEN:
        return _bad(f"key/value exceed {MAX_FACET_LEN} characters")
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        return _bad("limit must be an integer")
    if limit < 1:
        return _bad("limit must be >= 1")
    limit = min(limit, MAX_LOOKUP_LIMIT)

    # One indexed read. idx_facets_kv covers (key, value); the status filter is
    # a join back to atoms rather than a second round trip.
    rows = store._conn.execute(
        "SELECT f.atom_id FROM facets f JOIN atoms a ON a.id = f.atom_id "
        "WHERE f.key = ? AND f.value = ? AND a.status = 'live' "
        "ORDER BY f.atom_id LIMIT ?",
        (key, value, limit + 1),
    ).fetchall()

    ids = [r[0] for r in rows[:limit]]
    return 200, {
        "key": key,
        "value": value,
        "ids": ids,
        "limit": limit,
        # Reported rather than silent: a truncated set that looks complete is
        # the same class of lie as an empty result that looks like calm seas.
        "truncated": len(rows) > limit,
    }
