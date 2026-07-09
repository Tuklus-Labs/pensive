"""Kind-class stratification: which atom kinds form a recall population, and the
pure helpers that keep every population fairly represented at recall's two choke
points (candidate generation and the rerank head).

The store now holds two populations that must not compete on raw count: a large
bulk-imported code corpus (``document_chunk``) and a smaller, higher-value
reasoning memory (``atom``/``narrative``/``snapshot``). A single kind-blind
candidate pool lets the majority starve the minority before the cross-encoder
reranker (the real relevance judge) ever sees it. These helpers let the engine
generate candidates per class and fill the rerank head by round-robin, so both
populations reach the reranker and the winner is decided on relevance, not count.

Nothing here touches the model or the vector index; it is deliberately
dependency-free so both ``signals`` and ``vector_index`` can import it without a
cycle.
"""

__all__ = [
    "MEMORY_KINDS",
    "CODE_KINDS",
    "KIND_CLASSES",
    "classesForKinds",
    "kindInClause",
    "interleave",
    "splitByClass",
]

# The two populations. MEMORY is the reasoning memory; CODE is the bulk source
# corpus. Ordered tuples so round-robin interleaving is deterministic.
MEMORY_KINDS = ("atom", "narrative", "snapshot")
CODE_KINDS = ("document_chunk",)

# (className, kinds) in the order the rerank head round-robins them. Adding a
# third class later is a one-line edit here; every consumer reads this tuple.
KIND_CLASSES = (
    ("memory", MEMORY_KINDS),
    ("code", CODE_KINDS),
)


def classesForKinds(kinds):
    """The KIND_CLASSES entries to stratify over for a given ``kinds`` request.

    ``kinds=None`` (unscoped recall) returns every class, each with its full kind
    tuple: this is the case the whole feature exists for. A ``kinds`` subset
    returns only the classes that overlap it, each narrowed to the overlapping
    kinds, so an explicit ``kinds=['atom']`` request stratifies over just the
    memory class scoped to atoms. A ``kinds`` naming no known kind returns ``[]``,
    which the engine reads as "nothing eligible" and short-circuits to the
    low-confidence sentinel, matching the existing empty-kinds contract.
    """
    if kinds is None:
        return list(KIND_CLASSES)
    wanted = set(kinds)
    out = []
    for name, classKinds in KIND_CLASSES:
        overlap = tuple(k for k in classKinds if k in wanted)
        if overlap:
            out.append((name, overlap))
    return out


def kindInClause(kinds, alias="a"):
    """Build an ``AND <alias>.kind IN (?,?...)`` SQL fragment and its params.

    Returns ``("", ())`` for an empty/None ``kinds`` so callers can unconditionally
    concatenate the fragment and extend their params. ``alias`` is the atoms-table
    alias in the caller's query (every current caller uses ``a``). Values are bound
    parameters, never interpolated, so arbitrary kind strings are injection-safe.
    """
    if not kinds:
        return "", ()
    placeholders = ",".join("?" for _ in kinds)
    return f" AND {alias}.kind IN ({placeholders})", tuple(kinds)


def interleave(perClassLists, cap):
    """Round-robin merge of per-class ranked lists into one list of length <= cap.

    Takes one item from each list in turn (list order = class order), preserving
    each list's internal order, and backfills from the lists that still have items
    once others run dry. Stops as soon as ``cap`` items are collected. With a
    single non-empty list this is exactly that list's ``[:cap]`` prefix, so a
    single-class recall degrades to today's behavior. Empty lists are skipped.
    """
    result = []
    positions = [0] * len(perClassLists)
    while len(result) < cap:
        advanced = False
        for i, lst in enumerate(perClassLists):
            if positions[i] < len(lst):
                result.append(lst[positions[i]])
                positions[i] += 1
                advanced = True
                if len(result) >= cap:
                    break
        if not advanced:
            break
    return result


def splitByClass(fused, store, classNames):
    """Group ``fused`` ``[(atomId, score)]`` into per-class sublists.

    One batched ``SELECT id, kind`` maps each fused atom to its class via
    KIND_CLASSES. Returns ``{className: [(atomId, score), ...]}`` for every name in
    ``classNames``, each sublist in the input ``fused`` order. Atoms whose kind
    belongs to a class not in ``classNames`` (or to no class) are dropped, so a
    scoped recall never leaks an out-of-class atom into the rerank head. An empty
    ``fused`` returns a bucket-per-name dict of empty lists.
    """
    buckets = {name: [] for name in classNames}
    if not fused:
        return buckets
    kindToClass = {}
    for name, classKinds in KIND_CLASSES:
        for k in classKinds:
            kindToClass[k] = name
    atomIds = [atomId for atomId, _ in fused]
    placeholders = ",".join("?" for _ in atomIds)
    rows = store._conn.execute(
        f"SELECT id, kind FROM atoms WHERE id IN ({placeholders})",
        tuple(atomIds),
    ).fetchall()
    classById = {r[0]: kindToClass.get(r[1]) for r in rows}
    for atomId, score in fused:
        name = classById.get(atomId)
        if name in buckets:
            buckets[name].append((atomId, score))
    return buckets
