# Working on Pensive

Two related systems share this repository:

- `daemon/` is the active agent-memory service: natural-language recall,
  canonical SQLite data, provenance, corrections and session briefs. Start at
  [daemon/README.md](daemon/README.md); its tests live in `daemon/test/`.
- `src/pensive/` is the independently packaged `pypensive` retrieval library.
  Its graph query API is entity-exact; its tests live in `tests/`. The detailed
  library constraints in `STYLE.md` apply to this package.

Historical campaigns and evidence documents explain past decisions. Check their
claims against the current source and tests before treating them as an active
backlog. Retractions belong beside the claims they invalidate.

Use isolated stores and ports for changes. Preserve canonical text, provenance
and correction history. A source checkout, a running daemon and an installed
shell client can differ; verify the path that an agent actually calls before
claiming a live fix. Household `pensive-recall` and `engram-emit` sources are in
the sibling Engram repository's `tools/` directory.
