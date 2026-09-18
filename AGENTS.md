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
claiming a live fix. Some records mention `pensive-recall` and `engram-emit`;
those are shell wrappers from a private sibling repository and are not shipped
here. They call the MCP tools and HTTP routes documented in `daemon/README.md`.

Also here: `research/` (benchmarks and experiments; their private fixtures are
not shipped), `docs/superpowers/` (dated design specs and plans), and the
top-level campaign records, each carrying a header that dates it.
