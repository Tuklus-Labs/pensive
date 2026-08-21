# Sabotage log: first-class reader

Commit-before-plant. A `git checkout --` during an earlier plant destroyed
uncommitted `mcp.py` (STYLE.md: checkout restores committed, discards the rest).
Re-applied, then this log. Do not restore plants with checkout.

## test_agent_active_excludes_other_agents_recent_work

1. Production mutation: `provenance WHERE agent = ?` -> `agent != ?` in
   `_activeRanked`. Prediction: grok brief contains heph, drops grok.
   Observed: RED, `grok emit missing from grok brief`, only theia/heph handle
   remained. Load-bearing.
2. Restore: write the `=` predicate back. GREEN.

## test_pensive_recall_passes_l2_tier

1. Production mutation: `tier=DEFAULT_TIER` -> `rerankEnabled=False` only.
   Prediction: keywords have no `tier`. Observed: RED,
   `pensive_recall L2-default rule violated: keywords={..., rerankEnabled: False}`.
   Load-bearing.
2. Restore: re-apply `tier=DEFAULT_TIER`. GREEN.

## test_filter_agent_keeps_stamped_drops_null_and_other

1. Production mutation (mental, same shape as brief): `_filterAgent` `agent IN`
   inverted to `NOT IN` would keep heph and NULL, drop grok. Not re-run as a
   third live plant; the SELECT is the same join the brief test already killed.
2. Test mutation: drop the `hephId not in grokIds` equivalent (`grokIds == [grokId]`
   weakened to `grokId in grokIds`) would still pass if heph leaked. The equality
   to `[grokId]` is the load-bearing form. Kept.

## test_grok_hook_daemon_down_preserves_existing_brief

1. Production mutation: write `"unavailable\n"` on curl failure. Prediction:
   test sees clobber. Not planted in the script this round; the script has no
   else-write on the fail path (`|| exit 0` before any file open). Absence of
   a write is the mechanism.
