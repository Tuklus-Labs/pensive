#!/usr/bin/env bash
# session-brief-v3.sh -- SessionStart hook: inject the Pensive v3 working-set brief.
#
# ---------------------------------------------------------------------------
# WIRING (this block is documentation; the hook NEVER edits any settings file).
#
# To wire it after the Gary-gated cutover (Task 22), add ONE entry to
# ~/.claude/settings.json under hooks.SessionStart[].hooks[]:
#
#   {
#     "hooks": {
#       "SessionStart": [
#         { "hooks": [
#             { "type": "command",
#               "command": "/home/aegis/Projects/pensive/daemon/hooks/session-brief-v3.sh" }
#         ] }
#       ]
#     }
#   }
#
# It SUPPLEMENTS the existing Kairos SessionStart briefing -- it does not replace
# it. Both can be listed; each prints its own additionalContext block.
#
# The hook is OFF until you export PENSIVE_V3_BRIEF=1 in the session/daemon env.
# Until then it consumes its stdin, exits 0, and prints nothing, so wiring it
# early is harmless (no behaviour change until the flag flips).
# ---------------------------------------------------------------------------
#
# Env:
#   PENSIVE_V3_BRIEF   must equal "1" or the hook is a silent no-op (the flag).
#   PENSIVE_V3_PORT    daemon localhost port (default 5999, matches serve.daemon).
#   PENSIVE_V3_AGENT   agent name to brief for (default unset -> no loose ends).
#   PENSIVE_V3_BUDGET  token budget (default 1500).
#
# Fail-open contract: a SessionStart hook must NEVER block the session. Every
# error path (flag off, curl/jq missing, daemon down, malformed response, empty
# brief) exits 0 with no output.
set -uo pipefail

# Drain stdin (the hook payload) so the writer never blocks on a full pipe.
cat >/dev/null 2>&1 || true

# The flag. OFF unless explicitly "1".
[ "${PENSIVE_V3_BRIEF:-}" = "1" ] || exit 0

port="${PENSIVE_V3_PORT:-5999}"
agent="${PENSIVE_V3_AGENT:-}"
budget="${PENSIVE_V3_BUDGET:-1500}"

# Pull the brief off the daemon's localhost HTTP surface. -f: fail (non-zero) on
# HTTP errors; -s: silent; --max-time: never hang the session on a slow daemon.
resp="$(curl -sf --max-time 3 \
  "http://127.0.0.1:${port}/brief?agent=${agent}&budget=${budget}" 2>/dev/null)" || exit 0
[ -n "${resp}" ] || exit 0

# Extract the brief text; a parse failure or empty brief is a silent no-op.
context="$(printf '%s' "${resp}" | jq -r '.brief // empty' 2>/dev/null)" || exit 0
[ -n "${context}" ] || exit 0

# Emit as SessionStart additionalContext (the documented hook output shape).
jq -cn --arg c "${context}" \
  '{hookSpecificOutput: {hookEventName: "SessionStart", additionalContext: $c}}' \
  2>/dev/null || exit 0
exit 0
