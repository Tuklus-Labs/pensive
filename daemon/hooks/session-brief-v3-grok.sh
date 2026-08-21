#!/usr/bin/env bash
# session-brief-v3-grok.sh -- write Grok's working-set brief to a rules file.
#
# Grok Build SessionStart IGNORES stdout. Claude's additionalContext JSON from
# session-brief-v3.sh would be dropped. Grok auto-loads ~/.grok/rules/*.md, so
# this hook writes the brief there. Fail-open: a dead daemon, missing jq, or
# empty brief exits 0 and leaves any existing file. Never blocks session start.
#
# Env:
#   PENSIVE_V3_PORT       daemon port (default 5999)
#   PENSIVE_V3_AGENT      agent to brief (default grok)
#   PENSIVE_V3_BUDGET     token budget (default 1500)
#   PENSIVE_V3_GROK_BRIEF destination path (default ~/.grok/rules/pensive-brief.md)
#
# Register from ~/.grok/hooks/ as SessionStart and SessionEnd. SessionEnd
# refreshes the file so the next session is not empty if rules load before hooks.
set -uo pipefail

cat >/dev/null 2>&1 || true

port="${PENSIVE_V3_PORT:-5999}"
agent="${PENSIVE_V3_AGENT:-grok}"
budget="${PENSIVE_V3_BUDGET:-1500}"
dest="${PENSIVE_V3_GROK_BRIEF:-${HOME}/.grok/rules/pensive-brief.md}"

resp="$(curl -sf --max-time 3 \
  "http://127.0.0.1:${port}/brief?agent=${agent}&budget=${budget}" 2>/dev/null)" || exit 0
[ -n "${resp}" ] || exit 0

brief="$(printf '%s' "${resp}" | jq -r '.brief // empty' 2>/dev/null)" || exit 0
[ -n "${brief}" ] || exit 0

dir="$(dirname -- "${dest}")"
mkdir -p -- "${dir}" || exit 0

tmp="${dest}.tmp.$$"
{
  printf '%s\n' "# Pensive working set for ${agent} (generated; do not edit)"
  printf '%s\n'
  printf '%s\n' "${brief}"
} > "${tmp}" || { rm -f -- "${tmp}"; exit 0; }
mv -f -- "${tmp}" "${dest}" || { rm -f -- "${tmp}"; exit 0; }
exit 0
