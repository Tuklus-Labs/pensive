#!/usr/bin/env bash
# session-brief-v3-grok.sh -- write Grok's working-set brief to a rules file.
#
# Grok Build SessionStart IGNORES stdout. Claude's additionalContext JSON from
# session-brief-v3.sh would be dropped. Grok auto-loads ~/.grok/rules/*.md, so
# this hook writes the brief there. Session still starts if the daemon is
# down (exit 0). A dead daemon must NOT leave yesterday's VIEW in standing
# rules: we replace the file with "pensive brief unavailable" rather than
# keeping a stale working set as law.
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

_write_brief() {
  dir="$(dirname -- "${dest}")"
  mkdir -p -- "${dir}" || return 0
  tmp="${dest}.tmp.$$"
  {
    printf '%s\n' "# Pensive working set for ${agent} (generated; do not edit)"
    printf '%s\n'
    printf '%s\n' "$1"
  } > "${tmp}" || { rm -f -- "${tmp}"; return 0; }
  mv -f -- "${tmp}" "${dest}" || { rm -f -- "${tmp}"; return 0; }
}

resp="$(curl -sf --max-time 3 \
  "http://127.0.0.1:${port}/brief?agent=${agent}&budget=${budget}" 2>/dev/null)" || {
  _write_brief "pensive brief unavailable"
  exit 0
}
[ -n "${resp}" ] || { _write_brief "pensive brief unavailable"; exit 0; }

brief="$(printf '%s' "${resp}" | jq -r '.brief // empty' 2>/dev/null)" || {
  _write_brief "pensive brief unavailable"
  exit 0
}
[ -n "${brief}" ] || { _write_brief "pensive brief unavailable"; exit 0; }

_write_brief "${brief}"
exit 0
