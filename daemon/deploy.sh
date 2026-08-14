#!/usr/bin/env bash
# Deploy pensive-v3 behind its own gate.
#
# WHY THIS EXISTS: `daemon/gate/tiergate` measures L1/L2/L3 latency and quality
# together and knows how to REJECT. Until this script, nothing invoked it on
# deploy. A gate nothing invokes is a measurement, not a gate (STYLE.md), and I
# had that violation written into my own systemd drop-in comment as a known
# hole. This closes it.
#
# WHAT IT WILL NOT DO: auto-rollback. The gate rejecting is information, and
# what to do about it is a decision with context this script does not have
# (a rejection during a known-loaded box means something different from one on a
# quiet box). It exits nonzero, says exactly what failed, and leaves the call to
# the operator.
#
# SCAR, earned this campaign: never bind a success test to the tail of a pipe.
# `pytest ... | tail && commit` reports the exit status of `tail`, which is
# always 0, and it let two red tests through into a commit. Every status check
# here reads $? from the command itself, with no pipe between.

set -uo pipefail

cd "$(dirname "$0")" || exit 2
GATE_DIR="gate/tiergate"
LABEL="${1:-deploy-$(date +%H%M%S)}"

say() { printf '\n=== %s ===\n' "$*"; }
die() { printf '\nDEPLOY REFUSED: %s\n' "$*" >&2; exit 1; }

# --------------------------------------------------------------------------- #
# 1. tests, before anything is restarted
# --------------------------------------------------------------------------- #
say "tests"
python -m pytest test/ -q
testStatus=$?
[ "$testStatus" -eq 0 ] || die "pytest exited $testStatus. Nothing was restarted."

# --------------------------------------------------------------------------- #
# 2. the gate binary must be current with its own source
# --------------------------------------------------------------------------- #
say "gate build"
( cd "$GATE_DIR" && go build -o tiergate . )
buildStatus=$?
[ "$buildStatus" -eq 0 ] || die "tiergate failed to build (exit $buildStatus)."

# --------------------------------------------------------------------------- #
# 3. restart, then wait for the daemon to actually answer
# --------------------------------------------------------------------------- #
say "restart"
systemctl --user restart pensive-v3 || die "systemctl restart failed."

# /status, NOT /healthz. I wrote /healthz here from habit and it 404s: this
# daemon serves /brief /get /lookup /mcp /recall /status /viz and nothing else,
# so the original would have blocked 60s and refused every deploy. Verified by
# probing all six plausible spellings against the live daemon.
ready=0
for _ in $(seq 1 60); do
    code=$(curl -s -m 2 -o /dev/null -w '%{http_code}' "http://127.0.0.1:5999/status" 2>/dev/null)
    if [ "$code" = "200" ]; then ready=1; break; fi
    sleep 1
done
[ "$ready" -eq 1 ] || die "daemon did not answer /status within 60s of restart."

# --------------------------------------------------------------------------- #
# 4. the gate
# --------------------------------------------------------------------------- #
# Empty is not quiet: a gate that printed nothing has not passed, it has failed
# to run. The output is captured to a file and its size checked, because an
# exit-0 with no verdict is the failure mode that looks most like success.
say "gate"
outFile=$(mktemp)
"$GATE_DIR/tiergate" --epoch "$LABEL" 2>&1 | tee "$outFile"
gateStatus=${PIPESTATUS[0]}

if [ ! -s "$outFile" ]; then
    rm -f "$outFile"
    die "tiergate produced NO output. Empty is not quiet; treat as failure."
fi

if ! grep -q '^OUTCOME:' "$outFile"; then
    rm -f "$outFile"
    die "tiergate printed no OUTCOME line. It did not reach a verdict."
fi

outcome=$(grep '^OUTCOME:' "$outFile" | tail -1)
rm -f "$outFile"

if [ "$gateStatus" -ne 0 ]; then
    printf '\n%s\n' "$outcome"
    die "gate exited $gateStatus. The new code IS deployed and IS serving. Decide: fix forward, or 'git revert' and re-run this script."
fi

say "gate accepted"
printf '%s\n' "$outcome"
printf 'Deployed and certified under epoch %s.\n' "$LABEL"
