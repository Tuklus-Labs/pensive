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
# 3b. warm, and say so
# --------------------------------------------------------------------------- #
# Answering /status is not the same as being ready to serve at steady state. A
# gate run 90 seconds after restart measured L2 p50 28.5ms against a warm
# baseline nearer 19ms, which is startup cost, not serving cost. Production
# daemons here run for hours.
#
# This is a deliberate thumb on the scale and it is declared rather than hidden:
# the budgets describe normal operation, and a just-restarted process is not in
# normal operation. COLD LATENCY IS STILL REAL and is not measured by this
# script; if cold-start matters it needs its own unit with its own budget, not a
# silent inheritance of the warm number.
say "warm"
warmQ="pensive recall latency budget"
for _ in $(seq 1 12); do
    curl -s -m 10 -o /dev/null "http://127.0.0.1:5999/recall?q=$(printf %s "$warmQ" | tr ' ' '+')&tier=L2" || true
    curl -s -m 15 -o /dev/null "http://127.0.0.1:5999/recall?q=$(printf %s "$warmQ" | tr ' ' '+')&tier=L3" || true
done
printf 'warmed with 24 requests across L2 and L3\n'

# --------------------------------------------------------------------------- #
# 4. the gate
# --------------------------------------------------------------------------- #
# Empty is not quiet: a gate that printed nothing has not passed, it has failed
# to run. The output is captured to a file and its size checked, because an
# exit-0 with no verdict is the failure mode that looks most like success.
say "gate"
# RUN IT FROM ITS OWN DIRECTORY. tiergate's --src defaults to the RELATIVE
# "../../src" and it writes .gate-evidence relative to cwd, so both paths are
# only correct when cwd is gate/tiergate. Invoked from daemon/ (the first
# version of this script) --src resolved to ~/Projects/src, and the
# gate refused with "cannot resolve artifact sha ... refusing to certify an
# unidentified artifact" rather than measuring an artifact it could not name.
# That refusal is the instrument behaving correctly; the bug was here.
outFile=$(mktemp)
# SAMPLE SIZE IS NOT COSMETIC. tiergate refuses a P95 claim below n=59 (the
# Clopper-Pearson floor at which zero observed violations bounds the true
# violation rate under 5%). The defaults produce n=40 for L1 (from --iterations)
# and n=30 for L2/L3 (6 curated + 24 generated probes), so BOTH fail as
# "insufficient evidence" even at zero violations. A deploy gate that
# structurally cannot certify is not a gate, so the counts are raised here to
# clear the floor rather than left at values that guarantee a FAIL.
( cd "$GATE_DIR" && ./tiergate --epoch "$LABEL" \
    --iterations 60 --gen-probes 60 ) 2>&1 | tee "$outFile"
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
