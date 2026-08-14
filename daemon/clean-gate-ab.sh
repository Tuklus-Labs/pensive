#!/usr/bin/env bash
# The A/B that decides whether L2 meets its 20ms budget.
#
# WHY A SCRIPT AND NOT TYPING IT AT THE TIME: the whole result depends on
# changing exactly ONE variable between the two runs. Done by hand at the end of
# a long session, the easy mistake is to enable ONNX and wait for quiet in the
# same step, then credit ONNX with a delta that was mostly the box emptying out.
# That is a plausible, well-evidenced, completely wrong claim, and this file
# exists so it cannot be made by accident.
#
# THE ORDER:
#   A. restart on current source with ONNX OFF, gate it   -> clean baseline
#   B. enable ONNX, restart, gate it                      -> clean candidate
#   difference(A, B) is the ONNX effect and nothing else.
#
# Run A is the one worth protecting. No uncontaminated L2 number exists for the
# code as it stands, so A answers a question nobody has answered yet: was L2
# ever really 24ms over budget, or was most of that the neighbours?
#
# REFUSES TO RUN ON A LOADED BOX. tiergate reports contamination itself, but by
# then it has already restarted the daemon twice for nothing.

set -uo pipefail
cd "$(dirname "$0")" || exit 2

DROPIN_DIR="$HOME/.config/systemd/user/pensive-v3.service.d"
STAGED="$DROPIN_DIR/90-onnx-embedder.conf.staged"
ACTIVE="$DROPIN_DIR/90-onnx-embedder.conf"
MAX_LOAD=${MAX_LOAD:-8}

say() { printf '\n=== %s ===\n' "$*"; }
die() { printf '\nABORTED: %s\n' "$*" >&2; exit 1; }

load1=$(cut -d' ' -f1 /proc/loadavg)
if [ "${load1%%.*}" -ge "$MAX_LOAD" ]; then
    die "load is $load1, need under $MAX_LOAD. Latency taken now measures the neighbours.
     Override with MAX_LOAD=<n> only if you intend to publish a contaminated number."
fi
say "load $load1, proceeding"

# --------------------------------------------------------------------------- #
# A: baseline. Same source, torch encoder.
# --------------------------------------------------------------------------- #
if [ -f "$ACTIVE" ]; then
    mv "$ACTIVE" "$STAGED" || die "could not stage the ONNX drop-in aside"
    systemctl --user daemon-reload
fi
say "RUN A: clean baseline, ONNX OFF"
bash deploy.sh clean-baseline
aStatus=$?
printf 'run A exit: %s\n' "$aStatus"

# --------------------------------------------------------------------------- #
# B: candidate. Identical source, ONNX encoder.
# --------------------------------------------------------------------------- #
[ -f "$STAGED" ] || die "no staged drop-in at $STAGED; run B has nothing to enable"
mv "$STAGED" "$ACTIVE" || die "could not activate the ONNX drop-in"
systemctl --user daemon-reload

say "RUN B: clean candidate, ONNX ON"
bash deploy.sh clean-onnx
bStatus=$?
printf 'run B exit: %s\n' "$bStatus"

# --------------------------------------------------------------------------- #
# Read both, side by side. A gate that only prints its own verdict is not enough
# here: the QUESTION is the difference, so the difference gets printed.
# --------------------------------------------------------------------------- #
say "A/B"
python3 - <<'PY'
import json, os
base = os.path.expanduser(
    "~/Projects/pensive/daemon/gate/tiergate/.gate-evidence")


def read(epoch):
    p = os.path.join(base, epoch, "report.json")
    if not os.path.exists(p):
        return None
    out = {}
    for comp in json.load(open(p)).get("components", []):
        for unit in (comp if isinstance(comp, list) else [comp]):
            if isinstance(unit, dict) and "unit" in unit:
                out[unit["unit"]] = unit
    return out


a, b = read("clean-baseline"), read("clean-onnx")
if not a or not b:
    print("  one or both reports missing; nothing to compare")
    raise SystemExit(0)

print(f"  {'unit':34s} {'A baseline':>12s} {'B onnx':>12s}  {'delta':>9s}")
for name in sorted(set(a) | set(b)):
    ua, ub = a.get(name), b.get(name)
    va = (ua or {}).get("score", {}).get("value")
    vb = (ub or {}).get("score", {}).get("value")
    sa = (ua or {}).get("state", "-")
    sb = (ub or {}).get("state", "-")
    if va is None and vb is None:
        print(f"  {name:34s} {sa:>12s} {sb:>12s}")
    else:
        d = "" if (va is None or vb is None) else f"{vb - va:+.3f}"
        print(f"  {name:34s} {va if va is None else f'{va:10.3f}'} "
              f"{vb if vb is None else f'{vb:10.3f}'}  {d:>9s}  {sa}/{sb}")

for r, lbl in ((a, "A"), (b, "B")):
    c = r.get("contamination.background-traffic", {}).get("state")
    if c != "PASS":
        print(f"\n  WARNING: run {lbl} contamination is {c}. "
              "Its latency figures must not be quoted.")
PY
