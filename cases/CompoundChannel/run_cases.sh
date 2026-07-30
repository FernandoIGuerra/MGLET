#!/usr/bin/env bash
#
# Build MGLET and run the compound open channel cases.
#
#   ./run_cases.sh build              configure + compile
#   ./run_cases.sh check              run the MGLET test suite
#   ./run_cases.sh smoke   [case...]  20 steps at full size: does the grid load?
#   ./run_cases.sh spinup  [case...]  stage 1: spin up, no statistics
#   ./run_cases.sh run     [case...]  full production run (single stage)
#   ./run_cases.sh average [case...]  restart a spinup with statistics on
#                                    (TAVG=200 sets the averaging window)
#   ./run_cases.sh monitor [case...]  development / stationarity diagnostics
#   ./run_cases.sh stats   [case...]  assemble statistics into (y,z) fields
#   ./run_cases.sh all                build + check + smoke + spinup (NOT run)
#
# Environment overrides:
#   NP=64            MPI ranks
#   PRESET=gnu-release   CMake preset (gnu-release, intel-release, ...)
#   BUILD=<dir>      build directory
#   MGLET_BIN=<path> use an existing binary instead of building
#   CASES="Dr05 ..." which cases to act on by default
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

BUILD="${BUILD:-$ROOT/build}"
PRESET="${PRESET:-gnu-release}"
NP="${NP:-64}"
MGLET_BIN="${MGLET_BIN:-$BUILD/src/mglet}"
CASES="${CASES:-Dr05 Dr0375 Dr025}"

RED=$'\033[31m'; GRN=$'\033[32m'; YEL=$'\033[33m'; OFF=$'\033[0m'
say()  { printf '%s==>%s %s\n' "$GRN" "$OFF" "$*"; }
warn() { printf '%s!!%s  %s\n' "$YEL" "$OFF" "$*"; }
die()  { printf '%sxx%s  %s\n' "$RED" "$OFF" "$*" >&2; exit 1; }

need() { command -v "$1" >/dev/null 2>&1 || die "'$1' not found in PATH"; }

# ---------------------------------------------------------------- build ----
do_build() {
    need cmake; need make
    say "configuring with preset '$PRESET' into $BUILD"
    mkdir -p "$BUILD"
    ( cd "$BUILD" && cmake --preset="$PRESET" "$ROOT" ) || die "cmake failed.
If it stopped at FortranCInterface_VERIFY, your C and Fortran compilers are
from different toolchains (a classic conda-vs-system mix). Put one toolchain
first in PATH, or point CC/CXX/FC at a matching set, and delete $BUILD first."
    say "compiling with -j$(nproc)"
    make -C "$BUILD" -j"$(nproc)" || die "compilation failed"
    [[ -x "$MGLET_BIN" ]] || die "expected binary at $MGLET_BIN"
    say "built $MGLET_BIN"
}

do_check() {
    [[ -d "$BUILD" ]] || die "no build dir; run '$0 build' first"
    say "running MGLET test suite"
    ctest --output-on-failure --test-dir "$BUILD/tests"
}

# ----------------------------------------------------------------- runs ----
have_binary() {
    [[ -x "$MGLET_BIN" ]] || die "no MGLET binary at $MGLET_BIN; run '$0 build'"
}

check_ranks() {   # a grid cannot be split, so ngrid must be >= NP
    local dir="$1"
    python3 - "$dir/grids.h5" "$NP" <<'PY'
import sys, h5py
with h5py.File(sys.argv[1], "r") as f:
    n = len(f["GRIDINFO"])
np = int(sys.argv[2])
if n < np:
    sys.exit(f"grids.h5 has only {n} grids but NP={np}; MGLET needs ngrid >= ranks")
print(f"    {n} grids on {np} ranks ({n/np:.1f} per rank)")
PY
}

launch() {   # launch <dir> <label>
    local dir="$1" label="$2"
    have_binary
    [[ -f "$dir/parameters.json" ]] || die "no parameters.json in $dir"
    check_ranks "$dir"
    say "$label: mpirun -n $NP in $dir"
    ( cd "$dir" && mpirun -n "$NP" "$MGLET_BIN" 2>&1 | tee mglet.OUT )
    if grep -q "MGLET FINISHED SUCCESSFULLY" "$dir/mglet.OUT"; then
        say "$label finished"
    else
        warn "$label did not report success - check $dir/mglet.OUT"
    fi
}

# Stage a variant of a case: same grid (symlinked), patched parameters.json.
stage() {   # stage <case> <suffix>  ; patch script on stdin
    local case="$1" suffix="$2"
    local src="$HERE/$case" dst="$HERE/$case.$suffix"
    [[ -d "$src" ]] || die "no such case: $src"
    mkdir -p "$dst"
    ln -sf "$src/grids.h5" "$dst/grids.h5"
    python3 - "$src/parameters.json" "$dst/parameters.json"
    echo "$dst"
}

do_smoke() {
    for c in $@; do
        local dst
        dst=$(stage "$c" smoke <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
d["time"].update(mtstep=20, tend=1e9, tstat=1e9, itinfo=1)
d.pop("probes", None)          # probe setup is not what we are testing here
d.pop("statistics", None)
json.dump(d, open(sys.argv[2], "w"), indent=4)
PY
        )
        launch "$dst" "smoke[$c]"
    done
}

do_spinup() {
    for c in $@; do
        local dst
        dst=$(stage "$c" spinup <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
# Long enough to see transition, short enough to be cheap. Statistics off:
# the only question here is whether turbulence establishes itself.
d["time"].update(tend=30.0, tstat=1e9, itinfo=50)
d.pop("statistics", None)
d["probes"]["arrays"] = [a for a in d["probes"]["arrays"]
                         if not a["name"].startswith(("xsec", "surface"))]
json.dump(d, open(sys.argv[2], "w"), indent=4)
PY
        )
        launch "$dst" "spinup[$c]"
        say "spinup[$c]: check that UBULK levels off and VVBULK/WWBULK grow:"
        python3 "$HERE/monitor.py" "$dst" || true
    done
}

# Stage 2: restart from a finished spinup with statistics switched on.
# tstat and tend are decided from the observed spinup, not guessed in advance.
do_average() {
    local tavg="${TAVG:-200}"
    for c in $@; do
        local src="$HERE/$c.spinup" dst="$HERE/$c.avg"
        [[ -f "$src/fields.h5" ]] || die "no $src/fields.h5 - run '$0 spinup $c' first"
        mkdir -p "$dst"
        ln -sf "$HERE/$c/grids.h5" "$dst/grids.h5"
        cp -f "$src/fields.h5" "$dst/restart.h5"
        python3 - "$HERE/$c/parameters.json" "$dst/parameters.json" \
                  "$dst/restart.h5" "$tavg" <<'PY'
import json, sys, h5py
d = json.load(open(sys.argv[1]))
with h5py.File(sys.argv[3], "r") as f:
    t0 = float(f["RUNINFO"][-1]["TIMEPH"])
tavg = float(sys.argv[4])
d["io"]["infile"] = "restart.h5"
d["io"]["outfile"] = "fields.h5"
d["time"].update(read=True, continue_=True, tend=t0 + tavg, tstat=0.0)
d["time"]["continue"] = d["time"].pop("continue_")
# tstat = 0 with a restored timeph > 0 means statistics start immediately and
# dt is frozen from the first step (src/timeloop_mod.F90:272,313), which is
# what SPOD needs -- a strictly uniform sampling interval.
json.dump(d, open(sys.argv[2], "w"), indent=4)
print(f"    restarting at t = {t0:.2f}, averaging to t = {t0+tavg:.2f}")
PY
        launch "$dst" "average[$c]"
    done
}

do_run() {
    for c in $@; do launch "$HERE/$c" "run[$c]"; done
}

do_monitor() { for c in $@; do python3 "$HERE/monitor.py" "$HERE/$c" --plot; done; }
do_stats()   { for c in $@; do python3 "$HERE/postprocess.py" "$HERE/$c" --plot; done; }

# ----------------------------------------------------------------- main ----
cmd="${1:-}"; shift || true
targets="${*:-$CASES}"

case "$cmd" in
    build)   do_build ;;
    check)   do_check ;;
    smoke)   do_smoke $targets ;;
    spinup)  do_spinup $targets ;;
    run)     do_run $targets ;;
    average) do_average $targets ;;
    monitor) do_monitor $targets ;;
    stats)   do_stats $targets ;;
    all)
        do_build
        do_check
        do_smoke $CASES
        do_spinup $CASES
        say "staged checks done. Inspect the spinup, pick t_start, then: TAVG=200 $0 average"
        ;;
    *)
        sed -n '3,21p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
        exit 1 ;;
esac
