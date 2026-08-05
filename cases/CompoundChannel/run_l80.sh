#!/usr/bin/env bash
#
# L80 pipeline: prove the SPOD / free-surface analysis is obtainable, then
# produce the statistics from the ncy = 80 grid. Sized for 64 MPI ranks.
#
#   ./run_l80.sh plan                 grid, cell aspect ratio and 64-rank
#                                     decomposition report (no MPI, no build)
#   ./run_l80.sh build                fast build (ccache + ninja if available)
#   ./run_l80.sh feasibility [case]   THE FIRST THING TO RUN: a short run on the
#                                     real L80 grid with the full SPOD probe
#                                     set, then verify end to end that SPOD and
#                                     the free-surface decomposition come out
#   ./run_l80.sh ladder      [case]   40 (DNS) -> 80 (WALE), develop the flow
#   ./run_l80.sh stats       [case]   restart L80 with statistics AND uniform
#                                     SPOD sampling switched on
#   ./run_l80.sh spod        [case]   post-process: cross-plane SPOD per k_x,
#                                     free-surface SPOD, top view of omega_y
#   ./run_l80.sh monitor     [case]   development / stationarity diagnostics
#   ./run_l80.sh all         [case]   build + plan + feasibility
#
# Environment overrides:
#   NP=64             MPI ranks
#   NCY=80            cells over the main channel depth H (multiple of 8)
#   BLOCK=15          target cells per block edge -- this is the knob that
#                     decides parallel efficiency, see 'plan'
#   TAVG=200          averaging window of the statistics run, in H/u_tau
#   DTS=0.05          target SPOD sampling interval in H/u_tau
#   SPOD_PLANES=16    equally spaced y-z planes; k_x = 0 .. SPOD_PLANES/2
#   SPOD_SUB=4        probe subsampling relative to the LES grid
#   FEAS_STEPS=2000   timesteps in the feasibility run
#   CASES="Dr05 ..."  which cases to act on
#   PRESET=gnu-release, BUILD=<dir>, MGLET_BIN=<path>
#   NATIVE=1          add -march=native (faster, but ties the binary to this CPU)
#   WERROR=0          drop -Werror (gnu presets only) so a newer gfortran cannot
#                     fail the build on a new warning
#   MPIRUN_OPTS       default "--bind-to core"
#
# Directories are suffixed .spod.* so nothing here collides with run_cases.sh,
# whose ladder rungs use a different (much less efficient) blocking.
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

BUILD="${BUILD:-$ROOT/build}"
PRESET="${PRESET:-gnu-release}"
NP="${NP:-64}"
MGLET_BIN="${MGLET_BIN:-$BUILD/src/mglet}"
CASES="${CASES:-Dr05}"

NCY="${NCY:-80}"
BLOCK="${BLOCK:-15}"
COARSE_NCY="${COARSE_NCY:-40}"
COARSE_BLOCK="${COARSE_BLOCK:-10}"
COARSE_TEND="${COARSE_TEND:-60}"
L80_TEND="${L80_TEND:-25}"
TAVG="${TAVG:-200}"
DTS="${DTS:-0.05}"
SPOD_PLANES="${SPOD_PLANES:-16}"
SPOD_SUB="${SPOD_SUB:-4}"
FEAS_STEPS="${FEAS_STEPS:-2000}"
MPIRUN_OPTS="${MPIRUN_OPTS:---bind-to core}"

RED=$'\033[31m'; GRN=$'\033[32m'; YEL=$'\033[33m'; OFF=$'\033[0m'
say()  { printf '%s==>%s %s\n' "$GRN" "$OFF" "$*"; }
warn() { printf '%s!!%s  %s\n' "$YEL" "$OFF" "$*"; }
die()  { printf '%sxx%s  %s\n' "$RED" "$OFF" "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "'$1' not found in PATH"; }

# ---------------------------------------------------------------- build ----
# Where the time actually goes, and what each lever is worth:
#
#   * exprtk_wrapper.cxx includes a ~1 MB single-header expression parser and
#     is by far the most expensive translation unit (minutes, ~2 GB). ccache
#     caches it perfectly.
#   * ccache does NOT cache Fortran: gfortran emits .mod files as a side
#     effect that ccache cannot capture, so it declares every Fortran
#     compilation "uncacheable" and passes it straight through. So ccache buys
#     back the C/C++ half of a rebuild, and nothing of the Fortran half.
#   * Ninja schedules the Fortran module dependency graph better than Make,
#     which is where the remaining serialisation lives.
#   * Keeping $BUILD means FetchContent does not re-download nlohmann/json and
#     re-clone exprtk on every configure.
do_build() {
    need cmake
    local jobs="${JOBS:-$(nproc)}"
    local gen_note="Unix Makefiles"

    # exprtk at -O3 peaks near 2 GB; do not let -j oversubscribe memory.
    local memgb
    memgb=$(awk '/MemTotal/{printf "%d", $2/1048576}' /proc/meminfo 2>/dev/null || echo 8)
    local memjobs=$(( memgb / 2 )); [[ $memjobs -lt 1 ]] && memjobs=1
    [[ $jobs -gt $memjobs ]] && {
        warn "capping -j$jobs to -j$memjobs (${memgb} GB RAM; exprtk needs ~2 GB/job)"
        jobs=$memjobs
    }

    local -a cfg=()
    if command -v ccache >/dev/null 2>&1; then
        cfg+=(-DCMAKE_C_COMPILER_LAUNCHER=ccache
              -DCMAKE_CXX_COMPILER_LAUNCHER=ccache)
        say "ccache on for C/C++ (Fortran is uncacheable - .mod side effects)"
    fi
    if command -v ninja >/dev/null 2>&1; then
        if [[ -f "$BUILD/CMakeCache.txt" ]] && \
           ! grep -q "CMAKE_GENERATOR:INTERNAL=Ninja" "$BUILD/CMakeCache.txt"; then
            warn "$BUILD was configured with Make; keeping it (delete it to switch to Ninja)"
        else
            export CMAKE_GENERATOR=Ninja
            gen_note="Ninja"
        fi
    fi

    # These two only apply to the gnu presets, whose flag lists are reproduced
    # here; other toolchains keep their preset flags untouched.
    if [[ "$PRESET" == gnu-* ]]; then
        local fflags="-Wall;-Wextra;-Wpedantic;-Wno-maybe-uninitialized;-std=f2018;-fimplicit-none;-ffpe-trap=invalid,zero,overflow;-Wno-unused-dummy-argument;-Wno-uninitialized"
        local cflags="-Wall;-Wextra;-Wpedantic;-Wno-maybe-uninitialized"
        if [[ "${WERROR:-1}" == "0" ]]; then
            say "building without -Werror"
            cfg+=(-DMGLET_Fortran_FLAGS="$fflags"
                  -DMGLET_C_FLAGS="$cflags" -DMGLET_CXX_FLAGS="$cflags")
        fi
        if [[ "${NATIVE:-0}" == "1" ]]; then
            say "building with -march=native (binary is tied to this CPU)"
            local rel="-g;-fopenmp-simd;-march=native;-funroll-loops"
            cfg+=(-DMGLET_Fortran_FLAGS_RELEASE="$rel"
                  -DMGLET_C_FLAGS_RELEASE="$rel"
                  -DMGLET_CXX_FLAGS_RELEASE="$rel")
        fi
    elif [[ -n "${NATIVE:-}${WERROR:-}" ]]; then
        warn "NATIVE/WERROR are only wired up for the gnu-* presets; ignoring"
    fi

    say "configuring preset '$PRESET' ($gen_note) into $BUILD"
    mkdir -p "$BUILD"
    ( cd "$BUILD" && cmake --preset="$PRESET" "${cfg[@]}" "$ROOT" ) || die "cmake failed.
If it stopped at FortranCInterface_VERIFY, your C and Fortran compilers are
from different toolchains (a classic conda-vs-system mix). Put one toolchain
first in PATH, or point CC/CXX/FC at a matching set, and delete $BUILD first."

    say "building target 'mglet' with -j$jobs"
    cmake --build "$BUILD" -j"$jobs" --target mglet || die "compilation failed"
    [[ -x "$MGLET_BIN" ]] || die "expected binary at $MGLET_BIN"
    command -v ccache >/dev/null 2>&1 && ccache -s 2>/dev/null | \
        grep -Ei "hits|misses|uncacheable" | head -4 || true
    say "built $MGLET_BIN"
}

# ----------------------------------------------------------------- plan ----
# Answers, without running anything: what is the cell aspect ratio at this ncy,
# how many grids does the blocking produce, and how evenly do they land on NP
# ranks. MGLET cannot split a grid across ranks (dist_grids in grids_mod.F90
# hands out whole grids), so ngrid >= NP is mandatory and ngrid/NP decides the
# imbalance.
do_plan() {
    python3 - "$HERE" "$NCY" "$BLOCK" "$NP" "$COARSE_NCY" "$COARSE_BLOCK" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from make_case import (Blocking, closest_divisor, hydraulic_radius,
                       NYB, Z_JUNCTION, RE_TAU)

here, ncy, block, np_, cncy, cblock = (sys.argv[1], int(sys.argv[2]),
                                       int(sys.argv[3]), int(sys.argv[4]),
                                       int(sys.argv[5]), int(sys.argv[6]))
NG, LX, AX, AZ = 2, 12.0, 3.2, 1.6

def report(ncy, block, title):
    ncx = int(round(LX / (AX / ncy)))
    nzh = int(round(Z_JUNCTION / (AZ / ncy)))
    nxb = closest_divisor(ncx, max(1, round(ncx / block)))
    nzbh = closest_divisor(nzh, max(1, round(nzh / block)))
    dx, dy, dz = LX / ncx, 1.0 / ncy, 5.0 / (2 * nzh)
    print(f"\n=== {title}: ncy = {ncy}, --block {block} ===")
    print(f"  bounding box    {ncx} x {ncy} x {2*nzh} cells")
    print(f"  spacing         dx = {dx:.4f}H  dy = {dy:.4f}H  dz = {dz:.4f}H")
    print(f"  CELL ASPECT     dx/dy = {dx/dy:.2f}   dz/dy = {dz/dy:.2f}"
          f"   dx/dz = {dx/dz:.2f}")
    print(f"  {'Dr':>5} {'Re_tau':>7} {'dx+':>6} {'dy+':>6} {'dz+':>6} "
          f"{'y1+':>5} {'grids':>6} {'block':>12} {'Mcell':>7} "
          f"{'per rank':>10} {'imbal':>6} {'ghost':>6}")
    for dr in (0.500, 0.375, 0.250):
        b = Blocking(dr, LX, ncx, ncy, nzh, nxb, 2 * nzbh)
        ret = RE_TAU[dr]; dnu = 1.0 / ret
        cells = b.ngrid * b.cx * b.cy * b.cz
        alloc = b.ngrid * (b.cx+2*NG) * (b.cy+2*NG) * (b.cz+2*NG)
        mx = -(-b.ngrid // np_)
        print(f"  {dr:>5} {ret:>7.0f} {b.dx/dnu:>6.1f} {b.dy/dnu:>6.2f} "
              f"{b.dz/dnu:>6.1f} {0.5*b.dy/dnu:>5.2f} {b.ngrid:>6} "
              f"{f'{b.cx}x{b.cy}x{b.cz}':>12} {cells/1e6:>7.2f} "
              f"{b.ngrid/np_:>6.2f}->{mx:<3} {100*(mx/(b.ngrid/np_)-1):>5.1f}% "
              f"{alloc/cells:>5.2f}x")
        if b.ngrid < np_:
            print(f"        !! only {b.ngrid} grids for {np_} ranks - MGLET "
                  f"needs ngrid >= ranks")

report(cncy, cblock, "transition rung")
report(ncy, block, "L80 production rung")

# What --block does to efficiency at the L80 size
ncx = int(round(LX / (AX / ncy)))
nzh = int(round(Z_JUNCTION / (AZ / ncy)))
print(f"\n=== effect of --block at ncy = {ncy}, Dr = 0.5, NP = {np_} ===")
print(f"  {'block':>5} {'shape':>12} {'grids':>6} {'per rank':>11} "
      f"{'imbal':>6} {'ghost':>7} {'interior/rank':>14} {'alloc/rank':>11} "
      f"{'halo/rank':>10}")
best_c = best_m = None
rows = {}
for t in (8, 10, 12, 15, 20, 25, 30):
    nxb = closest_divisor(ncx, max(1, round(ncx / t)))
    nzbh = closest_divisor(nzh, max(1, round(nzh / t)))
    b = Blocking(0.5, LX, ncx, ncy, nzh, nxb, 2 * nzbh)
    if b.ngrid < np_:
        continue
    mx = -(-b.ngrid // np_)
    interior = b.cx * b.cy * b.cz
    ac = (b.cx+2*NG) * (b.cy+2*NG) * (b.cz+2*NG)
    compute, mem, halo = mx * interior, mx * ac, mx * (ac - interior)
    tag = " <- default" if t == int(sys.argv[3]) else ""
    print(f"  {t:>5} {f'{b.cx}x{b.cy}x{b.cz}':>12} {b.ngrid:>6} "
          f"{b.ngrid/np_:>6.2f}->{mx:<3} {100*(mx/(b.ngrid/np_)-1):>5.1f}% "
          f"{ac/interior:>6.2f}x {compute/1e3:>13.0f}k {mem/1e3:>10.0f}k "
          f"{halo/1e3:>9.0f}k{tag}")
    rows[t] = (compute, halo, mx / (b.ngrid / np_) - 1)
    if best_c is None or compute < best_c[1]:
        best_c = (t, compute)
    if best_m is None or halo < best_m[1]:
        best_m = (t, halo)

cspread = max(c for c, _, _ in rows.values()) / min(c for c, _, _ in rows.values())
hspread = max(h for _, h, _ in rows.values()) / min(h for _, h, _ in rows.values())
dflt = int(sys.argv[3])
print(f"""
  Read the two cost columns against each other:

    interior/rank  the busiest rank's flop count. Ghost cells are allocated
                   and exchanged but never integrated, so this is what the
                   interior loops and the SIP pressure solve actually cost.
                   Minimised by good LOAD BALANCE (many small grids).
    halo/rank      allocated minus interior: memory and MPI volume for the
                   2-cell halo every grid carries on all six faces.
                   Minimised by FEW LARGE grids.

  Across every option here flops vary by only {100*(cspread-1):.0f} % while halo varies by
  {hspread:.1f}x. Load balance is therefore NOT the thing to optimise at this size --
  all these blockings balance well enough. Halo is the discriminator.

  --block 8 is what run_cases.sh uses, and it is the one to avoid: its
  10x10x5 grids are so small that {rows[8][1]/(rows[8][0]+rows[8][1])*100:.0f} % of every allocated array is
  halo, for {rows[8][1]/rows[dflt][1]:.1f}x the exchange volume of --block {dflt} and no flop saving.

  --block {dflt} (default) is the knee: {100*rows[dflt][2]:.1f} % imbalance, {rows[dflt][1]/1e3:.0f}k halo cells.
  --block {best_m[0]} cuts halo a further {100*(1-rows[best_m[0]][1]/rows[dflt][1]):.0f} % for {100*rows[best_m[0]][2]:.1f} % imbalance; worth it only if
  mglet-perf-report.txt shows the run is communication bound (compare the
  CONNECT/COMM timers against the flow timers).""")
PY
}

# ----------------------------------------------------------------- runs ----
have_binary() {
    [[ -x "$MGLET_BIN" ]] || die "no MGLET binary at $MGLET_BIN; run '$0 build'"
}

check_ranks() {
    python3 - "$1/grids.h5" "$NP" <<'PY'
import sys, h5py
with h5py.File(sys.argv[1], "r") as f:
    gi = f["GRIDINFO"]
    n = len(gi)
    ii, jj, kk = int(gi[0]["II"]), int(gi[0]["JJ"]), int(gi[0]["KK"])
np_ = int(sys.argv[2])
if n < np_:
    sys.exit(f"grids.h5 has only {n} grids but NP={np_}; MGLET needs ngrid >= ranks")
mx = -(-n // np_)
print(f"    {n} grids of {ii-4}x{jj-4}x{kk-4}, {n/np_:.2f} per rank "
      f"(max {mx}, imbalance {100*(mx/(n/np_)-1):.1f} %)")
PY
}

launch() {   # launch <dir> <label>
    local dir="$1" label="$2"
    have_binary
    [[ -f "$dir/parameters.json" ]] || die "no parameters.json in $dir"
    check_ranks "$dir"
    say "$label: mpirun -n $NP $MPIRUN_OPTS in $(basename "$dir")"
    ( cd "$dir" && OMP_NUM_THREADS=1 mpirun -n "$NP" $MPIRUN_OPTS \
        "$MGLET_BIN" 2>&1 | tee mglet.OUT )
    if grep -q "MGLET FINISHED SUCCESSFULLY" "$dir/mglet.OUT"; then
        say "$label finished"
    else
        warn "$label did not report success - check $dir/mglet.OUT"
    fi
}

case_dr() {
    python3 -c "
import json;g=json.load(open('$HERE/$1/parameters.json'))['flow']['gradp'][0]
print(round((-1.0/g*7.0-2.5)/2.5,3))"
}

# Cost of the run just finished, and what it implies for the averaging run.
report_cost() {   # report_cost <dir> <tavg>
    python3 - "$1" "$2" <<'PY'
import os, sys, json
d, tavg = sys.argv[1], float(sys.argv[2])
log = os.path.join(d, "LOGS", "time.log")
if not os.path.exists(log):
    sys.exit(0)
rows = [l.split() for l in open(log) if l.strip() and not l.lstrip().startswith("#")]
rows = [r for r in rows if len(r) >= 5]
if len(rows) < 3:
    sys.exit(0)
it = [int(r[0]) for r in rows]
wt = [float(r[3]) for r in rows]
# skip the first interval: it carries grid setup and first-touch page faults
dsteps, dwall = it[-1] - it[1], wt[-1] - wt[1]
if dsteps <= 0:
    sys.exit(0)
sps = dwall / dsteps
dt = float(rows[-1][2])
print(f"    {sps*1e3:.1f} ms/step at this rank count ({dsteps} steps timed)")
print(f"    dt = {dt:.3g} -> {tavg/dt:,.0f} steps for TAVG = {tavg:g} H/u_tau"
      f"  ~ {sps*tavg/dt/3600:.1f} h")
PY
}

# --------------------------------------------------------- feasibility -----
# Everything the SPOD ambition depends on, exercised on the real production
# grid but for a few thousand steps: are all 19925 probe points inside a grid,
# is the sampling interval strictly uniform, is the surface plane written, what
# does it cost, and does the whole post-processing chain run to completion.
#
# tstat = 0 from a cold start does two things at once (timeloop_mod.F90:313):
# statistics sample from step 1, and dt is never adapted - so the probe series
# is uniformly spaced, which is the one property SPOD cannot do without.
do_feasibility() {
    for c in $@; do
        local dst="$HERE/$c.spod.feas" dr; dr=$(case_dr "$c")
        say "feasibility[$c]: ncy=$NCY, block=$BLOCK, $FEAS_STEPS steps, full SPOD probes"
        python3 "$HERE/make_case.py" --dr "$dr" --outdir "$dst" --ncy "$NCY" \
            --block "$BLOCK" --lesmodel wale --tend 1e9 --tstat 0.0 \
            --spod-planes "$SPOD_PLANES" --spod-sub "$SPOD_SUB" \
            --probe-itsamp "${FEAS_ITSAMP:-10}" || die "make_case failed"
        python3 - "$dst/parameters.json" "$FEAS_STEPS" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
d["time"].update(mtstep=int(sys.argv[2]), itinfo=100)
json.dump(d, open(sys.argv[1], "w"), indent=4)
PY
        rm -f "$dst/probes.h5"
        launch "$dst" "feasibility[$c]"
        report_cost "$dst" "$TAVG"

        say "feasibility[$c]: verifying the SPOD data path"
        python3 "$HERE/spod.py" "$dst" --check --nfft "${FEAS_NFFT:-64}" || \
            warn "the check above found a problem - fix it before the long run"
        say "feasibility[$c]: running the full decomposition on this short record"
        python3 "$HERE/spod.py" "$dst" --nfft "${FEAS_NFFT:-64}" || \
            warn "post-processing failed"
        say "feasibility[$c]: plots in $dst/spod/"
        warn "the physics here is meaningless (2000 steps from the initial"
        warn "condition, no turbulence yet) - this only proves the machinery."
    done
}

# --------------------------------------------------------------- ladder ----
patch_restart() {   # patch_restart <parameters.json> <tend>
    python3 - "$1" "$2" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
d["io"]["infile"] = "restart.h5"
d["io"]["outfile"] = "fields.h5"
d["time"].update(read=True, tend=float(sys.argv[2]), tstat=1e9)
d["time"]["continue"] = False     # clock restarts at 0; no RUNINFO needed
d.pop("statistics", None)
json.dump(d, open(sys.argv[1], "w"), indent=4)
PY
}

# Transition on the cheap grid as DNS (an eddy viscosity on a coarse grid
# suppresses transition), then one interpolation onto L80 and a short
# re-adjustment there. Probes are off on both rungs: nothing is being measured
# yet, and 20k probe points sampled every step is not free.
do_ladder() {
    for c in $@; do
        local dr; dr=$(case_dr "$c")
        local coarse="$HERE/$c.spod.L$COARSE_NCY" fine="$HERE/$c.spod.L$NCY"

        say "ladder[$c]: rung 1, ncy=$COARSE_NCY, DNS, tend=$COARSE_TEND"
        python3 "$HERE/make_case.py" --dr "$dr" --outdir "$coarse" \
            --ncy "$COARSE_NCY" --block "$COARSE_BLOCK" --lesmodel none \
            --tend "$COARSE_TEND" --tstat 1e9 --spod-planes 0 --no-surface
        python3 - "$coarse/parameters.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); d["time"]["itinfo"] = 20
d.pop("probes", None)
json.dump(d, open(sys.argv[1], "w"), indent=4)
PY
        launch "$coarse" "ladder[$c/ncy=$COARSE_NCY]"
        python3 "$HERE/monitor.py" "$coarse" || true

        say "ladder[$c]: rung 2, ncy=$NCY, WALE, tend=$L80_TEND"
        python3 "$HERE/make_case.py" --dr "$dr" --outdir "$fine" --ncy "$NCY" \
            --block "$BLOCK" --lesmodel wale --tend "$L80_TEND" --tstat 1e9 \
            --spod-planes 0 --no-surface
        python3 "$HERE/map_field.py" --from "$coarse" --to "$fine" \
            --out "$fine/restart.h5"
        patch_restart "$fine/parameters.json" "$L80_TEND"
        python3 - "$fine/parameters.json" <<'PY'
import json, sys
# The SPOD planes are off on this rung (nothing is being measured yet), but the
# x-rakes and profiles stay: they are only 436 points and they are what
# monitor.py needs for the R_uu domain-length check.
d = json.load(open(sys.argv[1])); d["time"]["itinfo"] = 20
json.dump(d, open(sys.argv[1], "w"), indent=4)
PY
        launch "$fine" "ladder[$c/ncy=$NCY]"
        python3 "$HERE/monitor.py" "$fine" || true
        report_cost "$fine" "$TAVG"
        say "ladder[$c] done. If UBULK has levelled off: $0 stats $c"
    done
}

# ---------------------------------------------------------------- stats ----
# Restart the developed L80 field with statistics on and SPOD sampling on.
#
# tstat = 0 is what makes the sampling SPOD-grade: statistics start at the
# first step, and dt is never adapted (timeloop_mod.F90:313 only adjusts while
# timeph < tstat), so the probe series is exactly equispaced.
#
# "continue" is deliberately FALSE, which is the opposite of what a restart
# usually wants. With continue = true the statistics fields are marked
# dread = dcont (statistics_mod.F90:75) and MGLET tries to read U_AVG and
# friends out of restart.h5. The ladder rung never computed them, so the field
# is missing; fieldio_read only tolerates that when its optional `required`
# argument is passed, which init_statistics does not do, and the read falls
# through to hdf5common_group_open, which tries to CREATE the missing group in
# a file opened read-only. The run dies at startup with
# "Group creation failed: U_AVG".
#
# With continue = false the statistics fields are simply allocated and zeroed,
# which is what a fresh averaging window wants anyway. The cost is that MGLET
# then also skips read_runinfo, so dt is taken from the JSON instead of from
# the restart file (timeloop_mod.F90:123) -- so the converged dt is read out of
# RUNINFO here and written into the JSON explicitly. The clock restarts at 0,
# which makes t = 0 the start of the averaging window.
do_stats() {
    for c in $@; do
        local src="$HERE/$c.spod.L$NCY" dst="$HERE/$c.spod.avg"
        local dr; dr=$(case_dr "$c")
        [[ -f "$src/fields.h5" ]] || \
            die "no developed field at $src/fields.h5 - run '$0 ladder $c' first"

        say "stats[$c]: staging from $(basename "$src"), TAVG=$TAVG"
        python3 "$HERE/make_case.py" --dr "$dr" --outdir "$dst" --ncy "$NCY" \
            --block "$BLOCK" --lesmodel wale --tend 1e9 --tstat 0.0 \
            --spod-planes "$SPOD_PLANES" --spod-sub "$SPOD_SUB" >/dev/null
        cp -f "$src/fields.h5" "$dst/restart.h5"
        [[ -f "$dst/probes.h5" ]] && mv -f "$dst/probes.h5" "$dst/probes.h5.bak"

        python3 - "$dst/parameters.json" "$dst/restart.h5" "$TAVG" "$DTS" <<'PY'
import json, sys, h5py
d = json.load(open(sys.argv[1]))
t0, dt0 = 0.0, None
with h5py.File(sys.argv[2], "r") as f:
    if "RUNINFO" in f:
        ri = f["RUNINFO"][-1]
        t0, dt0 = float(ri["TIMEPH"]), float(ri["DT"])
if not dt0 or dt0 <= 0.0:
    dt0 = float(d["time"]["dt"])
    print(f"    !! no usable RUNINFO/DT in the restart file; falling back to "
          f"the cold-start estimate dt = {dt0:.4g}. Check the CFL in "
          f"LOGS/general.log once the run starts.")
tavg, dts = float(sys.argv[3]), float(sys.argv[4])

d["io"]["infile"] = "restart.h5"
d["io"]["outfile"] = "fields.h5"
# continue = false on purpose (see the comment above do_stats): it keeps the
# statistics fields out of the restart read. dt must therefore be set here.
d["time"].update(read=True, tend=tavg, tstat=0.0, itinfo=100, dt=dt0)
d["time"]["continue"] = False
d["statistics"] = ["U_AVG", "V_AVG", "W_AVG", "P_AVG",
                   "UU_AVG", "VV_AVG", "WW_AVG",
                   "UV_AVG", "UW_AVG", "VW_AVG"]

# Sample the probes on a round number of steps closest to the target interval.
itsamp = max(1, round(dts / dt0))
d["probes"]["itsamp"] = itsamp
d["probes"]["tstart"] = 0.0
json.dump(d, open(sys.argv[1], "w"), indent=4)

nsamp = tavg / (itsamp * dt0)
nblk = max(0, 2 * int(nsamp // 256) - 1)
print(f"    source field is at t = {t0:.2f}; the averaging clock restarts at 0")
print(f"    dt = {dt0:.4g} (converged value from RUNINFO, frozen by tstat = 0)")
print(f"    tend = {tavg:g} -> {tavg/dt0:,.0f} timesteps")
print(f"    /probes/itsamp = {itsamp} -> dt_s = {itsamp*dt0:.4g} H/u_tau, "
      f"f_Nyquist = {0.5/(itsamp*dt0):.2f} u_tau/H")
print(f"    {nsamp:,.0f} probe samples -> {nblk} Welch blocks of 256 at 50 % "
      f"overlap ({100/max(nblk,1)**0.5:.0f} % error per eigenvalue)")
if nsamp < 1024:
    print(f"    !! fewer than 1024 samples; raise TAVG or lower DTS")
PY
        launch "$dst" "stats[$c]"
        say "stats[$c]: statistics are in $dst/fields.h5, SPOD input in $dst/probes.h5"
        say "next: $0 spod $c   and   python3 postprocess.py $dst --plot"
    done
}

do_spod() {
    for c in $@; do
        local dst="$HERE/$c.spod.avg"
        [[ -d "$dst" ]] || die "no $dst - run '$0 stats $c' first"
        python3 "$HERE/spod.py" "$dst" --nfft "${NFFT:-256}" ${KX:+--kx $KX}
    done
}

do_monitor() { for c in $@; do
    for d in "$HERE/$c.spod.avg" "$HERE/$c.spod.L$NCY" "$HERE/$c.spod.feas"; do
        [[ -d "$d" ]] && { say "monitor $(basename "$d")";
                           python3 "$HERE/monitor.py" "$d" || true; }
    done
done; }

# ----------------------------------------------------------------- main ----
cmd="${1:-}"; shift || true
targets="${*:-$CASES}"

case "$cmd" in
    plan)        do_plan ;;
    build)       do_build ;;
    feasibility) do_feasibility $targets ;;
    ladder)      do_ladder $targets ;;
    stats)       do_stats $targets ;;
    spod)        do_spod $targets ;;
    monitor)     do_monitor $targets ;;
    all)
        do_build
        do_plan
        do_feasibility $targets
        say "if the verdicts above are all OK: $0 ladder && $0 stats && $0 spod"
        ;;
    *)
        sed -n '3,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
        exit 1 ;;
esac
