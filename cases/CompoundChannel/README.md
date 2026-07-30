# Compound open channel — Tominaga & Nezu (1991) validation cases

LES of turbulent flow in an asymmetric compound open channel, set up to
reproduce the experiments of

> Tominaga, A. & Nezu, I. (1991) *Turbulent structure in compound open-channel
> flows*, J. Hydraulic Engineering **117**(1), 21–41

using the computational setup of

> Kara, S., Stoesser, T. & Sturm, T.W. (2012) *Turbulence statistics in
> compound channels with deep and shallow overbank flows*, J. Hydraulic
> Research **50**(5), 482–493.

Both papers are in `Reference/`.

## Cases

`Dr = h/H` is the floodplain-to-main-channel depth ratio.

| directory | Dr | experiment | Re_τ | Re = 4U_b R_h/ν | U_b/u_τ | cells |
|---|---|---|---|---|---|---|
| `Dr05/`   | 0.500 | **S-2** | 1193 | 5.44 × 10⁴ | 21.28 | 36.0 M |
| `Dr0375/` | 0.375 | *(none — interpolated)* | 1109 | 4.54 × 10⁴ | 20.85 | 33.0 M |
| `Dr025/`  | 0.250 | **S-3** | 1025 | 3.74 × 10⁴ | 20.43 | 30.0 M |

Tominaga & Nezu's S-1 (Dr = 0.75) and R-1 (Dr = 0.5, rough floodplain) are not
reproduced. Dr = 0.375 has no experimental counterpart; it is a prediction case
whose u_τ and U_b/u_τ are linearly interpolated between S-2 and S-3, with the
geometry computed exactly.

### Caveat on Table 1 of Tominaga & Nezu

The printed Reynolds and Froude numbers are reproducible from the printed
H, h, U_m for S-1, S-2 and R-1 with a single ν = 1.10 × 10⁻⁶ m²/s (≈16 °C), but
**not for S-3**: its printed R = 4.56 × 10⁴ and F = 0.402 both imply
U_m ≈ 0.35 m/s, whereas the printed U_m = 0.288 m/s (the value consistent with
the U_max/U_m trend) gives R = 3.77 × 10⁴ and F = 0.324. The setup here anchors
on the measured friction velocity U\* = 0.0141 m/s instead, which is unaffected.
Note that Kara et al. took the printed R = 45 600 at face value.

## Geometry and boundary conditions

Non-dimensional: **H = 1, u_τ = 1, ρ = 1**.

```
 y
 1 +--------------------------+--------------------------+   <- SLI (rigid lid)
   |                          |                          |
   |        main channel      |        floodplain        |
 D +                          +--------------------------+   <- NOS (fp bed)
   |                          |  solid step              |
 0 +--------------------------+                              <- NOS (bed)
   0                         2.5                         5   z
   ^ NOS (sidewall)           ^ NOS (step riser)         ^ NOS (sidewall)
```

- `x` streamwise, `[0, 12H]`, **periodic** (MGLET `CON` self-connection)
- `y` vertical, `[0, H]`; bed and floodplain bed `NOS`, free surface `SLI`
- `z` spanwise, `[0, 5H]`; both sidewalls and the step riser `NOS`
- step height `D = (1 − Dr)·H`, solid for `z > 2.5H` and `y < D`

The free surface is a frictionless rigid lid treated as a symmetry plane, as in
Kara et al. This is justified by Fr = 0.32–0.39 (surface deformation ~ Fr²) and
means the Froude number is deliberately *not* matched.

The L-shaped cross-section is built from **body-fitted Cartesian blocks**, so no
immersed boundary is needed (`"ib": {"type": "noib"}`).

## Forcing

The flow is driven by a uniform streamwise pressure gradient, which fixes the
global friction velocity unambiguously (as in Kara et al.). From the global
momentum balance ρ g I_e A = τ̄ P, i.e. u_τ² = g I_e R_h:

```
gmol  = 1/Re_τ                (ρ = 1, so ν = gmol)
gradp = [-H/R_h, 0, 0]        R_h = A/P, A = 2.5 + 2.5·Dr, P = 7  (units of H)
```

Note `gradp` is applied as literal dp/dx (`src/flow/tstle4_mod.F90:506`), so it
must be **negative** to drive flow in +x.

**Re and Re_τ cannot both be imposed.** Re_τ is the input; Re (equivalently
U_b/u_τ) is an *output* and is the primary validation metric.

## Grid

Uniform, matched to Kara et al.'s resolution:

| | Kara et al. | here (Dr = 0.5) |
|---|---|---|
| Δx⁺ | ≈26 | 23.9 |
| Δy⁺ | ≈6 | 7.46 |
| Δz⁺ | ≈13 | 11.9 |
| cells | 36 M / 30 M | 36 M / 30 M |

## Near-wall resolution

MGLET applies the **Werner–Wengle** wall model on `NOS` faces
(`src/flow/tstle4_mod.F90:1844`). Its switch from the linear branch to the 1/7
power law is at `|u|·Δ/ν = 69.7`, which in the linear region (U⁺ = y⁺,
Δ = 2y₁) is **y₁⁺ = 5.9**, where y₁ is the first cell-centre distance.

| | Δ⁺ | first cell centre y₁⁺ | verdict |
|---|---|---|---|
| bed / floodplain bed (y-normal) | 6.4 – 7.5 | **3.2 – 3.7** | inside the viscous sublayer, WW linear branch |
| sidewalls + step riser (z-normal) | 10.3 – 11.9 | **5.1 – 6.0** | at the sublayer edge, right *on* the WW crossover |
| streamwise | 20.5 – 23.9 | — | no wall normal to x |

Accounting for the spanwise variation of bed shear (τ/τ̄ ≈ 0.6–1.3, T&N Fig. 8),
the bed value spans y₁⁺ ≈ 2.9–4.3 — still inside the sublayer everywhere.

So: **yes for the beds, marginal for the sidewalls.** Be precise about what this
is, though — a *strictly* wall-resolved LES wants Δy⁺ ≈ 1 (several cells inside
the sublayer), not one cell centre inside it. This grid puts ~1.3 cells in the
sublayer, so it is a wall-*modelled* LES whose model happens to sit in its
resolved (linear) branch at the bed. At the sidewalls, y₁⁺ ≈ 6 means the linear
assumption U⁺ = y⁺ overestimates the true U⁺ (≈5.5) by ~8 %, so the sidewall
shear is underpredicted by roughly that much. The *global* balance is unaffected
— it is fixed by `gradp` — so the error redistributes rather than accumulating.

This is the same resolution Kara et al. used (Δy⁺ ≈ 6, Δz⁺ ≈ 13) and validated
against T&N, which is why it is the default. To put the sidewall first centre
inside the sublayer as well, use `--aspect-z 1.33`: Δz⁺ ≈ 10, z₁⁺ ≈ 5, at a cost
of about +19 % cells (≈43 M for Dr = 0.5).

## How "fully developed" is established

Two separate things, often conflated:

**Streamwise development** is enforced by construction — the domain is periodic,
so there is no development direction. This matches the reference data: T&N
measured at 7.5 m downstream where "a fully developed and uniform flow was
established" (p. 22).

**Temporal stationarity** is the thing that must actually be verified. The
rigorous statement is that for a periodic channel driven by constant dp/dx, the
volume-integrated streamwise momentum equation is

    dU_b/dt = (−dp/dx)/ρ − (1/V)∮ τ_w dA

so **dU_b/dt → 0 is exactly the statement that the wall shear balances the
applied forcing**. Watching the drift in `UBULK` is not a heuristic; it *is* the
global force-balance check. `monitor.py` reports that drift, and flags "still
developing" until |dU_b/dt| < 5×10⁻³ and the cross-flow variances are non-zero
(so a laminarised state is not mistaken for a converged one).

Relevant timescales, in H/u_τ:

| process | time |
|---|---|
| domain flush-through, L_x/U_b | 0.56 |
| eddy turnover | 1 |
| turbulent momentum diffusion over the depth, H²/ν_t | ~14 |
| secondary-current circulation | ~10 |

so O(50–100) is a sound development time. The default `tstat = 100` is generous
against Kara et al.'s 12.

**Averaging adequacy** is a third question, and the secondary currents set it,
not U — they are only a few per cent of U_max. With σ_v ≈ 1 u_τ, T_int ≈ 1 H/u_τ
and x-averaging over L_x/L_int ≈ 12 independent stations, an averaging window
T_avg gives N_indep ≈ 12·T_avg and a standard error σ/√N. T_avg = 300 gives
N ≈ 3600 and ≈0.017 u_τ on the mean cross-flow — about 3 % of a secondary
velocity of 0.5 u_τ. Adequate, but not by a large margin.

`monitor.py` measures all of this from the probe time series rather than
assuming it: split-half drift, the measured T_int, N_indep and the resulting
error bar, for U (fast) and for V, W (slow).

## On the streamwise domain length

Kara et al. used **L_x = 12H**, "nearly twice the recommended value of 2πH for
straight smooth channels", verified with two-point correlations.
**4πH = 12.566H**, i.e. within 4.7 % of 12H — the two are interchangeable. The
default here is 12H so that any discrepancy cannot be attributed to domain
length; pass `--lx 12.566371` for the 4π box.

This is checked empirically at run time: `monitor.py` computes R_uu(Δx) from the
streamwise probe rakes, which must decorrelate well before Δx = L_x/2.

## Runtime (on-the-fly) analysis

Enabled in the generated `parameters.json`:

| what | where | why |
|---|---|---|
| `"uvwbulk": true` | `LOGS/uvwbulk.log` | `UBULK` = U_b/u_τ vs time — development and the main validation number |
| `"probes"` | `probes.h5` | x-rakes (two-point correlation), vertical profiles, spanwise rake |
| `"statistics"` | `fields.h5` | time-averaged U,V,W,P and all Reynolds stresses |
| itinfo | `LOGS/general.log` | divmax, CFL, kinetic energy |

Probe rakes:

- `xrake_shear`, `xrake_main`, `xrake_fp` — 64 points along x → R_uu(Δx)
- `prof_main` (z = 1.25H), `prof_junc` (z = 2.40H), `prof_fp` (z = 3.75H) —
  48 points in y → log-law check (T&N Fig. 7), velocity dip (Fig. 5)
- `zrake_surf` — 100 points across the span at y = 0.95H → secondary-current cells

Monitor a run at any time:

```bash
python3 monitor.py Dr05 --plot
```

It reports whether the flow is developed, the drift in U_b, the deviation from
the experimental U_b/u_τ, and whether the domain is long enough.

⚠️ `snapshots` reads `tstart` from the JSON **root**, not from
`/snapshots/tstart` (`src/plugins/snapshots_mod.F90:49`) — unlike `probes`.

## Sampling for SPOD and resolvent analysis

### Why equally spaced planes, not "decorrelated" stations

The generated cases sample **16 equally spaced y–z cross-sections** plus a
near-surface x–z plane, as probe arrays (`xsec00`…`xsec15`, `surface`).

A tempting shortcut is to take a few cross-sections at one instant (x = 1, 4, 8)
and treat them as independent SPOD realisations. Two problems:

1. **Equal-time decorrelation is the wrong test.** R_uu(Δx, τ = 0) ≈ 0 does not
   mean independence — mean advection turns a distant station into a
   *phase-shifted copy*, so the cross-spectrum |S₁₂(ω)| stays large even where
   the equal-time correlation crosses zero. The relevant quantity is the
   magnitude-squared coherence γ²(Δx, ω) = |S₁₂|²/(S₁₁S₂₂).

2. **Coherence is frequency dependent, and the largest scales never decorrelate.**
   In a periodic box the λ_x = L_x mode is coherent across the *entire* domain by
   construction. No choice of stations makes them independent at low frequency —
   exactly the frequencies resolvent analysis usually targets.

The clean route is not to hunt for independence but to **Fourier transform in x**.
x is periodic and statistically homogeneous, so this is exact. It also happens to
be precisely what resolvent needs: for a streamwise-homogeneous base flow the
linearised operator is block-diagonal in k_x, so one forms

    R(k_x, ω) = (iω I − L_{k_x})⁻¹

and compares its leading response mode with the leading SPOD mode at the *same*
(k_x, ω). The SPOD–resolvent equivalence (Towne, Schmidt & Colonius, JFM 2018)
holds per (k_x, ω) under spatially white forcing.

16 planes over L_x = 12H give Δx = 0.75H, i.e. k_x modes n = 0…8
(λ_x = 12H down to 1.5H). Raise `--spod-planes` for higher k_x.

`monitor.py` prints the γ²(Δx, ω) table by frequency band so the shortcut can be
judged with data if you still want it — it is defensible in the high-frequency
band only.

### Base flow

The resolvent base flow is the 2-D mean U(y, z), obtained from `U_AVG` in
`fields.h5` averaged over x (exact, since x is homogeneous).

### Time sampling

⚠️ **dt is only frozen once `timeph >= tstat`** (`src/timeloop_mod.F90:313`).
Before that, adaptive time stepping makes the samples non-uniform in time.
**Discard t < tstat before any time FFT.**

With `--probe-itsamp 80` and dt ≈ 6e-4 (CFL 0.8), Δt_s ≈ 0.05 H/u_τ, so the
Nyquist frequency is f ≈ 10 u_τ/H. The domain-scale structure sweeps past in
L_x/U_b = 12/21.3 ≈ 0.56 H/u_τ (f ≈ 1.8), so it is well resolved.

Welch blocks vs. run length (50 % overlap, T_avail = tend − tstat):

| T_blk | f resolution | N_blk at T_avail = 300 | at 700 |
|---|---|---|---|
| 25 | 0.04 | 23 | 55 |
| 15 | 0.067 | 39 | 92 |

The defaults (`tend 400`, `tstat 100`) give ~39 blocks at T_blk = 15 — workable
for the leading SPOD mode, thin for the sub-leading ones. **For a serious SPOD
study raise `--tend` to ~800.** There is no shortcut: each k_x is its own SPOD
problem, and ±k_x are complex conjugates, so neither buys extra realisations.

Storage is not the constraint: ~70 k probe points × 4 variables is ≈1.1 MB per
sample in a single-precision build, so t = 100…800 at Δt_s = 0.05 is ≈16 GB.

### What this setup cannot answer

The domain is streamwise periodic, so **the spatial development of the junction
shear layer is not represented**. That is a deliberate match to the reference
data, not a defect: Tominaga & Nezu established "a fully developed and uniform
flow at the test section 7.5 m downstream from the entrance" (p. 22), so there is
no streamwise development in the experiment either. Kara et al. made the same
choice, and in fact attributed their slight *over*prediction of floodplain
velocity to "limited flow development in the experiment, infinite domain in the
LES" — i.e. they suspected the experiment, not the LES. Studying entrance-region
mixing-layer growth would need a different, inflow/outflow configuration.

The **rigid lid** is the sharper limitation for free-surface questions. A
symmetry plane gives zero normal velocity and finite tangential velocity at
y = H, so surface-parallel structures and the turbulence-anisotropy-driven
free-surface vortex (T&N Fig. 3) *are* captured. Surface deformation, boils and
surface-renewal events are not. Conclusions about small-scale surface eddies are
sound insofar as they concern anisotropy-driven motions, and outside the model's
reach insofar as they concern a deformable surface.

## Regenerating

```bash
python3 make_case.py --dr 0.5 --outdir Dr05                 # production, 36 M
python3 make_case.py --dr 0.5 --outdir test --ncy 24 --block 8   # coarse smoke test
```

Useful options: `--lx`, `--ncy`, `--re-tau`, `--tend`, `--tstat`, `--lesmodel`.

## Building and running

Everything goes through `run_cases.sh`. From this directory:

```bash
./run_cases.sh build          # cmake --preset=gnu-release + make -j
./run_cases.sh check          # MGLET's own test suite
./run_cases.sh smoke          # 20 steps at full size: does the grid load?
./run_cases.sh spinup         # to t = 30: does it become turbulent?
./run_cases.sh run            # full production
./run_cases.sh monitor        # development / stationarity diagnostics
./run_cases.sh stats          # assemble statistics into (y,z) fields
```

`./run_cases.sh all` chains build → check → smoke → spinup and deliberately
stops short of the production run.

Overrides: `NP=64` (MPI ranks), `PRESET=intel-release`, `BUILD=<dir>`,
`MGLET_BIN=<path>`, `CASES="Dr05"`.

Prerequisites: C/C++/Fortran compilers with Fortran 2008 + TS 29113, an MPI
library providing `MPI_f08`, HDF5 with MPI support, CMake, and Python with
`h5py`/`numpy` (`matplotlib` optional, for plots). nlohmann/json and exprtk are
fetched automatically by CMake.

If `cmake` stops at `FortranCInterface_VERIFY`, the C and Fortran compilers come
from different toolchains (a conda-vs-system mix does this). Put one toolchain
first in `PATH`, delete the build directory, and reconfigure.

`NP` must not exceed the grid count (3600 / 3300 / 3000) — a grid cannot be
split across ranks. `run_cases.sh` checks this before launching.

**Do the staged checks first.** `smoke` catches memory/decomposition problems in
minutes, and `spinup` answers the one question that can otherwise waste days:
whether the initial condition actually transitions to turbulence rather than
decaying back to laminar. In `LOGS/uvwbulk.log`, `VVBULK`/`WWBULK` should grow
and `UBULK` should bend over well below the laminar runaway. If it stalls, raise
the perturbation amplitude in `ic_expression()` in `make_case.py`.

## Getting the statistics

Everything MGLET writes is block-structured HDF5. `postprocess.py` collapses it
onto **one averaged transversal section** -- x is periodic and statistically
homogeneous, so averaging over it is exact and buys ~Lx/L_int extra independent
samples -- and writes plain readable files.

```bash
python3 postprocess.py Dr05                    # uses mgtools if importable
MGTOOLS_PATH=~/TUMHydro/mgtools-master python3 postprocess.py Dr05
python3 postprocess.py Dr05 --builtin          # force the fallback reader
```

Output in `<case>/post/`:

| file | contents |
|---|---|
| `xsec.csv` | the collapsed section, one row per (y,z): `y, z, fluid, U, V, W, P, uu, vv, ww, uv, uw, vw, k, omega_x` |
| `profiles_z.csv` | depth-averaged lateral distributions: `z, depth, Ud, Ud_over_Ub, tau_bed, tau_over_taubar, q_unit` (T&N Figs. 5 and 8) |
| `profiles_y.csv` | vertical profiles at four stations: `station, z, y, y_wall, y_plus, U, uu, vv, ww, uv` (T&N Fig. 7) |
| `meta.json` | Dr, Re_tau, nu, TSAMP, U_bulk vs the experimental target, unit conventions |
| `xsec.vtr` | VTK rectilinear grid for ParaView |

`xsec.csv` keeps solid points with `fluid=0` rather than dropping them, so it
stays reshapeable: `df.pivot(index='y', columns='z', values='U')` gives a matrix
for contouring and the L-shaped step stays explicit.

### Conventions

Everything is in wall units, because the case is set up with u_tau = H = rho = 1
-- the stored numbers *are* the normalised ones. `meta.json` carries `U_bulk` and
`U_max` so you can renormalise to U/U_max (T&N Fig. 5) without touching the
solver. `uv` is stored as the raw <u'v'>; T&N and Kara plot -<u'v'>/u_tau^2.

Two things worth knowing about how the numbers are formed:

**Stresses use the x-averaged mean**, `<ui'uj'> = mean_x[<ui uj>] - Ui*Uj`. With
finite sampling a per-x mean is noisy, and subtracting it would absorb genuine
turbulence into the mean and under-report the stress.

**The six stresses live on three different edge locations** -- `UV_AVG` on the
xy-edge, `UW_AVG` on xz, `VW_AVG` on yz (`src/flow/flowstat_mod.F90:100`) -- so
they must be projected to cell centres before they mean anything as a section.
With mgtools that is `Field.onto(p)`; the fallback reader does the same
interpolation. Note that interpolating a product is not the product of the
interpolations, an O(dx^2) inconsistency that is recorded in `meta.json`.

### Old notes on the assembler

`postprocess.py` (invoked by `run_cases.sh stats`) walks `grids.h5` + `fields.h5`,
strips ghost cells, de-staggers, averages over the homogeneous streamwise
direction and stitches the blocks into a single (y, z) map:

```bash
python3 postprocess.py Dr05 --plot
```

It writes `statistics.npz` (U, V, W, P and all six Reynolds stresses on a regular
(y, z) grid, with the mean products already subtracted) and `statistics.png`
(isovels and secondary-current vectors, comparable with T&N Figs. 3 and 5). It
also prints U_b/u_τ against the experimental target and the secondary-current
magnitude as a percentage of U_max, which T&N report as ~2–4 %.

Defaults are `tend = 400`, `tstat = 100` in units of H/u_τ. Kara et al. used
only 12 H/u_τ of development and 23 H/u_τ of averaging; that is short for
secondary currents, which are ~2–5 % of U_max and converge slowly, so the
defaults here are considerably longer. Use `LOGS/uvwbulk.log` to confirm the
flow has plateaued before `tstat`, and raise `tstat` if it has not.
