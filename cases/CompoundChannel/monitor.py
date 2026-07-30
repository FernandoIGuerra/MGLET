#!/usr/bin/env python3
"""Runtime monitoring / validation for the compound open channel cases.

Reads what MGLET writes while it runs and answers three questions:

  1. Is the flow developed?      LOGS/uvwbulk.log  -> U_b/u_tau vs. time
  2. Is the solver healthy?      LOGS/general.log  -> divmax, CFL, kin. energy
  3. Is the domain long enough?  probes.h5         -> R_uu(dx) on the x-rakes

Run it any time during or after a simulation:

    python3 monitor.py <casedir> [--plot]
"""

import argparse
import json
import os
import sys

import numpy as np

UB_TARGET = {0.500: 21.28, 0.375: 20.85, 0.250: 20.43}


def read_uvwbulk(path):
    """Columns: ITTOT TIME UBULK VBULK WBULK UUBULK VVBULK WWBULK."""
    rows = []
    with open(path) as f:
        for line in f:
            if line.lstrip().startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 8:
                rows.append([float(p) for p in parts[:8]])
    if not rows:
        return None
    a = np.array(rows)
    return dict(it=a[:, 0], t=a[:, 1], ub=a[:, 2], vb=a[:, 3], wb=a[:, 4],
                uu=a[:, 5], vv=a[:, 6], ww=a[:, 7])


def plateau(t, y, frac=0.3):
    """Mean/std over the last `frac` of the record, and a drift estimate."""
    n = max(2, int(len(y) * frac))
    tail_t, tail_y = t[-n:], y[-n:]
    slope = np.polyfit(tail_t, tail_y, 1)[0] if len(tail_y) > 2 else 0.0
    return tail_y.mean(), tail_y.std(), slope


def report_bulk(case, dr):
    path = os.path.join(case, "LOGS", "uvwbulk.log")
    if not os.path.exists(path):
        print(f"  [uvwbulk] {path} not found "
              f'(is "uvwbulk": true set in parameters.json?)')
        return None
    d = read_uvwbulk(path)
    if d is None or len(d["t"]) < 2:
        print("  [uvwbulk] no samples yet")
        return None

    mean, std, slope = plateau(d["t"], d["ub"])
    target = UB_TARGET.get(dr)
    print(f"  t = {d['t'][0]:.2f} .. {d['t'][-1]:.2f} H/u_tau "
          f"({len(d['t'])} samples)")
    print(f"  U_b/u_tau  now = {d['ub'][-1]:8.3f}   "
          f"last 30% mean = {mean:.3f} +/- {std:.3f}")
    print(f"  drift      dU_b/dt = {slope:+.2e} per H/u_tau")
    if target:
        print(f"  target     {target:.2f} (experiment)   "
              f"deviation = {100*(mean/target - 1):+.1f} %")

    # A developed flow has a flat U_b and non-zero cross-flow fluctuation
    settled = abs(slope) < 5e-3 and std < 0.15 * max(abs(mean), 1e-9)
    turbulent = d["vv"][-1] > 1e-4 and d["ww"][-1] > 1e-4
    print(f"  cross-flow <v^2> = {d['vv'][-1]:.4e}, <w^2> = {d['ww'][-1]:.4e}"
          f"   -> {'turbulent' if turbulent else 'NOT yet turbulent'}")
    developed = settled and turbulent
    print(f"  verdict    {'developed' if developed else 'still developing'}")
    d["developed"] = developed
    return d


def report_general(case):
    path = os.path.join(case, "LOGS", "general.log")
    if not os.path.exists(path):
        print(f"  [general] {path} not found")
        return
    with open(path) as f:
        lines = [l for l in f if l.strip() and not l.lstrip().startswith("#")]
    if not lines:
        print("  [general] empty")
        return
    print(f"  last line: {lines[-1].strip()}")


def two_point(case, plot=False):
    """Streamwise two-point correlation of u' on each x-rake."""
    path = os.path.join(case, "probes.h5")
    if not os.path.exists(path):
        print(f"  [probes] {path} not found yet")
        return
    try:
        import h5py
    except ImportError:
        print("  [probes] h5py not available")
        return

    with h5py.File(path, "r") as f:
        grp = f["PROBES"] if "PROBES" in f else f
        names = [n for n in grp if n.lower().startswith("xrake")]
        if not names:
            print(f"  [probes] no x-rakes found (groups: {list(grp)[:8]})")
            return
        for name in sorted(names):
            node = grp[name]
            key = next((k for k in node if k.upper().startswith("U")), None)
            if key is None:
                print(f"  [{name}] no U dataset (has {list(node)})")
                continue
            u = np.asarray(node[key])            # (ntime, npts) or (npts, ntime)
            if u.ndim != 2:
                print(f"  [{name}] unexpected shape {u.shape}")
                continue
            if u.shape[0] < u.shape[1]:
                u = u.T                          # -> (ntime, npts)
            if u.shape[0] < 8:
                print(f"  [{name}] only {u.shape[0]} time samples, skipping")
                continue

            up = u - u.mean(axis=0, keepdims=True)
            npts = up.shape[1]
            # Periodic correlation, averaged over time and over origin
            fft = np.fft.rfft(up, axis=1)
            corr = np.fft.irfft((fft * np.conj(fft)).real, n=npts, axis=1)
            r = corr.mean(axis=0)
            r = r / r[0] if r[0] != 0 else r
            half = npts // 2
            rmin = r[:half + 1].min()
            print(f"  [{name}] R_uu min over 0..Lx/2 = {rmin:+.3f}"
                  f"   R_uu(Lx/2) = {r[half]:+.3f}"
                  f"   -> {'OK, decorrelated' if abs(r[half]) < 0.15 else 'DOMAIN MAY BE TOO SHORT'}")
            if plot:
                _plot_corr(name, r[:half + 1], case)


def stationarity(case, tstat=None):
    """Is the flow statistically stationary, and is the averaging long enough?

    Streamwise development is enforced by the periodic domain, so the only
    thing to establish is stationarity in time.  Two tests:

      * split-half: mean over the first half of the window vs the second.
      * integral time scale T_int from the autocorrelation, giving the number
        of independent samples and hence the error bar on the mean.

    The secondary currents (V, W) are the slow ones: they are only a few per
    cent of U_max, so they set the required averaging time, not U.
    """
    path = os.path.join(case, "probes.h5")
    if not os.path.exists(path):
        print(f"  [stationarity] {path} not found yet")
        return
    try:
        import h5py
    except ImportError:
        return

    if tstat is None:
        pj = os.path.join(case, "parameters.json")
        if os.path.exists(pj):
            tstat = json.load(open(pj)).get("time", {}).get("tstat", 0.0)
        else:
            tstat = 0.0

    with h5py.File(path, "r") as f:
        grp = f["PROBES"] if "PROBES" in f else f
        t = probe_time(grp)
        if t is None:
            return
        sel = t >= tstat
        if sel.sum() < 16:
            print(f"  only {int(sel.sum())} samples with t >= tstat = {tstat:g}"
                  f" - too few to judge")
            return
        t = t[sel]
        dt = float(np.median(np.diff(t)))
        print(f"  window t = {t[0]:.1f} .. {t[-1]:.1f} ({len(t)} samples, "
              f"dt_s = {dt:.4g}); dt is frozen for t >= tstat")

        for name, comps in (("prof_main", ("U",)), ("zrake_surf", ("V", "W"))):
            if name not in grp:
                continue
            for c in comps:
                if c not in grp[name]:
                    continue
                a = np.asarray(grp[name][c])[sel]        # (ntime, npts)
                half = len(a) // 2
                m1, m2 = a[:half].mean(axis=0), a[half:2 * half].mean(axis=0)
                # Scale by the magnitude of the mean *profile*, not by its
                # spatial average: V and W average to ~0 over the span, so the
                # latter would blow the percentage up meaninglessly.
                scale = max(np.abs(a.mean(axis=0)).max(), 1e-12)
                drift = np.abs(m2 - m1).max() / scale

                # Integral time scale from the point-averaged autocorrelation
                ap = a - a.mean(axis=0, keepdims=True)
                var = (ap ** 2).mean(axis=0)
                nlag = min(len(ap) // 4, 200)
                rho = np.array([
                    (ap[:len(ap) - k] * ap[k:]).mean(axis=0).mean()
                    / max(var.mean(), 1e-30) for k in range(nlag)])
                cross = np.argmax(rho < 0.0) if (rho < 0.0).any() else nlag
                tint = max(rho[:cross].sum() * dt, dt)

                n_ind = (t[-1] - t[0]) / tint
                err = float(np.sqrt(var.mean()) / max(np.sqrt(n_ind), 1e-9))
                print(f"  {name:11s} {c}: split-half drift = {100*drift:5.1f} %"
                      f"   T_int = {tint:5.2f} H/u_tau"
                      f"   N_indep = {n_ind:6.0f}"
                      f"   sigma_mean = {err:.3f} u_tau")
        print("  (x-averaging multiplies N_indep by ~Lx/L_int ~ 12)")


def probe_time(grp):
    """PROBES/time is a compound dataset with fields ITTOT and TIME."""
    if "time" not in grp:
        return None
    raw = grp["time"][:]
    if raw.dtype.names and "TIME" in raw.dtype.names:
        return np.asarray(raw["TIME"], dtype=float)
    return np.asarray(raw, dtype=float)


def coherence(case, rake="xrake_shear", nseg=8):
    """Magnitude-squared coherence between x-stations, per frequency band.

    This is the test that decides whether cross-sections at different x may be
    treated as independent SPOD realisations.  Zero-lag decorrelation
    (R_uu(dx, tau=0) = 0) is NOT sufficient: mean advection turns a distant
    station into a phase-shifted copy, so |S_12(omega)| stays large even when
    the equal-time correlation has passed through zero.  What matters is

        gamma^2(dx, omega) = |S_12|^2 / (S_11 S_22)

    which is strongly frequency dependent: small scales decorrelate over short
    dx, the domain-scale structures never do (in a periodic box they cannot).
    """
    path = os.path.join(case, "probes.h5")
    if not os.path.exists(path):
        print(f"  [coherence] {path} not found yet")
        return
    try:
        import h5py
    except ImportError:
        return

    with h5py.File(path, "r") as f:
        grp = f["PROBES"] if "PROBES" in f else f
        if rake not in grp:
            print(f"  [coherence] {rake} not in probes.h5")
            return
        u = np.asarray(grp[rake]["U"])
        xs = np.asarray(grp[rake]["coordinates"])[:, 0]
        t = probe_time(grp)

    if u.ndim != 2:
        return
    if u.shape[0] < u.shape[1]:
        u = u.T
    ntime, npts = u.shape
    nfft = ntime // nseg
    if nfft < 16:
        print(f"  [coherence] only {ntime} samples, need >= {16*nseg}")
        return

    dt = float(np.median(np.diff(t))) if t is not None and len(t) > 1 else 1.0
    up = u - u.mean(axis=0, keepdims=True)

    # Welch segments -> (nseg, nfreq, npts)
    segs = np.stack([up[i * nfft:(i + 1) * nfft] for i in range(nseg)])
    win = np.hanning(nfft)[None, :, None]
    U = np.fft.rfft(segs * win, axis=1)
    freq = np.fft.rfftfreq(nfft, d=dt)

    # Average the cross-spectrum over segments and over all origins
    S11 = (np.abs(U) ** 2).mean(axis=0)                     # (nfreq, npts)
    lags = [d for d in (1, 2, 4, 8, npts // 4, npts // 2) if 0 < d <= npts // 2]
    dx_of = lambda d: d * (xs[-1] - xs[0]) / (npts - 1)

    bands = [("low  (large scales)", 0.0, 0.15),
             ("mid                ", 0.15, 0.4),
             ("high (small scales)", 0.4, 1.0)]
    fmax = freq[-1] if freq[-1] > 0 else 1.0

    print(f"  rake {rake}: {ntime} samples, dt_s = {dt:.4g}, "
          f"{nseg} Welch blocks of {nfft}")
    print(f"  {'dx/H':>7s} " + " ".join(f"{b[0]:>21s}" for b in bands))
    for d in sorted(set(lags)):
        S12 = (U * np.conj(np.roll(U, -d, axis=2))).mean(axis=0)
        g2 = (np.abs(S12) ** 2 /
              np.maximum(S11 * np.roll(S11, -d, axis=1), 1e-30))
        row = []
        for _, lo, hi in bands:
            m = (freq >= lo * fmax) & (freq < hi * fmax)
            row.append(f"{g2[m].mean():21.3f}" if m.any() else f"{'-':>21s}")
        print(f"  {dx_of(d):7.2f} " + " ".join(row))
    print("  gamma^2 << 0.1 in a band => stations are usable as independent")
    print("  realisations for THAT band only. Prefer the k_x-FFT route.")


def _plot_corr(name, r, case):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(np.arange(len(r)) / (len(r) - 1), r, lw=1.5)
    ax.axhline(0.0, color="k", lw=0.5)
    ax.axhline(0.15, color="r", ls=":", lw=0.8)
    ax.set_xlabel(r"$\Delta x / (L_x/2)$")
    ax.set_ylabel(r"$R_{uu}$")
    ax.set_title(name)
    fig.tight_layout()
    out = os.path.join(case, f"corr_{name}.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"      wrote {out}")


def plot_bulk(d, dr, case):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(d["t"], d["ub"], lw=1.2, label=r"$U_b/u_\tau$")
    if dr in UB_TARGET:
        ax.axhline(UB_TARGET[dr], color="r", ls="--", lw=1.0,
                   label=f"experiment {UB_TARGET[dr]:.2f}")
    ax.set_xlabel(r"$t\,u_\tau/H$")
    ax.set_ylabel(r"$U_b/u_\tau$")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(case, "convergence.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  wrote {out}")


def infer_dr(case):
    """Recover Dr from gradp in parameters.json: gradp = -P/A = -1/Rh."""
    path = os.path.join(case, "parameters.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        p = json.load(f)
    gradp = p.get("flow", {}).get("gradp", [0])[0]
    if not gradp:
        return None
    rh = -1.0 / gradp
    dr = (rh * 7.0 - 2.5) / 2.5      # A = 2.5 + 2.5*Dr, P = 7
    return min(UB_TARGET, key=lambda d: abs(d - dr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case", help="case directory")
    ap.add_argument("--dr", type=float, default=None)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    dr = args.dr if args.dr is not None else infer_dr(args.case)
    print(f"case: {args.case}   Dr = {dr}")
    print("\n-- development / bulk velocity --")
    d = report_bulk(args.case, dr)
    print("\n-- solver health --")
    report_general(args.case)
    print("\n-- streamwise decorrelation (domain length) --")
    if d is not None and not d.get("developed"):
        print("  NOTE: flow is not developed yet - R_uu still reflects the")
        print("        initial condition, not turbulence. Ignore the verdict.")
    two_point(args.case, args.plot)
    print("\n-- statistical stationarity / averaging adequacy --")
    stationarity(args.case)
    print("\n-- spectral coherence between x-stations (SPOD independence) --")
    coherence(args.case)
    if args.plot and d is not None:
        plot_bulk(d, dr, args.case)
    return 0


if __name__ == "__main__":
    sys.exit(main())
