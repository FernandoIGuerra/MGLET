#!/usr/bin/env python3
"""Assemble MGLET statistics into cross-sectional (y, z) fields.

MGLET stores every field as one flat block per grid. This walks grids.h5 +
fields.h5, strips ghost cells, de-staggers, averages over the homogeneous
streamwise direction and stitches the blocks into a single (y, z) map --
directly comparable with Tominaga & Nezu's isovel and Reynolds-stress figures.

    python3 postprocess.py <casedir> [--plot]

Writes <casedir>/statistics.npz with U, V, W, P, uu, vv, ww, uv, uw, vw on a
regular (y, z) grid, plus the derived bulk velocity and secondary-current
magnitude. Values are in wall units (u_tau = 1) because the case is set up
with u_tau = 1 by construction.
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np

NGHOST = 2
MEAN = {"U": "U_AVG", "V": "V_AVG", "W": "W_AVG", "P": "P_AVG"}
SECOND = {"uu": "UU_AVG", "vv": "VV_AVG", "ww": "WW_AVG",
          "uv": "UV_AVG", "uw": "UW_AVG", "vw": "VW_AVG"}
UB_TARGET = {0.500: 21.28, 0.375: 20.85, 0.250: 20.43}


def load_grids(path):
    with h5py.File(path, "r") as f:
        gi = f["GRIDINFO"][:]
    return gi


def block_view(flat, ii, jj, kk):
    """Fortran a(kk,jj,ii) stored contiguously -> numpy arr[i, j, k]."""
    return np.asarray(flat, dtype=np.float64).reshape(ii, jj, kk)


def destagger(arr, istag, jstag, kstag):
    """Move a staggered field to cell centres, returning only real cells."""
    i0, j0, k0 = NGHOST, NGHOST, NGHOST
    i1, j1, k1 = arr.shape[0] - NGHOST, arr.shape[1] - NGHOST, arr.shape[2] - NGHOST
    cur = arr[i0:i1, j0:j1, k0:k1]
    if istag:
        cur = 0.5 * (cur + arr[i0 - 1:i1 - 1, j0:j1, k0:k1])
    if jstag:
        cur = 0.5 * (cur + arr[i0:i1, j0 - 1:j1 - 1, k0:k1])
    if kstag:
        cur = 0.5 * (cur + arr[i0:i1, j0:j1, k0 - 1:k1 - 1])
    return cur


def assemble(casedir, wanted):
    gpath = os.path.join(casedir, "grids.h5")
    fpath = os.path.join(casedir, "fields.h5")
    for p in (gpath, fpath):
        if not os.path.exists(p):
            sys.exit(f"missing {p}")

    gi = load_grids(gpath)
    ny_cells = int(round(1.0 / (gi[0]["BBOX"][3] - gi[0]["BBOX"][2])
                         * (gi[0]["JJ"] - 2 * NGHOST)))
    # Derive the global cell counts from any block's size and extent
    dy = (gi[0]["BBOX"][3] - gi[0]["BBOX"][2]) / (gi[0]["JJ"] - 2 * NGHOST)
    dz = (gi[0]["BBOX"][5] - gi[0]["BBOX"][4]) / (gi[0]["KK"] - 2 * NGHOST)
    ny = int(round(1.0 / dy))
    nz = int(round(5.0 / dz))

    out, tsamp = {}, None
    with h5py.File(fpath, "r") as f:
        vol = f["VOLUMEFIELDS"]
        for key, name in wanted.items():
            if name not in vol:
                continue
            grp = vol[name]
            data = grp["LEVEL1"]
            idx = np.asarray(grp["IGRIDLVL"])
            a = grp.attrs
            istag, jstag, kstag = int(a["ISTAG"]), int(a["JSTAG"]), int(a["KSTAG"])
            tsamp = float(a["TSAMP"])

            acc = np.zeros((ny, nz))
            cnt = np.zeros((ny, nz))
            for row, igrid in enumerate(idx):
                g = gi[igrid - 1]
                ii, jj, kk = int(g["II"]), int(g["JJ"]), int(g["KK"])
                arr = block_view(data[row], ii, jj, kk)
                cell = destagger(arr, istag, jstag, kstag)
                prof = cell.mean(axis=0)                 # average over x
                j0 = int(round(g["BBOX"][2] / dy))
                k0 = int(round(g["BBOX"][4] / dz))
                acc[j0:j0 + prof.shape[0], k0:k0 + prof.shape[1]] += prof
                cnt[j0:j0 + prof.shape[0], k0:k0 + prof.shape[1]] += 1.0
            out[key] = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)

    y = (np.arange(ny) + 0.5) * dy
    z = (np.arange(nz) + 0.5) * dz
    return out, y, z, tsamp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    wanted = dict(MEAN)
    wanted.update(SECOND)
    fields, y, z, tsamp = assemble(args.case, wanted)
    if "U" not in fields:
        sys.exit("no U_AVG in fields.h5 - did the run reach tstat?")

    # Reynolds stresses: MGLET stores <u_i u_j>, subtract the mean product
    stats = {k: v for k, v in fields.items()}
    for key, (a, b) in dict(uu=("U", "U"), vv=("V", "V"), ww=("W", "W"),
                            uv=("U", "V"), uw=("U", "W"),
                            vw=("V", "W")).items():
        if key in stats:
            stats[key] = stats[key] - fields[a] * fields[b]

    valid = ~np.isnan(stats["U"])
    ub = float(np.nanmean(stats["U"][valid]))
    vsec = np.hypot(np.nan_to_num(stats["V"]), np.nan_to_num(stats["W"]))
    umax = float(np.nanmax(stats["U"]))

    dr = None
    pj = os.path.join(args.case, "parameters.json")
    if os.path.exists(pj):
        gradp = json.load(open(pj))["flow"]["gradp"][0]
        rh = -1.0 / gradp
        drv = (rh * 7.0 - 2.5) / 2.5
        dr = min(UB_TARGET, key=lambda d: abs(d - drv))

    print(f"case            {args.case}")
    print(f"averaging time  TSAMP = {tsamp:.2f} H/u_tau")
    print(f"grid            {stats['U'].shape[0]} x {stats['U'].shape[1]} (y, z),"
          f" {int(valid.sum())} fluid points")
    print(f"U_b/u_tau       {ub:.3f}", end="")
    if dr:
        print(f"   target {UB_TARGET[dr]:.2f}   deviation {100*(ub/UB_TARGET[dr]-1):+.1f} %")
    else:
        print()
    print(f"U_max/u_tau     {umax:.3f}")
    print(f"secondary curr. max |V,W| = {np.nanmax(vsec):.3f} u_tau"
          f"  = {100*np.nanmax(vsec)/umax:.1f} % of U_max"
          f"   (Tominaga & Nezu report ~2-4 %)")

    out = os.path.join(args.case, "statistics.npz")
    np.savez(out, y=y, z=z, tsamp=tsamp, **stats)
    print(f"wrote {out}")

    if args.plot:
        _plot(stats, y, z, args.case, umax)


def _plot(stats, y, z, case, umax):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return
    Z, Y = np.meshgrid(z, y)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    cs = axes[0].contour(Z, Y, stats["U"] / umax, levels=np.arange(0.4, 1.0, 0.05))
    axes[0].clabel(cs, fontsize=6)
    axes[0].set_ylabel("y/H")
    axes[0].set_title(r"$U/U_{max}$  (cf. Tominaga & Nezu Fig. 5)")
    step = 8
    axes[1].quiver(Z[::step, ::step], Y[::step, ::step],
                   stats["W"][::step, ::step], stats["V"][::step, ::step])
    axes[1].set_xlabel("z/H")
    axes[1].set_ylabel("y/H")
    axes[1].set_title("secondary currents (V, W)  (cf. Fig. 3)")
    for ax in axes:
        ax.set_aspect("equal")
    fig.tight_layout()
    p = os.path.join(case, "statistics.png")
    fig.savefig(p, dpi=130)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
