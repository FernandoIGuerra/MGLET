#!/usr/bin/env python3
"""Collapse MGLET statistics onto a single averaged transversal section.

x is periodic and statistically homogeneous, so averaging over it is exact and
simply buys ~Lx/L_int extra independent samples. The result is one (y, z)
cross-section from which every quantity of interest is derived.

Reads via mgtools when available (MGreadKMT8 matches MGLET-base's layout:
separate grids.h5/fields.h5, /GRIDINFO and /VOLUMEFIELDS). Field.onto(p)
projects the staggered velocities onto the pressure locations and
Field.average('x') collapses the streamwise direction. Falls back to a
built-in reader if mgtools is not installed.

    python3 postprocess.py <casedir> [--mgtools-path DIR]

Writes <casedir>/post/:
    xsec.csv        THE collapsed y-z plane, one row per (y, z)
    meta.json       Dr, Re_tau, normalisations, sampling time
    xsec.vtr        the same plane as a VTK rectilinear grid for ParaView

--profiles additionally writes the derived 1-D distributions
(profiles_z.csv, profiles_y.csv), which are not needed for the plane itself.
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np

NGHOST = 2
Z_JUNCTION, Z_TOTAL = 2.5, 5.0
MEANS = {"U": "U_AVG", "V": "V_AVG", "W": "W_AVG", "P": "P_AVG"}
STRESSES = {"uu": "UU_AVG", "vv": "VV_AVG", "ww": "WW_AVG",
            "uv": "UV_AVG", "uw": "UW_AVG", "vw": "VW_AVG"}
PAIRS = dict(uu=("U", "U"), vv=("V", "V"), ww=("W", "W"),
             uv=("U", "V"), uw=("U", "W"), vw=("V", "W"))
UB_TARGET = {0.500: 21.28, 0.375: 20.85, 0.250: 20.43}
STATIONS = {"main_centre": 1.25, "junction_mc": 2.40,
            "junction_fp": 2.60, "floodplain": 3.75}


# --------------------------------------------------------------- readers ---
def collapse_mgtools(case, mgpath):
    """Collapse using mgtools: onto() de-staggers, average('x') collapses."""
    if mgpath:
        sys.path.insert(0, mgpath)
    from mgtools import MGreadKMT8

    reader = MGreadKMT8(os.path.join(case, "fields.h5"),
                        os.path.join(case, "grids.h5"))
    with h5py.File(os.path.join(case, "grids.h5"), "r") as f:
        gi = f["GRIDINFO"][:]
    dy, dz, ny, nz = _geometry(gi)

    acc = {k: np.zeros((ny, nz)) for k in list(MEANS) + list(STRESSES)}
    cnt = np.zeros((ny, nz))
    tsamp = None
    for igrid in range(len(gi)):
        grid = reader[igrid]
        p = grid.read("P_AVG")
        if tsamp is None:
            tsamp = _tsamp(case)
        vals = {}
        for key, name in {**MEANS, **STRESSES}.items():
            f = grid.read(name)
            f = f if name == "P_AVG" else f.onto(p)
            vals[key] = f.average("x", strip=NGHOST).data[0]
        g = gi[igrid]
        j0 = int(round(g["BBOX"][2] / dy))
        k0 = int(round(g["BBOX"][4] / dz))
        sl = (slice(NGHOST, -NGHOST), slice(NGHOST, -NGHOST))
        for key, a in vals.items():
            blk = np.asarray(a)[sl]
            acc[key][j0:j0 + blk.shape[0], k0:k0 + blk.shape[1]] += blk
        blk = np.asarray(vals["U"])[sl]
        cnt[j0:j0 + blk.shape[0], k0:k0 + blk.shape[1]] += 1.0
    return _finish(acc, cnt, dy, dz, ny, nz, tsamp)


def collapse_builtin(case):
    """Fallback: assemble, de-stagger and collapse without mgtools."""
    with h5py.File(os.path.join(case, "grids.h5"), "r") as f:
        gi = f["GRIDINFO"][:]
    dy, dz, ny, nz = _geometry(gi)
    acc = {k: np.zeros((ny, nz)) for k in list(MEANS) + list(STRESSES)}
    cnt = np.zeros((ny, nz))
    with h5py.File(os.path.join(case, "fields.h5"), "r") as f:
        vol = f["VOLUMEFIELDS"]
        tsamp = float(vol["U_AVG"].attrs["TSAMP"])
        for key, name in {**MEANS, **STRESSES}.items():
            grp = vol[name]
            data, idx = grp["LEVEL1"], np.asarray(grp["IGRIDLVL"])
            st = (int(grp.attrs["ISTAG"]), int(grp.attrs["JSTAG"]),
                  int(grp.attrs["KSTAG"]))
            for row, igrid in enumerate(idx):
                g = gi[igrid - 1]
                ii, jj, kk = int(g["II"]), int(g["JJ"]), int(g["KK"])
                a = np.asarray(data[row], np.float64).reshape(ii, jj, kk)
                cell = _destagger(a, st)
                prof = cell.mean(axis=0)
                j0 = int(round(g["BBOX"][2] / dy))
                k0 = int(round(g["BBOX"][4] / dz))
                acc[key][j0:j0 + prof.shape[0], k0:k0 + prof.shape[1]] += prof
                if key == "U":
                    cnt[j0:j0 + prof.shape[0], k0:k0 + prof.shape[1]] += 1.0
    return _finish(acc, cnt, dy, dz, ny, nz, tsamp)


def _destagger(arr, stag):
    i0, j0, k0 = NGHOST, NGHOST, NGHOST
    i1, j1, k1 = (arr.shape[0] - NGHOST, arr.shape[1] - NGHOST,
                  arr.shape[2] - NGHOST)
    cur = arr[i0:i1, j0:j1, k0:k1]
    if stag[0]:
        cur = 0.5 * (cur + arr[i0 - 1:i1 - 1, j0:j1, k0:k1])
    if stag[1]:
        cur = 0.5 * (cur + arr[i0:i1, j0 - 1:j1 - 1, k0:k1])
    if stag[2]:
        cur = 0.5 * (cur + arr[i0:i1, j0:j1, k0 - 1:k1 - 1])
    return cur


def _geometry(gi):
    g0 = gi[0]
    dy = (g0["BBOX"][3] - g0["BBOX"][2]) / (g0["JJ"] - 2 * NGHOST)
    dz = (g0["BBOX"][5] - g0["BBOX"][4]) / (g0["KK"] - 2 * NGHOST)
    return dy, dz, int(round(1.0 / dy)), int(round(Z_TOTAL / dz))


def _tsamp(case):
    with h5py.File(os.path.join(case, "fields.h5"), "r") as f:
        return float(f["VOLUMEFIELDS/U_AVG"].attrs["TSAMP"])


def _finish(acc, cnt, dy, dz, ny, nz, tsamp):
    fluid = cnt > 0
    out = {k: np.where(fluid, v / np.maximum(cnt, 1), np.nan)
           for k, v in acc.items()}
    # Reynolds stresses relative to the x-averaged mean. Using a per-x mean
    # would absorb genuine turbulence into the mean and under-report these.
    for key, (a, b) in PAIRS.items():
        out[key] = out[key] - out[a] * out[b]
    y = (np.arange(ny) + 0.5) * dy
    z = (np.arange(nz) + 0.5) * dz
    return out, y, z, fluid, tsamp


# --------------------------------------------------------------- derived ---
def derive(f, y, z, fluid):
    f["k"] = 0.5 * (f["uu"] + f["vv"] + f["ww"])
    W = np.nan_to_num(f["W"]); V = np.nan_to_num(f["V"])
    dWdy = np.gradient(W, y, axis=0)
    dVdz = np.gradient(V, z, axis=1)
    f["omega_x"] = np.where(fluid, dWdy - dVdz, np.nan)
    return f


def ww_tau(u1, dds, nu, rho=1.0):
    """Werner-Wengle wall shear -- identical to src/flow/wernerwengle_mod.F90.

    Using the solver's own wall model is the self-consistent choice: the local
    tau_w(z) then integrates to the global balance that gradp imposes. A naive
    tau = mu*U1/y1 assumes U+ = y+ and is only valid deep in the viscous
    sublayer; it under-predicts by ~1 % at y1+ = 3.7 but ~8 % at y1+ = 7.5.
    """
    A, b = 8.3, 1.0 / 7.0
    cpo1, cpo2 = 1.0 - b, 1.0 + b
    cpo4 = cpo2 / A * nu ** b
    cpo5 = 0.5 * cpo1 * A ** (cpo2 / cpo1) * nu ** cpo2
    cpo6 = 0.5 * nu * A ** (2.0 / cpo1)
    cpo8 = 2.0 / cpo2
    un = np.abs(u1)
    lam = 2.0 * rho * nu * un / dds
    turb = rho * (dds ** (-b) * (cpo4 * un + cpo5 / dds)) ** cpo8
    return np.sign(u1) * np.where(un >= cpo6 / dds, turb, lam)


def clauser_utau(yw, u, nu, kappa=0.41, B=5.3):
    """u_tau from a log-law fit -- the method Tominaga & Nezu used.

    Needs only the log region, not the viscous sublayer, so it is insensitive
    to a coarse first cell. Note it *assumes* the universal kappa and B: if the
    LES has a log-layer mismatch the fit absorbs it, so a fitted u_tau is not
    the same quantity as the u_tau that gradp imposes. Report which one.
    """
    ut = 1.0
    for _ in range(60):
        yp = yw * ut / nu
        m = (yp > 30.0) & (yp < 0.25 * np.max(yp))
        if m.sum() < 3:
            return np.nan
        pred = (np.log(yp[m]) / kappa + B)
        ut_new = np.sum(u[m] * pred) / np.sum(pred * pred)
        if abs(ut_new - ut) < 1e-10:
            break
        ut = 0.5 * (ut + ut_new)
    return ut


def lateral(f, y, z, fluid, nu):
    """Depth-averaged lateral distributions and bed shear."""
    dy = y[1] - y[0]
    rows = []
    for kz in range(len(z)):
        col = fluid[:, kz]
        if not col.any():
            continue
        jj = np.where(col)[0]
        depth = (jj[-1] - jj[0] + 1) * dy
        Ud = np.nanmean(f["U"][jj, kz])
        u1 = f["U"][jj[0], kz]
        tau_ww = float(ww_tau(u1, dy, nu))       # dds = full cell size
        tau_lin = nu * u1 / (0.5 * dy)
        yw = (y[jj] - (y[jj[0]] - 0.5 * dy))
        ut_log = clauser_utau(yw, f["U"][jj, kz], nu)
        rows.append((z[kz], depth, Ud, tau_ww, tau_lin, ut_log, Ud * depth))
    a = np.array(rows)
    return dict(z=a[:, 0], depth=a[:, 1], Ud=a[:, 2], tau_bed=a[:, 3],
                tau_linear=a[:, 4], utau_loglaw=a[:, 5], q_unit=a[:, 6])


def vertical(f, y, z, fluid, nu):
    rows = []
    for name, zs in STATIONS.items():
        kz = int(np.argmin(np.abs(z - zs)))
        jj = np.where(fluid[:, kz])[0]
        if not len(jj):
            continue
        ybed = y[jj[0]] - 0.5 * (y[1] - y[0])
        for j in jj:
            yw = y[j] - ybed
            rows.append((name, z[kz], y[j], yw, yw / nu,
                         f["U"][j, kz], f["uu"][j, kz], f["vv"][j, kz],
                         f["ww"][j, kz], f["uv"][j, kz]))
    return rows


# ---------------------------------------------------------------- output ---
def write_vtr(path, y, z, fields):
    """Minimal ASCII VTK rectilinear grid of the (y,z) section."""
    ny, nz = len(y), len(z)
    yb = np.concatenate([[y[0] - 0.5 * (y[1] - y[0])],
                         0.5 * (y[:-1] + y[1:]),
                         [y[-1] + 0.5 * (y[1] - y[0])]])
    zb = np.concatenate([[z[0] - 0.5 * (z[1] - z[0])],
                         0.5 * (z[:-1] + z[1:]),
                         [z[-1] + 0.5 * (z[1] - z[0])]])
    with open(path, "w") as fh:
        fh.write('<?xml version="1.0"?>\n<VTKFile type="RectilinearGrid" '
                 'version="0.1" byte_order="LittleEndian">\n')
        fh.write(f'  <RectilinearGrid WholeExtent="0 0 0 {ny} 0 {nz}">\n')
        fh.write(f'    <Piece Extent="0 0 0 {ny} 0 {nz}">\n')
        fh.write('      <CellData>\n')
        for name, arr in fields.items():
            fh.write(f'        <DataArray type="Float32" Name="{name}" '
                     'format="ascii">\n          ')
            fh.write(" ".join(f"{v:.6g}" for v in
                              np.nan_to_num(arr).T.ravel()))
            fh.write("\n        </DataArray>\n")
        fh.write('      </CellData>\n      <Coordinates>\n')
        for nm, arr in (("x", np.array([0.0])), ("y", yb), ("z", zb)):
            fh.write(f'        <DataArray type="Float32" Name="{nm}" '
                     'format="ascii">\n          ')
            fh.write(" ".join(f"{v:.6g}" for v in arr))
            fh.write("\n        </DataArray>\n")
        fh.write('      </Coordinates>\n    </Piece>\n  </RectilinearGrid>\n'
                 '</VTKFile>\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--mgtools-path", default=os.environ.get("MGTOOLS_PATH"))
    ap.add_argument("--profiles", action="store_true",
                    help="also write the derived 1-D profile files")
    ap.add_argument("--builtin", action="store_true",
                    help="skip mgtools even if available")
    args = ap.parse_args()

    backend = "builtin"
    if not args.builtin:
        try:
            f, y, z, fluid, tsamp = collapse_mgtools(args.case, args.mgtools_path)
            backend = "mgtools"
        except ImportError:
            f, y, z, fluid, tsamp = collapse_builtin(args.case)
    else:
        f, y, z, fluid, tsamp = collapse_builtin(args.case)
    print(f"backend: {backend}")
    if not tsamp or tsamp <= 0.0:
        sys.exit(
            f"\nTSAMP = {tsamp} -- this run collected NO statistics.\n"
            "The ladder rungs (coarse/middle/finer) set tstat = 1e9 on purpose:\n"
            "they are field generators, not measurement runs.\n"
            "Restart the rung with averaging on:\n"
            f"    TAVG=50 ./run_cases.sh average {os.path.basename(os.path.abspath(args.case))}\n"
            "then post-process the resulting <rung>.avg directory.")

    with open(os.path.join(args.case, "parameters.json")) as fh:
        par = json.load(fh)
    nu = par["flow"]["gmol"] / par["flow"].get("rho", 1.0)
    re_tau = 1.0 / nu
    dr = min(UB_TARGET, key=lambda d: abs(d - ((-1.0 / par["flow"]["gradp"][0])
                                               * 7.0 - 2.5) / 2.5))
    f = derive(f, y, z, fluid)

    ub = float(np.nanmean(f["U"][fluid]))
    umax = float(np.nanmax(f["U"]))

    out = os.path.join(args.case, "post")
    os.makedirs(out, exist_ok=True)

    cols = ["U", "V", "W", "P", "uu", "vv", "ww", "uv", "uw", "vw",
            "k", "omega_x"]
    Y, Z = np.meshgrid(y, z, indexing="ij")
    with open(os.path.join(out, "xsec.csv"), "w") as fh:
        fh.write("y,z,fluid," + ",".join(cols) + "\n")
        for j in range(len(y)):
            for kz in range(len(z)):
                vals = ",".join("" if np.isnan(f[c][j, kz]) else
                                f"{f[c][j, kz]:.6g}" for c in cols)
                fh.write(f"{Y[j,kz]:.6g},{Z[j,kz]:.6g},"
                         f"{int(fluid[j,kz])},{vals}\n")

    if args.profiles:
        lat = lateral(f, y, z, fluid, nu)
        vert = vertical(f, y, z, fluid, nu)
        with open(os.path.join(out, "profiles_z.csv"), "w") as fh:
            fh.write("z,depth,Ud,Ud_over_Ub,tau_bed,tau_over_taubar,"
                     "utau_local,tau_linear,utau_loglaw,q_unit\n")
            tbar = float(np.mean(lat["tau_bed"]))
            for i in range(len(lat["z"])):
                fh.write(f"{lat['z'][i]:.6g},{lat['depth'][i]:.6g},"
                         f"{lat['Ud'][i]:.6g},{lat['Ud'][i]/ub:.6g},"
                         f"{lat['tau_bed'][i]:.6g},"
                         f"{lat['tau_bed'][i]/tbar:.6g},"
                         f"{np.sqrt(abs(lat['tau_bed'][i])):.6g},"
                         f"{lat['tau_linear'][i]:.6g},"
                         f"{lat['utau_loglaw'][i]:.6g},"
                         f"{lat['q_unit'][i]:.6g}\n")
        with open(os.path.join(out, "profiles_y.csv"), "w") as fh:
            fh.write("station,z,y,y_wall,y_plus,U,uu,vv,ww,uv\n")
            for r in vert:
                fh.write(f"{r[0]},{r[1]:.6g},{r[2]:.6g},{r[3]:.6g},"
                         f"{r[4]:.6g},{r[5]:.6g},{r[6]:.6g},{r[7]:.6g},"
                         f"{r[8]:.6g},{r[9]:.6g}\n")

    write_vtr(os.path.join(out, "xsec.vtr"), y, z,
              {c: f[c] for c in cols} | {"fluid": fluid.astype(float)})

    meta = {
        "case": os.path.basename(os.path.abspath(args.case)),
        "Dr": dr, "Re_tau": round(re_tau, 1), "nu": nu,
        "backend": backend, "tsamp": tsamp,
        "grid": {"ny": len(y), "nz": len(z),
                 "dy": float(y[1] - y[0]), "dz": float(z[1] - z[0])},
        "units": {"length": "H", "velocity": "u_tau", "stress": "u_tau^2"},
        "derived": {"U_bulk": ub, "U_max": umax,
                    "U_bulk_target": UB_TARGET[dr],
                    "deviation_pct": 100 * (ub / UB_TARGET[dr] - 1)},
        "u_tau": {
            "global": "exactly 1 by construction: tau_avg = |gradp|*A/P = 1. "
                      "No wall gradient is needed or used.",
            "local_tau_bed": "Werner-Wengle, identical to the solver's own "
                             "wall model, so tau(z) integrates to the global "
                             "balance",
            "utau_loglaw": "Clauser fit assuming kappa=0.41, B=5.3 - the same "
                           "method Tominaga & Nezu used for U*. Not the same "
                           "quantity as the imposed u_tau if the log law is "
                           "shifted.",
        },
        "conventions": {
            "reynolds_stress": "<ui'uj'> = mean_x[<ui uj>] - Ui*Uj, with Ui "
                               "the x-averaged mean",
            "sign": "uv is <u'v'>; Tominaga & Nezu plot -<u'v'>/u_tau^2",
            "note": "cross-stresses are stored on edges and interpolated to "
                    "cell centres, an O(dx^2) inconsistency vs U*V",
        },
    }
    with open(os.path.join(out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"  U_b/u_tau = {ub:.3f}  (target {UB_TARGET[dr]:.2f}, "
          f"{100*(ub/UB_TARGET[dr]-1):+.1f} %)")
    print(f"  U_max/u_tau = {umax:.3f}   TSAMP = {tsamp:.1f}")
    extra = " + profiles_z.csv, profiles_y.csv" if args.profiles else ""
    print(f"  wrote {out}/xsec.csv (the collapsed y-z plane), "
          f"meta.json, xsec.vtr{extra}")


if __name__ == "__main__":
    main()
