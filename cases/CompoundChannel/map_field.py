#!/usr/bin/env python3
"""Map a developed velocity field from a coarse grid onto a fine grid.

Grid sequencing: transition to turbulence is expensive and does not need the
final resolution, so it is done on a cheap coarse grid and the developed field
is then interpolated onto the fine grid. Only the velocity primitives u, v, w
are mapped -- mapping pressure produces artifacts, and it is unnecessary here
because the flow is driven by a prescribed gradp, not by the stored P. P is
written as zeros (MGLET requires the field to be present on restart,
src/flow/flowcore_mod.F90:132) and the first projection re-establishes it.

    python3 map_field.py --from Dr05_coarse.spinup --to Dr05 --out Dr05/restart.h5

Restart with "read": true and "continue": false, so the clock starts at zero
and no RUNINFO consistency is required.
"""

import argparse
import os
import sys

import h5py
import numpy as np

NGHOST = 2
Z_TOTAL = 5.0


def read_grid(path):
    with h5py.File(path, "r") as f:
        gi = f["GRIDINFO"][:]
    g0 = gi[0]
    dx = (g0["BBOX"][1] - g0["BBOX"][0]) / (g0["II"] - 2 * NGHOST)
    dy = (g0["BBOX"][3] - g0["BBOX"][2]) / (g0["JJ"] - 2 * NGHOST)
    dz = (g0["BBOX"][5] - g0["BBOX"][4]) / (g0["KK"] - 2 * NGHOST)
    lx = max(g["BBOX"][1] for g in gi)
    nx, ny, nz = (int(round(lx / dx)), int(round(1.0 / dy)),
                  int(round(Z_TOTAL / dz)))
    return gi, (dx, dy, dz), (nx, ny, nz), lx


def assemble(casedir, name, gi, d, n):
    """Scatter one field's blocks into a global array of real cells."""
    dx, dy, dz = d
    glob = np.zeros(n)
    with h5py.File(os.path.join(casedir, "fields.h5"), "r") as f:
        grp = f["VOLUMEFIELDS"][name]
        data, idx = grp["LEVEL1"], np.asarray(grp["IGRIDLVL"])
        stag = (int(grp.attrs["ISTAG"]), int(grp.attrs["JSTAG"]),
                int(grp.attrs["KSTAG"]))
        for row, igrid in enumerate(idx):
            g = gi[igrid - 1]
            ii, jj, kk = int(g["II"]), int(g["JJ"]), int(g["KK"])
            a = np.asarray(data[row], dtype=np.float64).reshape(ii, jj, kk)
            cell = a[NGHOST:ii - NGHOST, NGHOST:jj - NGHOST, NGHOST:kk - NGHOST]
            i0 = int(round(g["BBOX"][0] / dx))
            j0 = int(round(g["BBOX"][2] / dy))
            k0 = int(round(g["BBOX"][4] / dz))
            glob[i0:i0 + cell.shape[0],
                 j0:j0 + cell.shape[1],
                 k0:k0 + cell.shape[2]] = cell
    return glob, stag


def axis_coords(n, delta, stag):
    """Physical coordinate of each sample along one axis."""
    return (np.arange(n) + (1.0 if stag else 0.5)) * delta


def trilinear(src, sx, sy, sz, tx, ty, tz, lx):
    """Interpolate src(sx,sy,sz) at the outer product of (tx,ty,tz).

    x is periodic with period lx; y and z are clamped at the boundaries.
    """
    def weights(s, t, periodic=False, period=None):
        if periodic:
            ds = s[1] - s[0]
            pos = (t - s[0]) / ds
            i0 = np.floor(pos).astype(int)
            w = pos - i0
            return i0 % len(s), (i0 + 1) % len(s), w
        i0 = np.clip(np.searchsorted(s, t) - 1, 0, len(s) - 2)
        w = np.clip((t - s[i0]) / (s[i0 + 1] - s[i0]), 0.0, 1.0)
        return i0, i0 + 1, w

    ia, ib, wx = weights(sx, tx, periodic=True, period=lx)
    ja, jb, wy = weights(sy, ty)
    ka, kb, wz = weights(sz, tz)

    wx = wx[:, None, None]; wy = wy[None, :, None]; wz = wz[None, None, :]
    out = np.zeros((len(tx), len(ty), len(tz)))
    for ii, wi in ((ia, 1 - wx), (ib, wx)):
        for jj, wj in ((ja, 1 - wy), (jb, wy)):
            for kk, wk in ((ka, 1 - wz), (kb, wz)):
                out += wi * wj * wk * src[np.ix_(ii, jj, kk)]
    return out


def write_fields(path, gi, blocks, template):
    """Write a restart file in MGLET's layout (DATA + virtual LEVEL1)."""
    ngrid = len(gi)
    with h5py.File(path, "w") as f:
        vol = f.create_group("VOLUMEFIELDS")
        for name, (arrs, attrs) in blocks.items():
            grp = vol.create_group(name)
            lens = [a.size for a in arrs]
            flat = np.concatenate([a.ravel() for a in arrs]).astype(np.float32)
            grp.create_dataset("DATA", data=flat)
            grp.create_dataset("IGRIDLVL",
                               data=np.arange(1, ngrid + 1, dtype=np.int32))
            off = np.zeros(ngrid, dtype=[("OFFSET", "<i8"), ("LENGTH", "<i8")])
            pos = 1
            for i, L in enumerate(lens):
                off[i] = (pos, L)
                pos += L
            grp.create_dataset("OFFSET", data=off)

            layout = h5py.VirtualLayout(shape=(ngrid, lens[0]), dtype="f4")
            src = h5py.VirtualSource(".", f"/VOLUMEFIELDS/{name}/DATA",
                                     shape=(flat.size,))
            for i in range(ngrid):
                layout[i] = src[i * lens[0]:(i + 1) * lens[0]]
            grp.create_virtual_dataset("LEVEL1", layout)

            for k, v in attrs.items():
                grp.attrs[k] = v
        f.create_dataset("PARAMETERS", data=template)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", required=True, help="coarse case dir")
    ap.add_argument("--to", dest="dst", required=True, help="fine case dir")
    ap.add_argument("--out", default=None, help="output file (default <to>/restart.h5)")
    args = ap.parse_args()
    out = args.out or os.path.join(args.dst, "restart.h5")

    cgi, cd, cn, clx = read_grid(os.path.join(args.src, "grids.h5"))
    fgi, fd, fn, flx = read_grid(os.path.join(args.dst, "grids.h5"))
    if abs(clx - flx) > 1e-9:
        sys.exit(f"domain length differs: coarse {clx} vs fine {flx}")
    print(f"coarse {cn}  ->  fine {fn}   (Lx = {flx:g}H)")

    fields, attrs_out = {}, {}
    for name in ("U", "V", "W"):
        glob, stag = assemble(args.src, name, cgi, cd, cn)
        sx = axis_coords(cn[0], cd[0], stag[0])
        sy = axis_coords(cn[1], cd[1], stag[1])
        sz = axis_coords(cn[2], cd[2], stag[2])
        fields[name] = (glob, stag, sx, sy, sz)
        attrs_out[name] = stag
        print(f"  {name}: assembled coarse field, |max| = {np.abs(glob).max():.3f}")

    units_v = np.array([0, 1, -1, 0, 0, 0, 0], dtype=np.int32)
    units_p = np.array([1, -1, -2, 0, 0, 0, 0], dtype=np.int32)
    blocks = {}
    for name in ("U", "V", "W", "P"):
        arrs = []
        stag = attrs_out.get(name, (0, 0, 0))
        for g in fgi:
            ii, jj, kk = int(g["II"]), int(g["JJ"]), int(g["KK"])
            if name == "P":
                arrs.append(np.zeros((ii, jj, kk)))
                continue
            glob, _, sx, sy, sz = fields[name]
            tx = g["BBOX"][0] + (np.arange(ii) - NGHOST
                                 + (1.0 if stag[0] else 0.5)) * fd[0]
            ty = g["BBOX"][2] + (np.arange(jj) - NGHOST
                                 + (1.0 if stag[1] else 0.5)) * fd[1]
            tz = g["BBOX"][4] + (np.arange(kk) - NGHOST
                                 + (1.0 if stag[2] else 0.5)) * fd[2]
            ty = np.clip(ty, sy[0], sy[-1])
            tz = np.clip(tz, sz[0], sz[-1])
            arrs.append(trilinear(glob, sx, sy, sz, tx, ty, tz, flx))
        a = {"FIELDNAME": np.bytes_(name), "MINLEVEL": np.int32(1),
             "MAXLEVEL": np.int32(1),
             "ISTAG": np.int32(stag[0]), "JSTAG": np.int32(stag[1]),
             "KSTAG": np.int32(stag[2]),
             "UNITS": units_p if name == "P" else units_v}
        blocks[name] = (arrs, a)
        print(f"  {name}: mapped onto {len(arrs)} fine blocks")

    with h5py.File(os.path.join(args.src, "fields.h5"), "r") as f:
        template = f["PARAMETERS"][()]
    write_fields(out, fgi, blocks, template)
    print(f"wrote {out}")
    print('restart with "read": true, "continue": false, "infile": "restart.h5"')


if __name__ == "__main__":
    main()
