#!/usr/bin/env python3
"""Generate MGLET cases for the Tominaga & Nezu (1991) compound open channel.

Setup follows Kara, Stoesser & Sturm (2012), JHR 50(5), 482-493, who ran LES of
the same experiment:

  - main channel depth H, floodplain depth h, depth ratio Dr = h/H
  - main channel width 2.5H, floodplain width 2.5H, total width 5H
  - streamwise direction periodic, Lx = 12H (Kara) or 4*pi*H
  - bed and both sidewalls and the step riser: no-slip
  - free surface: frictionless rigid lid = plane of symmetry (MGLET 'SLI')
  - flow driven by a uniform streamwise pressure gradient which fixes the
    global friction velocity unambiguously

Non-dimensionalisation: H = 1, u_tau = 1, rho = 1.  Hence
    gmol  = nu = 1/Re_tau
    gradp = -H/Rh   (from rho*g*Ie*A = tau*P, i.e. u_tau^2 = g*Ie*Rh)

The fluid region is an L-shaped cross-section built from body-fitted Cartesian
blocks, so no immersed boundary is needed ("ib": {"type": "noib"}).

Coordinates:  x = streamwise, y = vertical (0 = main channel bed, 1 = free
surface), z = spanwise (0 = main channel sidewall, 2.5 = junction, 5 = outer
floodplain wall).  The solid step occupies z > 2.5, y < D with D = 1 - Dr.
"""

import argparse
import json
import os

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# Face / neighbour conventions, taken from src/core/connect2_mod.F90
#
#   face 1 = FRONT  (low  x)      face 4 = LEFT   (high y)
#   face 2 = BACK   (high x)      face 5 = BOTTOM (low  z)
#   face 3 = RIGHT  (low  y)      face 6 = TOP    (high z)
#
# facelist(4, 26) gives, for each of the 26 neighbour slots, the rank
# (1 = face, 2 = edge, 3 = corner) followed by the faces that are combined.
# ---------------------------------------------------------------------------
FACE_OFFSET = {1: (-1, 0, 0), 2: (1, 0, 0),
               3: (0, -1, 0), 4: (0, 1, 0),
               5: (0, 0, -1), 6: (0, 0, 1)}

FACELIST = [
    (1, 1, 0, 0), (1, 2, 0, 0), (1, 3, 0, 0), (1, 4, 0, 0),
    (1, 5, 0, 0), (1, 6, 0, 0),
    (2, 1, 3, 0), (2, 1, 4, 0), (2, 1, 5, 0), (2, 1, 6, 0),
    (2, 2, 3, 0), (2, 2, 4, 0), (2, 2, 5, 0), (2, 2, 6, 0),
    (2, 3, 5, 0), (2, 3, 6, 0), (2, 4, 5, 0), (2, 4, 6, 0),
    (3, 1, 3, 5), (3, 1, 3, 6), (3, 1, 4, 5), (3, 1, 4, 6),
    (3, 2, 3, 5), (3, 2, 3, 6), (3, 2, 4, 5), (3, 2, 4, 6),
]

NGHOST = 2  # ghost cells on each side, so II = ncells_x + 4

GRIDINFO_DTYPE = np.dtype([
    ("IGRID", "<i4"), ("II", "<i4"), ("JJ", "<i4"), ("KK", "<i4"),
    ("LEVEL", "<i4"), ("IPARENT", "<i4"),
    ("IPOSITION", "<i4"), ("JPOSITION", "<i4"), ("KPOSITION", "<i4"),
    ("FLAGS", "<i4"), ("IGRIDLVL", "<i4"),
    ("BBOX", "<f8", (6,)), ("NBRGRID", "<i4", (26,)),
])

BCOND_DTYPE = np.dtype([
    ("NBOCD", "<i4"), ("TYPE", "S8", (2,)),
    ("INTPRM", "<i4", (2, 2)), ("REALPRM", "<i4", (2, 2)),
])

# Geometry of the Tominaga & Nezu flume, scaled by H
Z_JUNCTION = 2.5   # main channel width / H
Z_TOTAL = 5.0      # total width / H
NYB = 8            # y-blocks: makes D/H = 4/8, 5/8, 6/8 land on a block edge


def closest_divisor(n, target):
    """Divisor of n closest to target (ties -> larger divisor)."""
    divs = [d for d in range(1, n + 1) if n % d == 0]
    return min(divs, key=lambda d: (abs(d - target), -d))


def hydraulic_radius(dr):
    """A/P for the compound cross-section, in units of H (H = 1)."""
    area = Z_JUNCTION * 1.0 + (Z_TOTAL - Z_JUNCTION) * dr
    perimeter = 2.0 * 1.0 + Z_TOTAL
    return area / perimeter


class Blocking:
    """Uniform brick decomposition of the L-shaped fluid cross-section."""

    def __init__(self, dr, lx, ncx, ncy, ncz_half, nxb, nzb):
        self.dr = dr
        self.step = 1.0 - dr            # D/H, height of the solid step
        self.lx = lx

        self.ncx, self.ncy = ncx, ncy
        self.ncz = 2 * ncz_half         # z = 2.5 must be a cell boundary
        self.nxb, self.nyb, self.nzb = nxb, NYB, nzb

        for name, nc, nb in (("x", self.ncx, nxb), ("y", ncy, NYB),
                             ("z", self.ncz, nzb)):
            if nc % nb:
                raise ValueError(f"{nc} cells in {name} not divisible by "
                                 f"{nb} blocks")
        if nzb % 2:
            raise ValueError("number of z-blocks must be even (junction at z=2.5)")

        self.cx, self.cy, self.cz = ncx // nxb, ncy // NYB, self.ncz // nzb
        self.dx, self.dy = lx / ncx, 1.0 / ncy
        self.dz = Z_TOTAL / self.ncz

        # Enumerate fluid blocks; index -> 1-based grid id
        self.ids = {}
        self.blocks = []
        for ibx in range(nxb):
            for iby in range(NYB):
                for ibz in range(nzb):
                    if self.is_fluid(iby, ibz):
                        self.ids[(ibx, iby, ibz)] = len(self.blocks) + 1
                        self.blocks.append((ibx, iby, ibz))
        self.ngrid = len(self.blocks)

    def is_fluid(self, iby, ibz):
        """A block is solid only if it lies fully inside the step."""
        yc = (iby + 0.5) / NYB
        zc = (ibz + 0.5) * Z_TOTAL / self.nzb
        return not (zc > Z_JUNCTION and yc < self.step)

    def bbox(self, ibx, iby, ibz):
        return (ibx * self.lx / self.nxb, (ibx + 1) * self.lx / self.nxb,
                iby / NYB, (iby + 1) / NYB,
                ibz * Z_TOTAL / self.nzb, (ibz + 1) * Z_TOTAL / self.nzb)

    def neighbour(self, ibx, iby, ibz, off):
        """Grid id of the block at the given offset, 0 if none. x is periodic."""
        jx = (ibx + off[0]) % self.nxb
        jy, jz = iby + off[1], ibz + off[2]
        if not (0 <= jy < NYB and 0 <= jz < self.nzb):
            return 0
        return self.ids.get((jx, jy, jz), 0)

    def face_bc(self, ibx, iby, ibz, iface):
        """Boundary condition string on one face of one block."""
        if self.neighbour(ibx, iby, ibz, FACE_OFFSET[iface]):
            return "CON"
        # No fluid neighbour -> a physical boundary.
        # The only non-wall boundary is the free surface at y = 1 (face 4).
        return "SLI" if iface == 4 else "NOS"


def write_grids(blk, path):
    gridinfo = np.zeros(blk.ngrid, dtype=GRIDINFO_DTYPE)
    faces = {n: np.zeros(blk.ngrid, dtype=BCOND_DTYPE)
             for n in ("FRONT", "BACK", "RIGHT", "LEFT", "BOTTOM", "TOP")}
    face_names = {1: "FRONT", 2: "BACK", 3: "RIGHT",
                  4: "LEFT", 5: "BOTTOM", 6: "TOP"}

    ii, jj, kk = (blk.cx + 2 * NGHOST, blk.cy + 2 * NGHOST, blk.cz + 2 * NGHOST)
    coords = {k: np.zeros((blk.ngrid, n)) for k, n in
              (("X", ii), ("Y", jj), ("Z", kk))}
    deltas = {k: np.zeros((blk.ngrid, n)) for k, n in
              (("DX", ii), ("DY", jj), ("DZ", kk))}

    for idx, (ibx, iby, ibz) in enumerate(blk.blocks):
        igrid = idx + 1
        bbox = blk.bbox(ibx, iby, ibz)

        nbr = np.zeros(26, dtype=np.int32)
        for slot, entry in enumerate(FACELIST):
            off = [0, 0, 0]
            for f in entry[1:]:
                if f:
                    for d in range(3):
                        off[d] += FACE_OFFSET[f][d]
            nbr[slot] = blk.neighbour(ibx, iby, ibz, off)

        gridinfo[idx] = (igrid, ii, jj, kk, 1, 0, -1, -1, -1, 1, igrid,
                         bbox, nbr)

        for iface in range(1, 7):
            rec = faces[face_names[iface]]
            rec[idx]["NBOCD"] = 1
            rec[idx]["TYPE"] = [blk.face_bc(ibx, iby, ibz, iface).encode(), b""]
            rec[idx]["INTPRM"] = [[-1, -1], [0, 0]]
            rec[idx]["REALPRM"] = [[-1, -1], [0, 0]]

        # Cell centres including ghost cells: c[i] = lo + (i - NGHOST + 0.5)*d
        for key, lo, d, n in (("X", bbox[0], blk.dx, ii),
                              ("Y", bbox[2], blk.dy, jj),
                              ("Z", bbox[4], blk.dz, kk)):
            coords[key][idx] = lo + (np.arange(n) - NGHOST + 0.5) * d
            deltas["D" + key][idx] = d

    with h5py.File(path, "w") as f:
        dset = f.create_dataset("GRIDINFO", data=gridinfo)
        dset.attrs["INTPRMS"] = np.array([0, 0], dtype=np.int32)
        dset.attrs["REALPRMS"] = np.array([-1.0, 1.0])
        for name, rec in faces.items():
            f.create_dataset(name, data=rec)
        for name, arr in list(coords.items()) + list(deltas.items()):
            grp = f.create_group(name)
            grp.attrs["MINLEVEL"] = np.int64(1)
            grp.attrs["MAXLEVEL"] = np.int64(1)
            grp.create_dataset("LEVEL1", data=arr)
            # Uniform grid: distance between centres equals the cell size
            if name.startswith("D"):
                grp2 = f.create_group("D" + name)
                grp2.attrs["MINLEVEL"] = np.int64(1)
                grp2.attrs["MAXLEVEL"] = np.int64(1)
                grp2.create_dataset("LEVEL1", data=arr)


def ic_expression(component, blk):
    """Initial condition: log-law on the local bed + deterministic modes.

    tu_level noise is only applied when uinf is a constant vector
    (src/flow/flow_mod.F90:195), so the perturbation has to live inside the
    expression.  Deterministic modes are used rather than 'rand' so that the
    expression stays time-independent.
    """
    step, lx, dr = blk.step, blk.lx, blk.dr
    common = f"""\
var step := {step:.6f};
var lx := {lx:.8f};
var dep;
var yw;
if (z > {Z_JUNCTION}) {{ dep := {dr:.6f}; yw := y - step; }}
else {{ dep := 1.0; yw := y; }};
if (yw < 0.0) {{ yw := 0.0; }};

var yp := yw/gmol;
var ulog;
if (yp < 11.6) {{ ulog := yp; }}
else {{ ulog := 2.43902*log(yp) + 5.3; }};

var eta := yw/dep;
var shape := sin(pi*eta);
var px := 2.0*pi*x/lx;
var pz := 2.0*pi*z/{Z_TOTAL};
"""
    if component == "u":
        body = ("ulog*(1.0 + 0.15*sin(3.0*px)*cos(6.0*pz)\n"
                "             + 0.10*sin(5.0*px)*cos(10.0*pz));\n")
    elif component == "v":
        body = ("0.8*shape*(sin(3.0*px)*cos(6.0*pz)\n"
                "           + 0.6*sin(5.0*px)*cos(11.0*pz));\n")
    else:
        body = ("0.8*shape*(cos(3.0*px)*sin(6.0*pz)\n"
                "           + 0.6*cos(5.0*px)*sin(11.0*pz));\n")
    return common + body


STATISTICS = [
    "U_AVG", "V_AVG", "W_AVG", "P_AVG",
    "UU_AVG", "VV_AVG", "WW_AVG",
    "UV_AVG", "UW_AVG", "VW_AVG",
]


def probe_arrays(blk, nx=64, ny=48, nz=100):
    """Runtime probe rakes.

    The streamwise rakes are the important ones: R_uu(dx) computed from them
    must decorrelate well before dx = Lx/2, otherwise the domain is too short.
    This is the empirical check Kara et al. used to justify Lx = 12H.
    """
    dr, step, lx = blk.dr, blk.step, blk.lx
    y_fp = step + 0.5 * dr          # mid-depth of the floodplain flow
    arrays = []

    for name, y, z in (("xrake_shear", y_fp, 2.55),      # junction shear layer
                       ("xrake_main", 0.5, 1.25),        # main channel core
                       ("xrake_fp", y_fp, 3.75)):        # floodplain core
        arrays.append({
            "name": name,
            "variables": ["U", "V", "W"],
            "positions": [[(i + 0.5) * lx / nx, y, z] for i in range(nx)],
        })

    # Vertical profiles: log-law check (Tominaga & Nezu Fig. 7) and the
    # velocity-dip in the main channel (Fig. 5)
    for name, z in (("prof_main", 1.25), ("prof_junc", 2.40),
                    ("prof_fp", 3.75)):
        ybed = step if z > Z_JUNCTION else 0.0
        arrays.append({
            "name": name,
            "variables": ["U", "V", "W", "P"],
            "positions": [[0.5 * lx, ybed + (1.0 - ybed) * (i + 0.5) / ny, z]
                          for i in range(ny)],
        })

    # Spanwise rake just below the free surface, where it stays in the fluid
    # across the full width: tracks the secondary-current cells
    arrays.append({
        "name": "zrake_surf",
        "variables": ["U", "V", "W"],
        "positions": [[0.5 * lx, 0.95, (i + 0.5) * Z_TOTAL / nz]
                      for i in range(nz)],
    })
    return arrays


def spod_arrays(blk, nplanes=16, sub=4, surface=True):
    """Cross-plane (y-z) sections at EQUALLY spaced x, plus a near-surface plane.

    Equal spacing is deliberate.  x is periodic and statistically homogeneous,
    so the streamwise direction should be Fourier-transformed rather than
    treated as a source of "independent" samples: SPOD is then computed per
    (k_x, omega), which is exactly the decomposition the resolvent operator is
    block-diagonal in.  That construction is exact and needs no decorrelation
    assumption between stations.

    A handful of unequally spaced stations (x = 1, 4, 8) cannot give k_x and,
    because advection makes distant stations phase-shifted copies of each
    other, they are not independent realisations at low frequency either.
    See coherence() in monitor.py.
    """
    step, lx = blk.step, blk.lx
    pts_yz = []
    for jy in range(0, blk.ncy, sub):
        y = (jy + 0.5) * blk.dy
        for kz in range(0, blk.ncz, sub):
            z = (kz + 0.5) * blk.dz
            if z > Z_JUNCTION and y < step:
                continue                      # inside the solid step
            pts_yz.append((y, z))

    arrays = []
    for ip in range(nplanes):
        # Half-cell offset: x = 0 sits exactly on the periodic boundary and
        # every probe there comes back "notfound". Still equally spaced, so
        # the streamwise FFT is unaffected apart from a constant phase.
        x = (ip + 0.5) * lx / nplanes
        arrays.append({
            "name": f"xsec{ip:02d}",
            "variables": ["U", "V", "W", "P"],
            "positions": [[x, y, z] for (y, z) in pts_yz],
        })

    if surface:
        # Top cell centre, just under the rigid lid. Equally spaced in x and z
        # so this plane can be Fourier transformed in both directions.
        ysurf = (blk.ncy - 0.5) * blk.dy
        nxs = max(4, blk.ncx // (2 * sub))
        arrays.append({
            "name": "surface",
            "variables": ["U", "V", "W", "P"],
            "positions": [[(i + 0.5) * lx / nxs, ysurf, (kz + 0.5) * blk.dz]
                          for i in range(nxs)
                          for kz in range(0, blk.ncz, sub)],
        })
    return arrays


def write_parameters(blk, re_tau, path, tend, tstat, dt, cfl, lesmodel,
                     spod_planes, spod_sub, spod_surface, probe_itsamp):
    rh = hydraulic_radius(blk.dr)
    params = {
        "time": {
            "dt": dt,
            "targetcflmax": cfl,
            "mtstep": 100000000,
            "tend": tend,
            "tstat": tstat,
            "itinfo": 100,
            "itsamp": 5,
            "read": False,
            "write": True,
            "continue": False,
        },
        "io": {
            "grids": "grids.h5",
            "infile": "none.h5",
            "outfile": "fields.h5",
        },
        "flow": {
            "gmol": 1.0 / re_tau,
            "rho": 1.0,
            "gradp": [-1.0 / rh, 0.0, 0.0],
            "uinf": [ic_expression(c, blk) for c in ("u", "v", "w")],
            "pressuresolver": {
                "type": "sip",
                "epcorr": 5e-5,
                "nouter_min": 2,
                "nouter": 12,
                "ninner": 6,
                "omega": 1.1,
                "loglevel": 3,
            },
            "lesmodel": {"model": lesmodel},
        },
        "ib": {"type": "noib"},
        "statistics": STATISTICS,
        # Bulk velocity time series -> LOGS/uvwbulk.log, written every
        # /time/itinfo steps.  UBULK is U_b/u_tau and is the primary
        # convergence and validation metric for this case.
        "uvwbulk": True,
        # NOTE for SPOD: dt is only frozen once timeph >= tstat
        # (src/timeloop_mod.F90:313).  Samples taken before tstat are NOT
        # uniformly spaced in time, so discard t < tstat before any time FFT.
        "probes": {
            "itsamp": probe_itsamp,
            "tstart": 0.0,
            "file": "probes.h5",
            "arrays": probe_arrays(blk) + spod_arrays(
                blk, spod_planes, spod_sub, spod_surface),
        },
    }
    with open(path, "w") as f:
        json.dump(params, f, indent=4)
        f.write("\n")


# Friction Reynolds numbers from the similarity analysis of Tominaga & Nezu
# Table 1 with a consistent nu = 1.10e-6 m^2/s (see README.md).
RE_TAU = {0.500: 1193.0, 0.375: 1109.0, 0.250: 1025.0}

# Target U_b/u_tau from the experiment (Dr = 0.375 interpolated).  This ratio
# is independent of the assumed viscosity, so it is the cleanest validation
# target: it is what UBULK in LOGS/uvwbulk.log must converge to.
UB_TARGET = {0.500: 21.28, 0.375: 20.85, 0.250: 20.43}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dr", type=float, required=True, choices=sorted(RE_TAU),
                   help="floodplain-to-main-channel depth ratio h/H")
    p.add_argument("--outdir", required=True)
    p.add_argument("--lx", type=float, default=12.0,
                   help="streamwise domain length in units of H (default 12, "
                        "Kara et al.; 4*pi = 12.566 is equivalent)")
    p.add_argument("--ncy", type=int, default=160,
                   help="cells over the main channel depth H (multiple of 8)")
    p.add_argument("--aspect-x", type=float, default=3.2, help="dx/dy")
    p.add_argument("--aspect-z", type=float, default=1.6, help="dz/dy")
    p.add_argument("--block", type=int, default=0,
                   help="target cells per block edge (default: ncy/8)")
    p.add_argument("--re-tau", type=float, default=None,
                   help="override the friction Reynolds number")
    p.add_argument("--tend", type=float, default=400.0, help="end time in H/u_tau")
    p.add_argument("--tstat", type=float, default=100.0,
                   help="time at which statistics sampling starts")
    p.add_argument("--dt", type=float, default=2.0e-3)
    p.add_argument("--cfl", type=float, default=0.8)
    p.add_argument("--lesmodel", default="wale", choices=["wale", "smagorinsky", "none"])
    p.add_argument("--spod-planes", type=int, default=16,
                   help="equally spaced y-z cross-sections for SPOD/resolvent "
                        "(0 disables). Must resolve the k_x range of interest.")
    p.add_argument("--spod-sub", type=int, default=4,
                   help="cross-plane point subsampling relative to the LES grid")
    p.add_argument("--no-surface", action="store_true",
                   help="omit the near-surface x-z plane")
    p.add_argument("--probe-itsamp", type=int, default=80,
                   help="probe sampling interval in timesteps")
    args = p.parse_args()

    if args.ncy % NYB:
        p.error(f"--ncy must be a multiple of {NYB}")

    ncx = int(round(args.lx / (args.aspect_x / args.ncy)))
    ncz_half = int(round(Z_JUNCTION / (args.aspect_z / args.ncy)))

    target = args.block or max(1, args.ncy // NYB)
    nxb = closest_divisor(ncx, max(1, round(ncx / target)))
    nzb_half = closest_divisor(ncz_half, max(1, round(ncz_half / target)))

    blk = Blocking(args.dr, args.lx, ncx, args.ncy, ncz_half, nxb, 2 * nzb_half)
    re_tau = args.re_tau if args.re_tau is not None else RE_TAU[args.dr]

    os.makedirs(args.outdir, exist_ok=True)
    write_grids(blk, os.path.join(args.outdir, "grids.h5"))
    write_parameters(blk, re_tau, os.path.join(args.outdir, "parameters.json"),
                     args.tend, args.tstat, args.dt, args.cfl, args.lesmodel,
                     args.spod_planes, args.spod_sub, not args.no_surface,
                     args.probe_itsamp)

    ncells = blk.ngrid * blk.cx * blk.cy * blk.cz
    dnu = 1.0 / re_tau
    print(f"Dr = {args.dr}  (h/H),  step D/H = {blk.step:.3f}")
    print(f"  domain          {args.lx:g}H x 1H x {Z_TOTAL:g}H, x periodic")
    print(f"  Re_tau          {re_tau:.0f}   gmol = {1/re_tau:.6e}")
    print(f"  R_h/H           {hydraulic_radius(args.dr):.4f}"
          f"   gradp_x = {-1/hydraulic_radius(args.dr):.4f}")
    print(f"  cells           {ncx} x {args.ncy} x {blk.ncz}"
          f"  (bounding box), {ncells/1e6:.2f} M in the fluid")
    print(f"  blocks          {nxb} x {NYB} x {2*nzb_half}"
          f"  -> {blk.ngrid} grids of {blk.cx}x{blk.cy}x{blk.cz}")
    print(f"  spacing in +    dx+ = {blk.dx/dnu:.1f}, dy+ = {blk.dy/dnu:.2f}, "
          f"dz+ = {blk.dz/dnu:.1f}")
    print(f"  target UBULK    {UB_TARGET[args.dr]:.2f}  "
          f"(U_b/u_tau; watch LOGS/uvwbulk.log)")

    if args.spod_planes:
        sp = spod_arrays(blk, args.spod_planes, args.spod_sub,
                         not args.no_surface)
        npts = sum(len(a["positions"]) for a in sp)
        nxsec = len(sp[0]["positions"])
        kmax = args.spod_planes // 2
        print(f"  SPOD sampling   {args.spod_planes} y-z planes"
              f" ({nxsec} pts each) + surface, {npts} probe points")
        print(f"                  dx_plane = {args.lx/args.spod_planes:.3f}H"
              f"  ->  k_x modes n = 0..{kmax}"
              f"  (lambda_x = {args.lx:g}H .. {args.lx/kmax:.2f}H)")
        print(f"                  {npts*4*8/1e6:.1f} MB per sample;"
              f" use only t >= tstat = {args.tstat:g} for time FFTs")


if __name__ == "__main__":
    main()
