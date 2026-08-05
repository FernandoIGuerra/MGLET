#!/usr/bin/env python3
"""Spectral POD of the compound-channel probe planes, and of the free surface.

Two decompositions, both fed by the runtime probe arrays in ``probes.h5``:

**Cross-planes** (``xsec00``…``xsecNN``): equally spaced y–z sections.  x is
periodic and statistically homogeneous, so it is Fourier transformed rather
than treated as a source of independent samples; SPOD is then computed
separately for every streamwise wavenumber k_x, which is the decomposition the
resolvent operator is block-diagonal in.

**Free surface** (``surface``): an x–z plane at the top cell centre, i.e. a top
view of the rigid lid.  The lid is a symmetry plane, so the tangential
vorticity vanishes there but the *surface-normal* component

    omega_y = du/dz - dw/dx

does not — it is exactly the signature of the vertical-axis vortices that
develop over the junction and carry fluid (and streamwise momentum) laterally
between main channel and floodplain.  This script gives both the instantaneous
top view of omega_y and the SPOD of the surface plane, plus the fraction of the
surface Reynolds stress <u'w'> that the leading SPOD mode accounts for.

Usage::

    python3 spod.py <casedir> --check        # can SPOD be obtained at all?
    python3 spod.py <casedir>                # both decompositions + plots
    python3 spod.py <casedir> --surface      # free surface only
    python3 spod.py --selftest               # verify the SPOD kernel

Sampling caveats, both enforced by ``--check``:

* MGLET freezes dt only once ``timeph >= /time/tstat``
  (``src/timeloop_mod.F90:313``).  Samples taken before that are not equally
  spaced in time and must not enter a time FFT.
* A stored probe sample is *not* instantaneous: ``sample_variable`` keeps a
  running mean over the ``/probes/itsamp`` steps of the window
  (``src/plugins/probes_mod.F90:592``).  That is a boxcar pre-filter of width
  dt_s, i.e. an anti-aliasing filter, but it also attenuates resolved
  frequencies by |sinc(f·dt_s)|.  The spectra are divided by |H(f)|^2 unless
  ``--no-boxcar-correction`` is given.
"""

import argparse
import json
import os
import sys

import numpy as np

Z_JUNCTION = 2.5
Z_TOTAL = 5.0


# --------------------------------------------------------------- probe I/O --
def probe_arrays_json(case):
    """Requested probe positions, keyed by array name.

    parameters.json is the authority on point ordering: MGLET scatters each
    rank's points back to their original global index before writing
    (``write_coordinates``), so row i of a probe dataset is position i of the
    request.
    """
    path = os.path.join(case, "parameters.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        d = json.load(f)
    out = {}
    for a in d.get("probes", {}).get("arrays", []):
        out[a["name"]] = dict(pos=np.asarray(a["positions"], dtype=float),
                              variables=list(a["variables"]))
    return out


def probe_meta(case):
    path = os.path.join(case, "parameters.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        d = json.load(f)
    return dict(itsamp=d.get("probes", {}).get("itsamp"),
                tstat=d.get("time", {}).get("tstat"),
                dt=d.get("time", {}).get("dt"),
                gmol=d.get("flow", {}).get("gmol"))


def open_probes(case):
    import h5py
    path = os.path.join(case, "probes.h5")
    if not os.path.exists(path):
        sys.exit(f"{path} not found - has the run written probes yet?")
    f = h5py.File(path, "r")
    return f, (f["PROBES"] if "PROBES" in f else f)


def read_time(grp):
    """PROBES/time is a compound dataset with fields ITTOT and TIME."""
    raw = grp["time"][:]
    if raw.dtype.names and "TIME" in raw.dtype.names:
        return np.asarray(raw["TIME"], float), np.asarray(raw["ITTOT"], np.int64)
    t = np.asarray(raw, float)
    return t, np.arange(len(t), dtype=np.int64)


def read_var(grp, name, var, nt, sel=None):
    """One variable of one probe array as (ntime, npts), float32.

    The datasets are created as ``[npts, ntime]`` in Fortran order
    (``probes_mod.F90:838``), which the HDF5 Fortran layer reverses, so h5py
    sees (ntime, npts).  Checked against the length of PROBES/time rather than
    assumed.
    """
    d = grp[name][var]
    if d.shape[0] == nt:
        a = d[sel] if sel is not None else d[:]
    elif d.shape[1] == nt:
        a = (d[:, sel] if sel is not None else d[:, :]).T
    else:
        sys.exit(f"{name}/{var}: shape {d.shape} matches neither "
                 f"ntime = {nt} axis")
    return np.ascontiguousarray(a, dtype=np.float32)


def notfound(grp, name):
    """Probes MGLET could not locate in any grid are flagged igrid == 0."""
    if name in grp and "igrid" in grp[name]:
        return int((np.asarray(grp[name]["igrid"]) == 0).sum())
    return -1


def coords_of(grp, name, jsonpos):
    if name in jsonpos:
        return jsonpos[name]["pos"]
    c = np.asarray(grp[name]["coordinates"])
    return c if c.shape[1] == 3 else c.T


def xsec_names(grp):
    return sorted(n for n in grp if n.startswith("xsec"))


# ------------------------------------------------------------- SPOD kernel --
def hann(n):
    """Periodic Hann window (the one that satisfies the Parseval bookkeeping)."""
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)


def block_starts(nt, nfft, overlap):
    step = max(1, int(round(nfft * (1.0 - overlap))))
    return list(range(0, nt - nfft + 1, step))


def spod(q, dt, nfft, overlap=0.5, weight=None, nmodes=6):
    """Spectral POD of q(t, ndof).  Towne, Schmidt & Colonius (2018), JFM 847.

    Returns (freq, lam, phi, csd_scale):

      freq  (nfft,)               two-sided, in 1/[t].  Complex input (a fixed
                                  k_x) has a genuinely asymmetric spectrum:
                                  +f and -f are downstream- and upstream-
                                  travelling, so the full range is kept.
      lam   (nfft, nmodes)        modal energies, descending
      phi   (nfft, ndof, nmodes)  modes, orthonormal under `weight`
      csd_scale                   scalar C such that

                                     <q q^H> = C * sum_f sum_j lam phi phi^H

                                  recovers the time covariance, i.e. the
                                  Reynolds stresses.  C = W1^2/(nfft*W2) with
                                  W1 = sum(win), W2 = sum(win^2), which is the
                                  Parseval factor for the windowed, amplitude-
                                  normalised transform used below.
    """
    q = np.asarray(q)
    nt, ndof = q.shape
    if weight is None:
        weight = np.ones(ndof)
    weight = np.broadcast_to(np.asarray(weight, float), (ndof,))
    sw = np.sqrt(weight)

    starts = block_starts(nt, nfft, overlap)
    nblk = len(starts)
    if nblk < 2:
        sys.exit(f"only {nblk} Welch block(s) of {nfft} in {nt} samples - "
                 f"lower --nfft or run longer")

    win = hann(nfft)
    w1, w2 = win.sum(), (win ** 2).sum()
    csd_scale = w1 ** 2 / (nfft * w2)

    qm = q - q.mean(axis=0, keepdims=True)
    nm = min(nmodes, nblk)
    # Sign convention: modes are exp(i(k_x x - 2 pi f t)) with the x-transform
    # taken as exp(-i k_x x), so a structure advected downstream must come out
    # at POSITIVE f. numpy's forward transform is exp(-2 pi i f t), which puts
    # it at -f; negating the axis here fixes the sense once, for every caller.
    # |H(f)|^2 below is even in f, so nothing else needs to know.
    freq = -np.fft.fftfreq(nfft, d=dt)
    lam = np.zeros((nfft, nm))
    phi = np.zeros((nfft, ndof, nm), dtype=np.complex128)

    # (nfft, nblk, ndof) is the natural layout for the per-frequency SVD
    Q = np.empty((nfft, nblk, ndof), dtype=np.complex128)
    for b, s in enumerate(starts):
        Q[:, b, :] = np.fft.fft(qm[s:s + nfft] * win[:, None], axis=0) / w1

    for i in range(nfft):
        # Y = sqrt(W) Q / sqrt(nblk); SPOD modes are W^-1/2 * left singular
        # vectors, eigenvalues are the squared singular values.
        Y = (Q[i].T * sw[:, None]) / np.sqrt(nblk)
        U, S, _ = np.linalg.svd(Y, full_matrices=False)
        lam[i, :] = (S ** 2)[:nm]
        phi[i, :, :] = (U[:, :nm] / sw[:, None])
    return freq, lam, phi, csd_scale


def boxcar_gain(freq, dt_s, itsamp):
    """|H(f)|^2 of the running mean MGLET applies inside each sample window.

    The stored value is the mean of `itsamp` solver steps spaced dt_s/itsamp,
    so H(f) = sin(pi f dt_s) / (N sin(pi f dt_s / N)), N = itsamp.
    """
    if not itsamp or itsamp <= 1:
        return np.ones_like(freq)
    n = float(itsamp)
    num = np.sin(np.pi * freq * dt_s)
    den = n * np.sin(np.pi * freq * dt_s / n)
    g = np.where(np.abs(den) < 1e-12, 1.0, num / np.where(den == 0, 1.0, den))
    return np.maximum(g ** 2, 1e-6)


# ------------------------------------------------------------ x-transform ---
def xfft_planes(grp, names, nt, kx_list, sel, variables):
    """DFT over the equally spaced x-stations, one plane at a time.

    Holding all planes in memory is what makes this expensive (16 planes x
    ~1e3 points x 4 vars x ~3e3 samples); accumulating the transform plane by
    plane keeps only the requested wavenumbers.
    """
    nx = len(names)
    qk = None
    for ip, name in enumerate(names):
        block = np.stack([read_var(grp, name, v, nt, sel) for v in variables],
                         axis=-1)                      # (nt, npts, nvar)
        if qk is None:
            qk = np.zeros((len(kx_list),) + block.shape, dtype=np.complex64)
        for j, k in enumerate(kx_list):
            qk[j] += block * np.exp(-2j * np.pi * k * ip / nx).astype(np.complex64)
    return qk / nx


def kx_multiplicity(kx_list, nx):
    """Weight for folding the k and -k halves of a real signal's x-spectrum."""
    return np.array([1.0 if (k == 0 or (nx % 2 == 0 and k == nx // 2)) else 2.0
                     for k in kx_list])


# ------------------------------------------------------------------ checks --
def do_check(case, args):
    f, grp = open_probes(case)
    jsonpos = probe_arrays_json(case)
    meta = probe_meta(case)
    names = list(grp)
    t, ittot = read_time(grp)
    nt = len(t)

    print(f"case: {case}")
    print(f"\n-- probe arrays ({len(names) - 1} of them) --")
    xs = xsec_names(grp)
    surf = "surface" if "surface" in grp else None
    listed = list(dict.fromkeys(xs[:1] + xs[-1:] + ([surf] if surf else [])))
    for name in listed:
        pos = coords_of(grp, name, jsonpos)
        nf = notfound(grp, name)
        vs = [v for v in grp[name] if v not in ("coordinates", "igrid")]
        flag = "OK" if nf == 0 else (f"{nf} NOT FOUND" if nf > 0 else "?")
        print(f"  {name:10s} {len(pos):6d} pts x {len(vs)} vars {str(vs):24s} {flag}")
    if xs:
        xstations = np.array([coords_of(grp, n, jsonpos)[0, 0] for n in xs])
        dxp = np.diff(xstations)
        lx = xstations[-1] - xstations[0] + dxp.mean() if len(dxp) else 0.0
        print(f"  {len(xs)} cross-planes, dx = {dxp.mean():.3f}H "
              f"(spread {dxp.std():.1e}), Lx = {lx:.3f}H")
        kmax = len(xs) // 2
        print(f"     -> k_x = 0..{kmax}, lambda_x = {lx:g}H .. {lx / kmax:.2f}H")
    if surf:
        p = coords_of(grp, "surface", jsonpos)
        ux, uz = np.unique(p[:, 0]), np.unique(p[:, 2])
        print(f"  surface plane {len(ux)} x {len(uz)} at y = {p[0, 1]:.4f}"
              f"   dx = {np.diff(ux).mean():.3f}H  dz = {np.diff(uz).mean():.3f}H")

    print(f"\n-- time sampling --")
    tstat = meta.get("tstat")
    print(f"  {nt} samples, t = {t[0]:.3f} .. {t[-1]:.3f}   "
          f"(/time/tstat = {tstat})")
    if nt < 3:
        print("  too few samples to say anything")
        f.close()
        return 1
    d = np.diff(t)
    dts = float(np.median(d))
    dev = float(np.abs(d - dts).max() / dts)

    # ITTOT is an integer, so uniformity in *timesteps* is exact. TIME is
    # REAL(c_realk), i.e. float32 in the default build (precision_mod.F90:16),
    # so a correct run still shows ~eps32*t/dt_s of scatter in diff(TIME).
    # Judging uniformity on that scatter alone would fail every healthy run.
    dit = np.diff(ittot)
    steps_uniform = bool(np.all(dit == dit[0])) if len(dit) else False
    quant = 4.0 * np.finfo(np.float32).eps * max(abs(t[-1]), 1.0) / dts
    uniform = steps_uniform and dev < max(quant, 1e-6)
    print(f"  dt_s = {dts:.6g}  (scatter {100 * dev:.2g} %, float32 storage "
          f"floor {100 * quant:.2g} %)")
    print(f"  d(ITTOT) = {dit[0] if steps_uniform else 'VARIES'}"
          f"  -> sampling is {'UNIFORM' if uniform else 'NOT UNIFORM'}")
    if not uniform:
        if not steps_uniform:
            print("     the sampling interval in timesteps itself varies - "
                  "the run was restarted with a different /probes/itsamp.")
        else:
            good = np.abs(d - d[-1]) / d[-1] < max(quant, 1e-6)
            first = int(np.argmax(good)) if good.any() else len(d)
            print(f"     dt was still being adapted (targetcflmax with "
                  f"timeph < tstat, timeloop_mod.F90:313); the frozen-dt tail "
                  f"starts at sample {first} (t = {t[first]:.3f}).")
            print(f"     Use --tstart {t[first]:.3f}, or restart with "
                  f"tstat <= the restart time so dt is frozen from step 1.")
    itsamp = meta.get("itsamp")
    if itsamp:
        print(f"  /probes/itsamp = {itsamp}  -> solver dt = {dts / itsamp:.4g}, "
              f"each sample is a running mean over that window")

    print(f"\n-- resolvable frequencies --")
    fny = 0.5 / dts
    nfft = args.nfft
    starts = block_starts(nt, nfft, args.overlap)
    print(f"  f_Nyquist = {fny:.3f} u_tau/H   (period {1 / fny:.3f} H/u_tau)")
    print(f"  nfft = {nfft}, overlap {args.overlap:g} -> {len(starts)} Welch "
          f"blocks, df = {1 / (nfft * dts):.4g}")
    if itsamp and itsamp > 1:
        g = boxcar_gain(np.array([fny]), dts, itsamp)[0]
        print(f"  running-mean pre-filter attenuates Nyquist by "
              f"{-10 * np.log10(g):.1f} dB "
              f"(corrected for unless --no-boxcar-correction)")
    print(f"  advective frequency of a lambda_x = 5H structure at U_c ~ 18 "
          f"u_tau: f ~ {18 / 5:.1f}  -> {'resolved' if fny > 18 / 5 else 'ALIASED'}")

    print(f"\n-- verdict --")
    ok = True
    for what, cond, why in (
            ("cross-plane SPOD", len(xs) >= 4 and uniform and len(starts) >= 4,
             f"{len(xs)} planes, {len(starts)} blocks, uniform={uniform}"),
            ("free-surface SPOD", surf is not None and uniform and len(starts) >= 4,
             f"surface array {'present' if surf else 'MISSING'}"),
            ("all probes located", all(notfound(grp, n) == 0 for n in xs + ([surf] if surf else [])),
             "igrid == 0 count"),
    ):
        good = bool(cond)
        ok &= good
        print(f"  {'OK  ' if good else 'FAIL'}  {what:22s} ({why})")
    if len(starts) < 8:
        print(f"  note: {len(starts)} blocks gives a ~{100 / np.sqrt(len(starts)):.0f} % "
              f"error on each eigenvalue; 16+ blocks is the usual target.")
    f.close()
    return 0 if ok else 1


# -------------------------------------------------------- cross-plane SPOD --
def do_xsec(case, args, outdir):
    f, grp = open_probes(case)
    jsonpos = probe_arrays_json(case)
    meta = probe_meta(case)
    names = xsec_names(grp)
    if len(names) < 4:
        print("  [xsec] fewer than 4 cross-planes - skipping")
        f.close()
        return
    t, _ = read_time(grp)
    sel = np.where(t >= args.tstart)[0]
    if args.stride > 1:
        sel = sel[::args.stride]
    ts = t[sel]
    dts = float(np.median(np.diff(ts)))
    nt = len(t)

    # Velocities only. The weights below are a plain volume quadrature, which
    # is the kinetic-energy norm for u,v,w; folding P into the same state
    # vector would add pressure^2 to velocity^2 and make the "energy" that
    # SPOD ranks modes by dimensionally meaningless.
    variables = [v for v in ("U", "V", "W") if v in grp[names[0]]]
    pos = coords_of(grp, names[0], jsonpos)
    y, z = pos[:, 1], pos[:, 2]
    npts = len(pos)

    nx = len(names)
    kx_list = [k for k in args.kx if k <= nx // 2]
    if len(kx_list) < len(args.kx):
        print(f"  (dropped k_x > {nx//2}: {nx} planes cannot resolve them)")
    xstations = np.array([coords_of(grp, n, jsonpos)[0, 0] for n in names])
    lxd = float(nx * np.diff(xstations).mean())
    print(f"\n== cross-plane SPOD ==")
    print(f"  {nx} planes, {npts} pts, vars {variables}, "
          f"{len(sel)} samples from t = {ts[0]:.2f}, dt_s = {dts:.5g}")
    qk = xfft_planes(grp, names, nt, kx_list, sel, variables)
    f.close()

    # Uniform probe spacing -> the quadrature weight is the same constant for
    # every point and every variable, so it only rescales lam.
    dy = np.diff(np.unique(y)).min()
    dz = np.diff(np.unique(z)).min()
    weight = np.full(npts * len(variables), dy * dz)

    mult = kx_multiplicity(kx_list, nx)
    lams, phis = [], []
    for j, k in enumerate(kx_list):
        q = qk[j].reshape(len(sel), -1)
        freq, lam, phi, C = spod(q, dts, args.nfft, args.overlap, weight,
                                 args.nmodes)
        if args.boxcar:
            lam = lam / boxcar_gain(freq, dts, meta.get("itsamp"))[:, None]
        lams.append(lam * mult[j])
        phis.append(phi)
        e = lam.sum(axis=0)
        tot = e.sum()
        i = int(np.argmax(lam[:, 0]))
        scale = (f"lambda_x = {lxd / k:.2f}H, c = "
                 f"{abs(freq[i]) * lxd / k:5.1f} u_tau" if k
                 else "x-mean (k_x = 0), no phase speed")
        print(f"  k_x = {k}: peak at f = {freq[i]:+.3f}  ({scale})"
              f"   lam1/sum = {e[0] / tot:.2f}")

    lam = np.stack(lams)                       # (nkx, nfft, nmodes)
    np.savez_compressed(
        os.path.join(outdir, "spod_xsec.npz"),
        freq=freq, kx=np.asarray(kx_list), lam=lam, dts=dts, csd_scale=C,
        y=y, z=z, variables=np.array(variables, dtype=object),
        phi_peak=phis[int(np.argmax([l[:, 0].max() for l in lams]))])
    print(f"  wrote {outdir}/spod_xsec.npz")

    _plot_xsec(outdir, freq, kx_list, lam, phis, y, z, variables)


def _plot_xsec(outdir, freq, kx_list, lam, phis, y, z, variables):
    plt = _mpl()
    if plt is None:
        return
    from matplotlib.colors import LogNorm
    pos = freq > 0
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for j, k in enumerate(kx_list):
        ax[0].loglog(freq[pos], lam[j, pos, 0], lw=1.3, label=f"$k_x = {k}$")
    ax[0].set_xlabel(r"$f\,H/u_\tau$")
    ax[0].set_ylabel(r"$\lambda_1$")
    ax[0].set_title("leading SPOD eigenvalue")
    ax[0].legend(fontsize=8)

    # (k_x, f) energy map: a straight ridge f = U_c k_x/L_x is the advection
    # of coherent structures, and its slope is the convection velocity.
    E = lam[:, :, 0]
    floor = max(E[E > 0].min(), E.max() * 1e-8) if (E > 0).any() else 1e-30
    order = np.argsort(freq)
    m = ax[1].pcolormesh(freq[order], np.asarray(kx_list),
                         np.maximum(E[:, order], floor),
                         shading="nearest", norm=LogNorm())
    fig.colorbar(m, ax=ax[1], label=r"$\lambda_1$")
    ax[1].set_xlabel(r"$f\,H/u_\tau$")
    ax[1].set_ylabel(r"$k_x$")
    ax[1].set_title(r"$\lambda_1(k_x, f)$  (ridge slope = $U_c$)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "spod_xsec_spectra.png"), dpi=140)
    plt.close(fig)

    # Leading mode of the most energetic (k_x, f) in the cross-plane
    j = int(np.argmax([l[:, 0].max() for l in lam]))
    i = int(np.argmax(lam[j, :, 0]))
    phi = phis[j][i, :, 0].reshape(len(y), len(variables))
    fig, axs = plt.subplots(1, len(variables), figsize=(4 * len(variables), 3.4))
    for c, (a, v) in enumerate(zip(np.atleast_1d(axs), variables)):
        s = a.scatter(z, y, c=phi[:, c].real, s=6, cmap="RdBu_r")
        a.axvline(Z_JUNCTION, color="k", lw=0.8, ls="--")
        a.set_xlabel("$z/H$")
        a.set_ylabel("$y/H$")
        a.set_aspect("equal")
        a.set_title(rf"$\Re\,\phi_{{{v}}}$")
        fig.colorbar(s, ax=a)
    fig.suptitle(rf"leading SPOD mode, $k_x = {kx_list[j]}$, "
                 rf"$f = {freq[i]:+.3f}$", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "spod_xsec_mode.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outdir}/spod_xsec_spectra.png, {outdir}/spod_xsec_mode.png")


# ------------------------------------------------------- free-surface SPOD --
def surface_grid(pos):
    """Map the flat surface point list onto its (nx, nz) lattice."""
    ux, uz = np.unique(pos[:, 0]), np.unique(pos[:, 2])
    ix = np.searchsorted(ux, pos[:, 0])
    iz = np.searchsorted(uz, pos[:, 2])
    idx = np.full((len(ux), len(uz)), -1, dtype=np.int64)
    idx[ix, iz] = np.arange(len(pos))
    if (idx < 0).any():
        sys.exit("surface probe points do not form a full x-z lattice")
    return ux, uz, idx


def omega_y(u, w, x, z):
    """du/dz - dw/dx on the surface lattice; x periodic, z bounded."""
    kx = 2j * np.pi * np.fft.fftfreq(len(x), d=x[1] - x[0])
    dwdx = np.real(np.fft.ifft(np.fft.fft(w, axis=-2) * kx[:, None], axis=-2))
    dudz = np.gradient(u, z, axis=-1)
    return dudz - dwdx


def do_surface(case, args, outdir):
    f, grp = open_probes(case)
    if "surface" not in grp:
        print("  [surface] no 'surface' probe array - regenerate the case "
              "without --no-surface")
        f.close()
        return
    jsonpos = probe_arrays_json(case)
    meta = probe_meta(case)
    t, _ = read_time(grp)
    nt = len(t)
    sel = np.where(t >= args.tstart)[0]
    if args.stride > 1:
        sel = sel[::args.stride]
    ts = t[sel]
    dts = float(np.median(np.diff(ts)))

    pos = coords_of(grp, "surface", jsonpos)
    ux, uz, idx = surface_grid(pos)
    variables = [v for v in ("U", "V", "W") if v in grp["surface"]]
    q = np.stack([read_var(grp, "surface", v, nt, sel)[:, idx]
                  for v in variables], axis=-1)        # (nt, nx, nz, nvar)
    f.close()

    nx, nz = len(ux), len(uz)
    lx = nx * (ux[1] - ux[0])
    print(f"\n== free-surface SPOD ==")
    print(f"  plane {nx} x {nz} at y = {pos[0, 1]:.4f}, vars {variables}, "
          f"{len(sel)} samples, dt_s = {dts:.5g}")

    iu, iw = variables.index("U"), variables.index("W")
    qm = q.mean(axis=0)
    up = q[..., iu] - qm[None, ..., iu]
    wp = q[..., iw] - qm[None, ..., iw]

    # Direct reference quantities, for the closure check below
    uw_direct = (up * wp).mean(axis=(0, 1))              # <u'w'>(z)
    wy = omega_y(q[..., iu], q[..., iw], ux, uz)

    # SPOD per streamwise wavenumber, dofs = (z, variable)
    kx_list = [k for k in args.kx if k <= nx // 2]
    mult = kx_multiplicity(kx_list, nx)
    dz = uz[1] - uz[0]
    weight = np.full(nz * len(variables), dz)
    qhat = np.fft.fft(q, axis=1) / nx

    lams, phis, uw_modal = [], [], []
    for j, k in enumerate(kx_list):
        qk = qhat[:, k].reshape(len(sel), -1)
        freq, lam, phi, C = spod(qk, dts, args.nfft, args.overlap, weight,
                                 args.nmodes)
        if args.boxcar:
            lam = lam / boxcar_gain(freq, dts, meta.get("itsamp"))[:, None]
        lams.append(lam * mult[j])
        phis.append(phi)
        # <u'w'>(z) carried by each mode:  C * sum_f lam_j * Re(phi_u phi_w*)
        p = phi.reshape(len(freq), nz, len(variables), -1)
        contrib = C * mult[j] * np.einsum(
            "fj,fzj->zj", lam, (p[:, :, iu, :] * np.conj(p[:, :, iw, :])).real)
        uw_modal.append(contrib)
        i = int(np.argmax(lam[:, 0]))
        scale = (f"lambda_x = {lx / k:.2f}H, c = {abs(freq[i]) * lx / k:5.1f}"
                 f" u_tau" if k else "x-mean (k_x = 0), no phase speed")
        print(f"  k_x = {k}: peak f = {freq[i]:+.3f}  ({scale})"
              f"  lam1 share = {lam[:, 0].sum() / lam.sum():.2f}")

    uw_modal = np.stack(uw_modal)                        # (nkx, nz, nmodes)
    uw_spod = uw_modal.sum(axis=(0, 2))
    frac = np.abs(uw_spod).sum() / max(np.abs(uw_direct).sum(), 1e-30)
    lead = np.abs(uw_modal[:, :, 0].sum(axis=0)).sum() / \
        max(np.abs(uw_spod).sum(), 1e-30)
    print(f"  closure: SPOD-summed <u'w'> recovers {100 * frac:.0f} % of the "
          f"directly measured profile")
    print(f"           (shortfall = the k_x and f bins not retained)")
    print(f"  the leading mode alone carries {100 * lead:.0f} % of that")
    ijun = int(np.argmin(np.abs(uz - Z_JUNCTION)))
    print(f"  at the junction z = {uz[ijun]:.2f}: <u'w'> = {uw_direct[ijun]:+.4f} "
          f"u_tau^2, leading-mode part {uw_modal[:, ijun, 0].sum():+.4f}")

    np.savez_compressed(
        os.path.join(outdir, "spod_surface.npz"),
        freq=freq, kx=np.asarray(kx_list), lam=np.stack(lams), x=ux, z=uz,
        dts=dts, uw_direct=uw_direct, uw_modal=uw_modal,
        omega_y_last=wy[-1], u_mean=qm[..., iu],
        variables=np.array(variables, dtype=object))
    print(f"  wrote {outdir}/spod_surface.npz")

    _plot_surface(outdir, ux, uz, wy, qm[..., iu], freq, kx_list,
                  np.stack(lams), phis, variables, uw_direct, uw_modal, lx)


def _plot_surface(outdir, x, z, wy, umean, freq, kx_list, lam, phis,
                  variables, uw_direct, uw_modal, lx):
    plt = _mpl()
    if plt is None:
        return
    # 1. top view: instantaneous surface-normal vorticity
    fig, axs = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    lim = float(np.percentile(np.abs(wy[-1]), 99))
    m = axs[0].pcolormesh(x, z, wy[-1].T, cmap="RdBu_r", vmin=-lim, vmax=lim,
                          shading="auto")
    fig.colorbar(m, ax=axs[0], label=r"$\omega_y H/u_\tau$")
    axs[0].axhline(Z_JUNCTION, color="k", lw=1.0)
    axs[0].set_ylabel("$z/H$")
    axs[0].set_title("free surface, instantaneous surface-normal vorticity "
                     "(top view; line = junction)")
    m = axs[1].pcolormesh(x, z, umean.T, cmap="viridis", shading="auto")
    fig.colorbar(m, ax=axs[1], label=r"$\overline{u}/u_\tau$")
    axs[1].axhline(Z_JUNCTION, color="w", lw=1.0)
    axs[1].set_xlabel("$x/H$")
    axs[1].set_ylabel("$z/H$")
    axs[1].set_title("time-mean surface velocity")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "surface_topview.png"), dpi=140)
    plt.close(fig)

    # 2. spectra + leading mode reconstructed as a top view
    j = int(np.argmax([l[:, 0].max() for l in lam]))
    i = int(np.argmax(lam[j, :, 0]))
    k = kx_list[j]
    nz = len(z)
    phi = phis[j][i, :, 0].reshape(nz, len(variables))
    iu = variables.index("U")

    fig, axs = plt.subplots(1, 3, figsize=(14, 3.8))
    pos = freq > 0
    for jj, kk in enumerate(kx_list):
        axs[0].loglog(freq[pos], lam[jj, pos, 0], lw=1.3, label=f"$k_x={kk}$")
    axs[0].set_xlabel(r"$f\,H/u_\tau$")
    axs[0].set_ylabel(r"$\lambda_1$")
    axs[0].set_title("surface SPOD spectrum")
    axs[0].legend(fontsize=8)

    mode2d = np.real(phi[:, iu][None, :] *
                     np.exp(2j * np.pi * k * x[:, None] / lx))
    lim = np.abs(mode2d).max()
    m = axs[1].pcolormesh(x, z, mode2d.T, cmap="RdBu_r", vmin=-lim, vmax=lim,
                          shading="auto")
    fig.colorbar(m, ax=axs[1])
    axs[1].axhline(Z_JUNCTION, color="k", lw=1.0)
    axs[1].set_xlabel("$x/H$")
    axs[1].set_ylabel("$z/H$")
    axs[1].set_title(rf"leading mode $u$, $k_x={k}$, $f={freq[i]:+.3f}$")

    axs[2].plot(uw_direct, z, "k", lw=1.5, label="measured")
    axs[2].plot(uw_modal.sum(axis=(0, 2)), z, "C0--", lw=1.2,
                label="SPOD, all modes")
    axs[2].plot(uw_modal[:, :, 0].sum(axis=0), z, "C3", lw=1.2,
                label="leading mode")
    axs[2].axhline(Z_JUNCTION, color="k", lw=0.8, ls=":")
    axs[2].set_xlabel(r"$\langle u'w'\rangle/u_\tau^2$")
    axs[2].set_ylabel("$z/H$")
    axs[2].set_title("lateral momentum flux at the surface")
    axs[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "surface_spod.png"), dpi=140)
    plt.close(fig)
    print(f"  wrote {outdir}/surface_topview.png, {outdir}/surface_spod.png")


def _mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("  (matplotlib not available - skipping plots)")
        return None


# ---------------------------------------------------------------- selftest --
def selftest():
    """Verify the kernel on a signal whose SPOD answer is known analytically.

    Two travelling waves with prescribed amplitudes plus noise: the leading
    eigenvalue must land in the right frequency bin, the modes must be
    orthonormal, and C*sum(lam*phi*phi^H) must reproduce the covariance.
    """
    rng = np.random.default_rng(7)
    nt, nz = 6000, 40
    dt = 0.05
    z = np.linspace(0, 1, nz)
    t = np.arange(nt) * dt
    f1, f2 = 0.30, 0.75
    s1 = np.exp(-((z - 0.35) / 0.10) ** 2)
    s2 = np.exp(-((z - 0.70) / 0.15) ** 2)
    q = (2.0 * np.cos(2 * np.pi * f1 * t)[:, None] * s1[None, :]
         + 0.7 * np.cos(2 * np.pi * f2 * t)[:, None] * s2[None, :]
         + 0.05 * rng.standard_normal((nt, nz)))

    w = np.full(nz, z[1] - z[0])
    freq, lam, phi, C = spod(q, dt, nfft=512, overlap=0.5, weight=w, nmodes=4)

    ok = True
    for fex in (f1, f2):
        i = int(np.argmin(np.abs(freq - fex)))
        peak = freq[np.argsort(lam[:, 0])[::-1][:6]]
        hit = np.any(np.abs(np.abs(peak) - fex) < 1.5 / (512 * dt))
        print(f"  peak at f = {fex}: {'found' if hit else 'MISSED'}")
        ok &= hit

    i = int(np.argmax(lam[:, 0]))
    gram = phi[i].conj().T @ (w[:, None] * phi[i])
    orth = np.abs(gram - np.eye(gram.shape[0])).max()
    print(f"  mode orthonormality error: {orth:.2e}")
    ok &= orth < 1e-9

    # covariance closure: C * sum_f sum_j lam phi phi^H == <q'q'^H>
    qm = q - q.mean(axis=0)
    cov_direct = (qm.T @ qm) / nt
    cov_spod = C * np.einsum("fj,fzj,fyj->zy", lam, phi, phi.conj()).real
    err = np.abs(cov_spod - cov_direct).max() / np.abs(cov_direct).max()
    print(f"  covariance closure error: {100 * err:.1f} % "
          f"(C = {C:.4f}; only {lam.shape[1]} modes kept)")
    ok &= err < 0.05

    print(f"\n  selftest {'PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


# -------------------------------------------------------------------- main --
def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("case", nargs="?", help="case directory holding probes.h5")
    p.add_argument("--check", action="store_true",
                   help="only report whether SPOD is obtainable from this run")
    p.add_argument("--xsec", action="store_true", help="cross-plane SPOD only")
    p.add_argument("--surface", action="store_true",
                   help="free-surface SPOD and top view only")
    p.add_argument("--selftest", action="store_true",
                   help="verify the SPOD kernel on synthetic data")
    p.add_argument("--nfft", type=int, default=256,
                   help="Welch block length in samples (default 256)")
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--nmodes", type=int, default=6)
    p.add_argument("--kx", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                   help="streamwise wavenumbers to analyse (default 0..4)")
    p.add_argument("--tstart", type=float, default=0.0,
                   help="discard samples before this time (dt must be frozen)")
    p.add_argument("--stride", type=int, default=1,
                   help="use every n-th sample (halves memory, halves f_Nyq)")
    p.add_argument("--no-boxcar-correction", dest="boxcar", action="store_false",
                   help="do not undo the running-mean pre-filter")
    p.add_argument("--outdir", default=None)
    args = p.parse_args()

    if args.selftest:
        return selftest()
    if not args.case:
        p.error("a case directory is required (or --selftest)")
    outdir = args.outdir or os.path.join(args.case, "spod")
    os.makedirs(outdir, exist_ok=True)

    if args.check:
        return do_check(args.case, args)
    rc = do_check(args.case, args)
    if not (args.surface and not args.xsec):
        do_xsec(args.case, args, outdir)
    if not (args.xsec and not args.surface):
        do_surface(args.case, args, outdir)
    return rc


if __name__ == "__main__":
    sys.exit(main())
