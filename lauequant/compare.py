"""Comparison of predictions with measured BLISS/EIGER2 Laue images (uses pyXRT's readers)."""
from __future__ import annotations

import ast
from typing import Optional

import numpy as np
import pandas as pd

from .crystal import Crystal
from .detector import PhotonCountingDetector
from .geometry import DetectorGeometry
from .simulate import LaueSimulator
from .spectrum import FocusedBeam, IncidentSpectrum, import_pyxrt_beamline


def read_lauetools_det(path):
    """(calib[5], pixelsize, shape, UB) from a LaueTools .det file (UB may be None)."""
    lines = open(path).read().splitlines()
    vals = [float(v) for v in lines[0].replace(",", " ").split()]
    UB = None
    for i, l in enumerate(lines):
        if l.startswith("Orientation Matrix"):
            UB = np.array(ast.literal_eval(lines[i + 1]))
    shape = (int(vals[6]), int(vals[7])) if len(vals) >= 8 else (2162, 2068)
    return vals[:5], vals[5], shape, UB


def read_scan(master, scan, pyxrt_path: Optional[str] = None, frame: int = 0):
    """Image (float), exposure (s) and EIGER2 / machine settings of a BLISS scan, via pyXRT."""
    import h5py
    import_pyxrt_beamline(pyxrt_path)                      # puts pyXRT on sys.path
    from bliss_hdf5_reader import BlissScan
    sc = BlissScan(master, scan)
    img = sc.read_frame(frame).astype(float)
    with h5py.File(master, "r") as f:
        e = f[scan]
        cs = e["instrument/eiger4m/camera_settings"]
        info = dict(threshold_kev=float(cs["threshold_energy"][()]) / 1e3,
                    threshold2_kev=float(cs["threshold_energy2"][()]) / 1e3,
                    exposure_s=float(np.atleast_1d(e["measurement/integration_time"][()])[0]),
                    mon=float(np.atleast_1d(e["measurement/mon"][()])[0]),
                    current_ma=float(e["instrument/machine/current"][()]),
                    title=e["title"][()].decode() if isinstance(e["title"][()], bytes) else str(e["title"][()]))
    if not img.any():
        raise IOError("empty frame: the image file was not found next to the master file "
                      "(expected <master dir>/scanNNNN/eiger4m_0000.h5)")
    return img, info


def roi_sums(img, mask, x, y, half: int = 5, r_in: int = 8, r_out: int = 12, pixel_origin: int = 1):
    """Box sum (2 half + 1)^2 around (x, y) minus the median of a square annulus [r_in, r_out].

    x, y are calibration-frame coordinates (LaueTools: first pixel = (1, 1), so the
    array element is img[y - 1, x - 1] for ``pixel_origin=1``).

    Returns (n, 3): net counts, background per pixel, max pixel.  NaN when the
    box touches a module gap / the frame edge.
    """
    ny, nx = img.shape
    yy, xx = np.mgrid[-r_out:r_out + 1, -r_out:r_out + 1]
    box = (np.abs(xx) <= half) & (np.abs(yy) <= half)
    annulus = np.maximum(np.abs(xx), np.abs(yy)) >= r_in
    out = np.full((len(x), 3), np.nan)
    cols = np.round(np.asarray(x, float) - pixel_origin).astype(int)
    rows = np.round(np.asarray(y, float) - pixel_origin).astype(int)
    for i, (xc, yc) in enumerate(zip(cols, rows)):
        y0, y1, x0, x1 = yc - r_out, yc + r_out + 1, xc - r_out, xc + r_out + 1
        if y0 < 0 or x0 < 0 or y1 > ny or x1 > nx:
            continue
        sub, m = img[y0:y1, x0:x1], mask[y0:y1, x0:x1]
        ann = annulus & m
        if (~m[box]).any() or ann.sum() < 50:
            continue
        bkg = np.median(sub[ann])
        out[i] = (sub[box].sum() - bkg * box.sum(), bkg, sub[box].max())
    return out


def build_simulator(master, scan, detfile, kb_angle_mrad: Optional[float] = None, model="dynamical",
                    threshold_kev: Optional[float] = None, pyxrt_path=None, crystal=None, **det_kw):
    """LaueSimulator with the slits, mirror angles and ring current of the scan (pyXRT bm32_from_h5)."""
    xb = import_pyxrt_beamline(pyxrt_path)
    kw = {} if kb_angle_mrad is None else dict(kbv_angle_mrad=kb_angle_mrad, kbh_angle_mrad=kb_angle_mrad)
    bl = xb.bm32_from_h5(master, scan, **kw)
    spec = IncidentSpectrum.from_pyxrt(beamline=bl)
    calib, pix, shape, UB = read_lauetools_det(detfile)
    beam = FocusedBeam.bm32(bl.settings.get("vg3", 0.2), bl.settings.get("hg3", 0.2))
    if threshold_kev is None:
        import h5py
        with h5py.File(master, "r") as f:
            threshold_kev = float(f[scan]["instrument/eiger4m/camera_settings/threshold_energy"][()]) / 1e3
    det = PhotonCountingDetector(threshold_kev=threshold_kev, **det_kw)
    return LaueSimulator(crystal or Crystal.germanium(), UB, spec, DetectorGeometry(calib, pix, shape),
                         beam=beam, detector=det, model=model)


def compare_spots(sim, img, exposure_s, half: int = 5, mask=None, rng=0, **ray_kw):
    """Spot table with predicted and measured (ROI-integrated) counts for the exposure."""
    mask = img > 0 if mask is None else mask
    sp = sim.spots()
    meas = roi_sums(img, mask, sp.X.values, sp.Y.values, half=half, pixel_origin=sim.detector_geometry.pixel_origin)
    sp["meas_counts"], sp["meas_bkg"], sp["meas_max"] = meas.T
    pc = [sim.spot_counts((r.h0, r.k0, r.l0), roi_half=half, rng=rng, **ray_kw) for r in sp.itertuples()]
    sp["pred_counts"] = [p["counts_s"] * exposure_s for p in pc]
    sp["pred_max"] = [p["roi"].max() * exposure_s for p in pc]
    sp["pred_photons"] = [p["photons_s"] * exposure_s for p in pc]
    return sp


def fit_scale(sp, min_meas=2000.0, min_pred=3000.0):
    """Median log ratio (global scale k) and rms (dex) over well-measured spots."""
    sel = (sp.meas_counts > min_meas) & (sp.pred_counts > min_pred)
    lr = np.log10(sp.meas_counts[sel] / sp.pred_counts[sel])
    return 10 ** float(np.median(lr)), float(np.std(lr)), int(sel.sum())


def block_median(img, mask, block: int = 32, min_valid: int = 200):
    ny, nx = img.shape
    out = np.full((ny // block, nx // block), np.nan)
    for i in range(ny // block):
        for j in range(nx // block):
            b = img[i * block:(i + 1) * block, j * block:(j + 1) * block]
            m = mask[i * block:(i + 1) * block, j * block:(j + 1) * block]
            if m.sum() >= min_valid:
                out[i, j] = np.median(b[m])
    return out
