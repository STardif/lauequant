"""Diffuse backgrounds on the detector.  Implemented: X-ray fluorescence of the sample.

For a thick homogeneous sample, the rate of line l emitted per steradian in a
direction leaving the surface with direction cosine gamma_out is

    I_l = 1/(4 pi) int Phi_E(E) w_Z sigma_l(E) / (mu(E) + mu(E_l) gamma_in / gamma_out) dE

(mass coefficients, cm2/g; sigma_l from xraylib CS_FluorLine_Kissel_Cascade,
which includes the photo-ionisation, fluorescence yield and radiative rate).
For Ge, Ka (9.89 keV) and Kb (10.98 keV) are above the usual 4 keV threshold
and every incident photon above 11.1 keV can produce them: this is the main
diffuse background of a Ge Laue pattern at low threshold.

TODO (see PLAN.md): Compton scattering, thermal diffuse scattering, air scattering.
"""
from __future__ import annotations

import numpy as np
import xraylib as xrl

from .materials import AirTable, mu_rho

try:
    from scipy.integrate import trapezoid
except ImportError:  # pragma: no cover
    from numpy import trapz as trapezoid

LINES = {"Ka1": xrl.KL3_LINE, "Ka2": xrl.KL2_LINE, "Kb1": xrl.KM3_LINE, "Kb3": xrl.KM2_LINE,
         "La1": xrl.L3M5_LINE, "Lb1": xrl.L2M4_LINE}


def fluorescence_lines(crystal, spectrum, min_energy_kev=1.0):
    """List of (label, E_line, I_line(gamma_out) callable) for the sample."""
    e = spectrum.energies_kev
    phi = spectrum.flux_per_ev * 1e3          # photons/s/keV
    frac = crystal.mass_fractions
    mu_e = mu_rho(frac, e)
    out = []
    for z, w in frac.items():
        for name, ln in LINES.items():
            try:
                el = xrl.LineEnergy(z, ln)
            except ValueError:
                continue
            if el < min_energy_kev:
                continue
            def _cs(x):
                try:
                    return xrl.CS_FluorLine_Kissel_Cascade(z, ln, float(x))
                except ValueError:          # below the edge
                    return 0.0
            sig = np.array([_cs(x) for x in e])
            mu_l = mu_rho(frac, [el])[0]
            label = f"{xrl.AtomicNumberToSymbol(z)} {name}"
            out.append((label, el, (e, phi, w * sig, mu_e, mu_l)))
    return out


def line_intensity_per_sr(data, gamma_in, gamma_out):
    """Photons/s/sr of one line towards directions with the given gamma_out (array)."""
    e, phi, wsig, mu_e, mu_l = data
    g = np.atleast_1d(np.asarray(gamma_out, float))
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = mu_e[None, :] + mu_l * gamma_in / g[:, None]
        val = trapezoid(phi[None, :] * wsig[None, :] / denom, e, axis=1) / (4 * np.pi)
    return np.where(g > 0, val, 0.0)


def fluorescence_background(sim, step: int = 16, min_energy_kev=None, return_lines=False):
    """Expected fluorescence counts/s per pixel on the detector of a LaueSimulator.

    Returned as a numpy array (array indices).  Computed on a grid with ``step`` pixels and bilinearly interpolated (the
    background is smooth).  Includes the pixel solid angle, air absorption,
    and the detector flat-field response (QE x charge-sharing multiplicity).
    """
    from scipy.interpolate import RegularGridInterpolator

    geo, smp, det = sim.detector_geometry, sim.sample, sim.detector
    ny, nx = geo.shape
    xs = np.r_[np.arange(0, nx, step), nx - 1].astype(float)
    ys = np.r_[np.arange(0, ny, step), ny - 1].astype(float)
    X, Y = np.meshgrid(xs, ys)
    Xc, Yc = geo.from_index(X.ravel(), Y.ravel())          # array indices -> calibration frame
    uf, r = geo.directions_from_pixels(Xc, Yc)
    omega = geo.pixel_solid_angle(Xc, Yc)
    g_in = smp.gamma_in()
    g_out = smp.gamma_out(uf)
    cosi = geo.incidence_cos(uf)
    air = AirTable()
    thr = det.threshold_kev if min_energy_kev is None else min_energy_kev
    total = np.zeros(X.size)
    per_line = {}
    for label, el, data in fluorescence_lines(sim.crystal, sim.spectrum, min_energy_kev=1.0):
        I = line_intensity_per_sr(data, g_in, g_out)
        resp = det.flat_response([el], n=20000)[0] if el > 0.5 * thr else 0.0
        # obliquity: flat_response is at normal incidence; QE of CdTe ~1 for these lines anyway
        rate = I * omega * air.transmission(el, r) * resp
        per_line[label] = (el, rate.reshape(X.shape))
        total += rate
    interp = RegularGridInterpolator((ys, xs), total.reshape(X.shape))
    YY, XX = np.mgrid[0:ny, 0:nx]
    img = interp(np.c_[YY.ravel(), XX.ravel()]).reshape(ny, nx)
    if return_lines:
        return img, per_line, (xs, ys)
    return img
