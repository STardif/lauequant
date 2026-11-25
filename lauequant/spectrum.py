"""Incident white-beam spectrum, polarisation and focusing (convergence) of the beam.

The absolute spectral flux is taken from the pyXRT beamline model
(``xrf_beamline.BM32_WHITE_BEAM``, Tardif 2026), which integrates the
bending-magnet angular density over the accepted vertical window and multiplies
by the mirror reflectivities, Be windows and air path.  This module only wraps
it, adds the degree of linear polarisation of the accepted fan, and the angular
convergence produced by the KB mirrors.

Units: energies in keV, flux density in photons/s/eV.
"""
from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
from scipy.special import kv

from .materials import Medium

try:
    from scipy.integrate import trapezoid
except ImportError:  # pragma: no cover
    from numpy import trapz as trapezoid


# ----------------------------------------------------------------------------
# pyXRT import
# ----------------------------------------------------------------------------

def import_pyxrt_beamline(pyxrt_path: Optional[str] = None):
    """Import pyXRT's ``xrf_beamline`` module.

    pyXRT is a folder of scripts, not an installed package: the folder is taken
    from ``pyxrt_path``, else from the ``PYXRT_PATH`` environment variable.
    """
    path = pyxrt_path or os.environ.get("PYXRT_PATH")
    if path is not None and str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return importlib.import_module("xrf_beamline")


# ----------------------------------------------------------------------------
# Polarisation of the bending-magnet fan
# ----------------------------------------------------------------------------

def bending_magnet_sigma_fraction(energies_kev, critical_energy_kev: float, gamma: float,
                                  vertical_acceptance_urad: Optional[float],
                                  vertical_offset_urad: float = 0.0, n_angles: int = 41):
    """Fraction of the accepted intensity polarised in the orbit plane (sigma).

    Uses the sigma (K_{2/3}) and pi (X^2/(1+X^2) K_{1/3}) terms of the
    Schwinger angular density, integrated over the accepted vertical window.
    On BM32 (a few urad accepted, 1/gamma = 85 urad) this is > 0.999.
    """
    e = np.atleast_1d(np.asarray(energies_kev, float))
    if vertical_acceptance_urad is None or not np.isfinite(vertical_acceptance_urad):
        psi = np.linspace(-600, 600, 601)
    else:
        h = 0.5 * vertical_acceptance_urad
        psi = np.linspace(vertical_offset_urad - h, vertical_offset_urad + h, n_angles)
    y = (e / critical_energy_kev)[:, None]
    x2 = (gamma * psi * 1e-6)[None, :] ** 2
    xi = 0.5 * y * (1 + x2) ** 1.5
    with np.errstate(all="ignore"):
        s = np.nan_to_num((1 + x2) ** 2 * kv(2 / 3, xi) ** 2)
        p = np.nan_to_num((1 + x2) ** 2 * x2 / (1 + x2) * kv(1 / 3, xi) ** 2)
    S, P = trapezoid(s, psi, axis=1), trapezoid(p, psi, axis=1)
    with np.errstate(all="ignore"):
        return np.where(S + P > 0, S / (S + P), 1.0)


# ----------------------------------------------------------------------------
# Spectrum container
# ----------------------------------------------------------------------------

@dataclass
class IncidentSpectrum:
    """Tabulated incident spectrum at the sample.

    Parameters
    ----------
    energies_kev : (n,) array, increasing
    flux_per_ev : (n,) array
        Photons/s/eV integrated over the whole (focused) beam.
    sigma_fraction : float or (n,) array
        Fraction of intensity linearly polarised along the horizontal axis
        (LT2 x).  The remainder is treated as polarised vertically (z), i.e. an
        incoherent mixture (circular components are irrelevant for intensities).
    meta : dict
        Provenance (beamline settings, ring current, ...).
    """

    energies_kev: np.ndarray
    flux_per_ev: np.ndarray
    sigma_fraction: object = 1.0
    meta: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self):
        self.energies_kev = np.asarray(self.energies_kev, float)
        self.flux_per_ev = np.asarray(self.flux_per_ev, float)
        if np.ndim(self.sigma_fraction) == 0:
            self.sigma_fraction = np.full_like(self.energies_kev, float(self.sigma_fraction))
        else:
            self.sigma_fraction = np.asarray(self.sigma_fraction, float)

    # -- construction ---------------------------------------------------------
    @classmethod
    def from_pyxrt(cls, energies_kev=None, pyxrt_path: Optional[str] = None,
                   beamline=None, **bm32_kwargs) -> "IncidentSpectrum":
        """Spectrum at the sample from pyXRT's BM32 beamline model.

        ``bm32_kwargs`` go to ``xrf_beamline.BM32_WHITE_BEAM`` (slit gaps
        ``vg1, hg1, vg3, hg3`` in mm, ``current_a``, ``kbv_angle_mrad``, ...).
        A prebuilt ``beamline`` (e.g. from ``bm32_from_h5``) may be given instead.
        """
        xb = import_pyxrt_beamline(pyxrt_path)
        if beamline is None:
            beamline = xb.BM32_WHITE_BEAM(**bm32_kwargs)
        e = np.arange(3.0, 45.0, 0.02) if energies_kev is None else np.asarray(energies_kev, float)
        flux = beamline.photon_flux(e)
        src = beamline.source
        sig = bending_magnet_sigma_fraction(e, src.critical_energy_kev, src.gamma,
                                            beamline.effective_vertical_acceptance(),
                                            beamline.vertical_offset_urad)
        meta = dict(source="pyXRT BM32_WHITE_BEAM", settings=dict(beamline.settings))
        meta.update(beamline.describe())
        return cls(e, flux, sig, meta)

    @classmethod
    def from_csv(cls, path) -> "IncidentSpectrum":
        d = np.loadtxt(path, delimiter=",", comments="#")
        sig = d[:, 2] if d.shape[1] > 2 else 1.0
        return cls(d[:, 0], d[:, 1], sig, dict(source=str(path)))

    def to_csv(self, path, header: str = ""):
        np.savetxt(path, np.c_[self.energies_kev, self.flux_per_ev, self.sigma_fraction],
                   delimiter=",", header=(header + "\n" if header else "") +
                   "energy_keV,flux_photons_per_s_per_eV,sigma_fraction")

    # -- evaluation -------------------------------------------------------------
    def flux(self, energies_kev):
        """Photons/s/eV at the given energies (0 outside the table)."""
        return np.interp(energies_kev, self.energies_kev, self.flux_per_ev, left=0.0, right=0.0)

    __call__ = flux

    def sigma(self, energies_kev):
        return np.interp(energies_kev, self.energies_kev, self.sigma_fraction)

    def total_flux(self, emin=None, emax=None) -> float:
        m = np.ones_like(self.energies_kev, bool)
        if emin is not None:
            m &= self.energies_kev >= emin
        if emax is not None:
            m &= self.energies_kev <= emax
        return float(trapezoid(self.flux_per_ev[m], self.energies_kev[m] * 1e3))

    def mean_energy(self) -> float:
        return float(trapezoid(self.flux_per_ev * self.energies_kev, self.energies_kev)
                     / trapezoid(self.flux_per_ev, self.energies_kev))

    def band(self, frac: float = 1e-3):
        """Energies where the flux per eV exceeds ``frac`` of its maximum."""
        ok = np.flatnonzero(self.flux_per_ev >= frac * self.flux_per_ev.max())
        return float(self.energies_kev[ok[0]]), float(self.energies_kev[ok[-1]])

    # -- modification -------------------------------------------------------------
    def scaled(self, factor: float) -> "IncidentSpectrum":
        """Global scale (ring current, slit-zero error, calibration factor)."""
        meta = dict(self.meta, scale=self.meta.get("scale", 1.0) * factor)
        return IncidentSpectrum(self.energies_kev, self.flux_per_ev * factor, self.sigma_fraction, meta)

    def filtered(self, formula: str, thickness_um: float, density: float) -> "IncidentSpectrum":
        """Insert an absorber (attenuator, Be dome, Kapton window...) in the beam."""
        t = Medium.from_formula(formula, density).transmission(self.energies_kev, thickness_um)
        meta = dict(self.meta)
        meta.setdefault("filters", []).append((formula, thickness_um, density))
        return IncidentSpectrum(self.energies_kev, self.flux_per_ev * t, self.sigma_fraction, meta)

    def multiplied(self, transfer: Callable, label: str = "") -> "IncidentSpectrum":
        """Multiply by an arbitrary transfer function T(E) (e.g. measured correction)."""
        meta = dict(self.meta)
        meta.setdefault("transfers", []).append(label)
        return IncidentSpectrum(self.energies_kev, self.flux_per_ev * transfer(self.energies_kev),
                                self.sigma_fraction, meta)


# ----------------------------------------------------------------------------
# Focused beam: angular convergence at the sample
# ----------------------------------------------------------------------------

@dataclass
class FocusedBeam:
    """Angular distribution of the incident rays at the focus.

    The KB mirrors image the source: each point of the illuminated mirror sends
    a ray to the focus, so the full convergence angle is (illuminated length
    projected normal to the beam) / (mirror-to-focus distance).  On BM32 the
    S3 slits define the illumination.  The angular profile is taken uniform.

    Angles in mrad, full widths; LT2 frame (beam // y, z vertical, x horizontal).
    The beam size at the focus (0.2 x 0.3 um) is negligible for the detector.
    """

    convergence_v_mrad: float = 0.56
    convergence_h_mrad: float = 1.67

    @classmethod
    def bm32(cls, vg3_mm: float = 0.2, hg3_mm: float = 0.2,
             q_v_m: float = 0.355, q_h_m: float = 0.120) -> "FocusedBeam":
        """BM32 LaueMAX KB pair: demagnification 355/45000 (V) and 120/45000 (H)."""
        return cls(vg3_mm / q_v_m, hg3_mm / q_h_m)

    def sample_directions(self, n_v: int = 7, n_h: int = 15, rng=None, jitter: bool = True):
        """Unit incident vectors (n, 3) in LT2 on a (jittered) stratified grid."""
        rng = np.random.default_rng(rng)
        uv = rng.random((n_v, n_h)) if jitter else np.full((n_v, n_h), 0.5)
        uh = rng.random((n_v, n_h)) if jitter else np.full((n_v, n_h), 0.5)
        av = (np.arange(n_v)[:, None] + uv) / n_v - 0.5
        ah = (np.arange(n_h)[None, :] + uh) / n_h - 0.5
        tv = (av * self.convergence_v_mrad * 1e-3).ravel()
        th = (ah * self.convergence_h_mrad * 1e-3).ravel()
        d = np.stack([np.tan(th), np.ones_like(tv), np.tan(tv)], axis=1)
        return d / np.linalg.norm(d, axis=1, keepdims=True)
