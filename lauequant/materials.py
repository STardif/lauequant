"""Thin xraylib helpers: attenuation of compounds, cached form factors."""
from __future__ import annotations

from functools import lru_cache
from typing import Dict

import numpy as np
import xraylib as xrl

from .constants import AIR_DENSITY, AIR_MASS_FRACTIONS


def z_of(symbol) -> int:
    return symbol if isinstance(symbol, (int, np.integer)) else xrl.SymbolToAtomicNumber(symbol)


def mass_fractions(formula: str) -> Dict[int, float]:
    """Mass fractions {Z: w} of a chemical formula (xraylib parser)."""
    cd = xrl.CompoundParser(formula)
    return {int(z): float(w) for z, w in zip(cd["Elements"], cd["massFractions"])}


def _vec(f, *args):
    return np.vectorize(f, otypes=[float])(*args)


def mu_rho(fractions: Dict[int, float], energies_kev, kind: str = "total") -> np.ndarray:
    """Mass attenuation coefficient (cm2/g) of a mixture given as {Z: w}.

    kind: "total" (photo + Rayleigh + Compton), "photo".
    """
    e = np.atleast_1d(np.asarray(energies_kev, dtype=float))
    fn = {"total": xrl.CS_Total, "photo": xrl.CS_Photo}[kind]
    out = np.zeros_like(e)
    for z, w in fractions.items():
        out += w * _vec(fn, int(z), e)
    return out


class Medium:
    """A homogeneous absorber: mass fractions + density."""

    def __init__(self, fractions: Dict[int, float], density: float, label: str = ""):
        self.fractions = dict(fractions)
        self.density = float(density)
        self.label = label

    @classmethod
    def from_formula(cls, formula: str, density: float):
        return cls(mass_fractions(formula), density, formula)

    @classmethod
    def air(cls, density: float = AIR_DENSITY):
        return cls({z_of(s): w for s, w in AIR_MASS_FRACTIONS.items()}, density, "air")

    def mu(self, energies_kev) -> np.ndarray:
        """Linear attenuation coefficient in 1/cm."""
        return self.density * mu_rho(self.fractions, energies_kev)

    def transmission(self, energies_kev, path_um) -> np.ndarray:
        return np.exp(-self.mu(energies_kev) * np.asarray(path_um) * 1e-4)


class AirTable:
    """Fast interpolated air attenuation (1/mm) on 1-60 keV."""

    def __init__(self, density: float = AIR_DENSITY):
        self.e = np.linspace(1.0, 60.0, 2361)
        self.mu_mm = Medium.air(density).mu(self.e) * 0.1

    def transmission(self, energies_kev, path_mm):
        return np.exp(-np.interp(energies_kev, self.e, self.mu_mm) * path_mm)


@lru_cache(maxsize=64)
def f0_table(z: int, smax: float = 3.0, n: int = 1201):
    """Tabulated non-resonant form factor f0(s), s = sin(theta)/lambda (1/A)."""
    s = np.linspace(0.0, smax, n)
    return s, _vec(xrl.FF_Rayl, z, s)


@lru_cache(maxsize=64)
def anomalous_table(z: int, emin: float = 2.0, emax: float = 80.0, step: float = 0.002):
    """Tabulated f'(E), f''(E) (xraylib convention, f'' <= 0)."""
    e = np.arange(emin, emax + step, step)
    return e, _vec(xrl.Fi, z, e), _vec(xrl.Fii, z, e)


def f0(z: int, s):
    sg, f = f0_table(int(z))
    return np.interp(s, sg, f)


def f_anom(z: int, energies_kev):
    e, fp, fpp = anomalous_table(int(z))
    return np.interp(energies_kev, e, fp), np.interp(energies_kev, e, fpp)
