"""Crystal structure: lattice, B0 matrix (LaueTools convention), structure factors, absorption."""
from __future__ import annotations

from dataclasses import dataclass, field
from math import gcd
from typing import List, Sequence

import numpy as np
import xraylib as xrl

from .constants import HC_KEV_A, N_A, R_E_A
from .materials import f0, f_anom, mu_rho, z_of

DEG = np.pi / 180.0


@dataclass
class Atom:
    element: str
    xyz: Sequence[float]
    occupancy: float = 1.0
    B_iso: float = 0.0  # Debye-Waller B (A^2), f -> f exp(-B s^2)

    @property
    def Z(self) -> int:
        return z_of(self.element)


@dataclass
class Crystal:
    """Unit cell + atoms in fractional coordinates (full cell, no symmetry expansion)."""

    name: str
    lattice: Sequence[float]  # a, b, c (A), alpha, beta, gamma (deg)
    atoms: List[Atom] = field(default_factory=list)

    # -- lattice -----------------------------------------------------------------
    @property
    def direct_matrix(self):
        """Columns a, b, c in an orthonormal frame (a // x)."""
        a, b, c, al, be, ga = self.lattice
        al, be, ga = al * DEG, be * DEG, ga * DEG
        ax = np.array([a, 0, 0])
        bx = np.array([b * np.cos(ga), b * np.sin(ga), 0])
        cx = c * np.cos(be)
        cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
        cz = np.sqrt(c ** 2 - cx ** 2 - cy ** 2)
        return np.c_[ax, bx, np.array([cx, cy, cz])]

    @property
    def volume(self) -> float:
        return float(abs(np.linalg.det(self.direct_matrix)))

    @property
    def B0(self):
        """LaueTools B0 (Busing-Levy, upper triangular): q = U B0 [h k l], 1/A without 2 pi."""
        a, b, c, al, be, ga = self.lattice
        al, be, ga = al * DEG, be * DEG, ga * DEG
        V = self.volume
        ast, bst, cst = b * c * np.sin(al) / V, a * c * np.sin(be) / V, a * b * np.sin(ga) / V
        cosal_s = (np.cos(be) * np.cos(ga) - np.cos(al)) / (np.sin(be) * np.sin(ga))
        cosbe_s = (np.cos(al) * np.cos(ga) - np.cos(be)) / (np.sin(al) * np.sin(ga))
        cosga_s = (np.cos(al) * np.cos(be) - np.cos(ga)) / (np.sin(al) * np.sin(be))
        sinbe_s, singa_s = np.sqrt(1 - cosbe_s ** 2), np.sqrt(1 - cosga_s ** 2)
        return np.array([[ast, bst * cosga_s, cst * cosbe_s],
                         [0.0, bst * singa_s, -cst * sinbe_s * np.cos(al)],
                         [0.0, 0.0, 1.0 / c]])

    def d_spacing(self, hkl):
        q = np.atleast_2d(np.asarray(hkl, float)) @ self.B0.T
        return 1.0 / np.linalg.norm(q, axis=1)

    # -- composition ---------------------------------------------------------------
    @property
    def mass_fractions(self):
        m = {}
        for at in self.atoms:
            m[at.Z] = m.get(at.Z, 0.0) + at.occupancy * xrl.AtomicWeight(at.Z)
        tot = sum(m.values())
        return {z: v / tot for z, v in m.items()}

    @property
    def density(self) -> float:
        """g/cm3."""
        mass = sum(at.occupancy * xrl.AtomicWeight(at.Z) for at in self.atoms)
        return mass / (N_A * self.volume * 1e-24)

    def mu(self, energies_kev, kind: str = "total"):
        """Linear attenuation coefficient (1/cm)."""
        return self.density * mu_rho(self.mass_fractions, energies_kev, kind)

    # -- structure factor -------------------------------------------------------------
    def geometric_factor(self, hkl):
        """sum_j exp(2 pi i h.r_j) with unit scatterers (systematic absences -> 0)."""
        hkl = np.atleast_2d(np.asarray(hkl, float))
        xyz = np.array([a.xyz for a in self.atoms], float)
        return np.exp(2j * np.pi * hkl @ xyz.T).sum(axis=1)

    def structure_factor(self, hkl, energies_kev, anomalous: bool = True, debye_waller: bool = True):
        """F(hkl, E) in electrons, xraylib convention f = f0 + f' + i f'' (f'' <= 0).

        hkl : (n, 3); energies : scalar or (n,).
        """
        hkl = np.atleast_2d(np.asarray(hkl, float))
        e = np.broadcast_to(np.asarray(energies_kev, float), (len(hkl),))
        s = 0.5 / self.d_spacing(hkl)                     # sin(theta)/lambda
        F = np.zeros(len(hkl), complex)
        for at in self.atoms:
            f = f0(at.Z, s).astype(complex)
            if anomalous:
                fp, fpp = f_anom(at.Z, e)
                f = f + fp + 1j * fpp
            if debye_waller and at.B_iso:
                f = f * np.exp(-at.B_iso * s ** 2)
            F += at.occupancy * f * np.exp(2j * np.pi * hkl @ np.asarray(at.xyz, float))
        return F

    def F0(self, energies_kev):
        """Forward structure factor F(000, E) (complex)."""
        e = np.atleast_1d(np.asarray(energies_kev, float))
        F = np.zeros(len(e), complex)
        for at in self.atoms:
            fp, fpp = f_anom(at.Z, e)
            F += at.occupancy * (at.Z + fp + 1j * fpp)
        return F

    def chi(self, F, energies_kev):
        """Fourier coefficient of the susceptibility: chi = -r_e lambda^2 F / (pi V)."""
        lam = HC_KEV_A / np.asarray(energies_kev, float)
        return -R_E_A * lam ** 2 * F / (np.pi * self.volume)

    # -- reflections ----------------------------------------------------------------------
    def allowed(self, hkl, tol: float = 1e-6):
        return np.abs(self.geometric_factor(hkl)) > tol * len(self.atoms)

    # -- factories ------------------------------------------------------------------------
    @classmethod
    def germanium(cls, a: float = 5.6575, B_iso: float = 0.565) -> "Crystal":
        """Diamond-structure Ge. a = 5.6575 A as in LaueTools' dict_Materials
        (5.6579 A at 298 K); B_iso = 0.565 A2 at room temperature."""
        base = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
        pos = base + [(x + 0.25, y + 0.25, z + 0.25) for x, y, z in base]
        return cls("Ge", [a, a, a, 90, 90, 90], [Atom("Ge", p, 1.0, B_iso) for p in pos])

    @classmethod
    def silicon(cls, a: float = 5.4309, B_iso: float = 0.46) -> "Crystal":
        c = cls.germanium(a, B_iso)
        return cls("Si", c.lattice, [Atom("Si", at.xyz, 1.0, B_iso) for at in c.atoms])


def enumerate_hkl(crystal: Crystal, qmax: float):
    """All non-zero hkl with |q| = 1/d <= qmax (1/A), as an (n, 3) int array."""
    B = crystal.B0
    Binv = np.linalg.inv(B)          # h = Binv q  ->  |h_i| <= |Binv[i]| qmax
    hmax = [int(np.ceil(qmax * np.linalg.norm(Binv[i]))) for i in range(3)]
    rng = [np.arange(-m, m + 1) for m in hmax]
    H = np.stack(np.meshgrid(*rng, indexing="ij"), -1).reshape(-1, 3)
    H = H[np.any(H != 0, axis=1)]
    q = np.linalg.norm(H @ B.T, axis=1)
    return H[q <= qmax]


def primitive(hkl):
    """Split integer hkl into (primitive direction, harmonic order n = gcd)."""
    hkl = np.asarray(hkl, int)
    n = np.gcd.reduce(np.abs(hkl), axis=1)
    return hkl // n[:, None], n
