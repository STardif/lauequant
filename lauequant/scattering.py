r"""Energy-integrated Laue reflection power: kinematic (mosaic) and dynamical (perfect crystal).

For a stationary crystal in a white beam, each reflection h selects a narrow
band of wavelengths.  With :math:`\Phi_E` the incident flux per unit energy
(photons/s/eV), the diffracted photon rate is

.. math::  P_h = \Phi_E(E_h)\, E_h\, \zeta_h,
           \qquad \zeta_h = \int R(\lambda)\, d\lambda/\lambda = \cot\theta \int R\, d\theta_{in}

(at fixed incidence :math:`\delta\lambda/\lambda = \cot\theta\,\delta\theta`).

* Kinematic, absorbing slab (ideally imperfect / mosaic crystal):
  :math:`\int R d\theta = Q\, t_{eff}`, :math:`Q = r_e^2\lambda^3|F|^2 P/(V^2\sin 2\theta)`,
  :math:`t_{eff} = (1/\gamma_0)\,[1-e^{-\mu T A}]/(\mu A)`, :math:`A = 1/\gamma_0+1/\gamma_h`.
  This gives the classical Laue formula
  :math:`P_h = \Phi_E E\, r_e^2 \lambda^3 |F|^2 P\, t_{eff}/(2 V^2 \sin^2\theta)`.
* Dynamical, semi-infinite perfect crystal, Bragg case, with absorption
  (Darwin-Prins / Takagi-Taupin thick-crystal solution), integrated numerically:
  :math:`R(\eta)=|\chi_h/\chi_{\bar h}|\,|\eta\pm\sqrt{\eta^2-1}|^2`,
  :math:`\eta = [b\Delta\theta\sin 2\theta + (1-b)\chi_0/2]/(|C|\sqrt{|b|}\sqrt{\chi_h\chi_{\bar h}})`,
  :math:`b=-\gamma_0/\gamma_h`.  Without absorption the integral is
  :math:`(8/3)|C||\chi_h|/(\sqrt{|b|}\sin 2\theta)`.

Polarisation: the beam is an incoherent mixture of linear polarisations along
LT2 x (fraction ``p``) and z; each is split into sigma/pi components of the
scattering plane (C = 1 and C = cos 2theta).
"""
from __future__ import annotations

import numpy as np

from .constants import HC_KEV_A, R_E_A


def polarization_weights(ki, uf, sigma_fraction):
    """Weights (w_sigma, w_pi) of the incident intensity relative to the scattering plane.

    ki, uf : (..., 3) unit vectors; sigma_fraction : fraction polarised along x (LT2).
    """
    ki = np.asarray(ki, float)
    uf = np.asarray(uf, float)
    n = np.cross(ki, uf)
    nn = np.linalg.norm(n, axis=-1, keepdims=True)
    n = np.where(nn > 1e-12, n / np.where(nn > 0, nn, 1), np.array([1.0, 0, 0]))
    p = np.asarray(sigma_fraction, float)
    ws = p * n[..., 0] ** 2 + (1 - p) * n[..., 2] ** 2
    return ws, 1.0 - ws


def kinematic_zeta(F, energy_kev, theta, volume, mu_cm, gamma0, gammah, w_sigma, w_pi,
                   thickness_um=np.inf):
    """Kinematic integrated reflectivity in d(lambda)/lambda (dimensionless)."""
    lam = HC_KEV_A / np.asarray(energy_kev, float)
    c2 = np.cos(2 * theta)
    P = w_sigma + w_pi * c2 ** 2
    Q = R_E_A ** 2 * lam ** 3 * np.abs(F) ** 2 * P / (volume ** 2 * np.sin(2 * theta))   # 1/A
    mu_A = np.asarray(mu_cm, float) * 1e-8
    A = 1.0 / gamma0 + 1.0 / gammah
    T = np.asarray(thickness_um, float) * 1e4
    with np.errstate(over="ignore", invalid="ignore"):
        att = np.where(np.isfinite(T), -np.expm1(-mu_A * T * A), 1.0)
    teff = att / (gamma0 * mu_A * A)                                                    # A
    return Q * teff / np.tan(theta)


def _eta_grid(n_core=2401, core=8.0, tail=2.0e4, n_tail=600):
    u_core = np.linspace(-core, core, n_core)
    u_tail = np.geomspace(core, tail, n_tail)[1:]
    return np.concatenate([-u_tail[::-1], u_core, u_tail])


_U = _eta_grid()


def dynamical_theta_integral(chi0, chih, chihb, theta, gamma0, gammah, C):
    """Integral of the thick-crystal Bragg reflectivity over the incident angle (rad)."""
    s2 = np.sin(2 * theta)
    b = -gamma0 / gammah
    sq = np.sqrt(chih * chihb + 0j)
    if abs(sq) == 0 or C == 0:
        return 0.0
    w = abs(C) * abs(sq) / (np.sqrt(abs(b)) * s2)          # half width of total reflection
    center = -0.5 * (1 - b) * chi0.real / (b * s2)
    dth = center + w * _U
    eta = (b * dth * s2 + 0.5 * (1 - b) * chi0) / (abs(C) * np.sqrt(abs(b)) * sq)
    root = np.sqrt(eta * eta - 1 + 0j)
    r1, r2 = eta + root, eta - root
    r = np.where(np.abs(r1) < np.abs(r2), r1, r2)
    R = np.abs(chih / chihb) * np.abs(r) ** 2
    integ = np.trapezoid(R, dth) if hasattr(np, "trapezoid") else np.trapz(R, dth)
    return float(integ + w / (2 * _U[-1]))                 # analytic 1/(4 u^2) tails


def dynamical_zeta(chi0, chih, chihb, theta, gamma0, gammah, w_sigma, w_pi):
    """Dynamical integrated reflectivity in d(lambda)/lambda, sigma and pi weighted."""
    cot = 1.0 / np.tan(theta)
    zs = dynamical_theta_integral(chi0, chih, chihb, theta, gamma0, gammah, 1.0) * cot
    zp = dynamical_theta_integral(chi0, chih, chihb, theta, gamma0, gammah, np.cos(2 * theta)) * cot
    return w_sigma * zs + w_pi * zp


def darwin_zeta_nonabsorbing(F_abs, energy_kev, theta, volume, gamma0=1.0, gammah=1.0, C=1.0):
    """Closed form (8/3)|C||chi_h|/(sqrt|b| sin 2theta) * cot(theta), for tests."""
    lam = HC_KEV_A / energy_kev
    chih = R_E_A * lam ** 2 * F_abs / (np.pi * volume)
    b = gamma0 / gammah
    return 8.0 / 3.0 * abs(C) * chih / (np.sqrt(b) * np.sin(2 * theta)) / np.tan(theta)


def extinction_depth_um(F_abs, energy_kev, volume, gamma0, gammah, C=1.0):
    """Extinction depth (normal to the surface) Lambda_0 / (2 pi), in um."""
    lam = HC_KEV_A / energy_kev
    lam0 = np.pi * volume * np.sqrt(gamma0 * gammah) / (R_E_A * lam * np.abs(C) * F_abs)  # A
    return lam0 / (2 * np.pi) * 1e-4


def dynamical_depth_profile(chi0, chih, chihb, theta, gamma0, gammah, C, abs_depth_um, energy_kev):
    """Depth sensitivity of the thick-crystal Bragg reflection, resolved over the Darwin curve.

    Returns (weights, depth_um): for each point of the deviation grid, the reflected
    power weight R(dtheta) d(dtheta) and the 1/e depth (normal to the surface) of the
    wavefield intensity, approximated by the sum of rates
        1/z(eta) = 1/z_abs + (2 pi / Lambda_0) Im sqrt(eta^2 - 1),
    Lambda_0 = lambda sqrt(gamma0 gammah) / (|C| |sqrt(chi_h chi_hbar)|).
    Inside the total-reflection plateau (|eta| < 1) the depth is limited by extinction
    (Lambda_0/2pi at the centre); in the tails (|eta| > 1, ~1/4 of the integrated
    reflectivity) only by absorption, as in the kinematic limit.
    """
    s2 = np.sin(2 * theta)
    b = -gamma0 / gammah
    sq = np.sqrt(chih * chihb + 0j)
    if abs(sq) == 0 or C == 0:
        return np.zeros(1), np.full(1, abs_depth_um)
    w = abs(C) * abs(sq) / (np.sqrt(abs(b)) * s2)
    center = -0.5 * (1 - b) * chi0.real / (b * s2)
    dth = center + w * _U
    eta = (b * dth * s2 + 0.5 * (1 - b) * chi0) / (abs(C) * np.sqrt(abs(b)) * sq)
    root = np.sqrt(eta * eta - 1 + 0j)
    r1, r2 = eta + root, eta - root
    R = np.abs(chih / chihb) * np.abs(np.where(np.abs(r1) < np.abs(r2), r1, r2)) ** 2
    eta_r = (b * dth * s2 + 0.5 * (1 - b) * chi0.real) / (abs(C) * np.sqrt(abs(b)) * abs(sq))
    lam_um = HC_KEV_A / energy_kev * 1e-4
    lambda0_um = lam_um * np.sqrt(gamma0 * gammah) / (abs(C) * abs(sq))
    rate = 1.0 / abs_depth_um + 2 * np.pi / lambda0_um * np.abs(np.sqrt(eta_r ** 2 - 1 + 0j).imag)
    weights = R * np.gradient(dth)
    return weights, 1.0 / rate
