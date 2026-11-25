"""Response of a hybrid photon-counting pixel detector (EIGER2 X CdTe) to polychromatic photons.

Why a dedicated model
---------------------
Photon-counting detectors are designed and calibrated for monochromatic beams:
the threshold is set to ~50 % of the photon energy so that a photon whose
charge is shared between pixels is counted exactly once, and the flat-field and
count-rate corrections are tabulated per energy.  In Laue microdiffraction the
photons of a spot span ~5-30 keV (fundamental + harmonics) while the threshold
is set as low as possible (4 keV).  Then

* a photon absorbed near a pixel border can be counted in 2 (or 4) pixels
  (charge sharing with E * fraction > threshold), so the mean number of counts
  per photon exceeds 1 and depends on E;
* above the Cd (26.71 keV) and Te (31.81 keV) K edges, K fluorescence escapes
  the pixel and may be counted in a neighbour, or leave the sensor;
* at oblique incidence the absorption point is displaced laterally (parallax);
* the count rate saturates (retrigger dead time) on strong spots;
* the factory flat field is not valid.

This module tracks individual photons (Monte Carlo) through these processes
and returns expected counts per pixel.  Geometry: photons enter at z = 0,
pixels (electron collection) at z = T; pixel i covers [i - 0.5, i + 0.5].
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import xraylib as xrl
from scipy.special import erf

from .materials import Medium, mass_fractions, mu_rho

KT_Q = 0.02585  # V at 300 K


@dataclass
class PhotonCountingDetector:
    """Hybrid pixel detector with a high-Z sensor (default: EIGER2 X CdTe).

    Parameters
    ----------
    threshold_kev, threshold2_kev : lower / optional upper energy threshold.
    charge_sigma0_um : rms size of the charge cloud at creation (+ repulsion).
    bias_v : sensor bias, sets diffusion during drift, sigma_d^2 = 2 kT/q T d / V.
    noise_kev : rms electronic noise / threshold dispersion added to the comparison.
    dead_time_ns : non-paralysable dead time used for the instant-retrigger mode.
    rate_cutoff_cps : maximum rate the vendor count-rate correction handles.
    windows : list of (formula, thickness_um, density) in front of the sensor.
    """

    sensor_formula: str = "CdTe"
    sensor_density: float = 5.85
    thickness_um: float = 750.0
    pixel_um: float = 75.0
    threshold_kev: float = 4.0
    threshold2_kev: Optional[float] = None
    charge_sigma0_um: float = 8.0
    bias_v: float = 500.0
    noise_kev: float = 0.3
    dead_time_ns: float = 50.0
    rate_cutoff_cps: float = 2.0e7
    windows: List[Tuple[str, float, float]] = field(default_factory=list)
    fluorescence: bool = True

    def __post_init__(self):
        self._e = np.arange(1.0, 80.0, 0.01)
        frac = mass_fractions(self.sensor_formula)
        self._frac = frac
        self._mu_um = self.sensor_density * mu_rho(frac, self._e) * 1e-4        # 1/um
        tw = np.ones_like(self._e)
        for formula, t_um, rho in self.windows:
            tw *= Medium.from_formula(formula, rho).transmission(self._e, t_um)
        self._tw = tw
        # K fluorescence of each element: probability per interaction and lines
        self._kfl = []
        mu_mass = mu_rho(frac, self._e)
        for z, w in frac.items():
            edge = xrl.EdgeEnergy(z, xrl.K_SHELL)
            if edge > 80:
                continue
            tauK = np.vectorize(lambda e: xrl.CS_Photo_Partial(z, xrl.K_SHELL, e) if e > edge else 0.0)(self._e)
            pK = w * tauK / mu_mass * xrl.FluorYield(z, xrl.K_SHELL)
            lines, probs = [], []
            for ln in (xrl.KL3_LINE, xrl.KL2_LINE, xrl.KM3_LINE, xrl.KM2_LINE, xrl.KN3_LINE):
                try:
                    r = xrl.RadRate(z, ln)
                    lines.append(xrl.LineEnergy(z, ln))
                    probs.append(r)
                except ValueError:
                    pass
            probs = np.array(probs) / np.sum(probs)
            self._kfl.append(dict(Z=z, edge=edge, pK=pK, E=np.array(lines), p=probs,
                                  mu_um=self.sensor_density * mu_rho(frac, np.array(lines)) * 1e-4))
        self._sigd2 = 2 * KT_Q * self.thickness_um ** 2 / self.bias_v          # um^2 (full drift)

    # ------------------------------------------------------------------ efficiency
    def mu_um(self, e):
        return np.interp(e, self._e, self._mu_um)

    def window_transmission(self, e):
        return np.interp(e, self._e, self._tw)

    def interaction_probability(self, e, cos_inc=1.0):
        """Probability that a photon interacts in the sensor (window included)."""
        L = self.thickness_um / np.asarray(cos_inc, float)
        return self.window_transmission(e) * -np.expm1(-self.mu_um(e) * L)

    # ------------------------------------------------------------------ Monte Carlo
    def _cloud_sigma_px(self, z_um):
        drift = np.clip(self.thickness_um - z_um, 0, None)
        s2 = self.charge_sigma0_um ** 2 + self._sigd2 * drift / self.thickness_um
        return np.sqrt(s2) / self.pixel_um

    @staticmethod
    def _fractions_3x3(px, py, sig):
        """Charge fractions in the 3x3 pixels around the pixel containing (px, py)."""
        ix, iy = np.floor(px + 0.5).astype(int), np.floor(py + 0.5).astype(int)
        edges = np.array([-1.5, -0.5, 0.5, 1.5])
        ux = (ix[:, None] + edges[None, :] - px[:, None]) / (np.sqrt(2) * sig[:, None])
        uy = (iy[:, None] + edges[None, :] - py[:, None]) / (np.sqrt(2) * sig[:, None])
        cx = 0.5 * np.diff(erf(ux), axis=1)       # (n, 3)
        cy = 0.5 * np.diff(erf(uy), axis=1)
        return ix, iy, cx[:, None, :] * cy[:, :, None]   # frac[n, dy, dx]

    def respond(self, x, y, energies, weights, directions_inplane=None, cos_inc=None, rng=None):
        """Expected counts per pixel for a set of incident photons (rays).

        Parameters
        ----------
        x, y : float pixel coordinates of the impact on the sensor surface.
        energies : keV.  weights : incident photons (or photons/s) carried by each ray.
        directions_inplane : (n, 2) unit in-plane direction of the ray (pixel axes),
            used with ``cos_inc`` for the parallax inside the sensor.
        Returns
        -------
        dict with 'ix', 'iy' (int pixel indices) and 'c1' (counts, lower threshold),
        'c2' (upper threshold, if set), 'interacting' (sum of interacting photons).
        """
        rng = np.random.default_rng(rng)
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        e = np.asarray(energies, float)
        w = np.asarray(weights, float)
        n = len(e)
        c = np.ones(n) if cos_inc is None else np.clip(np.asarray(cos_inc, float), 1e-3, 1)
        mu = self.mu_um(e)
        L = self.thickness_um / c
        pint = self.interaction_probability(e, c)
        weff = w * pint
        # depth along the path, conditional on interaction
        s = -np.log1p(-rng.random(n) * -np.expm1(-mu * L)) / mu
        z = s * c
        lat = s * np.sqrt(1 - c ** 2) / self.pixel_um
        if directions_inplane is None:
            dx = dy = np.zeros(n)
        else:
            d = np.asarray(directions_inplane, float)
            dn = np.linalg.norm(d, axis=1)
            dn[dn == 0] = 1
            dx, dy = d[:, 0] / dn, d[:, 1] / dn
        px, py = x + lat * dx, y + lat * dy
        e_prim = e.copy()
        # K fluorescence
        fl_x = np.full(n, np.nan)
        fl_y = np.full(n, np.nan)
        fl_z = np.full(n, np.nan)
        fl_e = np.zeros(n)
        if self.fluorescence and self._kfl:
            u = rng.random(n)
            acc = np.zeros(n)
            for k in self._kfl:
                p = np.interp(e, self._e, k["pK"])
                sel = (u >= acc) & (u < acc + p)
                acc += p
                m = int(sel.sum())
                if m == 0:
                    continue
                li = rng.choice(len(k["E"]), size=m, p=k["p"])
                ef, muf = k["E"][li], k["mu_um"][li]
                ct = 2 * rng.random(m) - 1
                ph = 2 * np.pi * rng.random(m)
                st = np.sqrt(1 - ct ** 2)
                r = -np.log(rng.random(m)) / muf
                idx = np.flatnonzero(sel)
                fl_z[idx] = z[idx] + r * ct
                fl_x[idx] = px[idx] + r * st * np.cos(ph) / self.pixel_um
                fl_y[idx] = py[idx] + r * st * np.sin(ph) / self.pixel_um
                fl_e[idx] = ef
                e_prim[idx] = e[idx] - ef
            escaped = (fl_e > 0) & ((fl_z < 0) | (fl_z > self.thickness_um))
            fl_e[escaped] = 0.0
        # charge deposits -> per-pixel charge (keV)
        deposits = [(px, py, z, e_prim, np.arange(n))]
        has_fl = fl_e > 0
        if has_fl.any():
            deposits.append((fl_x[has_fl], fl_y[has_fl], fl_z[has_fl], fl_e[has_fl], np.flatnonzero(has_fl)))
        rid, gx, gy, q = [], [], [], []
        off = np.array([-1, 0, 1])
        for (ax, ay, az, ae, ar) in deposits:
            ix, iy, fr = self._fractions_3x3(ax, ay, self._cloud_sigma_px(az))
            GX = np.broadcast_to(ix[:, None, None] + off[None, None, :], fr.shape)
            GY = np.broadcast_to(iy[:, None, None] + off[None, :, None], fr.shape)
            rid.append(np.broadcast_to(ar[:, None, None], fr.shape).ravel())
            gx.append(GX.ravel())
            gy.append(GY.ravel())
            q.append((ae[:, None, None] * fr).ravel())
        rid, gx, gy, q = map(np.concatenate, (rid, gx, gy, q))
        # sum charges of the same ray in the same pixel (primary + fluorescence)
        key = (rid.astype(np.int64) * 100_001 + (gx + 50_000)) * 100_001 + (gy + 50_000)
        uk, inv = np.unique(key, return_inverse=True)
        qsum = np.bincount(inv, weights=q)
        first = np.zeros(len(uk), int)
        first[inv[::-1]] = np.arange(len(inv))[::-1]
        r_u, x_u, y_u = rid[first], gx[first], gy[first]
        if self.noise_kev > 0:
            qsum = qsum + rng.normal(0, self.noise_kev, len(qsum))
        hit1 = qsum > self.threshold_kev
        out = {}
        pix_key = (x_u.astype(np.int64) + 50_000) * 200_000 + (y_u + 50_000)
        pk, pinv = np.unique(pix_key, return_inverse=True)
        out["ix"] = pk // 200_000 - 50_000
        out["iy"] = pk % 200_000 - 50_000
        out["c1"] = np.bincount(pinv, weights=weff[r_u] * hit1, minlength=len(pk))
        if self.threshold2_kev is not None:
            hit2 = qsum > self.threshold2_kev
            out["c2"] = np.bincount(pinv, weights=weff[r_u] * hit2, minlength=len(pk))
        out["interacting"] = float(weff.sum())
        out["incident"] = float(w.sum())
        return out

    def flat_response(self, energies, n: int = 20000, rng=0, cos_inc=1.0):
        """Mean counts per incident photon under uniform illumination (QE x multiplicity)."""
        rng = np.random.default_rng(rng)
        res = []
        for e in np.atleast_1d(energies):
            x = rng.random(n) - 0.5
            y = rng.random(n) - 0.5
            r = self.respond(x, y, np.full(n, e), np.full(n, 1.0 / n),
                             cos_inc=np.full(n, cos_inc), rng=rng)
            res.append(r["c1"].sum())
        return np.array(res)

    def multiplicity(self, energies, n: int = 20000, rng=0):
        """Mean number of counts per *interacting* photon under uniform illumination."""
        return self.flat_response(energies, n, rng) / self.interaction_probability(np.atleast_1d(energies))

    # ------------------------------------------------------------------ count rate
    def measured_rate(self, true_rate_cps):
        """Non-paralysable dead-time model of the instant-retrigger mode: m = n / (1 + n tau)."""
        n = np.asarray(true_rate_cps, float)
        return n / (1 + n * self.dead_time_ns * 1e-9)

    def corrected_rate(self, true_rate_cps):
        """What the image shows if the vendor count-rate correction is on:
        the true rate up to the cutoff, flagged (returned as the cutoff) above it."""
        n = np.asarray(true_rate_cps, float)
        return np.minimum(n, self.rate_cutoff_cps), n > self.rate_cutoff_cps
