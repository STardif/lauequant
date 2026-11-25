"""Quantitative Laue pattern: spot list with absolute photon rates, and expected counts per pixel."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .constants import HC_KEV_A, KI_LT2, M_LT_TO_LT2
from .crystal import Crystal, enumerate_hkl, primitive
from .detector import PhotonCountingDetector
from .geometry import DetectorGeometry, SampleGeometry
from .materials import AirTable
from .scattering import (dynamical_depth_profile, dynamical_zeta, extinction_depth_um, kinematic_zeta,
                         polarization_weights)
from .spectrum import FocusedBeam, IncidentSpectrum

DEG = np.pi / 180.0


@dataclass
class LaueSimulator:
    """Forward model: spectrum x crystal x geometry x detector -> counts per pixel.

    Parameters
    ----------
    crystal : Crystal
    UB : 3x3 orientation matrix of LaueTools (q = UB B0 hkl in the LT frame),
        or in LT2 if ``ub_frame='LT2'``.
    spectrum : IncidentSpectrum (photons/s/eV)
    detector_geometry : DetectorGeometry
    sample : SampleGeometry (surface at 40 deg, thickness)
    beam : FocusedBeam (KB convergence)
    detector : PhotonCountingDetector
    model : "dynamical" (perfect crystal) or "kinematic" (mosaic crystal)
    air : include absorption in air between sample and detector
    """

    crystal: Crystal
    UB: np.ndarray
    spectrum: IncidentSpectrum
    detector_geometry: DetectorGeometry
    sample: SampleGeometry = field(default_factory=SampleGeometry)
    beam: FocusedBeam = field(default_factory=FocusedBeam.bm32)
    detector: PhotonCountingDetector = field(default_factory=PhotonCountingDetector)
    model: str = "dynamical"
    ub_frame: str = "LT"
    band_frac: float = 1e-4
    air: bool = True
    depth_model: str = "darwin"   # "darwin" (resolved over the Darwin curve), "simple" (one exponential), "none"

    def __post_init__(self):
        UB = np.asarray(self.UB, float)
        if self.ub_frame.upper() == "LT":
            UB = M_LT_TO_LT2 @ UB
        self.Rq = UB @ self.crystal.B0                  # q_LT2 = Rq @ hkl
        self.emin, self.emax = self.spectrum.band(self.band_frac)
        self._air = AirTable()
        self._refl = None

    # ------------------------------------------------------------------ reflections
    def reflections(self) -> pd.DataFrame:
        """All allowed reflections h reaching the detector, with their photon rates (photons/s)."""
        if self._refl is not None:
            return self._refl
        cr, ki = self.crystal, KI_LT2
        H = enumerate_hkl(cr, 2 * self.emax / HC_KEV_A)
        H = H[cr.allowed(H)]
        q = H @ self.Rq.T
        qq = np.sum(q ** 2, axis=1)
        qk = q @ ki
        ok = qk < 0
        H, q, qq, qk = H[ok], q[ok], qq[ok], qk[ok]
        E = HC_KEV_A * qq / (2 * -qk)
        ok = (E >= self.emin) & (E <= self.emax)
        H, q, qq, E = H[ok], q[ok], qq[ok], E[ok]
        uf = ki + q * (HC_KEV_A / E)[:, None]
        uf /= np.linalg.norm(uf, axis=1, keepdims=True)
        g0 = self.sample.gamma_in(ki)
        ge = self.sample.gamma_out(uf)
        thin = np.isfinite(self.sample.thickness_um)
        ok = ge > 1e-3 if not thin else np.abs(ge) > 1e-3   # reflection geometry (transmission: TODO)
        x, y, dist, hit = self.detector_geometry.intersect(uf)
        ok &= hit & self.detector_geometry.in_frame(x, y)
        H, q, E, uf, ge, x, y, dist = H[ok], q[ok], E[ok], uf[ok], ge[ok], x[ok], y[ok], dist[ok]
        theta = 0.5 * np.arccos(np.clip(uf @ ki, -1, 1))
        prim, nh = primitive(H)
        # structure factors and susceptibilities
        F = cr.structure_factor(H, E)
        Fb = cr.structure_factor(-H, E)
        F0 = cr.F0(E)
        mu = cr.mu(E)
        ws, wp = polarization_weights(ki, uf, self.spectrum.sigma(E))
        zeta_k = kinematic_zeta(F, E, theta, cr.volume, mu, g0, ge, ws, wp, self.sample.thickness_um)
        zeta_d = np.array([dynamical_zeta(cr.chi(F0[i], E[i]), cr.chi(F[i], E[i]), cr.chi(Fb[i], E[i]),
                                          theta[i], g0, ge[i], ws[i], wp[i]) for i in range(len(E))])
        zeta = zeta_d if self.model == "dynamical" else zeta_k
        flux = self.spectrum.flux(E)
        P = flux * E * 1e3 * zeta
        tair = self._air.transmission(E, dist) if self.air else np.ones_like(E)
        tth = 2 * theta / DEG
        chi = np.arctan(-uf[:, 0] / (uf[:, 2] + 1e-17)) / DEG
        ext = extinction_depth_um(np.abs(F), E, cr.volume, g0, ge)
        absd = 1e4 / (mu * (1 / g0 + 1 / ge))                       # 1/e depth (um, normal)
        df = pd.DataFrame(dict(h=H[:, 0], k=H[:, 1], l=H[:, 2], h0=prim[:, 0], k0=prim[:, 1], l0=prim[:, 2],
                               n=nh, E_keV=E, two_theta=tth, chi=chi, X=x, Y=y, dist_mm=dist,
                               theta_deg=theta / DEG, gamma_out=ge, d_A=cr.d_spacing(H),
                               absF=np.abs(F), flux_per_eV=flux, sigma_w=ws,
                               zeta_kin=zeta_k, zeta_dyn=zeta_d, photons_s=P, air_T=tair,
                               photons_s_at_det=P * tair, ext_depth_um=ext, abs_depth_um=absd,
                               qx=q[:, 0], qy=q[:, 1], qz=q[:, 2]))
        self._refl = df.sort_values("photons_s", ascending=False).reset_index(drop=True)
        return self._refl

    def spots(self) -> pd.DataFrame:
        """One row per Laue spot (all harmonics of a direction summed)."""
        r = self.reflections()
        g = r.groupby(["h0", "k0", "l0"], sort=False)
        s = g.agg(X=("X", "first"), Y=("Y", "first"), two_theta=("two_theta", "first"), chi=("chi", "first"),
                  photons_s=("photons_s", "sum"), photons_s_at_det=("photons_s_at_det", "sum"),
                  n_harmonics=("n", "size"))
        top = r.loc[g["photons_s"].idxmax(), ["h0", "k0", "l0", "h", "k", "l", "E_keV"]]
        top = top.rename(columns=dict(h="h_main", k="k_main", l="l_main", E_keV="E_main_keV")).set_index(["h0", "k0", "l0"])
        s = s.join(top)
        s["E_mean_keV"] = g.apply(lambda d: np.average(d.E_keV, weights=d.photons_s), include_groups=False)
        s["harmonics"] = g.apply(lambda d: ",".join(str(v) for v in sorted(d.n)), include_groups=False)
        return s.sort_values("photons_s", ascending=False).reset_index()

    # ------------------------------------------------------------------ rays for one spot
    def spot_rays(self, h0k0l0, n_v: int = 9, n_h: int = 25, n_depth: int = 4, rng=None):
        """Monte Carlo rays (x, y, E, photons/s, in-plane dir, cos incidence) for one spot.

        Each harmonic is traced for every incident direction of the convergent
        beam (energy and exit direction change with the incident direction) and
        for several scattering depths (parallax along the beam).
        """
        rng = np.random.default_rng(rng)
        r = self.reflections()
        sel = r[(r.h0 == h0k0l0[0]) & (r.k0 == h0k0l0[1]) & (r.l0 == h0k0l0[2])]
        if sel.empty:
            raise KeyError(f"spot {h0k0l0} not on the detector")
        geo, smp = self.detector_geometry, self.sample
        out = {k: [] for k in ("x", "y", "E", "w", "dirx", "diry", "cos", "n")}
        for _, row in sel.iterrows():
            q = np.array([row.qx, row.qy, row.qz])
            kis = self.beam.sample_directions(n_v, n_h, rng=rng)
            kis = np.repeat(kis, n_depth, axis=0)
            m = len(kis)
            qk = kis @ q
            E = HC_KEV_A * (q @ q) / (2 * -qk)
            uf = kis + q[None, :] * (HC_KEV_A / E)[:, None]
            uf /= np.linalg.norm(uf, axis=1, keepdims=True)
            depth = self._sample_depth(row, m, rng)
            if np.isfinite(smp.thickness_um):
                depth = np.minimum(depth, smp.thickness_um)
            src = smp.source_points(depth, kis)
            x, y, dist, hit = geo.intersect(uf, src)
            ratio = self.spectrum.flux(E) * E / (row.flux_per_eV * row.E_keV)
            w = row.photons_s * ratio / m
            if self.air:
                w = w * self._air.transmission(E, np.nan_to_num(dist))
            cosi = geo.incidence_cos(uf)
            dirx, diry = uf @ geo.ex, uf @ geo.ey
            ok = hit & geo.in_frame(x, y)
            for k, v in (("x", x), ("y", y), ("E", E), ("w", w), ("dirx", dirx), ("diry", diry),
                         ("cos", cosi), ("n", np.full(m, row.n))):
                out[k].append(v[ok])
        return {k: np.concatenate(v) for k, v in out.items()}

    def _sample_depth(self, row, m, rng):
        """Scattering depths (um, normal to the surface) for m rays of one reflection."""
        if self.depth_model == "none":
            return np.zeros(m)
        if self.model != "dynamical":          # kinematic: exp(-z / z_abs) for incident + exit paths
            return -row.abs_depth_um * np.log(rng.random(m))
        if self.depth_model == "simple":
            scale = 1.0 / (1.0 / row.abs_depth_um + 1.0 / row.ext_depth_um)
            return -scale * np.log(rng.random(m))
        key = (row.h, row.k, row.l)
        cache = self.__dict__.setdefault("_depth_cache", {})
        if key not in cache:
            cr, hkl, E = self.crystal, np.array([[row.h, row.k, row.l]]), row.E_keV
            F, Fb, F0 = cr.structure_factor(hkl, E)[0], cr.structure_factor(-hkl, E)[0], cr.F0(E)[0]
            th, g0 = np.radians(row.theta_deg), self.sample.gamma_in()
            ws, zs = [], []
            for C, wpol in ((1.0, row.sigma_w), (np.cos(2 * th), 1 - row.sigma_w)):
                w, z = dynamical_depth_profile(cr.chi(F0, E), cr.chi(F, E), cr.chi(Fb, E), th, g0,
                                               row.gamma_out, C, row.abs_depth_um, E)
                ws.append(w * wpol)
                zs.append(z)
            w, z = np.concatenate(ws), np.concatenate(zs)
            cache[key] = (np.cumsum(w) / w.sum(), z)
        cdf, z = cache[key]
        idx = np.searchsorted(cdf, rng.random(m))
        return -z[np.minimum(idx, len(z) - 1)] * np.log(rng.random(m))

    def mean_depth_um(self, row, n=20000, rng=0):
        return float(self._sample_depth(row, n, np.random.default_rng(rng)).mean())

    def spot_counts(self, h0k0l0, roi_half: int = 4, rng=None, **ray_kw):
        """Expected count rate (counts/s) in a ROI around a spot, before count-rate effects.

        Returns dict with 'roi' (2 roi_half+1 square, counts/s, lower threshold),
        'roi2' (upper threshold, if set), 'x0', 'y0' (calibration-frame coordinates of the
        ROI's first pixel), 'col0', 'row0' (the same as numpy array indices: ``img[row0:, col0:]``),
        'photons_s' (photons/s reaching the sensor) and 'counts_s' (sum of roi).
        """
        rays = self.spot_rays(h0k0l0, rng=rng, **ray_kw)
        res = self.detector.respond(rays["x"], rays["y"], rays["E"], rays["w"],
                                    np.c_[rays["dirx"], rays["diry"]], rays["cos"], rng=rng)
        xc, yc = int(np.round(np.average(rays["x"], weights=rays["w"]))), int(np.round(np.average(rays["y"], weights=rays["w"])))
        n = 2 * roi_half + 1
        roi = np.zeros((n, n))
        roi2 = np.zeros((n, n)) if "c2" in res else None
        ix, iy = res["ix"] - (xc - roi_half), res["iy"] - (yc - roi_half)
        m = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
        np.add.at(roi, (iy[m], ix[m]), res["c1"][m])
        if roi2 is not None:
            np.add.at(roi2, (iy[m], ix[m]), res["c2"][m])
        o = self.detector_geometry.pixel_origin
        return dict(roi=roi, roi2=roi2, x0=xc - roi_half, y0=yc - roi_half, col0=xc - roi_half - o,
                    row0=yc - roi_half - o, photons_s=float(rays["w"].sum()),
                    counts_s=float(roi.sum()), rays=rays)

    # ------------------------------------------------------------------ full image
    def image(self, exposure_s: float = 1.0, spots: Optional[Sequence] = None, max_spots: Optional[int] = None,
              background: Optional[np.ndarray] = None, rate_effects: bool = True, noise: bool = False,
              rng=None, **ray_kw):
        """Expected (or Poisson-sampled) counts per pixel for an exposure, as a numpy array
        (array index = calibration coordinate - ``detector_geometry.pixel_origin``).

        background : counts/s per pixel (e.g. from :func:`fluorescence_background`).
        rate_effects : apply the non-paralysable dead time to the true count rate
            (``measured``), and also return the vendor-corrected image.
        """
        rng = np.random.default_rng(rng)
        geo = self.detector_geometry
        ny, nx = geo.shape
        rate = np.zeros((ny, nx))
        sp = self.spots()
        if spots is None:
            spots = list(zip(sp.h0, sp.k0, sp.l0))
        if max_spots is not None:
            spots = spots[:max_spots]
        for s in spots:
            rays = self.spot_rays(s, rng=rng, **ray_kw)
            res = self.detector.respond(rays["x"], rays["y"], rays["E"], rays["w"],
                                        np.c_[rays["dirx"], rays["diry"]], rays["cos"], rng=rng)
            col, row = res["ix"] - geo.pixel_origin, res["iy"] - geo.pixel_origin
            m = (col >= 0) & (col < nx) & (row >= 0) & (row < ny)
            np.add.at(rate, (row[m], col[m]), res["c1"][m])
        if background is not None:
            rate = rate + background
        out = dict(true_rate=rate)
        if rate_effects:
            meas = self.detector.measured_rate(rate)
            corr, flag = self.detector.corrected_rate(rate)
            out.update(measured_rate=meas, corrected_rate=corr, saturated=flag)
        base = out.get("corrected_rate", rate)
        counts = base * exposure_s
        out["counts"] = rng.poisson(counts).astype(float) if noise else counts
        return out
